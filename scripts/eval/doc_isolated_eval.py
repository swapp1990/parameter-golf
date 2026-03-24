"""
Document-isolated eval: score each document independently, starting from its BOS token.
Compare against standard non-overlapping 2048-window eval.

Quick estimate: first 500 documents (~5 min on CPU).
"""
import torch, os, sys, time, math, json
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import model from layer_xray (handles SmearGate, BigramHash)
# But we need 11L + SwiGLU + XSA — need to build manually
print('=== Document-Isolated Eval (CPU) ===')

# Load val tokens
print('Loading val tokens...')
header_bytes = 256 * np.dtype("<i4").itemsize
tokens_np = np.fromfile('data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin',
                        dtype="<u2", offset=header_bytes)
val_tokens = torch.from_numpy(tokens_np.astype(np.int64))
print(f'Val tokens: {val_tokens.numel()}')

# Find document boundaries
BOS_ID = 1
bos_positions = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
print(f'Documents: {len(bos_positions)}')

# Build model (11L SwiGLU + SmearGate + BigramHash + XSA)
print('Loading model...')
os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'

from dashboard.layer_xray import GPT as BaseGPT

# We need a model that matches the Exp 17 architecture
# layer_xray.py GPT has SmearGate + BigramHash but not SwiGLU or XSA
# Load state dict and let strict=False handle mismatches
# Actually, let's build it properly

import torch.nn as nn

class SwiGLU_MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(2 * mlp_mult * dim / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.up(x))


# Monkey-patch the MLP class in layer_xray
import dashboard.layer_xray as lx
original_MLP = lx.MLP
lx.MLP = SwiGLU_MLP

# Now create the model
model = lx.GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8,
               num_kv_heads=4, mlp_mult=3, tie_embeddings=True,
               tied_embed_init_std=0.02, logit_softcap=30.0,
               rope_base=10000.0, qk_gain_init=1.5)

# Load checkpoint
sd = torch.load('dashboard/checkpoints/exp17_xsa/final_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Loaded: {sum(p.numel() for p in model.parameters())} params')

# Forward function (no XSA on CPU — minor difference, XSA only adds ~0.005)
def get_per_token_loss(model, token_ids):
    """Get per-token loss for a sequence. token_ids shape: (seq_len,)"""
    x = token_ids.unsqueeze(0)  # (1, seq_len)
    target = token_ids[1:].unsqueeze(0)  # (1, seq_len-1)
    input_ids = token_ids[:-1].unsqueeze(0)  # (1, seq_len-1)

    with torch.no_grad():
        # Manual forward
        xemb = model.tok_emb(input_ids)
        if hasattr(model, 'bigram'):
            xemb = xemb + model.bigram(input_ids)
        xemb = F.rms_norm(xemb, (xemb.size(-1),))
        if hasattr(model, 'smear'):
            xemb = model.smear(xemb)
        x0 = xemb
        skips = []
        for i in range(model.num_encoder_layers):
            xemb = model.blocks[i](xemb, x0)
            skips.append(xemb)
        for i in range(model.num_decoder_layers):
            if skips:
                xemb = xemb + model.skip_weights[i].to(dtype=xemb.dtype)[None, None, :] * skips.pop()
            xemb = model.blocks[model.num_encoder_layers + i](xemb, x0)
        xemb = model.final_norm(xemb)
        logits = F.linear(xemb, model.tok_emb.weight)
        logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

        per_token = F.cross_entropy(logits[0], target[0], reduction='none')
    return per_token


# === Standard eval (non-overlapping 2048 windows, first 500 docs worth) ===
MAX_DOCS = 500
last_doc_end = bos_positions[MAX_DOCS] if MAX_DOCS < len(bos_positions) else val_tokens.numel()
n_standard_tokens = (last_doc_end // 2048) * 2048

print(f'\nEvaluating first {MAX_DOCS} documents (~{last_doc_end} tokens)')
print(f'\n--- Standard eval (2048 non-overlapping windows) ---')
t0 = time.time()
standard_nll = 0.0
standard_count = 0
n_windows = n_standard_tokens // 2048
for w in range(n_windows):
    start = w * 2048
    chunk = val_tokens[start:start + 2048]
    losses = get_per_token_loss(model, chunk)
    standard_nll += losses.sum().item()
    standard_count += losses.numel()
    if (w + 1) % 5 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (w + 1) * (n_windows - w - 1)
        print(f'  Window {w+1}/{n_windows}, elapsed={elapsed:.0f}s, eta={eta:.0f}s')

standard_loss = standard_nll / standard_count
print(f'Standard: val_loss={standard_loss:.4f} tokens={standard_count} time={time.time()-t0:.0f}s')

# === Document-isolated eval ===
print(f'\n--- Document-isolated eval (each doc scored independently) ---')
t1 = time.time()
doc_isolated_nll = 0.0
doc_isolated_count = 0
doc_losses_by_position = {}  # position_bucket -> (total_loss, count)

for d in range(min(MAX_DOCS, len(bos_positions))):
    doc_start = bos_positions[d]
    doc_end = bos_positions[d + 1] if d + 1 < len(bos_positions) else val_tokens.numel()
    doc_tokens = val_tokens[doc_start:doc_end]
    doc_len = doc_tokens.numel()

    if doc_len < 3:
        continue

    # Score in chunks of 2048 (or less for short docs)
    chunk_size = min(2048, doc_len)
    chunk = doc_tokens[:chunk_size]
    losses = get_per_token_loss(model, chunk)
    doc_isolated_nll += losses.sum().item()
    doc_isolated_count += losses.numel()

    # Track loss by position within document
    for i, loss_val in enumerate(losses):
        bucket = (i // 50) * 50  # 0, 50, 100, 150, ...
        if bucket not in doc_losses_by_position:
            doc_losses_by_position[bucket] = [0.0, 0]
        doc_losses_by_position[bucket][0] += loss_val.item()
        doc_losses_by_position[bucket][1] += 1

    if (d + 1) % 50 == 0:
        elapsed = time.time() - t1
        eta = elapsed / (d + 1) * (MAX_DOCS - d - 1)
        print(f'  Doc {d+1}/{MAX_DOCS}, elapsed={elapsed:.0f}s, eta={eta:.0f}s')

doc_isolated_loss = doc_isolated_nll / doc_isolated_count
print(f'Doc-isolated: val_loss={doc_isolated_loss:.4f} tokens={doc_isolated_count} time={time.time()-t1:.0f}s')

# === Comparison ===
print(f'\n=== Results ===')
print(f'Standard (2048 windows):  val_loss={standard_loss:.4f}')
print(f'Document-isolated:        val_loss={doc_isolated_loss:.4f}')
print(f'Delta:                    {doc_isolated_loss - standard_loss:+.4f}')

# === Loss by position within document ===
print(f'\n--- Loss by position within document ---')
for bucket in sorted(doc_losses_by_position.keys()):
    total, count = doc_losses_by_position[bucket]
    avg = total / count if count > 0 else 0
    bar = '#' * int(avg * 5)
    print(f'  Position {bucket:>4d}-{bucket+50:>4d}: avg_loss={avg:.3f} n={count:>6d} {bar}')

# Save results
results = {
    'standard_loss': float(standard_loss),
    'doc_isolated_loss': float(doc_isolated_loss),
    'delta': float(doc_isolated_loss - standard_loss),
    'docs_evaluated': int(min(MAX_DOCS, len(bos_positions))),
    'standard_tokens': int(standard_count),
    'doc_isolated_tokens': int(doc_isolated_count),
    'loss_by_doc_position': {
        str(k): {'avg_loss': v[0]/v[1], 'count': v[1]}
        for k, v in sorted(doc_losses_by_position.items())
    },
}

with open('dashboard/doc_isolated_eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to dashboard/doc_isolated_eval_results.json')
print('=== Done ===')
