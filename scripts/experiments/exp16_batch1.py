"""
Batch 1: Eval-only experiments on Exp 15 checkpoint.
No training. Just forward passes with different configurations.

Run on pod: python exp16_batch1.py
"""
import torch, os, sys, time, glob, math
import torch.nn.functional as F
from pathlib import Path

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard

SEQ_LEN = 2048
EVAL_WINDOWS = 300  # enough for reliable comparison


def load_model():
    model = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
                mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
                logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
    sd = torch.load('final_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


def forward_logits(model, x):
    xemb = model.tok_emb(x)
    if hasattr(model, 'bigram'):
        xemb = xemb + model.bigram(x)
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
    return model.logit_softcap * torch.tanh(logits / model.logit_softcap)


def eval_loss(model, val_tokens, windows=EVAL_WINDOWS):
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for w in range(windows):
            start = w * SEQ_LEN
            x = val_tokens[start:start + SEQ_LEN].unsqueeze(0)
            y = val_tokens[start + 1:start + SEQ_LEN + 1].unsqueeze(0)
            loss = model(x, y)
            total_nll += loss.item() * SEQ_LEN
            total_tok += SEQ_LEN
    return total_nll / total_tok


print('=' * 60)
print('Batch 1: Eval-Only Experiments')
print('=' * 60)

model = load_model()
print(f'Model loaded: {sum(p.numel() for p in model.parameters())} params')

val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
print(f'Val tokens: {val_tokens.numel()}')

# === Baseline ===
print('\n--- Baseline (standard eval, seq=2048) ---')
t0 = time.time()
base_loss = eval_loss(model, val_tokens)
print(f'Baseline val_loss={base_loss:.6f} time={time.time()-t0:.0f}s')

# === Exp 2: Sliding window stride=64 ===
print('\n--- Sliding window stride=64, seq=2048 ---')
stride = 64
batch_size = 4
total_nll_sw = 0.0
total_scored_sw = 0
starts = list(range(0, val_tokens.numel() - SEQ_LEN, stride))
# Limit to ~100K windows for time (full would be ~960K)
max_windows = min(len(starts), 100000)
starts = starts[:max_windows]
t1 = time.time()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for batch_off in range(0, len(starts), batch_size):
        batch_starts = starts[batch_off:batch_off + batch_size]
        x = torch.stack([val_tokens[s:s + SEQ_LEN] for s in batch_starts])
        y = torch.stack([val_tokens[s + 1:s + SEQ_LEN + 1] for s in batch_starts])
        logits = forward_logits(model, x)
        sf = SEQ_LEN - stride
        nll = F.cross_entropy(logits[:, sf:].reshape(-1, 1024).float(),
                              y[:, sf:].reshape(-1), reduction='sum')
        total_nll_sw += nll.item()
        total_scored_sw += len(batch_starts) * stride
        done = batch_off + batch_size
        if done % 5000 < batch_size:
            pct = done / len(starts) * 100
            print(f'  {pct:.0f}% ({done}/{len(starts)})')
sw_loss = total_nll_sw / total_scored_sw
print(f'Sliding(stride=64) val_loss={sw_loss:.6f} delta={sw_loss-base_loss:.6f} time={time.time()-t1:.0f}s')

# === Exp 7: Head-level ablation ===
print('\n--- Head ablation (zero each head individually) ---')
t2 = time.time()
abl_windows = 100
head_dim = 512 // 8  # 64

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    # Baseline for ablation
    abl_base = 0.0
    for w in range(abl_windows):
        start = w * SEQ_LEN
        x = val_tokens[start:start + SEQ_LEN].unsqueeze(0)
        y = val_tokens[start + 1:start + SEQ_LEN + 1].unsqueeze(0)
        loss = model(x, y)
        abl_base += loss.item() * SEQ_LEN
    abl_base /= abl_windows * SEQ_LEN

    for layer_idx in range(11):
        for head_idx in range(8):
            # Zero out this head's Q output contribution
            saved = model.blocks[layer_idx].attn.q_gain.data[head_idx].clone()
            model.blocks[layer_idx].attn.q_gain.data[head_idx] = 0.0

            abl_nll = 0.0
            for w in range(abl_windows):
                start = w * SEQ_LEN
                x = val_tokens[start:start + SEQ_LEN].unsqueeze(0)
                y = val_tokens[start + 1:start + SEQ_LEN + 1].unsqueeze(0)
                loss = model(x, y)
                abl_nll += loss.item() * SEQ_LEN
            abl_loss = abl_nll / (abl_windows * SEQ_LEN)
            impact = abl_loss - abl_base

            model.blocks[layer_idx].attn.q_gain.data[head_idx] = saved

            if impact > 0.01 or impact < -0.01:  # only print notable heads
                print(f'  L{layer_idx:2d} H{head_idx}: impact={impact:+.4f}')

print(f'Head ablation time={time.time()-t2:.0f}s')

# === Exp 11: MLP removal from L5-L7 ===
print('\n--- MLP removal: L5, L6, L7 ---')
for remove_layers in [[5], [6], [7], [5, 6, 7]]:
    saved_scales = {}
    for li in remove_layers:
        saved_scales[li] = model.blocks[li].mlp_scale.data.clone()
        model.blocks[li].mlp_scale.data.zero_()

    mlp_loss = eval_loss(model, val_tokens, windows=100)

    for li in remove_layers:
        model.blocks[li].mlp_scale.data = saved_scales[li]

    label = ','.join(str(l) for l in remove_layers)
    print(f'  Remove MLP L[{label}]: val_loss={mlp_loss:.6f} delta={mlp_loss-abl_base:+.6f}')

# === Exp 12: SmearGate vs BigramHash ablation ===
print('\n--- SmearGate vs BigramHash ablation ---')

# Save originals
smear_gate_saved = model.smear.gate.data.clone()
bigram_scale_saved = model.bigram.scale.data.clone()

# Test without SmearGate (set gate to -inf so sigmoid=0, no blending)
model.smear.gate.data.fill_(-20.0)
no_smear_loss = eval_loss(model, val_tokens, windows=100)
model.smear.gate.data = smear_gate_saved
print(f'  Without SmearGate: val_loss={no_smear_loss:.6f} delta={no_smear_loss-abl_base:+.6f}')

# Test without BigramHash (set scale to 0)
model.bigram.scale.data.fill_(0.0)
no_bigram_loss = eval_loss(model, val_tokens, windows=100)
model.bigram.scale.data = bigram_scale_saved
print(f'  Without BigramHash: val_loss={no_bigram_loss:.6f} delta={no_bigram_loss-abl_base:+.6f}')

# Test without both
model.smear.gate.data.fill_(-20.0)
model.bigram.scale.data.fill_(0.0)
no_both_loss = eval_loss(model, val_tokens, windows=100)
model.smear.gate.data = smear_gate_saved
model.bigram.scale.data = bigram_scale_saved
print(f'  Without both: val_loss={no_both_loss:.6f} delta={no_both_loss-abl_base:+.6f}')

print(f'\n{"="*60}')
print('Batch 1 complete')
print(f'{"="*60}')
