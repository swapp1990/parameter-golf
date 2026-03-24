"""
Debug: compare sliding window per-document vs full-stream.
Proves which approach gives lower loss.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dashboard.layer_xray import GPT

SEQ_LEN = 512
STRIDE = 128

print('=== Sliding Window Debug ===')

model = GPT()
sd = torch.load('dashboard/checkpoints/model_step_2000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()

header_bytes = 256 * np.dtype("<i4").itemsize
tokens_np = np.fromfile('data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin',
                        dtype="<u2", offset=header_bytes)
val_tokens = torch.from_numpy(tokens_np.astype(np.int64))

def forward_logits(mdl, x):
    result = mdl.forward_full(x)
    return result["logits"]

# === 1. Standard eval (non-overlapping, full stream) ===
print('\n--- 1. Standard (non-overlapping, full stream) ---')
std_nll = 0.0
std_tok = 0
n_win = 20
with torch.no_grad():
    for w in range(n_win):
        s = w * SEQ_LEN
        x = val_tokens[s:s+SEQ_LEN].unsqueeze(0)
        y = val_tokens[s+1:s+SEQ_LEN+1].unsqueeze(0)
        logits = forward_logits(model, x)
        per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
        std_nll += per_tok.sum().item()
        std_tok += per_tok.numel()
print(f'Standard: loss={std_nll/std_tok:.4f} tokens={std_tok}')

# === 2. Sliding window FULL STREAM (correct way) ===
print(f'\n--- 2. Sliding window FULL STREAM (stride={STRIDE}) ---')
sw_nll = 0.0
sw_tok = 0
max_pos = n_win * SEQ_LEN
starts = list(range(0, max_pos - SEQ_LEN, STRIDE))
with torch.no_grad():
    for s in starts:
        x = val_tokens[s:s+SEQ_LEN].unsqueeze(0)
        y = val_tokens[s+1:s+SEQ_LEN+1].unsqueeze(0)
        logits = forward_logits(model, x)
        # Score ONLY last stride tokens
        score_logits = logits[0, -STRIDE:, :]
        score_targets = y[0, -STRIDE:]
        per_tok = F.cross_entropy(score_logits.float(), score_targets, reduction='none')
        sw_nll += per_tok.sum().item()
        sw_tok += per_tok.numel()
print(f'SW full stream: loss={sw_nll/sw_tok:.4f} tokens={sw_tok}')

# === 3. Sliding window PER DOCUMENT (our buggy approach) ===
print(f'\n--- 3. Sliding window PER DOCUMENT (stride={STRIDE}) ---')
bos_pos = (val_tokens == 1).nonzero(as_tuple=True)[0].numpy()
pd_nll = 0.0
pd_tok = 0
n_docs = min(20, len(bos_pos))
with torch.no_grad():
    for d in range(n_docs):
        ds = int(bos_pos[d])
        de = int(bos_pos[d+1]) if d+1 < len(bos_pos) else val_tokens.numel()
        doc = val_tokens[ds:de]
        if doc.numel() <= SEQ_LEN + 1:
            # Short: score all
            x = doc[:-1].unsqueeze(0)
            y = doc[1:].unsqueeze(0)
            logits = forward_logits(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            pd_nll += per_tok.sum().item()
            pd_tok += per_tok.numel()
        else:
            # Long: sliding within doc
            starts = list(range(0, doc.numel()-SEQ_LEN, STRIDE))
            scored = set()
            for s in starts:
                x = doc[s:s+SEQ_LEN].unsqueeze(0)
                y = doc[s+1:s+SEQ_LEN+1].unsqueeze(0)
                logits = forward_logits(model, x)
                score_logits = logits[0, -STRIDE:, :]
                score_targets = y[0, -STRIDE:]
                per_tok = F.cross_entropy(score_logits.float(), score_targets, reduction='none')
                abs_start = s + SEQ_LEN - STRIDE + 1
                for i in range(per_tok.numel()):
                    ap = abs_start + i
                    if ap not in scored and ap < doc.numel():
                        scored.add(ap)
                        pd_nll += per_tok[i].item()
                        pd_tok += 1
print(f'SW per-doc: loss={pd_nll/pd_tok:.4f} tokens={pd_tok}')

# === 4. Standard eval PER DOCUMENT (no sliding, just doc-by-doc) ===
print(f'\n--- 4. Standard PER DOCUMENT (no sliding) ---')
sd_nll = 0.0
sd_tok = 0
for d in range(n_docs):
    ds = int(bos_pos[d])
    de = int(bos_pos[d+1]) if d+1 < len(bos_pos) else val_tokens.numel()
    doc = val_tokens[ds:de]
    if doc.numel() < 3: continue
    x = doc[:min(SEQ_LEN, doc.numel()-1)].unsqueeze(0)
    y = doc[1:min(SEQ_LEN+1, doc.numel())].unsqueeze(0)
    with torch.no_grad():
        logits = forward_logits(model, x)
        per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
    sd_nll += per_tok.sum().item()
    sd_tok += per_tok.numel()
print(f'Std per-doc: loss={sd_nll/sd_tok:.4f} tokens={sd_tok}')

print(f'\n=== Summary ===')
print(f'1. Standard full stream:    {std_nll/std_tok:.4f}')
print(f'2. SW full stream:          {sw_nll/sw_tok:.4f}  delta={sw_nll/sw_tok - std_nll/std_tok:+.4f}')
print(f'3. SW per-document:         {pd_nll/pd_tok:.4f}  delta={pd_nll/pd_tok - std_nll/std_tok:+.4f}')
print(f'4. Standard per-document:   {sd_nll/sd_tok:.4f}  delta={sd_nll/sd_tok - std_nll/std_tok:+.4f}')
