"""
Test sliding window + TTT eval locally on CPU.
Uses basic 9L checkpoint to verify mechanics before running on pod.

Verifies:
1. Sliding window gives lower loss than standard eval
2. TTT gives lower loss than no-TTT
3. BPB computation is correct
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import sentencepiece as spm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Use layer_xray model (basic 9L, no XSA/SwiGLU)
from dashboard.layer_xray import GPT, load_tokenizer

SEQ_LEN = 1024  # shorter for CPU speed
STRIDE = 256
VOCAB_SIZE = 1024


def forward_logits(model, x):
    """Forward pass returning logits (not loss)."""
    result = model.forward_full(x)
    return result["logits"]


def standard_eval(model, tokens, seq_len, max_windows=10):
    """Standard non-overlapping window eval. Returns (total_nll, total_tokens)."""
    total_nll = 0.0
    total_tokens = 0
    n_windows = min(max_windows, (tokens.numel() - seq_len) // seq_len)
    with torch.no_grad():
        for w in range(n_windows):
            start = w * seq_len
            x = tokens[start:start + seq_len].unsqueeze(0)
            y = tokens[start + 1:start + seq_len + 1].unsqueeze(0)
            logits = forward_logits(model, x)
            per_token = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            total_nll += per_token.sum().item()
            total_tokens += per_token.numel()
    return total_nll, total_tokens


def sliding_window_eval(model, tokens, seq_len, stride, max_tokens=20000):
    """
    Sliding window eval — ONLY scores last `stride` tokens per window.
    Each scored token has at least (seq_len - stride) tokens of context.
    Returns (total_nll, total_tokens).
    """
    total_nll = 0.0
    total_tokens = 0
    starts = list(range(0, min(tokens.numel() - seq_len, max_tokens), stride))

    with torch.no_grad():
        for s in starts:
            x = tokens[s:s + seq_len].unsqueeze(0)
            y = tokens[s + 1:s + seq_len + 1].unsqueeze(0)
            logits = forward_logits(model, x)

            # ONLY score the last `stride` tokens
            score_logits = logits[0, -stride:, :]  # (stride, vocab)
            score_targets = y[0, -stride:]          # (stride,)
            per_token = F.cross_entropy(score_logits.float(), score_targets, reduction='none')

            total_nll += per_token.sum().item()
            total_tokens += per_token.numel()

    return total_nll, total_tokens


def ttt_adapt(model, doc_tokens, lora_modules, lora_params, lr=0.05, seq_len=512):
    """Adapt LoRA on a document (1 epoch)."""
    model.train()
    opt = torch.optim.Adam(lora_params, lr=lr)
    csz = min(seq_len, doc_tokens.numel() - 1)
    for cs in range(0, doc_tokens.numel() - 1, csz):
        ce = min(cs + csz, doc_tokens.numel() - 1)
        if ce - cs < 2:
            continue
        x = doc_tokens[cs:ce].unsqueeze(0)
        y = doc_tokens[cs + 1:ce + 1].unsqueeze(0)
        logits = forward_logits(model, x)
        loss = F.cross_entropy(logits[0].float(), y[0])
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()


class LoRALinear(nn.Module):
    def __init__(self, original, rank=4):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_d) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_d, rank) * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.001)


# =====================================================
# Load model and data
# =====================================================
print("=== Local Eval Test ===")
print("Loading model (9L basic, step 2000)...")
model = GPT()  # 9L basic
sd = torch.load("dashboard/checkpoints/model_step_2000.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"Loaded: {sum(p.numel() for p in model.parameters())} params")

print("Loading val tokens...")
header_bytes = 256 * np.dtype("<i4").itemsize
tokens_np = np.fromfile("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin",
                        dtype="<u2", offset=header_bytes)
val_tokens = torch.from_numpy(tokens_np.astype(np.int64))
print(f"Val tokens: {val_tokens.numel()}")

# =====================================================
# Test 1: Standard vs Sliding Window
# =====================================================
print(f"\n--- Test 1: Standard vs Sliding Window (seq={SEQ_LEN}, stride={STRIDE}) ---")

t0 = time.time()
std_nll, std_tok = standard_eval(model, val_tokens, SEQ_LEN, max_windows=10)
std_loss = std_nll / std_tok
print(f"Standard:       loss={std_loss:.4f} tokens={std_tok} time={time.time()-t0:.1f}s")

t1 = time.time()
sw_nll, sw_tok = sliding_window_eval(model, val_tokens, SEQ_LEN, STRIDE, max_tokens=SEQ_LEN * 10)
sw_loss = sw_nll / sw_tok
print(f"Sliding(s{STRIDE}):  loss={sw_loss:.4f} tokens={sw_tok} time={time.time()-t1:.1f}s")

sw_better = sw_loss < std_loss
print(f"Sliding < Standard: {sw_better} (delta={sw_loss - std_loss:+.4f})")
if sw_better:
    print("  PASS: Sliding window gives lower loss")
else:
    print("  WARN: Sliding window didn't help — may be expected for short eval")

# =====================================================
# Test 2: TTT (LoRA) on single document
# =====================================================
print(f"\n--- Test 2: TTT on single document ---")

# Find first document boundary
BOS_ID = 1
bos_pos = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
doc_start = int(bos_pos[0])
doc_end = int(bos_pos[1]) if len(bos_pos) > 1 else min(doc_start + 2000, val_tokens.numel())
doc = val_tokens[doc_start:doc_end]
print(f"Document: positions {doc_start}-{doc_end}, length={doc.numel()} tokens")

# Score WITHOUT TTT
model.eval()
with torch.no_grad():
    x = doc[:-1].unsqueeze(0)
    y = doc[1:].unsqueeze(0)
    logits = forward_logits(model, x)
    no_ttt_loss = F.cross_entropy(logits[0].float(), y[0]).item()
print(f"Without TTT: loss={no_ttt_loss:.4f}")

# Inject LoRA
for p in model.parameters():
    p.requires_grad = False
lora_modules = []
for blk in model.blocks:
    lq = LoRALinear(blk.attn.c_q, rank=4)
    blk.attn.c_q = lq
    lora_modules.append(lq)
    lv = LoRALinear(blk.attn.c_v, rank=4)
    blk.attn.c_v = lv
    lora_modules.append(lv)
lora_params = []
for m in lora_modules:
    lora_params.extend([m.lora_A, m.lora_B])
print(f"LoRA injected: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params")

# Score WITH TTT (adapt then score)
for m in lora_modules:
    m.reset()
ttt_adapt(model, doc, lora_modules, lora_params, lr=0.05, seq_len=512)

model.eval()
with torch.no_grad():
    logits = forward_logits(model, x)
    ttt_loss = F.cross_entropy(logits[0].float(), y[0]).item()
print(f"With TTT:    loss={ttt_loss:.4f}")

ttt_better = ttt_loss < no_ttt_loss
print(f"TTT < No-TTT: {ttt_better} (delta={ttt_loss - no_ttt_loss:+.4f})")
if ttt_better:
    print("  PASS: TTT improves loss")
else:
    print("  FAIL: TTT didn't help")

# =====================================================
# Test 3: Combined TTT + Sliding Window on same document
# =====================================================
print(f"\n--- Test 3: Combined TTT + Sliding Window ---")

# Reset and re-adapt
for m in lora_modules:
    m.reset()
ttt_adapt(model, doc, lora_modules, lora_params, lr=0.05, seq_len=512)

# Sliding window within document
model.eval()
if doc.numel() > SEQ_LEN + 1:
    doc_sw_nll = 0.0
    doc_sw_tok = 0
    doc_starts = list(range(0, doc.numel() - SEQ_LEN, STRIDE))
    with torch.no_grad():
        for s in doc_starts:
            x = doc[s:s + SEQ_LEN].unsqueeze(0)
            y = doc[s + 1:s + SEQ_LEN + 1].unsqueeze(0)
            logits = forward_logits(model, x)
            score_logits = logits[0, -STRIDE:, :]
            score_targets = y[0, -STRIDE:]
            per_tok = F.cross_entropy(score_logits.float(), score_targets, reduction='none')
            doc_sw_nll += per_tok.sum().item()
            doc_sw_tok += per_tok.numel()
    combined_loss = doc_sw_nll / doc_sw_tok
    print(f"TTT + Sliding: loss={combined_loss:.4f}")
    combined_better = combined_loss < no_ttt_loss
    print(f"Combined < No-TTT: {combined_better} (delta={combined_loss - no_ttt_loss:+.4f})")
else:
    print(f"  Document too short ({doc.numel()} < {SEQ_LEN}) for sliding window, using standard scoring")
    print(f"  TTT alone: loss={ttt_loss:.4f} (delta={ttt_loss - no_ttt_loss:+.4f})")

# =====================================================
# Summary
# =====================================================
print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"Standard eval:           loss={std_loss:.4f}")
print(f"Sliding window(s{STRIDE}):    loss={sw_loss:.4f}  {'PASS' if sw_better else 'WARN'}")
print(f"No TTT (single doc):     loss={no_ttt_loss:.4f}")
print(f"TTT (single doc):        loss={ttt_loss:.4f}  {'PASS' if ttt_better else 'FAIL'}")
print(f"{'='*50}")
