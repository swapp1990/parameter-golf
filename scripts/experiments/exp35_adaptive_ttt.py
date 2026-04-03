"""
Exp 35: Per-document adaptive TTT.
Instead of fixed SGD hyperparams for all docs, adapt based on training loss:
  - Run 1 step of SGD, measure loss drop
  - If loss drops fast → doc is "easy to adapt" → use standard lr
  - If loss barely drops → doc is "hard to adapt" → try higher lr or more steps
  - If loss increases → doc might be adversarial → use lower lr or skip TTT

Variants:
  A: Fixed baseline (lr=0.005, 1 epoch) for comparison
  B: Adaptive lr (start lr=0.005, double if loss drops <10%, halve if loss increases)
  C: Adaptive epochs (1 epoch default, add epoch if training loss still high)
  D: Per-doc lr grid search (try lr=0.002,0.005,0.01, pick best training loss)
  E: Short-doc boost (higher lr for docs <500 tokens, since they're harder)
"""
import torch, os, sys, glob, time, math, io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
MIXED_MODEL_PATH = os.path.join(BASE_DIR, "final_model.mixed.ptz")
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "500"))
TTT_MIN_DOC_LEN = 32
TTT_TRAIN_CHUNK = 1024
SCORE_CAP = 2048

os.chdir(BASE_DIR)
os.environ["XSA_LAST_N"] = os.environ.get("XSA_LAST_N", "11")
os.environ.setdefault("MLP_MULT", "3")
os.environ.setdefault("NUM_LAYERS", "11")
os.environ.setdefault("VOCAB_SIZE", "1024")
os.environ.setdefault("TRAIN_SEQ_LEN", "2048")

import importlib.util
_spec = importlib.util.spec_from_file_location("tgp", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPT = _mod.GPT; load_data_shard = _mod.load_data_shard
dequantize_state_dict_mixed = _mod.dequantize_state_dict_mixed
Hyperparameters = _mod.Hyperparameters; build_sentencepiece_luts = _mod.build_sentencepiece_luts

import sentencepiece as spm; import zstandard
device = torch.device("cuda")
args = Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

print("Loading model...")
blob = open(MIXED_MODEL_PATH, "rb").read()
raw = zstandard.ZstdDecompressor().decompress(blob)
mixed_state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
base_state_dict = dequantize_state_dict_mixed(mixed_state)
del blob, raw, mixed_state

print("Loading val tokens...")
val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()

bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
n_all_docs = min(len(bos_positions), TTT_MAX_DOCS) if TTT_MAX_DOCS > 0 else len(bos_positions)
doc_list = []
for d in range(n_all_docs):
    ds = int(bos_positions[d])
    de = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
    dl = de - ds
    if dl >= 5: doc_list.append((ds, dl))
short_docs = [(ds, dl) for ds, dl in doc_list if dl < TTT_MIN_DOC_LEN]
long_docs = [(ds, dl) for ds, dl in doc_list if dl >= TTT_MIN_DOC_LEN]
print(f"Documents: {len(doc_list)} (short={len(short_docs)}, long={len(long_docs)})")


def create_model():
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
    ).to(device)
    model.load_state_dict(base_state_dict, strict=True)
    return model


def score_doc(model, doc, score_len):
    """Score a document, return (nll, bytes, tokens)."""
    sx = doc[:score_len].unsqueeze(0)
    sy = doc[1:score_len + 1].unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(sx, sy).detach()
    nll = loss.to(torch.float64) * score_len
    pi = sx.reshape(-1)[:score_len]; ti = sy.reshape(-1)[:score_len]
    tb = base_bytes_lut[ti].to(torch.float64)
    tb += (has_leading_space_lut[ti] & ~is_boundary_token_lut[pi]).to(torch.float64)
    return nll, tb.sum(), score_len


def train_one_epoch(model, doc, lr, momentum, chunk_sz):
    """Train one epoch, return final chunk loss."""
    doc_len = doc.numel()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    last_loss = None
    for cs in range(0, doc_len - 1, chunk_sz):
        ce = min(cs + chunk_sz, doc_len - 1)
        if ce - cs < 2: continue
        tx = doc[cs:ce].unsqueeze(0)
        ty = doc[cs + 1:ce + 1].unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tloss = model(tx, ty)
        last_loss = tloss.item()
        tloss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return last_loss


def run_variant(name, ttt_fn):
    """Generic runner that calls ttt_fn(model, doc, base_state) for each doc."""
    model = create_model()
    base_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
    ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
    ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    # Short docs
    model.eval()
    with torch.no_grad():
        for ds, dl in short_docs:
            nll, tb, n = score_doc(model, val_tokens[ds:ds+dl], dl - 1)
            ttt_nll += nll; ttt_bytes += tb; ttt_tokens += n

    for di, (ds, dl) in enumerate(long_docs):
        doc = val_tokens[ds:ds + dl]
        doc_len = doc.numel()
        score_len = min(SCORE_CAP, doc_len - 1)

        # Restore and run TTT
        model.load_state_dict(base_model_state, strict=True)
        ttt_fn(model, doc, doc_len)

        # Score
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        nll, tb, n = score_doc(model, doc, score_len)
        ttt_nll += nll; ttt_bytes += tb; ttt_tokens += n

        if (di + 1) % 100 == 0:
            bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            print(f"  [{name}] Doc {di+1}/{len(long_docs)}: bpb={bpb:.4f} t={time.perf_counter()-t0:.0f}s", flush=True)

    final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] DONE: bpb={final_bpb:.4f} time={elapsed:.0f}s", flush=True)
    del model, base_model_state; torch.cuda.empty_cache()
    return final_bpb, elapsed


# =========================================================================
# Variant A: Fixed baseline
# =========================================================================
def ttt_fixed(model, doc, doc_len):
    for p in model.parameters(): p.requires_grad = True
    model.train()
    train_one_epoch(model, doc, lr=0.005, momentum=0.9, chunk_sz=TTT_TRAIN_CHUNK)


# =========================================================================
# Variant B: Adaptive LR based on first-chunk loss drop
# =========================================================================
def ttt_adaptive_lr(model, doc, doc_len):
    for p in model.parameters(): p.requires_grad = True
    model.train()
    chunk_sz = min(TTT_TRAIN_CHUNK, doc_len - 1)

    # Measure pre-training loss on first chunk
    tx = doc[:min(chunk_sz, doc_len-1)].unsqueeze(0)
    ty = doc[1:min(chunk_sz, doc_len-1)+1].unsqueeze(0)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pre_loss = model(tx, ty).item()

    # One step with base lr
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tloss = model(tx, ty)
    post_loss = tloss.item()
    tloss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Adapt lr based on loss drop
    drop_pct = (pre_loss - post_loss) / max(pre_loss, 1e-6)
    if drop_pct < 0:  # loss increased — lower lr
        adapted_lr = 0.002
    elif drop_pct < 0.05:  # barely dropped — try higher lr
        adapted_lr = 0.01
    else:  # good drop — keep or slightly increase
        adapted_lr = 0.005

    # Continue training remaining chunks with adapted lr
    optimizer = torch.optim.SGD(model.parameters(), lr=adapted_lr, momentum=0.9)
    for cs in range(chunk_sz, doc_len - 1, chunk_sz):
        ce = min(cs + chunk_sz, doc_len - 1)
        if ce - cs < 2: continue
        tx2 = doc[cs:ce].unsqueeze(0)
        ty2 = doc[cs + 1:ce + 1].unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tloss2 = model(tx2, ty2)
        tloss2.backward()
        optimizer.step()
        optimizer.zero_grad()


# =========================================================================
# Variant C: Adaptive epochs — add epoch if training loss still high
# =========================================================================
def ttt_adaptive_epochs(model, doc, doc_len):
    for p in model.parameters(): p.requires_grad = True
    model.train()
    chunk_sz = min(TTT_TRAIN_CHUNK, doc_len - 1)

    # Epoch 1
    last_loss = train_one_epoch(model, doc, lr=0.005, momentum=0.9, chunk_sz=chunk_sz)

    # If training loss is still high (>2.5), do another epoch
    if last_loss is not None and last_loss > 2.5:
        train_one_epoch(model, doc, lr=0.003, momentum=0.9, chunk_sz=chunk_sz)


# =========================================================================
# Variant D: Per-doc LR grid search (try 3 LRs, pick best)
# =========================================================================
def ttt_lr_search(model, doc, doc_len):
    chunk_sz = min(TTT_TRAIN_CHUNK, doc_len - 1)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_lr = 0.005
    best_loss = float('inf')

    # Measure first-chunk loss with each lr after 1 step
    for lr in [0.002, 0.005, 0.01]:
        model.load_state_dict(base_state, strict=True)
        for p in model.parameters(): p.requires_grad = True
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        tx = doc[:min(chunk_sz, doc_len-1)].unsqueeze(0)
        ty = doc[1:min(chunk_sz, doc_len-1)+1].unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tloss = model(tx, ty)
        tloss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Measure post-step loss
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                post_loss = model(tx, ty).item()
        if post_loss < best_loss:
            best_loss = post_loss
            best_lr = lr

    # Now train with best lr from scratch
    model.load_state_dict(base_state, strict=True)
    for p in model.parameters(): p.requires_grad = True
    model.train()
    train_one_epoch(model, doc, lr=best_lr, momentum=0.9, chunk_sz=chunk_sz)
    del base_state


# =========================================================================
# Variant E: Short-doc boost (higher lr for docs <500 tokens)
# =========================================================================
def ttt_short_boost(model, doc, doc_len):
    for p in model.parameters(): p.requires_grad = True
    model.train()
    # Short docs get higher lr since they have less context and are harder
    lr = 0.01 if doc_len < 500 else 0.005
    train_one_epoch(model, doc, lr=lr, momentum=0.9, chunk_sz=TTT_TRAIN_CHUNK)


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 35: Adaptive TTT ({TTT_MAX_DOCS} docs)")
    print(f"{'#'*60}")

    variants = [
        ("A: Fixed lr=0.005", ttt_fixed),
        ("B: Adaptive LR", ttt_adaptive_lr),
        ("C: Adaptive epochs", ttt_adaptive_epochs),
        ("D: LR grid search", ttt_lr_search),
        ("E: Short-doc boost", ttt_short_boost),
    ]

    results = {}
    for name, fn in variants:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        bpb, elapsed = run_variant(name, fn)
        results[name] = (bpb, elapsed)

    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({TTT_MAX_DOCS} docs)")
    print(f"{'='*70}")
    print(f"{'Variant':<45} {'BPB':>8} {'Time':>8}")
    print(f"{'-'*45} {'-'*8} {'-'*8}")
    for name, (bpb, elapsed) in results.items():
        print(f"{name:<45} {bpb:>8.4f} {elapsed:>7.0f}s")
    print(f"{'='*70}")

    best_name = min(results, key=lambda k: results[k][0])
    baseline_bpb = results["A: Fixed lr=0.005"][0]
    print(f"\nBaseline: A = {baseline_bpb:.4f}")
    print(f"Best: {best_name} = {results[best_name][0]:.4f} (delta={results[best_name][0]-baseline_bpb:+.4f})")
