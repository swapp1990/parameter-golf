"""
Exp 34: SGD TTT combo tests on 500 docs.
Test removing remaining constraints:
  F: lr=0.005, 1 epoch, 1024 chunks (best from exp33)
  G: lr=0.005, 1 epoch, 2048 chunks (full doc training)
  H: lr=0.005, 1 epoch, 1024 chunks, grad_clip=1.0
  I: lr=0.005, 2 epochs, 1024 chunks
  J: lr=0.005, 1 epoch, 2048 chunks, grad_clip=1.0 (combo)
  K: lr=0.005, 2 epochs, 2048 chunks, grad_clip=1.0 (full combo)
  L: lr=0.01, 1 epoch, 2048 chunks, grad_clip=1.0 (push LR higher)
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


def run_variant(name, lr, momentum, n_epochs, train_chunk, grad_clip):
    model = create_model()
    base_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    ttt_nll = torch.zeros((), device=device, dtype=torch.float64)
    ttt_bytes = torch.zeros((), device=device, dtype=torch.float64)
    ttt_tokens = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for ds, dl in short_docs:
            x = val_tokens[ds:ds+dl-1].unsqueeze(0)
            y = val_tokens[ds+1:ds+dl].unsqueeze(0)
            n = dl - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            ttt_nll += loss.to(torch.float64) * n; ttt_tokens += n
            pi = x.reshape(-1)[:n]; ti = y.reshape(-1)[:n]
            tb = base_bytes_lut[ti].to(torch.float64)
            tb += (has_leading_space_lut[ti] & ~is_boundary_token_lut[pi]).to(torch.float64)
            ttt_bytes += tb.sum()

    for di, (ds, dl) in enumerate(long_docs):
        doc = val_tokens[ds:ds+dl]
        doc_len = doc.numel()

        model.load_state_dict(base_model_state, strict=True)
        for p in model.parameters(): p.requires_grad = True

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        chunk_sz = min(train_chunk, doc_len - 1)

        for epoch in range(n_epochs):
            for cs in range(0, doc_len - 1, chunk_sz):
                ce = min(cs + chunk_sz, doc_len - 1)
                if ce - cs < 2: continue
                tx = doc[cs:ce].unsqueeze(0)
                ty = doc[cs+1:ce+1].unsqueeze(0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tloss = model(tx, ty)
                tloss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        for p in model.parameters(): p.requires_grad = False
        score_len = min(SCORE_CAP, doc_len - 1)
        sx = doc[:score_len].unsqueeze(0)
        sy = doc[1:score_len+1].unsqueeze(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sloss = model(sx, sy).detach()
            ttt_nll += sloss.to(torch.float64) * score_len; ttt_tokens += score_len
            pi = sx.reshape(-1)[:score_len]; ti = sy.reshape(-1)[:score_len]
            tb = base_bytes_lut[ti].to(torch.float64)
            tb += (has_leading_space_lut[ti] & ~is_boundary_token_lut[pi]).to(torch.float64)
            ttt_bytes += tb.sum()

        if (di + 1) % 100 == 0:
            bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
            print(f"  [{name}] Doc {di+1}/{len(long_docs)}: bpb={bpb:.4f} t={time.perf_counter()-t0:.0f}s", flush=True)

    final_bpb = (ttt_nll.item() / math.log(2.0)) / max(ttt_bytes.item(), 1.0)
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] DONE: bpb={final_bpb:.4f} time={elapsed:.0f}s", flush=True)
    del model, base_model_state; torch.cuda.empty_cache()
    return final_bpb, elapsed


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 34: SGD Combo Tests ({TTT_MAX_DOCS} docs)")
    print(f"{'#'*60}")

    # (name, lr, momentum, epochs, train_chunk, grad_clip)
    variants = [
        ("F: lr=0.005 chunk=1024",           0.005, 0.9, 1, 1024, 0),
        ("G: lr=0.005 chunk=2048",           0.005, 0.9, 1, 2048, 0),
        ("H: lr=0.005 chunk=1024 clip=1",    0.005, 0.9, 1, 1024, 1.0),
        ("I: lr=0.005 2ep chunk=1024",       0.005, 0.9, 2, 1024, 0),
        ("J: lr=0.005 chunk=2048 clip=1",    0.005, 0.9, 1, 2048, 1.0),
        ("K: lr=0.005 2ep chunk=2048 clip=1", 0.005, 0.9, 2, 2048, 1.0),
        ("L: lr=0.01 chunk=2048 clip=1",     0.010, 0.9, 1, 2048, 1.0),
    ]

    results = {}
    for name, lr, mom, epochs, chunk, clip in variants:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        bpb, elapsed = run_variant(name, lr, mom, epochs, chunk, clip)
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
    print(f"\nBest: {best_name} (bpb={results[best_name][0]:.4f})")
    print(f"Ref: exp28 full eval baseline = 1.1429 (lr=0.002, 1ep, chunk=1024, no clip)")
