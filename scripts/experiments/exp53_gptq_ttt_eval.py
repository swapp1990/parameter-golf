"""
Exp 53: GPTQ + SGD all-weights TTT eval on QK-Gain=4.0 checkpoint.
"""
import torch, os, sys, glob, time, math, io, json, random
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
FLOAT_MODEL_PATH = os.path.join(BASE_DIR, "final_model.pt")
N_CALIB = 64
TTT_MAX_DOCS = int(os.environ.get("TTT_MAX_DOCS", "0"))  # 0 = all docs
TTT_MIN_DOC_LEN = 32
TTT_TRAIN_CHUNK = 2048
SCORE_CAP = 2048
SGD_LR = 0.005
SGD_MOMENTUM = 0.9
SEED = 42

os.chdir(BASE_DIR)
for k, v in {"XSA_LAST_N": "7", "MLP_MULT": "3", "NUM_LAYERS": "7", "MODEL_DIM": "624",
             "VOCAB_SIZE": "1024", "TRAIN_SEQ_LEN": "2048",
             "LAYER_SCHEDULE": "0,1,2,3,4,3,4,3,4,5,6,5,6",
             "QK_GAIN_INIT": "4.0"}.items():
    os.environ.setdefault(k, v)

import importlib.util
_spec = importlib.util.spec_from_file_location("tgp", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

GPT = _mod.GPT; load_data_shard = _mod.load_data_shard
dequantize_state_dict_mixed = _mod.dequantize_state_dict_mixed
Hyperparameters = _mod.Hyperparameters; build_sentencepiece_luts = _mod.build_sentencepiece_luts
MLP_QUANT_PATTERNS = _mod.MLP_QUANT_PATTERNS; EMBED_QUANT_PATTERNS = _mod.EMBED_QUANT_PATTERNS
CONTROL_TENSOR_NAME_PATTERNS = _mod.CONTROL_TENSOR_NAME_PATTERNS
import sentencepiece as spm, zstandard

device = torch.device("cuda")


def compute_bpb(total_nll, total_bytes):
    return (total_nll / math.log(2.0)) / max(total_bytes, 1.0)


def collect_layer_inputs(model, calib_data, module_name):
    inputs_list = []
    hooks = []
    def hook_fn(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        inputs_list.append(inp.detach().reshape(-1, inp.shape[-1]).float())
    for name, module in model.named_modules():
        if name == module_name:
            hooks.append(module.register_forward_hook(hook_fn))
            break
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for x in calib_data:
            _ = model(x, torch.roll(x, -1, dims=1))
    for h in hooks:
        h.remove()
    return torch.cat(inputs_list, dim=0) if inputs_list else None


def gptq_quantize_weight(W, H, bits, block_size=128, dampening=0.01):
    rows, cols = W.shape
    max_val = 2 ** (bits - 1) - 1
    scale = W.abs().amax(dim=1) / max_val
    scale = scale.clamp_min(1e-12)
    damp = dampening * H.diag().mean()
    H_damped = H + damp * torch.eye(cols, device=H.device, dtype=H.dtype)
    try:
        L = torch.linalg.cholesky(H_damped)
        H_inv = torch.cholesky_inverse(L)
    except Exception:
        H_inv = torch.diag(1.0 / H_damped.diag().clamp_min(1e-12))
    Q = torch.zeros_like(W)
    W_work = W.clone()
    for col_start in range(0, cols, block_size):
        col_end = min(col_start + block_size, cols)
        for j in range(col_start, col_end):
            w_col = W_work[:, j]
            q_col = torch.clamp(torch.round(w_col / scale), -max_val - 1, max_val)
            Q[:, j] = q_col * scale
            err = w_col - Q[:, j]
            if j + 1 < cols:
                h_jj = H_inv[j, j].clamp_min(1e-12)
                W_work[:, j+1:cols] -= err.unsqueeze(1) * H_inv[j, j+1:cols].unsqueeze(0) / h_jj
    return Q


def eval_ttt(model, val_tokens, bbl, hsl, ibt, max_docs, score_cap):
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    doc_list = []
    for d in range(len(bos_positions)):
        ds = int(bos_positions[d])
        de = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
        dl = de - ds
        if dl >= 5:
            doc_list.append((ds, dl))
    if max_docs > 0:
        doc_list = doc_list[:max_docs]
    short = [(s, l) for s, l in doc_list if l < TTT_MIN_DOC_LEN]
    long = [(s, l) for s, l in doc_list if l >= TTT_MIN_DOC_LEN]
    nll = byt = tok = 0.0
    t0 = time.perf_counter()

    model.eval()
    for p in model.parameters(): p.requires_grad = False
    with torch.no_grad():
        for ds, dl in short:
            x = val_tokens[ds:ds+dl-1].unsqueeze(0)
            y = val_tokens[ds+1:ds+dl].unsqueeze(0)
            n = dl - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            nll += loss.to(torch.float64).item() * n; tok += n
            tb = bbl[y.reshape(-1)[:n]].to(torch.float64)
            tb += (hsl[y.reshape(-1)[:n]] & ~ibt[x.reshape(-1)[:n]]).to(torch.float64)
            byt += tb.sum().item()

    for di, (ds, dl) in enumerate(long):
        doc = val_tokens[ds:ds+dl]
        model.load_state_dict(base_state, strict=True)
        for p in model.parameters(): p.requires_grad = True
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=SGD_LR, momentum=SGD_MOMENTUM)
        for cs in range(0, dl-1, TTT_TRAIN_CHUNK):
            ce = min(cs+TTT_TRAIN_CHUNK, dl-1)
            if ce-cs < 2: continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(doc[cs:ce].unsqueeze(0), doc[cs+1:ce+1].unsqueeze(0))
            loss.backward(); opt.step(); opt.zero_grad()

        model.eval()
        for p in model.parameters(): p.requires_grad = False
        sl = min(score_cap, dl-1)
        sx = doc[:sl].unsqueeze(0); sy = doc[1:sl+1].unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sloss = model(sx, sy).detach()
        nll += sloss.to(torch.float64).item() * sl; tok += sl
        tb = bbl[sy.reshape(-1)[:sl]].to(torch.float64)
        tb += (hsl[sy.reshape(-1)[:sl]] & ~ibt[sx.reshape(-1)[:sl]]).to(torch.float64)
        byt += tb.sum().item()

        if (di+1) % 500 == 0:
            elapsed = time.perf_counter()-t0
            print(f"  Doc {di+1}/{len(long)}: bpb={compute_bpb(nll,byt):.4f} elapsed={elapsed:.0f}s")

    model.load_state_dict(base_state, strict=True)
    return compute_bpb(nll, byt), nll/max(tok,1), time.perf_counter()-t0


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 53: GPTQ + SGD TTT eval (QK-Gain=4.0)")
    print(f"{'#'*60}")

    args = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    bbl, hsl, ibt = build_sentencepiece_luts(sp, args.vocab_size, device)

    print("Loading float model...")
    fs = torch.load(FLOAT_MODEL_PATH, map_location="cpu", weights_only=False)
    if "model" in fs: fs = fs["model"]
    model = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
        layer_schedule=args.layer_schedule if args.layer_schedule else None).to(device)
    model.load_state_dict(fs, strict=False)
    del fs

    print("Loading tokens...")
    vt = torch.cat([load_data_shard(Path(s)).to(device) for s in sorted(glob.glob(
        "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))]).long()
    tt = torch.cat([load_data_shard(Path(s)).to(device) for s in sorted(glob.glob(
        "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))[:2]]).long()
    random.seed(SEED)
    cd = [tt[random.randint(0, tt.numel()-2049):][:2048].unsqueeze(0) for _ in range(N_CALIB)]
    del tt

    # Apply GPTQ
    print("\nApplying GPTQ...")
    gptq_state = {k: v.cpu() for k, v in model.state_dict().items()}
    for name, param in model.named_parameters():
        if param.ndim != 2 or param.numel() < 4096: continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or "smear" in name: continue
        if any(p in name for p in EMBED_QUANT_PATTERNS): continue
        bits = 5 if any(p in name for p in MLP_QUANT_PATTERNS) else 6
        mn = ".".join(name.split(".")[:-1])
        X = collect_layer_inputs(model, cd[:32], mn)
        if X is not None and X.shape[0] > 0:
            H = (X.float().T @ X.float()) / X.shape[0]
            gptq_state[name] = gptq_quantize_weight(param.data.float(), H, bits).cpu()
    del cd

    # Quantize -> compress -> decompress -> dequantize
    print("Quantize -> compress -> decompress -> dequantize...")
    sys.path.insert(0, os.path.join(BASE_DIR, "scripts", "experiments"))
    from exp48_per_block_gptq import quantize_state_dict_mixed_custom
    qm = quantize_state_dict_mixed_custom(gptq_state, {})
    buf = io.BytesIO(); torch.save(qm, buf)
    comp = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
    artifact_size = len(comp)
    print(f"  Artifact size: {artifact_size} bytes ({artifact_size/1e6:.2f} MB)")
    dec = zstandard.ZstdDecompressor().decompress(comp)
    dq = dequantize_state_dict_mixed(torch.load(io.BytesIO(dec), map_location="cpu", weights_only=False))
    model.load_state_dict({k: v.to(device) for k, v in dq.items()}, strict=True)
    del gptq_state, qm, buf, comp, dec, dq

    # SGD TTT eval
    n_docs = "all" if TTT_MAX_DOCS == 0 else TTT_MAX_DOCS
    print(f"\n=== SGD TTT eval ({n_docs} docs, SCORE_CAP={SCORE_CAP}) ===")
    bpb, loss, elapsed = eval_ttt(model, vt, bbl, hsl, ibt, TTT_MAX_DOCS, SCORE_CAP)
    print(f"\n  FINAL: val_bpb={bpb:.4f} val_loss={loss:.4f} time={elapsed:.0f}s")

    results = {
        "ttt_val_bpb": bpb,
        "ttt_val_loss": loss,
        "ttt_time_s": elapsed,
        "artifact_bytes": artifact_size,
        "score_cap": SCORE_CAP,
        "sgd_lr": SGD_LR,
    }
    with open("/runpod-volume/exp53_gptq_ttt_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /runpod-volume/exp53_gptq_ttt_results.json")
    print("Done!")
