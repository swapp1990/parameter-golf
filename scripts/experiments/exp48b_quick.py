"""Exp 48b: Just run variant B (B6 int8) pre-TTT eval. Compare to A=1.1640."""
import torch, os, sys, glob, math, io, random, time
from pathlib import Path
os.chdir("/runpod-volume/parameter-golf")
for k,v in {"XSA_LAST_N":"7","MLP_MULT":"3","NUM_LAYERS":"7","MODEL_DIM":"624",
            "VOCAB_SIZE":"1024","TRAIN_SEQ_LEN":"2048",
            "LAYER_SCHEDULE":"0,1,2,3,4,3,4,3,4,5,6,5,6"}.items():
    os.environ.setdefault(k,v)

import importlib.util
_s = importlib.util.spec_from_file_location("tgp",
    "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100/train_gpt.py")
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
GPT=_m.GPT; load_data_shard=_m.load_data_shard
dequantize_state_dict_mixed=_m.dequantize_state_dict_mixed
Hyperparameters=_m.Hyperparameters; build_sentencepiece_luts=_m.build_sentencepiece_luts
MLP_QUANT_PATTERNS=_m.MLP_QUANT_PATTERNS; EMBED_QUANT_PATTERNS=_m.EMBED_QUANT_PATTERNS
CONTROL_TENSOR_NAME_PATTERNS=_m.CONTROL_TENSOR_NAME_PATTERNS
import sentencepiece as spm, zstandard
sys.path.insert(0, "scripts/experiments")
from exp48_per_block_gptq import apply_gptq_custom, quantize_state_dict_mixed_custom

device = torch.device("cuda")
args = Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
bbl,hsl,ibt = build_sentencepiece_luts(sp, args.vocab_size, device)

print("Loading model + data...")
fs = torch.load("exp40d_model.pt", map_location="cpu", weights_only=False)
if "model" in fs: fs=fs["model"]
model = GPT(vocab_size=args.vocab_size,num_layers=args.num_layers,model_dim=args.model_dim,
    num_heads=args.num_heads,num_kv_heads=args.num_kv_heads,mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,xsa_last_n=args.xsa_last_n,
    layer_schedule=args.layer_schedule if args.layer_schedule else None).to(device)
model.load_state_dict(fs, strict=False)

vt = torch.cat([load_data_shard(Path(s)).to(device) for s in sorted(glob.glob(
    "./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))]).long()
tt = torch.cat([load_data_shard(Path(s)).to(device) for s in sorted(glob.glob(
    "./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))[:2]]).long()
random.seed(42)
cd = [tt[random.randint(0,tt.numel()-2049):][:2048].unsqueeze(0) for _ in range(64)]
del tt

def no_ttt(mdl):
    sl=args.train_seq_len; ns=(vt.numel()-1)//sl; ls=tc=bc=0.
    mdl.eval()
    with torch.no_grad():
        for i in range(ns):
            s=i*sl; x=vt[s:s+sl].unsqueeze(0); y=vt[s+1:s+sl+1].unsqueeze(0)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): l=mdl(x,y).detach()
            ls+=l.to(torch.float64).item()*sl; tc+=sl
            tb=bbl[y.reshape(-1)].to(torch.float64)
            tb+=(hsl[y.reshape(-1)]&~ibt[x.reshape(-1)]).to(torch.float64); bc+=tb.sum().item()
    return ls/tc, (ls/math.log(2))/bc

print("\n=== Variant B: B6 at int8 ===")
model.load_state_dict({k:v.to(device) for k,v in fs.items()}, strict=False)
gs = apply_gptq_custom(model, cd, {6: 8})
qm = quantize_state_dict_mixed_custom(gs, {6: 8})
buf=io.BytesIO(); torch.save(qm,buf)
comp=zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
print(f"Artifact: {len(comp):,} bytes ({len(comp)/1e6:.2f} MB)")
dec=zstandard.ZstdDecompressor().decompress(comp)
dq=dequantize_state_dict_mixed(torch.load(io.BytesIO(dec),map_location="cpu",weights_only=False))
model.load_state_dict({k:v.to(device) for k,v in dq.items()}, strict=True)
del comp,dec,dq,qm,gs,buf
t0=time.perf_counter()
vl,vb=no_ttt(model)
print(f"Pre-TTT: val_loss={vl:.6f} val_bpb={vb:.6f} time={time.perf_counter()-t0:.0f}s")
print(f"\nA baseline pre-TTT: val_bpb=1.164032")
print(f"B (B6 int8) pre-TTT: val_bpb={vb:.6f}")
print(f"Delta: {vb - 1.164032:+.6f}")
