"""
Int5-MLP + Int6-Attn + Zstd-22 quantization for Exp 17 checkpoint.
Produces submission-ready artifact under 16MB.

Usage: python quantize_mixed.py
Input:  dashboard/checkpoints/exp17_xsa/final_model.pt
Output: dashboard/checkpoints/exp17_xsa/final_model.mixed.ptz
"""
import torch, io, os, sys, json
import numpy as np

try:
    import zstandard
    print("zstd available")
except ImportError:
    print("ERROR: pip install zstandard")
    sys.exit(1)

CHECKPOINT = "dashboard/checkpoints/exp17_xsa/final_model.pt"
OUTPUT = "dashboard/checkpoints/exp17_xsa/final_model.mixed.ptz"

# Patterns to classify tensors
MLP_PATTERNS = ("mlp.", "gate.", "up.")  # SwiGLU has gate, up, proj
ATTN_PATTERNS = ("attn.", "c_q.", "c_k.", "c_v.", "proj.")
FP16_KEEP = ("tok_emb",)  # Tied embedding stays fp16
CONTROL_PATTERNS = ("scale", "gain", "mix", "skip_weight", "smear", "bigram")


def quantize_int5_per_row(t):
    """Quantize to int5 range [-16, 15]."""
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 15.0).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -16, 15).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 15.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -16, 15).to(torch.int8)
    return q, scale


def quantize_int6_per_row(t):
    """Quantize to int6 range [-32, 31]."""
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 31.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def dequantize(q, scale, dtype):
    """Dequantize int5 or int6 tensor."""
    if q.ndim == 2 and scale.ndim == 1:
        return (q.float() * scale.float()[:, None]).to(dtype)
    return (q.float() * scale.float()).to(dtype)


def classify_tensor(name):
    """Classify tensor as mlp, attn, fp16_keep, control, or other."""
    if any(p in name for p in FP16_KEEP):
        return "fp16"
    if any(p in name for p in CONTROL_PATTERNS):
        return "control"
    if any(p in name for p in MLP_PATTERNS):
        return "mlp"
    if any(p in name for p in ATTN_PATTERNS):
        return "attn"
    return "other"


print(f"Loading checkpoint: {CHECKPOINT}")
sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
print(f"Loaded {len(sd)} tensors, {sum(p.numel() for p in sd.values())} params")

# Classify and quantize
result = {}
stats = {"mlp_int5": 0, "attn_int6": 0, "fp16": 0, "control": 0, "other": 0}

for name, t in sd.items():
    t_cpu = t.detach().cpu()
    cat = classify_tensor(name)

    # Small tensors: always passthrough as fp16
    if t_cpu.numel() <= 896 or not t_cpu.is_floating_point():
        result[name] = t_cpu.to(torch.float16) if t_cpu.is_floating_point() else t_cpu
        stats["control"] += 1
        continue

    if cat == "fp16":
        result[name] = t_cpu.to(torch.float16).contiguous()
        stats["fp16"] += 1
        print(f"  {name:50s} fp16 passthrough ({t_cpu.numel()} params)")

    elif cat == "mlp":
        q, scale = quantize_int5_per_row(t_cpu)
        result[name + ".__q"] = q
        result[name + ".__scale"] = scale
        result[name + ".__dtype"] = str(t_cpu.dtype)
        result[name + ".__bits"] = "int5"
        stats["mlp_int5"] += 1
        print(f"  {name:50s} int5 ({t_cpu.shape})")

    elif cat in ("attn", "other"):
        q, scale = quantize_int6_per_row(t_cpu)
        result[name + ".__q"] = q
        result[name + ".__scale"] = scale
        result[name + ".__dtype"] = str(t_cpu.dtype)
        result[name + ".__bits"] = "int6"
        stats["attn_int6"] += 1
        print(f"  {name:50s} int6 ({t_cpu.shape})")

    elif cat == "control":
        result[name] = t_cpu.to(torch.float16).contiguous()
        stats["control"] += 1

result["__quant_format__"] = "mixed_int5_int6_v1"

print(f"\nStats: {stats}")

# Serialize
print("Serializing...")
buf = io.BytesIO()
torch.save(result, buf)
raw = buf.getvalue()
print(f"Raw serialized: {len(raw)} bytes ({len(raw)/1024/1024:.2f} MB)")

# Compress with zstd-22
print("Compressing with zstd level 22...")
compressed = zstandard.ZstdCompressor(level=22).compress(raw)
print(f"Compressed: {len(compressed)} bytes ({len(compressed)/1024/1024:.2f} MB)")

# Also try zlib for comparison
import zlib
compressed_zlib = zlib.compress(raw, 9)
print(f"Zlib-9:    {len(compressed_zlib)} bytes ({len(compressed_zlib)/1024/1024:.2f} MB)")

# Code size
code_path = "train_gpt.py"
code_bytes = len(open(code_path, "rb").read()) if os.path.exists(code_path) else 55000
total = len(compressed) + code_bytes
print(f"\nCode: {code_bytes} bytes")
print(f"Total submission: {total} bytes ({total/1024/1024:.2f} MB)")
print(f"Under 16MB: {total < 16_000_000}")

# Save
with open(OUTPUT, "wb") as f:
    f.write(compressed)
print(f"\nSaved: {OUTPUT}")

# Verify roundtrip (dequantize only, no eval)
print("\nVerifying roundtrip dequantization...")
quant_state = torch.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(compressed)),
                         map_location="cpu", weights_only=False)
fmt = quant_state.pop("__quant_format__", None)
print(f"Format: {fmt}")

recovered = {}
seen = set()
for key in list(quant_state.keys()):
    if key.endswith(".__q"):
        name = key[:-4]
        if name in seen:
            continue
        seen.add(name)
        q = quant_state[name + ".__q"]
        scale = quant_state[name + ".__scale"]
        dtype_str = quant_state[name + ".__dtype"]
        dtype = getattr(torch, dtype_str.split(".")[-1])
        recovered[name] = dequantize(q, scale, dtype)
    elif not any(key.endswith(s) for s in (".__scale", ".__dtype", ".__bits")):
        recovered[key] = quant_state[key]

print(f"Recovered {len(recovered)} tensors")

# Compare a few tensors
for name in list(sd.keys())[:5]:
    if name in recovered:
        orig = sd[name].float()
        rec = recovered[name].float()
        mse = (orig - rec).pow(2).mean().item()
        max_err = (orig - rec).abs().max().item()
        print(f"  {name:50s} MSE={mse:.6f} max_err={max_err:.4f}")

print("\n=== Done ===")
