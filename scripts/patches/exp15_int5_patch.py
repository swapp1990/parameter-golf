"""
Patch train_gpt.py for Exp 15:
- Int5 quantization for MLP weights (range [-16, 15])
- Int6 for attention weights (range [-32, 31])
- Zstd-22 compression
- Mixed quantize/dequantize functions

Apply AFTER patch_exp10.py + exp14_patch.py + WD fix.
Run on pod: python exp15_int5_patch.py
"""
import os

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

changes = []

# 1. Add int5 quantization functions (before existing int6 functions)
int5_code = '''

def quantize_int5_per_row(t):
    """Quantize to int5 range [-16, 15] with per-row scaling."""
    import torch
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 15.0).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -16, 15).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 15.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -16, 15).to(torch.int8)
    return q, scale


def dequantize_int5_per_row(q, scale, dtype):
    """Dequantize int5 tensor."""
    import torch
    if q.ndim == 2 and scale.ndim == 1:
        return (q.float() * scale.float()[:, None]).to(dtype)
    return (q.float() * scale.float()).to(dtype)


# Patterns to identify MLP weights (get int5) vs attention weights (get int6)
MLP_NAME_PATTERNS = ("mlp.", "gate.", "up.")
FP16_KEEP_PATTERNS = ("tok_emb",)


def quantize_state_dict_mixed(state_dict):
    """Int5 for MLP weights, Int6 for attention weights, fp16 for sensitive tensors."""
    import torch
    result = {}
    meta = {}
    for name, t in state_dict.items():
        t_cpu = t.detach().cpu()
        # FP16 passthrough for sensitive tensors
        if any(pat in name for pat in FP16_KEEP_PATTERNS):
            result[name] = t_cpu.to(torch.float16).contiguous()
            meta[name] = "fp16"
            continue
        # Small/non-float: passthrough
        if t_cpu.numel() <= 896 or not t_cpu.is_floating_point():
            result[name] = t_cpu.to(torch.float16).contiguous() if t_cpu.is_floating_point() else t_cpu
            meta[name] = "passthrough"
            continue
        # Determine int5 (MLP) or int6 (attention/other)
        is_mlp = any(pat in name for pat in MLP_NAME_PATTERNS)
        if is_mlp:
            q, scale = quantize_int5_per_row(t_cpu)
            result[name + ".__q"] = q
            result[name + ".__scale"] = scale
            result[name + ".__dtype"] = str(t_cpu.dtype)
            result[name + ".__bits"] = "int5"
            meta[name] = "int5"
        else:
            q, scale = quantize_int6_per_row(t_cpu)
            result[name + ".__q"] = q
            result[name + ".__scale"] = scale
            result[name + ".__dtype"] = str(t_cpu.dtype)
            result[name + ".__bits"] = "int6"
            meta[name] = "int6"
    result["__quant_format__"] = "mixed_int5_int6_v1"
    return result, meta


def dequantize_state_dict_mixed(quant_dict):
    """Dequantize mixed int5/int6 state dict."""
    import torch
    fmt = quant_dict.pop("__quant_format__", None)
    result = {}
    seen = set()
    for key in list(quant_dict.keys()):
        if key.endswith(".__q"):
            name = key[:-4]
            if name in seen:
                continue
            seen.add(name)
            q = quant_dict[name + ".__q"]
            scale = quant_dict[name + ".__scale"]
            dtype_str = quant_dict[name + ".__dtype"]
            bits = quant_dict.get(name + ".__bits", "int6")
            dtype = getattr(torch, dtype_str.split(".")[-1])
            if bits == "int5":
                result[name] = dequantize_int5_per_row(q, scale, dtype)
            else:
                result[name] = dequantize_int6_per_row(q, scale, dtype)
        elif not any(key.endswith(s) for s in (".__scale", ".__dtype", ".__bits")):
            result[key] = quant_dict[key]
    return result

'''

# Insert before quantize_state_dict_int8
if "def quantize_int5_per_row" not in code:
    code = code.replace(
        "def quantize_state_dict_int8(",
        int5_code + "\ndef quantize_state_dict_int8("
    )
    changes.append("int5_functions")

# 2. Replace the serialization section to use mixed int5/int6 + zstd
# Add mixed quant after the int8 serialization
mixed_serial = '''
        # Mixed int5-MLP + int6-attn + zstd-22 (submission format)
        try:
            import zstandard
            quant_mixed, quant_mixed_meta = quantize_state_dict_mixed(base_model.state_dict())
            mixed_buf = io.BytesIO()
            torch.save(quant_mixed, mixed_buf)
            mixed_raw = mixed_buf.getvalue()
            mixed_blob = zstandard.ZstdCompressor(level=22).compress(mixed_raw)
            with open("final_model.mixed.ptz", "wb") as f:
                f.write(mixed_blob)
            log0(f"Serialized model mixed int5/int6+zstd: {len(mixed_blob)} bytes")
            log0(f"Total submission size mixed: {len(mixed_blob) + code_bytes} bytes")
            # Roundtrip validation
            mixed_state = torch.load(
                io.BytesIO(zstandard.ZstdDecompressor().decompress(mixed_blob)),
                map_location="cpu", weights_only=False
            )
            base_model.load_state_dict(dequantize_state_dict_mixed(mixed_state), strict=True)
            torch.cuda.synchronize()
            t_mqeval = time.perf_counter()
            mq_val_loss, mq_val_bpb = eval_val(
                args, model, rank, world_size, grad_accum_steps, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            torch.cuda.synchronize()
            log0(f"final_mixed_roundtrip val_loss:{mq_val_loss:.4f} val_bpb:{mq_val_bpb:.4f} "
                 f"eval_time:{1000.0 * (time.perf_counter() - t_mqeval):.0f}ms")
            log0(f"final_mixed_roundtrip_exact val_loss:{mq_val_loss:.8f} val_bpb:{mq_val_bpb:.8f}")
        except ImportError:
            log0("WARNING: zstandard not installed, skipping mixed quantization")
        except Exception as e:
            log0(f"WARNING: mixed quantization failed: {e}")
'''

# Insert after the int8 roundtrip logging
if "final_mixed_roundtrip" not in code:
    code = code.replace(
        'log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")',
        'log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")\n'
        + mixed_serial
    )
    changes.append("mixed_serialization")

# 3. Default seq_len to 2048
code = code.replace(
    'train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))',
    'train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))'
)
changes.append("seq2048_default")

# Verify
try:
    compile(code, TRAIN_GPT, 'exec')
    print(f"Exp 15 patch: Syntax OK, applied: {', '.join(changes)}")
except SyntaxError as e:
    print(f"Exp 15 patch: SYNTAX ERROR: {e}")
    import sys; sys.exit(1)

with open(TRAIN_GPT, "w") as f:
    f.write(code)
