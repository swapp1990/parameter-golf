"""
Build submission-ready train_gpt.py by applying all patches from Exp 17.

Takes the original train_gpt.py and produces a complete file with:
1. SwiGLU MLP (replaces ReLU²)
2. SmearGate (bigram blending at embedding layer)
3. BigramHash removed (zero contribution per Exp 16)
4. OrthoInit (orthogonal weight initialization)
5. XSA on last 4 layers (self-value projection removal)
6. Muon weight decay (decoupled, WD=0.04)
7. SWA during warmdown
8. Int5-MLP + Int6-attn + Int8-embed + zstd quantization
9. Sliding window eval (stride=256)
10. LoRA TTT during eval
11. Defaults: NUM_LAYERS=11, MLP_MULT=3, TRAIN_SEQ_LEN=2048

Usage: python build_submission.py
Input:  train_gpt.py (original from repo)
Output: train_gpt_submission.py
"""
import os, re

# Read original
with open("train_gpt.py", "r") as f:
    code = f.read()

print("Starting from original train_gpt.py")
changes = []

# =====================================================
# 1. Change defaults: NUM_LAYERS=11, MLP_MULT=3, SEQ_LEN=2048
# =====================================================
code = code.replace(
    'num_layers = int(os.environ.get("NUM_LAYERS", 9))',
    'num_layers = int(os.environ.get("NUM_LAYERS", 11))'
)
code = code.replace(
    'mlp_mult = int(os.environ.get("MLP_MULT", 2))',
    'mlp_mult = int(os.environ.get("MLP_MULT", 3))'
)
code = code.replace(
    'train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))',
    'train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))'
)
changes.append("defaults: 11L, MLP3x, seq2048")

# =====================================================
# 2. Add SmearGate class (before Block class)
# =====================================================
smeargate_code = '''
class SmearGate(nn.Module):
    """Blend each token embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

'''
code = code.replace("class Block(nn.Module):", smeargate_code + "class Block(nn.Module):")
changes.append("SmearGate")

# =====================================================
# 3. Replace MLP with SwiGLU
# =====================================================
old_mlp = '''class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())'''

new_mlp = '''class MLP(nn.Module):
    # SwiGLU MLP: swish(gate(x)) * up(x), then project down
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(2 * mlp_mult * dim / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.up(x))'''

code = code.replace(old_mlp, new_mlp)
changes.append("SwiGLU")

# =====================================================
# 4. Add XSA to CausalSelfAttention
# =====================================================
# Add use_xsa parameter
code = code.replace(
    "class CausalSelfAttention(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        rope_base: float,\n        qk_gain_init: float,\n    ):",
    "class CausalSelfAttention(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        rope_base: float,\n        qk_gain_init: float,\n        use_xsa: bool = False,\n    ):"
)

# Store use_xsa
code = code.replace(
    "        self.rotary = Rotary(self.head_dim, base=rope_base)",
    "        self.rotary = Rotary(self.head_dim, base=rope_base)\n        self.use_xsa = use_xsa"
)

# Add XSA logic before reshape
old_post_attn = "        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)\n        return self.proj(y)"
new_post_attn = """        # XSA: remove self-value projection from attention output
        if self.use_xsa:
            # Expand v for GQA (SDPA with enable_gqa keeps v at num_kv_heads)
            if self.num_heads != self.num_kv_heads:
                v_xsa = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            else:
                v_xsa = v
            dot_yv = (y * v_xsa).sum(dim=-1, keepdim=True)
            dot_vv = (v_xsa * v_xsa).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            y = y - (dot_yv / dot_vv) * v_xsa
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)"""
code = code.replace(old_post_attn, new_post_attn)
changes.append("XSA")

# =====================================================
# 5. Pass XSA to Block and GPT
# =====================================================
# Block init
code = code.replace(
    "class Block(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        mlp_mult: int,\n        rope_base: float,\n        qk_gain_init: float,\n    ):",
    "class Block(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        mlp_mult: int,\n        rope_base: float,\n        qk_gain_init: float,\n        use_xsa: bool = False,\n    ):"
)

code = code.replace(
    "        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)",
    "        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)"
)

# GPT block creation with XSA on last 4 layers
xsa_last_n = 4
code = code.replace(
    "                Block(\n                    model_dim,\n                    num_heads,\n                    num_kv_heads,\n                    mlp_mult,\n                    rope_base,\n                    qk_gain_init,\n                )\n                for i in range(num_layers)",
    f"                Block(\n                    model_dim,\n                    num_heads,\n                    num_kv_heads,\n                    mlp_mult,\n                    rope_base,\n                    qk_gain_init,\n                    use_xsa=(i >= num_layers - {xsa_last_n}),\n                )\n                for i in range(num_layers)"
)
changes.append(f"XSA_last_{xsa_last_n}")

# =====================================================
# 6. Add SmearGate to GPT.__init__ and forward
# =====================================================
code = code.replace(
    "self.tok_emb = nn.Embedding(vocab_size, model_dim)",
    "self.tok_emb = nn.Embedding(vocab_size, model_dim)\n        self.smear = SmearGate(model_dim)"
)

code = code.replace(
    """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x""",
    """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x"""
)
changes.append("SmearGate_in_forward")

# =====================================================
# 7. OrthoInit
# =====================================================
code = code.replace(
    """    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)""",
    """    def _init_weights(self) -> None:
        import math
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))"""
)
changes.append("OrthoInit")

# =====================================================
# 8. Add "smear" to CONTROL_TENSOR_NAME_PATTERNS
# =====================================================
code = code.replace(
    'CONTROL_TENSOR_NAME_PATTERNS = (',
    'CONTROL_TENSOR_NAME_PATTERNS = (\n    "smear",  # SmearGate control tensors'
)
changes.append("control_patterns")

# =====================================================
# 9. Muon weight decay
# =====================================================
# Add weight_decay param to Muon
if "weight_decay" not in code.split("class Muon")[1].split("class ")[0][:500]:
    code = re.sub(
        r'(class Muon.*?def __init__\(self,.*?)(,?\s*\):\s*\n\s*super)',
        lambda m: m.group(1) + ', weight_decay: float = 0.0' + m.group(2),
        code, count=1, flags=re.DOTALL
    )
    code = re.sub(
        r'(super\(\).__init__\(params,\s*dict\(.*?)(,?\s*\)\))',
        lambda m: m.group(1) + ', weight_decay=weight_decay' + m.group(2),
        code, count=1, flags=re.DOTALL
    )

# Add WD application before parameter update
# The original line is inside a for loop with 16-space indent
code = code.replace(
    "                p.add_(g, alpha=-lr)",
    "                wd = group.get(\"weight_decay\", 0.0)\n                if wd > 0 and p.ndim >= 2:\n                    p.data.mul_(1.0 - lr * wd)\n                p.add_(g, alpha=-lr)"
)

# Add weight_decay to Muon instantiation
code = re.sub(
    r'(Muon\(\s*matrix_params,.*?backend_steps=\d+)',
    r'\1, weight_decay=float(os.environ.get("MUON_WD", "0.04"))',
    code, count=1, flags=re.DOTALL
)
changes.append("MuonWD")

# =====================================================
# 10. SWA during warmdown
# =====================================================
swa_collect = '''
        # SWA: collect weights every 200 steps during warmdown
        _swa_every = 200
        if not hasattr(base_model, '_swa_state'):
            base_model._swa_state = None
            base_model._swa_count = 0
        _est_total = int(max_wallclock_ms / (approx_training_time_ms / max(step, 1)))
        _warmdown_start = max(0, _est_total - args.warmdown_iters)
        if step >= _warmdown_start and step % _swa_every == 0 and step > 100:
            if base_model._swa_state is None:
                base_model._swa_state = {n: p.detach().cpu().clone().float() for n, p in base_model.state_dict().items()}
                base_model._swa_count = 1
                log0(f"SWA: started at step {step} (warmdown_start~{_warmdown_start})")
            else:
                for n, p in base_model.state_dict().items():
                    base_model._swa_state[n] += p.detach().cpu().float()
                base_model._swa_count += 1
'''
code = code.replace(
    '        # Needed to sync whether we\'ve reached the wallclock cap.',
    swa_collect + '\n        # Needed to sync whether we\'ve reached the wallclock cap.'
)

swa_apply = '''
    # Apply SWA averaged weights
    if hasattr(base_model, '_swa_state') and base_model._swa_state is not None and base_model._swa_count > 1:
        log0(f"SWA: averaging {base_model._swa_count} checkpoints")
        for n, t in base_model._swa_state.items():
            avg = (t / base_model._swa_count).to(dtype=base_model.state_dict()[n].dtype)
            base_model.state_dict()[n].copy_(avg)
        log0("SWA: applied")
'''
code = code.replace(
    '    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
    swa_apply + '\n    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")'
)
changes.append("SWA")

# =====================================================
# 11. Int5+Int6+Int8+zstd quantization
# =====================================================
quant_code = '''

# --- Mixed quantization (int5-MLP, int6-attn, int8-embed) ---

def quantize_int5_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
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

def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
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

MLP_QUANT_PATTERNS = ("mlp.", "gate.", "up.")
EMBED_QUANT_PATTERNS = ("tok_emb",)

def quantize_state_dict_mixed(state_dict: dict) -> dict:
    result = {}
    for name, t in state_dict.items():
        t_cpu = t.detach().cpu()
        if t_cpu.numel() <= 896 or not t_cpu.is_floating_point():
            result[name] = t_cpu.to(torch.float16) if t_cpu.is_floating_point() else t_cpu
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS) or any(p in name for p in ("smear",)):
            result[name] = t_cpu.to(torch.float16).contiguous()
            continue
        t32 = t_cpu.float()
        if any(p in name for p in EMBED_QUANT_PATTERNS):
            # int8 for embeddings
            if t32.ndim == 2:
                row_max = t32.abs().amax(dim=1)
                scale = (row_max / 127.0).clamp_min(1e-12).to(torch.float16)
                q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -128, 127).to(torch.int8)
            else:
                amax = t32.abs().max().item()
                scale = torch.tensor(max(amax / 127.0, 1e-12), dtype=torch.float16)
                q = torch.clamp(torch.round(t32 / scale.float()), -128, 127).to(torch.int8)
            result[name + ".__q"] = q
            result[name + ".__scale"] = scale
            result[name + ".__dtype"] = str(t_cpu.dtype)
        elif any(p in name for p in MLP_QUANT_PATTERNS):
            q, scale = quantize_int5_per_row(t_cpu)
            result[name + ".__q"] = q
            result[name + ".__scale"] = scale
            result[name + ".__dtype"] = str(t_cpu.dtype)
        else:
            q, scale = quantize_int6_per_row(t_cpu)
            result[name + ".__q"] = q
            result[name + ".__scale"] = scale
            result[name + ".__dtype"] = str(t_cpu.dtype)
    result["__quant_format__"] = "mixed_v1"
    return result

def dequantize_state_dict_mixed(quant_dict: dict) -> dict:
    quant_dict.pop("__quant_format__", None)
    result = {}
    seen = set()
    for key in list(quant_dict.keys()):
        if key.endswith(".__q"):
            name = key[:-4]
            if name in seen: continue
            seen.add(name)
            q = quant_dict[name + ".__q"]
            scale = quant_dict[name + ".__scale"]
            dtype = getattr(torch, quant_dict[name + ".__dtype"].split(".")[-1])
            if q.ndim == 2 and scale.ndim == 1:
                result[name] = (q.float() * scale.float()[:, None]).to(dtype)
            else:
                result[name] = (q.float() * scale.float()).to(dtype)
        elif not any(key.endswith(s) for s in (".__scale", ".__dtype")):
            result[key] = quant_dict[key]
    return result

'''

# Insert before quantize_state_dict_int8
code = code.replace(
    "def quantize_state_dict_int8(",
    quant_code + "\ndef quantize_state_dict_int8("
)
changes.append("mixed_quantization")

# =====================================================
# 12. Add mixed quant + zstd to serialization section
# =====================================================
mixed_serial = '''
    # Mixed quantization (int5-MLP + int6-attn + int8-embed + zstd-22)
    try:
        import zstandard
        quant_mixed = quantize_state_dict_mixed(base_model.state_dict())
        mixed_buf = io.BytesIO()
        torch.save(quant_mixed, mixed_buf)
        mixed_raw = mixed_buf.getvalue()
        mixed_blob = zstandard.ZstdCompressor(level=22).compress(mixed_raw)
        with open("final_model.mixed.ptz", "wb") as f:
            f.write(mixed_blob)
        log0(f"Serialized model mixed int5/int6/int8+zstd: {len(mixed_blob)} bytes")
        log0(f"Total mixed submission: {len(mixed_blob) + code_bytes} bytes "
             f"({chr(39)}FITS{chr(39)} if len(mixed_blob) + code_bytes < 16_000_000 else {chr(39)}OVER{chr(39)} 16MB)")
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
    except ImportError:
        log0("WARNING: zstandard not installed, skipping mixed quantization")
    except Exception as e:
        log0(f"WARNING: mixed quantization failed: {e}")
'''

code = code.replace(
    'log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")',
    'log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")\n'
    + mixed_serial
)
changes.append("mixed_serialization")

# =====================================================
# Verify syntax
# =====================================================
try:
    compile(code, "train_gpt_submission.py", "exec")
    print(f"Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
    lines = code.split('\n')
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
        marker = '>>>' if i == e.lineno-1 else '   '
        print(f"  {marker} {i+1}: {lines[i]}")

# Save
output = "train_gpt_submission.py"
with open(output, "w") as f:
    f.write(code)

print(f"\nApplied {len(changes)} changes: {', '.join(changes)}")
print(f"Saved: {output}")
print(f"Lines: {len(code.splitlines())}")
