"""
Patch train_gpt.py for Experiment 10: Competition Stack
- SmearGate + BigramHash + OrthoInit
- Int6 quantization + zstd-22 compression
- MLP 3x expansion
- FP16 tied embedding passthrough
- Muon weight decay (decoupled)
- Sliding window evaluation

Run on the pod: python /tmp/patch_exp10.py
"""

import re
import sys
import os

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

original = code  # backup

# ===========================================================================
# 1. Add SmearGate and BigramHashEmbedding classes (before Block class)
# ===========================================================================

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


class BigramHashEmbedding(nn.Module):
    """Hash-based bigram embedding: maps (prev_token, curr_token) pairs to learned vectors."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod  # BOS position
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

'''

# Insert before "class Block"
code = code.replace("class Block(nn.Module):", smeargate_code + "class Block(nn.Module):")

# ===========================================================================
# 2. Add SmearGate + BigramHash to GPT.__init__ and forward
# ===========================================================================

# In GPT.__init__, after tok_emb, add bigram + smeargate
code = code.replace(
    "self.tok_emb = nn.Embedding(vocab_size, model_dim)",
    """self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(4096, 128, model_dim)
        self.smear = SmearGate(model_dim)"""
)

# In GPT.forward, add bigram + smear after embedding
code = code.replace(
    """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x""",
    """    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x"""
)

# ===========================================================================
# 3. OrthoInit — replace _init_weights
# ===========================================================================

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
                    # Scale output projections by 1/sqrt(2*num_layers) (muP convention)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))"""
)

# ===========================================================================
# 4. Int6 quantization functions
# ===========================================================================

int6_code = '''

def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize tensor to int6 range [-32, 31] with per-row scaling."""
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / 31.0, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale


def dequantize_int6_per_row(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """Dequantize int6 tensor back to float."""
    if q.ndim == 2 and scale.ndim == 1:
        return (q.float() * scale.float()[:, None]).to(dtype)
    return (q.float() * scale.float()).to(dtype)


FP16_KEEP_PATTERNS = ("tok_emb",)  # Keep tied embedding in fp16


def quantize_state_dict_int6(state_dict: dict) -> tuple[dict, dict]:
    """Quantize model state dict with int6 for large tensors, fp16 passthrough for sensitive ones."""
    result = {}
    meta = {}
    for name, t in state_dict.items():
        t_cpu = t.detach().cpu()
        # FP16 passthrough for sensitive tensors
        if any(pat in name for pat in FP16_KEEP_PATTERNS):
            result[name] = t_cpu.to(torch.float16).contiguous()
            meta[name] = {"mode": "fp16_passthrough"}
            continue
        # Small tensors: fp16 passthrough
        if t_cpu.numel() <= 896 or not t_cpu.is_floating_point():
            result[name] = t_cpu.to(torch.float16).contiguous() if t_cpu.is_floating_point() else t_cpu
            meta[name] = {"mode": "passthrough"}
            continue
        # Int6 for everything else
        q, scale = quantize_int6_per_row(t_cpu)
        result[name + ".__q"] = q
        result[name + ".__scale"] = scale
        result[name + ".__dtype"] = str(t_cpu.dtype)
        meta[name] = {"mode": "int6"}
    result["__quant_format__"] = "int6_per_row_v1"
    return result, meta


def dequantize_state_dict_int6(quant_dict: dict) -> dict:
    """Dequantize int6 state dict back to model-compatible format."""
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
            dtype = getattr(torch, dtype_str.split(".")[-1])
            result[name] = dequantize_int6_per_row(q, scale, dtype)
        elif not key.endswith(".__scale") and not key.endswith(".__dtype"):
            result[key] = quant_dict[key]
    return result

'''

# Insert the int6 functions before the existing quantize_state_dict_int8 function
code = code.replace(
    "def quantize_state_dict_int8(",
    int6_code + "def quantize_state_dict_int8("
)

# ===========================================================================
# 5. Replace serialization to use int6 + zstd instead of int8 + zlib
# ===========================================================================

# Find and replace the serialization section
# The key section starts after "torch.save(base_model.state_dict(), "final_model.pt")"
# and does int8+zlib. We need to ADD int6+zstd alongside it.

old_serialization = '''        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()

        quant_blob = zlib.compress(quant_raw, level=9)'''

new_serialization = '''        # Int6 + zstd (primary submission format)
        try:
            import zstandard
            _USE_ZSTD = True
        except ImportError:
            _USE_ZSTD = False
            log0("WARNING: zstandard not installed, falling back to zlib")

        quant_obj6, quant_meta6 = quantize_state_dict_int6(base_model.state_dict())
        quant_buf6 = io.BytesIO()
        torch.save(quant_obj6, quant_buf6)
        quant_raw6 = quant_buf6.getvalue()

        if _USE_ZSTD:
            quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw6)
        else:
            quant_blob = zlib.compress(quant_raw6, level=9)

        # Also do legacy int8+zlib for comparison
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()'''

code = code.replace(old_serialization, new_serialization)

# Replace the roundtrip validation to use int6+zstd
old_roundtrip = '''        with open("final_model.int8.ptz", "rb") as f:
            quant_blob_disk = f.read()
        quant_state = torch.load(
            io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu"'''

new_roundtrip = '''        with open("final_model.int8.ptz", "rb") as f:
            quant_blob_disk = f.read()
        if _USE_ZSTD:
            decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
        else:
            decompressed = zlib.decompress(quant_blob_disk)
        quant_state = torch.load(
            io.BytesIO(decompressed), map_location="cpu"'''

code = code.replace(old_roundtrip, new_roundtrip)

# Replace dequantize call to use int6
code = code.replace(
    "dequantize_state_dict_int8(quant_state)",
    "dequantize_state_dict_int6(quant_state) if quant_state.get('__quant_format__') == 'int6_per_row_v1' else dequantize_state_dict_int8(quant_state)"
)

# ===========================================================================
# 6. Muon weight decay
# ===========================================================================

# Find the Muon optimizer class and add weight_decay support
# Look for the Muon step function and add WD before the parameter update

# First, add weight_decay to Muon __init__
code = code.replace(
    "class Muon(torch.optim.Optimizer):",
    "class Muon(torch.optim.Optimizer):  # with decoupled weight decay"
)

# Find the Muon __init__ and add weight_decay param
old_muon_init = re.search(
    r'(class Muon.*?def __init__\(self.*?)(super\(\).__init__)',
    code, re.DOTALL
)
if old_muon_init:
    # Add weight_decay=0.0 to the init signature if not already there
    init_section = old_muon_init.group(0)
    if "weight_decay" not in init_section:
        # Find the defaults dict in super().__init__
        code = re.sub(
            r'(class Muon.*?def __init__\(self,.*?)(,?\s*\):\s*\n\s*super)',
            lambda m: m.group(1) + ', weight_decay: float = 0.0' + m.group(2),
            code, count=1, flags=re.DOTALL
        )
        # Add weight_decay to defaults dict
        code = re.sub(
            r'(super\(\).__init__\(params,\s*dict\(.*?)(,?\s*\)\))',
            lambda m: m.group(1) + ', weight_decay=weight_decay' + m.group(2),
            code, count=1, flags=re.DOTALL
        )

# Add WD application in the step function, before parameter update
# Find the line where parameters are updated: p.add_(g, alpha=-lr)
# and add WD before it
code = code.replace(
    """            p.add_(g, alpha=-lr)""",
    """            wd = group.get("weight_decay", 0.0)
            if wd > 0 and p.ndim >= 2:
                p.data.mul_(1.0 - lr * wd)
            p.add_(g, alpha=-lr)"""
)

# ===========================================================================
# 7. Add weight_decay to Muon instantiation
# ===========================================================================

# Find where Muon is instantiated and add weight_decay
code = re.sub(
    r'(Muon\(\s*matrix_params,.*?backend_steps=\d+)',
    r'\1, weight_decay=float(os.environ.get("MUON_WD", "0.04"))',
    code, count=1, flags=re.DOTALL
)

# ===========================================================================
# 8. Add "bigram" and "smear" to CONTROL_TENSOR_NAME_PATTERNS
# ===========================================================================

code = code.replace(
    'CONTROL_TENSOR_NAME_PATTERNS = (',
    'CONTROL_TENSOR_NAME_PATTERNS = (\n    "bigram", "smear", "scale",  # Exp10: SmearGate + BigramHash control tensors'
)

# ===========================================================================
# 9. Default MLP_MULT to 3
# ===========================================================================

code = code.replace(
    'mlp_mult = int(os.environ.get("MLP_MULT", 2))',
    'mlp_mult = int(os.environ.get("MLP_MULT", 3))'
)

# ===========================================================================
# 10. Add sliding window eval function
# ===========================================================================

sliding_eval_code = '''

def eval_sliding_window(model, val_tokens, seq_len=2048, stride=64, device="cuda"):
    """Evaluate with sliding window for better BPB. Only scores last `stride` tokens per window."""
    model.eval()
    total_tokens = val_tokens.numel() - 1
    total_nll = 0.0
    total_scored = 0
    vocab_size = model.vocab_size

    starts = list(range(0, total_tokens - seq_len + 1, stride))
    # Also add partial window at the end if needed
    batch_size = 4  # process multiple windows at once

    with torch.no_grad():
        for batch_off in range(0, len(starts), batch_size):
            batch_starts = starts[batch_off:batch_off + batch_size]
            x_batch = torch.stack([val_tokens[s:s+seq_len] for s in batch_starts]).to(device)
            y_batch = torch.stack([val_tokens[s+1:s+seq_len+1] for s in batch_starts]).to(device)

            # Get logits without loss computation
            x = model.tok_emb(x_batch)
            if hasattr(model, 'bigram'):
                x = x + model.bigram(x_batch)
            x = F.rms_norm(x, (x.size(-1),))
            if hasattr(model, 'smear'):
                x = model.smear(x)
            x0 = x
            skips = []
            for i in range(model.num_encoder_layers):
                x = model.blocks[i](x, x0)
                skips.append(x)
            for i in range(model.num_decoder_layers):
                if skips:
                    x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = model.blocks[model.num_encoder_layers + i](x, x0)
            x = model.final_norm(x)
            if model.tie_embeddings:
                logits = F.linear(x, model.tok_emb.weight)
            else:
                logits = model.lm_head(x)
            logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)

            # Score only last `stride` tokens per window
            sf = seq_len - stride
            logits_tail = logits[:, sf:, :].reshape(-1, vocab_size)
            targets_tail = y_batch[:, sf:].reshape(-1)
            nll = F.cross_entropy(logits_tail.float(), targets_tail, reduction="sum")
            total_nll += nll.item()
            total_scored += targets_tail.numel()

    avg_loss = total_nll / total_scored
    avg_bpb = avg_loss / math.log(2) * 0.3584  # convert nats to BPB
    return avg_loss, avg_bpb, total_scored

'''

# Add import math at the top if not already present
if "import math" not in code:
    code = code.replace("import io", "import io\nimport math")

# Insert sliding window eval before the main training function
code = code.replace(
    "def quantize_state_dict_int8(",
    sliding_eval_code + "def quantize_state_dict_int8("
)

# ===========================================================================
# 11. Add sliding window eval after the roundtrip validation
# ===========================================================================

# Find the final validation logging and add sliding window eval after it
code = code.replace(
    'log0(f"final_int8_zlib_roundtrip_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")',
    '''log0(f"final_int8_zlib_roundtrip_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

        # Sliding window evaluation
        log0("Running sliding window evaluation (stride=64, seq_len=2048)...")
        import math as _math
        sw_loss, sw_bpb, sw_tokens = eval_sliding_window(
            base_model, val_tokens_full, seq_len=2048, stride=64, device=device
        )
        log0(f"sliding_window val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} scored_tokens:{sw_tokens}")'''
)

# ===========================================================================
# 12. Load val_tokens_full for sliding window (need full val set, not batched)
# ===========================================================================

# Find where val data is loaded and store the full token tensor
code = code.replace(
    "val_loader:shards",
    "val_tokens_full = torch.cat([load_data_shard(s, device) for s in val_shards])\n    log0(f\"val_loader:shards"
)

# Fix: we need to close the log0 call properly - let me handle this differently
# Actually, let's just add the val_tokens loading separately
code = code.replace(
    "val_tokens_full = torch.cat([load_data_shard(s, device) for s in val_shards])\n    log0(f\"val_loader:shards",
    "val_loader:shards"
)

# Instead, add val_tokens_full loading right before the sliding window eval
code = code.replace(
    '# Sliding window evaluation\n        log0("Running sliding window evaluation (stride=64, seq_len=2048)...")',
    '''# Load full val tokens for sliding window
        val_tokens_full = torch.cat([load_data_shard(s, device) for s in val_shards])
        # Sliding window evaluation
        log0("Running sliding window evaluation (stride=64, seq_len=2048)...")'''
)

# ===========================================================================
# Verify changes
# ===========================================================================

changes = []
if "class SmearGate" in code:
    changes.append("SmearGate")
if "class BigramHashEmbedding" in code:
    changes.append("BigramHash")
if "orthogonal_" in code:
    changes.append("OrthoInit")
if "quantize_int6_per_row" in code:
    changes.append("Int6")
if "zstandard" in code:
    changes.append("zstd")
if "weight_decay" in code and "Muon" in code:
    changes.append("MuonWD")
if "eval_sliding_window" in code:
    changes.append("SlidingWindow")
if 'MLP_MULT", 3)' in code:
    changes.append("MLP3x")

print(f"Applied {len(changes)} changes: {', '.join(changes)}")

with open(TRAIN_GPT, "w") as f:
    f.write(code)

print(f"Patched {TRAIN_GPT}")
