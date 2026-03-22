"""
Patch train_gpt.py to add XSA (Cross-token Self-Attention bias removal).
Applied to last N layers only (default 4).

XSA removes the self-value projection from attention output:
  z = y - (dot(y, v_self) / dot(v_self, v_self)) * v_self

This forces attention to contribute only contextual information
rather than redundantly passing through the token's own features.

Apply AFTER patch_exp10.py + exp14_patch.py + WD fix.
Run on pod: python exp17_xsa_patch.py
"""
import os

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

changes = []

# 1. Add XSA flag to CausalSelfAttention
# Modify __init__ to accept use_xsa parameter
code = code.replace(
    "class CausalSelfAttention(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        rope_base: float,\n        qk_gain_init: float,\n    ):",
    "class CausalSelfAttention(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        rope_base: float,\n        qk_gain_init: float,\n        use_xsa: bool = False,\n    ):"
)
changes.append("xsa_param")

# 2. Store use_xsa flag
code = code.replace(
    "        self.rotary = Rotary(self.head_dim, base=rope_base)",
    "        self.rotary = Rotary(self.head_dim, base=rope_base)\n        self.use_xsa = use_xsa"
)
changes.append("xsa_flag")

# 3. Add XSA logic after SDPA, before the reshape/proj
# Find the line after SDPA output and before reshape
old_post_attn = "        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)\n        return self.proj(y)"

new_post_attn = """        # XSA: remove self-value projection from attention output
        if self.use_xsa:
            # v may need expanding for GQA
            if self.num_heads != self.num_kv_heads:
                v_expanded = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            else:
                v_expanded = v
            # y shape: (bsz, num_heads, seqlen, head_dim)
            # v_expanded shape: (bsz, num_heads, seqlen, head_dim)
            dot_yv = (y * v_expanded).sum(dim=-1, keepdim=True)
            dot_vv = (v_expanded * v_expanded).sum(dim=-1, keepdim=True).clamp_min(1e-8)
            y = y - (dot_yv / dot_vv) * v_expanded
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)"""

code = code.replace(old_post_attn, new_post_attn)
changes.append("xsa_logic")

# 4. Pass use_xsa to last N layers in Block creation
# Modify Block.__init__ to accept and pass use_xsa
code = code.replace(
    "class Block(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        mlp_mult: int,\n        rope_base: float,\n        qk_gain_init: float,\n    ):",
    "class Block(nn.Module):\n    def __init__(\n        self,\n        dim: int,\n        num_heads: int,\n        num_kv_heads: int,\n        mlp_mult: int,\n        rope_base: float,\n        qk_gain_init: float,\n        use_xsa: bool = False,\n    ):"
)
changes.append("block_xsa_param")

code = code.replace(
    "        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)",
    "        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)"
)
changes.append("block_pass_xsa")

# 5. In GPT.__init__, enable XSA for last 4 layers
xsa_last_n = 4  # Apply XSA to last 4 layers

code = code.replace(
    "                Block(\n                    model_dim,\n                    num_heads,\n                    num_kv_heads,\n                    mlp_mult,\n                    rope_base,\n                    qk_gain_init,\n                )\n                for i in range(num_layers)",
    f"                Block(\n                    model_dim,\n                    num_heads,\n                    num_kv_heads,\n                    mlp_mult,\n                    rope_base,\n                    qk_gain_init,\n                    use_xsa=(i >= num_layers - {xsa_last_n}),\n                )\n                for i in range(num_layers)"
)
changes.append(f"xsa_last_{xsa_last_n}")

# Verify
try:
    compile(code, TRAIN_GPT, 'exec')
    print(f"XSA patch: Syntax OK, applied: {', '.join(changes)}")
except SyntaxError as e:
    print(f"XSA patch: SYNTAX ERROR at line {e.lineno}: {e.msg}")
    # Show context
    lines = code.split('\n')
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
        marker = '>>>' if i == e.lineno-1 else '   '
        print(f"  {marker} {i+1}: {lines[i]}")
    import sys; sys.exit(1)

with open(TRAIN_GPT, "w") as f:
    f.write(code)
