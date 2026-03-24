"""
Patch train_gpt.py to replace ReLU-squared MLP with SwiGLU.
Keeps total MLP params constant by adjusting hidden dim.

ReLU²: hidden = mlp_mult * dim (2 matrices: fc, proj)
  params = 2 * dim * hidden
SwiGLU: hidden = 2/3 * mlp_mult * dim (3 matrices: gate, up, proj)
  params = 3 * dim * hidden = 3 * dim * (2/3 * mlp_mult * dim) = 2 * dim * mlp_mult * dim
  Same total params!

Run on pod: python exp13_swiglu_patch.py
"""
import os

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

# Replace the MLP class
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
    # 3 matrices at 2/3 hidden dim = same param count as 2 matrices at full hidden
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(2 * mlp_mult * dim / 3)
        # Round to nearest multiple of 64 for efficiency
        hidden = ((hidden + 63) // 64) * 64
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.up = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.up(x))'''

if old_mlp in code:
    code = code.replace(old_mlp, new_mlp)
    print("SwiGLU patch: Replaced MLP class")
else:
    print("SwiGLU patch: WARNING - Could not find MLP class to replace")
    # Try to find it with different whitespace
    import re
    if re.search(r'class MLP.*relu.*square', code, re.DOTALL):
        print("  Found MLP class but exact match failed - check whitespace")
    import sys; sys.exit(1)

# Verify
try:
    compile(code, TRAIN_GPT, 'exec')
    print("SwiGLU patch: Syntax OK")
except SyntaxError as e:
    print(f"SwiGLU patch: SYNTAX ERROR: {e}")
    import sys; sys.exit(1)

with open(TRAIN_GPT, "w") as f:
    f.write(code)
print(f"SwiGLU patch applied to {TRAIN_GPT}")
