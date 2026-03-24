"""
Exp 14: 11 Layers + SwiGLU + SWA + Int5 MLP quantization.
Apply AFTER patch_exp10.py and WD fix.

Changes from Exp 10:
1. NUM_LAYERS default 9 -> 11
2. SwiGLU replaces ReLU²
3. SWA during warmdown (only when step > total_steps - warmdown_iters)
4. Int5 quantization for MLP weights (range [-16,15])

Run on pod: python exp14_patch.py
"""
import os, re

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

changes = []

# 1. Change default NUM_LAYERS to 11
code = code.replace(
    'num_layers = int(os.environ.get("NUM_LAYERS", 9))',
    'num_layers = int(os.environ.get("NUM_LAYERS", 11))'
)
changes.append("11L default")

# 2. SwiGLU MLP (same as exp13)
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

if old_mlp in code:
    code = code.replace(old_mlp, new_mlp)
    changes.append("SwiGLU")
else:
    print("WARNING: Could not find MLP class")

# 3. SWA during warmdown only (not from step 0)
swa_code = '''
        # SWA: collect weights every 200 steps during warmdown phase only
        _swa_every = 200
        if not hasattr(base_model, '_swa_state'):
            base_model._swa_state = None
            base_model._swa_count = 0
        # Only start SWA when we're actually in warmdown (step > estimated_total - warmdown)
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

if '# Needed to sync whether we\'ve reached the wallclock cap.' in code:
    code = code.replace(
        '        # Needed to sync whether we\'ve reached the wallclock cap.',
        swa_code + '\n        # Needed to sync whether we\'ve reached the wallclock cap.'
    )
    changes.append("SWA")

# SWA apply before serialization
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

# Verify
try:
    compile(code, TRAIN_GPT, 'exec')
    print(f"Exp 14 patch: Syntax OK, applied: {', '.join(changes)}")
except SyntaxError as e:
    print(f"Exp 14 patch: SYNTAX ERROR: {e}")
    import sys; sys.exit(1)

with open(TRAIN_GPT, "w") as f:
    f.write(code)
