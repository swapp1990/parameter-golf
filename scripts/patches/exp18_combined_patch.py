"""
Exp 18: XSA + Partial RoPE + EMA combined patch.
Apply AFTER patch_exp10.py + WD fix + GQA fix + exp14_patch.py + exp17_xsa_patch.py + XSA GQA fix.

Changes:
1. Partial RoPE: only 16 of 64 dims get positional encoding
2. EMA: exponential moving average (decay=0.997) replaces SWA checkpoint averaging

Run on pod: python exp18_combined_patch.py
"""
import os

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

changes = []

# 1. Partial RoPE: only apply to first 16 dims (8 pairs)
old_apply = '''def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)'''

new_apply = '''def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Partial RoPE: only first 8 pairs (16 dims) get positional encoding
    rope_dims = 8
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    x1r, x1p = x1[..., :rope_dims], x1[..., rope_dims:]
    x2r, x2p = x2[..., :rope_dims], x2[..., rope_dims:]
    cr, sr = cos[..., :rope_dims], sin[..., :rope_dims]
    return torch.cat((x1r * cr + x2r * sr, x1p, x1r * (-sr) + x2r * cr, x2p), dim=-1)'''

if old_apply in code:
    code = code.replace(old_apply, new_apply)
    changes.append("partial_rope")
else:
    print("WARNING: Could not find apply_rotary_emb to replace")

# 2. EMA: replace SWA with exponential moving average
# The SWA code accumulates a running sum and divides at the end.
# EMA instead maintains: ema = decay * ema + (1-decay) * current

# Replace the SWA collection logic with EMA
old_swa_collect = """        # SWA: collect weights every 200 steps during warmdown phase only
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
                base_model._swa_count += 1"""

new_ema_collect = """        # EMA: exponential moving average (decay=0.997), collected every 10 steps during warmdown
        _ema_decay = 0.997
        _ema_every = 10
        if not hasattr(base_model, '_ema_state'):
            base_model._ema_state = None
            base_model._ema_started = False
        _est_total = int(max_wallclock_ms / (approx_training_time_ms / max(step, 1)))
        _warmdown_start = max(0, _est_total - args.warmdown_iters)
        if step >= _warmdown_start and step % _ema_every == 0 and step > 100:
            if base_model._ema_state is None:
                base_model._ema_state = {n: p.detach().cpu().clone().float() for n, p in base_model.state_dict().items()}
                base_model._ema_started = True
                log0(f"EMA: started at step {step} (warmdown_start~{_warmdown_start}, decay={_ema_decay})")
            else:
                for n, p in base_model.state_dict().items():
                    base_model._ema_state[n] = _ema_decay * base_model._ema_state[n] + (1 - _ema_decay) * p.detach().cpu().float()"""

if old_swa_collect in code:
    code = code.replace(old_swa_collect, new_ema_collect)
    changes.append("ema_collect")
else:
    print("WARNING: Could not find SWA collection code to replace")

# Replace SWA apply with EMA apply
old_swa_apply = """    # Apply SWA averaged weights
    if hasattr(base_model, '_swa_state') and base_model._swa_state is not None and base_model._swa_count > 1:
        log0(f"SWA: averaging {base_model._swa_count} checkpoints")
        for n, t in base_model._swa_state.items():
            avg = (t / base_model._swa_count).to(dtype=base_model.state_dict()[n].dtype)
            base_model.state_dict()[n].copy_(avg)
        log0("SWA: applied")"""

new_ema_apply = """    # Apply EMA weights
    if hasattr(base_model, '_ema_state') and base_model._ema_state is not None:
        log0("EMA: applying exponential moving average weights")
        for n, t in base_model._ema_state.items():
            base_model.state_dict()[n].copy_(t.to(dtype=base_model.state_dict()[n].dtype))
        log0("EMA: applied")"""

if old_swa_apply in code:
    code = code.replace(old_swa_apply, new_ema_apply)
    changes.append("ema_apply")
else:
    print("WARNING: Could not find SWA apply code to replace")

# Verify
try:
    compile(code, TRAIN_GPT, 'exec')
    print(f"Exp 18 patch: Syntax OK, applied: {', '.join(changes)}")
except SyntaxError as e:
    print(f"Exp 18 patch: SYNTAX ERROR at line {e.lineno}: {e.msg}")
    import sys; sys.exit(1)

with open(TRAIN_GPT, "w") as f:
    f.write(code)
