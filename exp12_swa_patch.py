"""
Patch train_gpt.py to add SWA during warmdown.
Accumulates model states every SWA_EVERY steps when cosine LR scale < 0.5.
After training, averages all accumulated states before serialization.

Run on pod: python exp12_swa_patch.py
"""
import os

TRAIN_GPT = os.environ.get("TRAIN_GPT_PATH", "/runpod-volume/parameter-golf/train_gpt.py")

with open(TRAIN_GPT, "r") as f:
    code = f.read()

# 1. Add SWA env var
code = code.replace(
    'warmdown_iters = int(os.environ.get("WARMDOWN_ITERS',
    'swa_every = int(os.environ.get("SWA_EVERY", "100"))\n'
    '    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS'
)

# 2. Add SWA accumulation in the training loop
# Find the wallclock check and insert SWA logic before it
swa_logic = '''
        # SWA: accumulate weights during warmdown (when LR scale < 0.5)
        if not hasattr(model, '_swa_state'):
            model._swa_state = None
            model._swa_count = 0
        if hasattr(args, 'warmdown_iters') and args.warmdown_iters > 0:
            total_est_steps = step + max(1, int((max_wallclock_ms - approx_training_time_ms) / (approx_training_time_ms / max(step, 1))))
            warmdown_start = max(0, total_est_steps - args.warmdown_iters)
            if step >= warmdown_start and step % swa_every == 0:
                if model._swa_state is None:
                    model._swa_state = {n: p.detach().cpu().clone().float() for n, p in base_model.state_dict().items()}
                    model._swa_count = 1
                    log0(f"SWA: started collecting at step {step}")
                else:
                    for n, p in base_model.state_dict().items():
                        model._swa_state[n] += p.detach().cpu().float()
                    model._swa_count += 1
'''

code = code.replace(
    '        # Needed to sync whether we\'ve reached the wallclock cap.',
    swa_logic + '\n        # Needed to sync whether we\'ve reached the wallclock cap.'
)

# 3. Apply SWA average before serialization
swa_apply = '''
    # Apply SWA if we collected checkpoints
    if hasattr(model, '_swa_state') and model._swa_state is not None and model._swa_count > 1:
        log0(f"SWA: averaging {model._swa_count} checkpoints")
        avg_state = {}
        for n, t in model._swa_state.items():
            avg_state[n] = (t / model._swa_count).to(dtype=base_model.state_dict()[n].dtype)
        base_model.load_state_dict(avg_state, strict=True)
        log0("SWA: applied averaged weights")
'''

code = code.replace(
    '    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")',
    swa_apply + '\n    if master_process:\n        torch.save(base_model.state_dict(), "final_model.pt")'
)

# Verify
try:
    compile(code, TRAIN_GPT, 'exec')
    print("SWA patch: Syntax OK")
except SyntaxError as e:
    print(f"SWA patch: SYNTAX ERROR: {e}")
    import sys; sys.exit(1)

with open(TRAIN_GPT, "w") as f:
    f.write(code)
print(f"SWA patch applied to {TRAIN_GPT}")
