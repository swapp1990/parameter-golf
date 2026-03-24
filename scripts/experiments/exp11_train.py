"""
Exp 11 training wrapper. Patches train_gpt.py's data loading to use enriched shards.
Usage: python exp11_train.py [a|b|c]
  a = juncture-enriched data after step 500
  b = rare-bigram-enriched data after step 500
  c = control (standard data, same as Exp 10)

Each run: 300s wallclock, saves checkpoint to /runpod-volume/exp11X_checkpoint.pt
"""
import os, sys, re

os.chdir('/runpod-volume/parameter-golf')
EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else 'c'
WALLCLOCK = 300

print(f'=== Exp 11{EXPERIMENT} Training ===')

# Read the current (already patched with Exp10 changes) train_gpt.py
code = open('train_gpt.py').read()

# Patch 1: Save checkpoint with experiment name
code = code.replace(
    'torch.save(base_model.state_dict(), "final_model.pt")',
    f'torch.save(base_model.state_dict(), "/runpod-volume/exp11{EXPERIMENT}_checkpoint.pt")\n'
    '        torch.save(base_model.state_dict(), "final_model.pt")'
)

if EXPERIMENT in ('a', 'b'):
    enriched_file = {
        'a': '/runpod-volume/enriched_juncture.bin',
        'b': '/runpod-volume/enriched_rare_bigram.bin',
    }[EXPERIMENT]

    # Patch 2: After step 500, switch dataloader to enriched shard
    # Find where train_shards is defined and inject switching logic
    # The dataloader cycles through shards. We need to replace the shard list after step 500.

    # Strategy: patch the shard loading function to detect step count via a global
    # and return enriched data instead
    inject = f'''
# EXP11{EXPERIMENT.upper()}: Hard example data injection
import threading
_exp11_step_counter = [0]
_exp11_enriched_path = "{enriched_file}"
_exp11_original_load = load_data_shard

def _exp11_load_data_shard(file):
    \"\"\"After step 500, mix in enriched data 50% of the time.\"\"\"
    import random
    if _exp11_step_counter[0] > 500 and random.random() < 0.5:
        from pathlib import Path
        return _exp11_original_load(Path(_exp11_enriched_path))
    return _exp11_original_load(file)

load_data_shard = _exp11_load_data_shard
'''
    # Insert after all imports but before main training logic
    # Find a safe insertion point - after the model class definitions
    insert_point = code.find('def main(')
    if insert_point == -1:
        # No main function, insert before the first "if __name__" or "if master_process"
        insert_point = code.find('if __name__')
    if insert_point == -1:
        # Last resort: insert before "# Training loop" or similar
        insert_point = code.find('log0(f"train_batch_tokens')

    if insert_point != -1:
        code = code[:insert_point] + inject + code[insert_point:]
        print(f'  Injected enriched data loader for 11{EXPERIMENT}')
    else:
        print('  WARNING: Could not find injection point!')

    # Also patch the training loop to increment the counter
    # Find "step += 1" in the training loop
    code = code.replace(
        'step += 1',
        f'step += 1\n        _exp11_step_counter[0] = step'
    )
    print(f'  Patched step counter')

# Verify syntax
try:
    compile(code, 'train_gpt.py', 'exec')
    print('  Syntax OK')
except SyntaxError as e:
    print(f'  SYNTAX ERROR: {e}')
    sys.exit(1)

open('train_gpt.py', 'w').write(code)
print(f'  Patched train_gpt.py')

# Run training
print(f'\nStarting training ({WALLCLOCK}s)...')
print(f'{"="*40}')
os.environ.update({
    'TRAIN_BATCH_TOKENS': '65536',
    'WARMDOWN_ITERS': '3000',
    'MAX_WALLCLOCK_SECONDS': str(WALLCLOCK),
    'MLP_MULT': '3',
    'MUON_WD': '0.04',
    'PYTHONUNBUFFERED': '1',
})
os.execvp('python', ['python', 'train_gpt.py'])
