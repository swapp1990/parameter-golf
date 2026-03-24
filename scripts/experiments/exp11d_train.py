"""
Exp 11d: Loss-based hard example mining (one-time scoring at 80% mark)

Steps 1-1260 (80%): Normal training
Step 1260: Score 1000 batches, cache top 5% highest-loss sequences
Steps 1260-1580 (20%): Train on cached hard set only

Usage: python exp11d_train.py
Saves checkpoint to /runpod-volume/exp11d_checkpoint.pt
"""
import os, sys, time, struct, glob, math, random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')

os.environ['MLP_MULT'] = '3'
os.environ['MUON_WD'] = '0.04'
os.environ['TRAIN_BATCH_TOKENS'] = '65536'
os.environ['TRAIN_SEQ_LEN'] = '1024'

TOTAL_WALLCLOCK = 300  # 5 minutes
SWITCH_FRACTION = 0.80  # switch to hard data at 80% of training
SCORING_BATCHES = 1000  # number of batches to score
TOP_FRACTION = 0.05  # keep top 5% hardest
SEQ_LEN = 1024
HEADER_SIZE = 256


def load_shard_tokens(path):
    with open(path, 'rb') as f:
        f.read(HEADER_SIZE * 4)
        data = np.frombuffer(f.read(), dtype=np.uint16)
    return data


def write_shard(path, tokens):
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


print('=== Exp 11d: Loss-based hard example mining ===')
print(f'Total wallclock: {TOTAL_WALLCLOCK}s')
print(f'Switch to hard data at: {SWITCH_FRACTION*100:.0f}% ({TOTAL_WALLCLOCK*SWITCH_FRACTION:.0f}s)')
print(f'Scoring batches: {SCORING_BATCHES}')
print(f'Top fraction: {TOP_FRACTION*100:.0f}%')
print()

# Phase 1: Normal training for 80% of wallclock
phase1_wallclock = int(TOTAL_WALLCLOCK * SWITCH_FRACTION)
print(f'Phase 1: Normal training for {phase1_wallclock}s...')

os.environ['MAX_WALLCLOCK_SECONDS'] = str(phase1_wallclock)

# Phase 1 runs train_gpt.py as subprocess. Imports happen after phase 1.
import subprocess

print('Starting Phase 1 training...')
env = os.environ.copy()
env['MAX_WALLCLOCK_SECONDS'] = str(phase1_wallclock)
env['PYTHONUNBUFFERED'] = '1'
env['WARMDOWN_ITERS'] = '3000'

proc = subprocess.run(
    ['python', 'train_gpt.py'],
    env=env, timeout=phase1_wallclock + 300  # extra time for compile
)
print(f'Phase 1 complete (exit code {proc.returncode})')

# Import model classes now (after phase 1 is done)
sys.path.insert(0, '/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard

# Load the checkpoint from phase 1
print('\nLoading phase 1 checkpoint...')
model = GPT(vocab_size=1024, num_layers=9, model_dim=512, num_heads=8, num_kv_heads=4,
            mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
sd = torch.load('final_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()
print(f'Model loaded: {sum(p.numel() for p in model.parameters())} params')

# Phase 2: Score training sequences and find hardest ones
print(f'\nPhase 2: Scoring {SCORING_BATCHES} batches to find hard examples...')
t_score = time.time()

train_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
all_sequences = []

# Load sequences from random shards
for shard_path in random.sample(train_shards, min(10, len(train_shards))):
    tokens = load_shard_tokens(shard_path)
    n_seqs = len(tokens) // SEQ_LEN
    for si in range(min(n_seqs, SCORING_BATCHES // 5)):
        seq = tokens[si * SEQ_LEN:(si + 1) * SEQ_LEN]
        if len(seq) == SEQ_LEN:
            all_sequences.append(seq)
    if len(all_sequences) >= SCORING_BATCHES:
        break

all_sequences = all_sequences[:SCORING_BATCHES]
print(f'  Loaded {len(all_sequences)} sequences to score')

# Score each sequence
seq_losses = []
batch_size = 32
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for i in range(0, len(all_sequences), batch_size):
        batch_seqs = all_sequences[i:i+batch_size]
        x = torch.tensor(np.array(batch_seqs), dtype=torch.long, device='cuda')
        y_input = x[:, :-1]  # not needed, model takes input_ids and target_ids
        # Use model's forward which takes (input_ids, target_ids)
        # input_ids = tokens[:-1], target_ids = tokens[1:]
        for j, seq in enumerate(batch_seqs):
            seq_t = torch.tensor(seq, dtype=torch.long, device='cuda')
            input_ids = seq_t[:-1].unsqueeze(0)
            target_ids = seq_t[1:].unsqueeze(0)
            loss = model(input_ids, target_ids)
            seq_losses.append((loss.item(), i + j))

        if (i // batch_size) % 10 == 0:
            print(f'  Scored {i+len(batch_seqs)}/{len(all_sequences)} sequences')

# Select top 5% hardest
seq_losses.sort(key=lambda x: x[0], reverse=True)
n_hard = max(1, int(len(seq_losses) * TOP_FRACTION))
hard_indices = [idx for _, idx in seq_losses[:n_hard]]
print(f'\n  Scoring took {time.time()-t_score:.1f}s')
print(f'  Hardest loss: {seq_losses[0][0]:.4f}')
print(f'  Easiest loss: {seq_losses[-1][0]:.4f}')
print(f'  Hard cutoff: {seq_losses[n_hard-1][0]:.4f}')
print(f'  Selected {n_hard} hard sequences (top {TOP_FRACTION*100:.0f}%)')

# Build hard data shard
hard_tokens = []
for idx in hard_indices:
    hard_tokens.extend(all_sequences[idx])
# Repeat to fill enough data for the remaining training
# We need ~320 steps * 65536 tokens = ~21M tokens, but we only have n_hard * 1024
# Repeat the hard set multiple times
tokens_needed = 500 * 65536  # generous estimate
repeats = max(1, tokens_needed // len(hard_tokens) + 1)
hard_tokens = np.tile(np.array(hard_tokens, dtype=np.uint16), repeats)
hard_shard = '/runpod-volume/hard_sequences_11d.bin'
write_shard(hard_shard, hard_tokens)
print(f'  Wrote hard shard: {len(hard_tokens)} tokens ({repeats}x repeated)')

# Phase 3: Continue training on hard data only
remaining_wallclock = TOTAL_WALLCLOCK - phase1_wallclock - int(time.time() - t_score)
print(f'\nPhase 3: Training on hard data for ~{remaining_wallclock}s...')

if remaining_wallclock < 30:
    print('  WARNING: Not enough time for phase 3, skipping')
else:
    # Replace training shards with just the hard shard
    # We need to modify train_gpt.py to only use this shard
    # Simplest: symlink the hard shard into the data directory and remove others temporarily

    # Actually, let's just run train_gpt.py with the shard path modified
    # The dataloader globs for fineweb_train_*.bin, so rename our hard shard
    import shutil
    backup_dir = '/runpod-volume/train_shards_backup'
    data_dir = './data/datasets/fineweb10B_sp1024'

    # Move all train shards to backup
    os.makedirs(backup_dir, exist_ok=True)
    for f in glob.glob(f'{data_dir}/fineweb_train_*.bin'):
        shutil.move(f, backup_dir)

    # Copy hard shard as the only training shard
    shutil.copy(hard_shard, f'{data_dir}/fineweb_train_000000.bin')
    print(f'  Swapped training data to hard shard only')

    # Run phase 3 training (loads from the phase 1 checkpoint via the saved model)
    # Problem: train_gpt.py starts from scratch. We need to save/load a training state.
    # For simplicity: just run train_gpt.py fresh — it won't resume from checkpoint.
    # This means phase 3 starts from random init, which is wrong.
    #
    # Better approach: skip train_gpt.py and do manual training loop here.

    model.train()

    # Load hard data
    hard_data = load_data_shard(Path(f'{data_dir}/fineweb_train_000000.bin')).cuda().long()
    print(f'  Hard data loaded: {hard_data.numel()} tokens')

    # Simple training loop with the existing optimizer settings
    # Reconstruct optimizer from model params
    matrix_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'tok_emb' not in n]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'tok_emb' in n]

    optimizer = torch.optim.AdamW([
        {'params': matrix_params, 'lr': 0.01, 'weight_decay': 0.04},
        {'params': scalar_params, 'lr': 0.04, 'weight_decay': 0.0},
    ])

    n_seqs = hard_data.numel() // SEQ_LEN
    t_phase3 = time.time()
    step = 0
    total_loss = 0.0

    while time.time() - t_phase3 < remaining_wallclock - 5:  # 5s buffer
        # Random sequence from hard data
        idx = random.randint(0, n_seqs - 2)
        x = hard_data[idx * SEQ_LEN:(idx + 1) * SEQ_LEN].unsqueeze(0)
        y = hard_data[idx * SEQ_LEN + 1:(idx + 1) * SEQ_LEN + 1].unsqueeze(0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y)

        loss.backward()

        if (step + 1) % 8 == 0:  # grad accumulation like train_gpt.py
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg = total_loss / step
            elapsed = time.time() - t_phase3
            print(f'  Phase 3 step {step}: avg_loss={avg:.4f} elapsed={elapsed:.0f}s')

    print(f'  Phase 3 complete: {step} steps, avg_loss={total_loss/max(step,1):.4f}')

    # Restore original training shards
    for f in glob.glob(f'{backup_dir}/fineweb_train_*.bin'):
        shutil.move(f, data_dir)
    os.rmdir(backup_dir)
    print('  Training shards restored')

# Save final checkpoint
torch.save(model.state_dict(), '/runpod-volume/exp11d_checkpoint.pt')
torch.save(model.state_dict(), 'final_model.pt')
print(f'\nCheckpoint saved to /runpod-volume/exp11d_checkpoint.pt')
print('=== Exp 11d complete ===')
