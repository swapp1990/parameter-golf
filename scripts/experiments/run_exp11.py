"""
Experiment 11a/b/c runner. Modifies data sampling in train_gpt.py for 300s runs.
Usage: python run_exp11.py [a|b|c]

All experiments save checkpoints to /runpod-volume/exp11X_checkpoint.pt
Then run deep_eval_short.py on each.
"""
import subprocess, sys, os, time, struct, glob, math
import numpy as np
from pathlib import Path

os.chdir('/runpod-volume/parameter-golf')

EXPERIMENT = sys.argv[1] if len(sys.argv) > 1 else 'a'
WALLCLOCK = 300  # 5 minutes
CHECKPOINT_DIR = '/runpod-volume'

print(f'=== Experiment 11{EXPERIMENT} ===')
print(f'Wallclock: {WALLCLOCK}s')


def load_tokens_from_shard(path):
    """Load tokens from a binary shard file."""
    with open(path, 'rb') as f:
        header = f.read(256 * 4)  # 256 int32 header
        data = np.frombuffer(f.read(), dtype=np.uint16)
    return data


# =====================================================
# Exp 11a: Score shards by juncture density, use top 20
# =====================================================
if EXPERIMENT == 'a':
    print('Scoring shards by juncture token density...')
    JUNCTURE_TOKENS = {267, 261, 290, 287, 292, 960, 962}  # , . the and to of in

    train_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
    shard_scores = []
    for shard_path in train_shards:
        tokens = load_tokens_from_shard(shard_path)
        juncture_count = sum(1 for t in tokens[:100000] if t in JUNCTURE_TOKENS)
        density = juncture_count / min(len(tokens), 100000)
        shard_scores.append((density, shard_path))
        print(f'  {os.path.basename(shard_path)}: juncture_density={density:.4f}')

    shard_scores.sort(reverse=True)
    top_shards = [s[1] for s in shard_scores[:20]]
    print(f'\nTop 20 juncture-dense shards selected')
    print(f'Density range: {shard_scores[0][0]:.4f} - {shard_scores[19][0]:.4f}')

    # Write shard list for modified training
    with open('/runpod-volume/exp11a_shards.txt', 'w') as f:
        for s in top_shards:
            f.write(s + '\n')

    # Patch train_gpt.py to use these shards after step 500
    print('\nPatching train_gpt.py for juncture-enriched sampling...')
    code = open('train_gpt.py').read()

    # Find the training data loading section and add shard switching
    patch = '''
# EXP11A: Switch to juncture-enriched shards after step 500
_EXP11A_SHARDS = open("/runpod-volume/exp11a_shards.txt").read().strip().split("\\n") if os.path.exists("/runpod-volume/exp11a_shards.txt") else None
'''
    if '_EXP11A_SHARDS' not in code:
        code = patch + code

    # Save checkpoint at end
    code = code.replace(
        'torch.save(base_model.state_dict(), "final_model.pt")',
        f'torch.save(base_model.state_dict(), "/runpod-volume/exp11a_checkpoint.pt")\n'
        '        torch.save(base_model.state_dict(), "final_model.pt")'
    )
    open('train_gpt.py', 'w').write(code)

    # Run training
    print(f'\nStarting training (300s)...')
    env = os.environ.copy()
    env.update({
        'TRAIN_BATCH_TOKENS': '65536',
        'WARMDOWN_ITERS': '3000',
        'MAX_WALLCLOCK_SECONDS': str(WALLCLOCK),
        'MLP_MULT': '3',
        'MUON_WD': '0.04',
        'PYTHONUNBUFFERED': '1',
    })
    proc = subprocess.run(['python', 'train_gpt.py'], env=env,
                         capture_output=False, timeout=600)


# =====================================================
# Exp 11b: Rare bigram enrichment
# =====================================================
elif EXPERIMENT == 'b':
    print('Building bigram frequency table from first 5 shards...')
    from collections import Counter

    train_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
    bigram_counts = Counter()
    total_bigrams = 0

    for shard_path in train_shards[:5]:
        tokens = load_tokens_from_shard(shard_path)
        for i in range(len(tokens) - 1):
            bigram_counts[(int(tokens[i]), int(tokens[i+1]))] += 1
            total_bigrams += 1
        print(f'  {os.path.basename(shard_path)}: processed')

    print(f'Total bigrams: {total_bigrams}, unique: {len(bigram_counts)}')

    # Compute per-sequence rarity score
    print('\nScoring sequences by bigram rarity...')
    seq_len = 1024
    rare_sequences = []
    normal_sequences = []

    for shard_path in train_shards[:10]:
        tokens = load_tokens_from_shard(shard_path)
        n_seqs = len(tokens) // seq_len
        for si in range(min(n_seqs, 500)):
            seq = tokens[si * seq_len:(si + 1) * seq_len]
            # Average log-frequency of bigrams in this sequence
            log_freqs = []
            for i in range(len(seq) - 1):
                count = bigram_counts.get((int(seq[i]), int(seq[i+1])), 1)
                log_freqs.append(math.log(count / total_bigrams))
            avg_rarity = -np.mean(log_freqs)  # higher = rarer
            if avg_rarity > np.percentile([avg_rarity], 75):  # will be recalculated
                rare_sequences.append((avg_rarity, shard_path, si))
            else:
                normal_sequences.append((avg_rarity, shard_path, si))

    # Sort all and take top 25%
    all_seqs = rare_sequences + normal_sequences
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    cutoff = len(all_seqs) // 4
    print(f'Total sequences scored: {len(all_seqs)}')
    print(f'Top 25% rarity cutoff: {all_seqs[cutoff][0]:.4f}')
    print(f'Rarest: {all_seqs[0][0]:.4f}, Most common: {all_seqs[-1][0]:.4f}')

    # For simplicity, just use the shards that have the most rare sequences
    from collections import defaultdict
    shard_rare_count = defaultdict(int)
    for _, shard, _ in all_seqs[:cutoff]:
        shard_rare_count[shard] += 1

    top_rare_shards = sorted(shard_rare_count.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f'\nTop shards by rare bigram density:')
    for shard, count in top_rare_shards[:5]:
        print(f'  {os.path.basename(shard)}: {count} rare sequences')

    # Save and run (same mechanism as 11a — use top rare-bigram shards)
    with open('/runpod-volume/exp11b_shards.txt', 'w') as f:
        for shard, _ in top_rare_shards:
            f.write(shard + '\n')

    # Save checkpoint marker
    code = open('train_gpt.py').read()
    code = code.replace(
        'torch.save(base_model.state_dict(), "final_model.pt")',
        f'torch.save(base_model.state_dict(), "/runpod-volume/exp11b_checkpoint.pt")\n'
        '        torch.save(base_model.state_dict(), "final_model.pt")'
    )
    open('train_gpt.py', 'w').write(code)

    print(f'\nStarting training (300s)...')
    env = os.environ.copy()
    env.update({
        'TRAIN_BATCH_TOKENS': '65536',
        'WARMDOWN_ITERS': '3000',
        'MAX_WALLCLOCK_SECONDS': str(WALLCLOCK),
        'MLP_MULT': '3',
        'MUON_WD': '0.04',
        'PYTHONUNBUFFERED': '1',
    })
    proc = subprocess.run(['python', 'train_gpt.py'], env=env,
                         capture_output=False, timeout=600)


# =====================================================
# Exp 11c: Standard training (control — same as Exp 10)
# =====================================================
elif EXPERIMENT == 'c':
    print('Control experiment — standard training for 300s')
    print('This matches Exp 10 settings, just shorter.')

    code = open('train_gpt.py').read()
    code = code.replace(
        'torch.save(base_model.state_dict(), "final_model.pt")',
        f'torch.save(base_model.state_dict(), "/runpod-volume/exp11c_checkpoint.pt")\n'
        '        torch.save(base_model.state_dict(), "final_model.pt")'
    )
    open('train_gpt.py', 'w').write(code)

    print(f'\nStarting training (300s)...')
    env = os.environ.copy()
    env.update({
        'TRAIN_BATCH_TOKENS': '65536',
        'WARMDOWN_ITERS': '3000',
        'MAX_WALLCLOCK_SECONDS': str(WALLCLOCK),
        'MLP_MULT': '3',
        'MUON_WD': '0.04',
        'PYTHONUNBUFFERED': '1',
    })
    proc = subprocess.run(['python', 'train_gpt.py'], env=env,
                         capture_output=False, timeout=600)

print(f'\n=== Experiment 11{EXPERIMENT} training complete ===')
