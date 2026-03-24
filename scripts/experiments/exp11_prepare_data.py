"""
Pre-filter training data by bigram rarity. Creates enriched shards for Exp 11a/b.
Run on pod: python exp11_prepare_data.py

Output:
  /runpod-volume/enriched_rare_bigram.bin  - sequences with rare bigrams (top 25%)
  /runpod-volume/enriched_juncture.bin     - sequences with high juncture diversity
  /runpod-volume/data_stats.txt            - statistics about the filtering
"""
import numpy as np
import struct, glob, os, math, time
from collections import Counter, defaultdict
from pathlib import Path

os.chdir('/runpod-volume/parameter-golf')

SEQ_LEN = 1024
HEADER_SIZE = 256  # int32 header entries
JUNCTURE_TOKENS = {267, 261, 290, 287, 292, 960, 962}  # , . the and to of in


def load_shard_tokens(path):
    with open(path, 'rb') as f:
        f.read(HEADER_SIZE * 4)  # skip header
        data = np.frombuffer(f.read(), dtype=np.uint16)
    return data


def write_shard(path, tokens):
    """Write tokens as a binary shard compatible with train_gpt.py's load_data_shard."""
    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 7         # version
    header[2] = len(tokens)
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())
    print(f'  Wrote {path}: {len(tokens)} tokens ({len(tokens)/SEQ_LEN:.0f} sequences)')


print('=== Exp 11 Data Preparation ===')
print()

# Step 1: Build bigram frequency table from first 10 shards
print('Step 1: Building bigram frequency table (10 shards)...')
train_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin'))
t0 = time.time()

bigram_counts = Counter()
total_bigrams = 0
for shard_path in train_shards[:10]:
    tokens = load_shard_tokens(shard_path)
    # Sample every 10th bigram for speed
    for i in range(0, min(len(tokens) - 1, 500000), 1):
        bigram_counts[(int(tokens[i]), int(tokens[i+1]))] += 1
        total_bigrams += 1
    print(f'  {os.path.basename(shard_path)}: {total_bigrams} bigrams so far')

print(f'  Total: {total_bigrams} bigrams, {len(bigram_counts)} unique')
print(f'  Time: {time.time()-t0:.1f}s')
print()

# Step 2: Score all sequences by bigram rarity + juncture diversity
print('Step 2: Scoring sequences from 20 shards...')
t0 = time.time()

all_scores_rare = []    # (rarity_score, shard_idx, seq_idx)
all_scores_juncture = []  # (juncture_diversity, shard_idx, seq_idx)
shard_tokens_cache = {}  # cache loaded shards

for shard_idx, shard_path in enumerate(train_shards[:5]):
    tokens = load_shard_tokens(shard_path)
    shard_tokens_cache[shard_idx] = tokens
    n_seqs = len(tokens) // SEQ_LEN

    # Vectorized scoring
    n_seqs = min(n_seqs, 5000)
    usable = n_seqs * SEQ_LEN
    tok_flat = tokens[:usable].astype(np.int32)
    seqs = tok_flat.reshape(n_seqs, SEQ_LEN)
    prev_t = seqs[:, :-1]  # (n_seqs, 1023)
    curr_t = seqs[:, 1:]   # (n_seqs, 1023)

    # Bigram hash for lookup
    bigram_keys = prev_t.astype(np.int64) * 1024 + curr_t.astype(np.int64)

    # Build flat lookup from bigram_counts
    if shard_idx == 0:
        # Build once
        global bigram_log_freq
        bigram_log_freq = {}
        for (p, c), cnt in bigram_counts.items():
            bigram_log_freq[p * 1024 + c] = -math.log(cnt / total_bigrams)
        default_rarity = -math.log(1 / total_bigrams)

    # Score each sequence
    for si in range(n_seqs):
        bk = bigram_keys[si]
        rarity_vals = np.array([bigram_log_freq.get(int(k), default_rarity) for k in bk])
        avg_rarity = float(rarity_vals.mean())

        # Juncture diversity
        juncture_mask = np.isin(seqs[si, :-1], list(JUNCTURE_TOKENS))
        if juncture_mask.any():
            followers = set(seqs[si, 1:][juncture_mask].tolist())
            juncture_diversity = len(followers)
        else:
            juncture_diversity = 0

        all_scores_rare.append((avg_rarity, shard_idx, si))
        all_scores_juncture.append((juncture_diversity, shard_idx, si))

    print(f'  Shard {shard_idx}: {n_seqs} sequences scored')

total_seqs = len(all_scores_rare)
print(f'  Total sequences: {total_seqs}')
print(f'  Time: {time.time()-t0:.1f}s')
print()

# Step 3: Select top 25% by each metric and build enriched shards
print('Step 3: Building enriched shards (top 25%)...')

# Rare bigram enrichment
all_scores_rare.sort(key=lambda x: x[0], reverse=True)
cutoff_rare = total_seqs // 4
rare_seqs = all_scores_rare[:cutoff_rare]
print(f'  Rare bigram: selecting {cutoff_rare} sequences')
print(f'  Rarity range: {rare_seqs[0][0]:.4f} (rarest) to {rare_seqs[-1][0]:.4f} (cutoff)')

rare_tokens = []
for _, shard_idx, seq_idx in rare_seqs:
    seq = shard_tokens_cache[shard_idx][seq_idx * SEQ_LEN:(seq_idx + 1) * SEQ_LEN]
    rare_tokens.extend(seq)
rare_tokens = np.array(rare_tokens, dtype=np.uint16)
write_shard('/runpod-volume/enriched_rare_bigram.bin', rare_tokens)

# Juncture diversity enrichment
all_scores_juncture.sort(key=lambda x: x[0], reverse=True)
cutoff_juncture = total_seqs // 4
juncture_seqs = all_scores_juncture[:cutoff_juncture]
print(f'  Juncture diversity: selecting {cutoff_juncture} sequences')
print(f'  Diversity range: {juncture_seqs[0][0]} (most diverse) to {juncture_seqs[-1][0]} (cutoff)')

juncture_tokens = []
for _, shard_idx, seq_idx in juncture_seqs:
    seq = shard_tokens_cache[shard_idx][seq_idx * SEQ_LEN:(seq_idx + 1) * SEQ_LEN]
    juncture_tokens.extend(seq)
juncture_tokens = np.array(juncture_tokens, dtype=np.uint16)
write_shard('/runpod-volume/enriched_juncture.bin', juncture_tokens)

# Stats
stats = f"""=== Data Preparation Stats ===
Total sequences scored: {total_seqs}
Bigram frequency table: {len(bigram_counts)} unique bigrams from {total_bigrams} total

Rare Bigram Shard:
  Sequences: {cutoff_rare} (top 25%)
  Tokens: {len(rare_tokens)}
  Rarity range: {rare_seqs[0][0]:.4f} - {rare_seqs[-1][0]:.4f}

Juncture Diversity Shard:
  Sequences: {cutoff_juncture} (top 25%)
  Tokens: {len(juncture_tokens)}
  Diversity range: {juncture_seqs[0][0]} - {juncture_seqs[-1][0]}
"""
with open('/runpod-volume/data_stats.txt', 'w') as f:
    f.write(stats)
print(stats)
print('=== Done ===')
