"""
Bits Budget Analysis — Per-token loss map for Exp 17 checkpoint.
Runs on pod with GPU. Saves results as JSON for local dashboard.

Usage: python bits_budget_analysis.py
Output: /runpod-volume/bits_budget_results.json (~50MB)
"""
import torch, os, sys, time, glob, math, json
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard
import sentencepiece as spm

SEQ_LEN = 2048
MAX_WINDOWS = 1000  # ~2M tokens, enough for statistical significance

print('=== Bits Budget Analysis ===')

# Load model
print('Loading model...')
model = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
            mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
sd = torch.load('final_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()
print(f'Loaded: {sum(p.numel() for p in model.parameters())} params')

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.Load('./data/tokenizers/fineweb_1024_bpe.model')

# Load val tokens
print('Loading val tokens...')
val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
print(f'Val tokens: {val_tokens.numel()}')

# =====================================================
# Step 1: Per-token loss computation
# =====================================================
print(f'\nStep 1: Computing per-token losses ({MAX_WINDOWS} windows)...')

def forward_logits(model, x):
    xemb = model.tok_emb(x)
    if hasattr(model, 'bigram'):
        xemb = xemb + model.bigram(x)
    xemb = F.rms_norm(xemb, (xemb.size(-1),))
    if hasattr(model, 'smear'):
        xemb = model.smear(xemb)
    x0 = xemb
    skips = []
    for i in range(model.num_encoder_layers):
        xemb = model.blocks[i](xemb, x0)
        skips.append(xemb)
    for i in range(model.num_decoder_layers):
        if skips:
            xemb = xemb + model.skip_weights[i].to(dtype=xemb.dtype)[None, None, :] * skips.pop()
        xemb = model.blocks[model.num_encoder_layers + i](xemb, x0)
    xemb = model.final_norm(xemb)
    logits = F.linear(xemb, model.tok_emb.weight)
    return model.logit_softcap * torch.tanh(logits / model.logit_softcap)

# Collect per-token data
all_losses = []       # per-token loss (nats)
all_positions = []    # position in window
all_token_ids = []    # target token ID
all_prev_ids = []     # previous token ID
all_top1_correct = [] # whether top-1 prediction was correct
all_entropies = []    # output entropy
all_top1_probs = []   # probability of top prediction

windows = min(MAX_WINDOWS, (val_tokens.numel() - SEQ_LEN) // SEQ_LEN)
t0 = time.time()

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for w in range(windows):
        start = w * SEQ_LEN
        x = val_tokens[start:start + SEQ_LEN].unsqueeze(0)
        y = val_tokens[start + 1:start + SEQ_LEN + 1].unsqueeze(0)

        logits = forward_logits(model, x)
        logits_f = logits[0].float()  # (seq_len, vocab)

        # Per-token loss
        per_token_loss = F.cross_entropy(logits_f, y[0], reduction='none')

        # Top-1 prediction
        top1 = logits_f.argmax(dim=-1)
        top1_correct = (top1 == y[0])

        # Entropy and top-1 probability
        probs = F.softmax(logits_f, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        top1_prob = probs.max(dim=-1).values

        # Store
        all_losses.append(per_token_loss.cpu().numpy())
        all_positions.append(np.arange(SEQ_LEN))
        all_token_ids.append(y[0].cpu().numpy())
        all_prev_ids.append(x[0].cpu().numpy())
        all_top1_correct.append(top1_correct.cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())
        all_top1_probs.append(top1_prob.cpu().numpy())

        if (w + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  {w+1}/{windows} windows, {elapsed:.0f}s')

# Flatten
losses = np.concatenate(all_losses)
positions = np.concatenate(all_positions)
token_ids = np.concatenate(all_token_ids)
prev_ids = np.concatenate(all_prev_ids)
top1_correct = np.concatenate(all_top1_correct)
entropies = np.concatenate(all_entropies)
top1_probs = np.concatenate(all_top1_probs)

total_tokens = len(losses)
avg_loss = losses.mean()
print(f'\nTotal tokens analyzed: {total_tokens}')
print(f'Average loss: {avg_loss:.4f} nats')
print(f'Time: {time.time()-t0:.0f}s')

# =====================================================
# Step 2: Token classification
# =====================================================
print('\nStep 2: Token classification...')

# Get token strings
token_strings = {}
for tid in range(1024):
    try:
        token_strings[tid] = sp.IdToPiece(tid)
    except:
        token_strings[tid] = f'[{tid}]'

# Classify tokens
# Juncture tokens (syntactic boundaries)
JUNCTURE_IDS = set()
for tid, s in token_strings.items():
    if s in ['▁,', ',', '▁.', '.', '▁the', '▁and', '▁to', '▁of', '▁in', '▁a', '▁is']:
        JUNCTURE_IDS.add(tid)

# Word-initial tokens (start with ▁)
WORD_INITIAL_IDS = set()
for tid, s in token_strings.items():
    if s.startswith('▁') and len(s) > 1:
        WORD_INITIAL_IDS.add(tid)

# Number/digit tokens
NUMBER_IDS = set()
for tid, s in token_strings.items():
    clean = s.replace('▁', '')
    if clean and all(c.isdigit() or c in '.,/-:' for c in clean):
        NUMBER_IDS.add(tid)

# Classify each token
categories = np.zeros(total_tokens, dtype=np.int8)
# 0=easy, 1=medium, 2=hard_learnable, 3=unpredictable
for i in range(total_tokens):
    loss = losses[i]
    tid = token_ids[i]
    if loss < 1.0:
        categories[i] = 0  # easy
    elif loss < 3.0:
        categories[i] = 1  # medium
    elif loss < 5.0:
        if tid in NUMBER_IDS:
            categories[i] = 3  # numbers are unpredictable
        else:
            categories[i] = 2  # hard but learnable
    else:
        if tid in NUMBER_IDS or loss > 8.0:
            categories[i] = 3  # very high loss = likely unpredictable
        else:
            categories[i] = 2  # hard but learnable

cat_names = ['easy(<1)', 'medium(1-3)', 'hard_learnable(3-5)', 'unpredictable(5+)']
for c in range(4):
    mask = categories == c
    count = mask.sum()
    avg = losses[mask].mean() if mask.any() else 0
    total_bits = losses[mask].sum()
    pct_tokens = count / total_tokens * 100
    pct_bits = total_bits / losses.sum() * 100
    print(f'  {cat_names[c]:25s}: {pct_tokens:.1f}% of tokens, {pct_bits:.1f}% of bits, avg_loss={avg:.2f}')

# =====================================================
# Step 3: Position analysis
# =====================================================
print('\nStep 3: Position analysis...')
pos_loss = np.zeros(SEQ_LEN)
pos_count = np.zeros(SEQ_LEN)
for i in range(total_tokens):
    p = positions[i]
    pos_loss[p] += losses[i]
    pos_count[p] += 1
pos_avg = pos_loss / np.maximum(pos_count, 1)

# Position ranges
pos_ranges = {}
for s in range(0, SEQ_LEN, 128):
    e = min(s + 128, SEQ_LEN)
    pos_ranges[f'{s}-{e}'] = float(pos_avg[s:e].mean())
    print(f'  {s:4d}-{e:4d}: {pos_avg[s:e].mean():.4f}')

# =====================================================
# Step 4: Bigram analysis
# =====================================================
print('\nStep 4: Top-50 most expensive bigrams...')
bigram_total_loss = defaultdict(float)
bigram_count = defaultdict(int)

for i in range(total_tokens):
    bg = (int(prev_ids[i]), int(token_ids[i]))
    bigram_total_loss[bg] += losses[i]
    bigram_count[bg] += 1

# Sort by total cost
top_bigrams = sorted(bigram_total_loss.items(), key=lambda x: x[1], reverse=True)[:50]
bigram_data = []
for (prev_id, cur_id), total_cost in top_bigrams:
    count = bigram_count[(prev_id, cur_id)]
    avg = total_cost / count
    prev_str = token_strings.get(prev_id, f'[{prev_id}]')
    cur_str = token_strings.get(cur_id, f'[{cur_id}]')
    bigram_data.append({
        'prev': prev_str, 'cur': cur_str,
        'prev_id': int(prev_id), 'cur_id': int(cur_id),
        'total_cost': float(total_cost), 'count': int(count), 'avg_loss': float(avg)
    })
    if len(bigram_data) <= 20:
        print(f'  {prev_str:10s} → {cur_str:10s}: total={total_cost:.0f} count={count} avg={avg:.2f}')

# =====================================================
# Step 5: Entropy analysis (quadrants)
# =====================================================
print('\nStep 5: Entropy quadrants...')
loss_threshold = 2.0  # nats
entropy_threshold = np.median(entropies)

quadrants = {
    'confident_right': ((losses < loss_threshold) & (entropies < entropy_threshold)).sum(),
    'uncertain_right': ((losses < loss_threshold) & (entropies >= entropy_threshold)).sum(),
    'confident_wrong': ((losses >= loss_threshold) & (entropies < entropy_threshold)).sum(),
    'uncertain_wrong': ((losses >= loss_threshold) & (entropies >= entropy_threshold)).sum(),
}
for k, v in quadrants.items():
    print(f'  {k:20s}: {v/total_tokens*100:.1f}% ({v})')

# Average loss per quadrant
for name, mask in [
    ('confident_right', (losses < loss_threshold) & (entropies < entropy_threshold)),
    ('uncertain_right', (losses < loss_threshold) & (entropies >= entropy_threshold)),
    ('confident_wrong', (losses >= loss_threshold) & (entropies < entropy_threshold)),
    ('uncertain_wrong', (losses >= loss_threshold) & (entropies >= entropy_threshold)),
]:
    if mask.sum() > 0:
        print(f'    avg_loss={losses[mask].mean():.3f} avg_entropy={entropies[mask].mean():.3f}')

# =====================================================
# Step 6: Highlighted text samples
# =====================================================
print('\nStep 6: Generating highlighted text samples...')
text_samples = []
for w_idx in [0, 100, 500]:  # 3 sample windows
    start = w_idx * SEQ_LEN
    tokens_in_window = val_tokens[start + 1:start + SEQ_LEN + 1].cpu().numpy()
    losses_in_window = all_losses[w_idx] if w_idx < len(all_losses) else None
    if losses_in_window is None:
        continue
    sample = []
    for i in range(min(200, SEQ_LEN)):  # First 200 tokens
        tid = int(tokens_in_window[i])
        tok_str = token_strings.get(tid, f'[{tid}]')
        sample.append({
            'token': tok_str,
            'loss': float(losses_in_window[i]),
            'position': i,
        })
    text_samples.append(sample)

# =====================================================
# Step 7: Loss distribution histogram
# =====================================================
print('\nStep 7: Loss distribution...')
hist_bins = np.arange(0, 12, 0.25)
hist_counts, _ = np.histogram(losses, bins=hist_bins)
loss_histogram = {f'{hist_bins[i]:.2f}-{hist_bins[i+1]:.2f}': int(hist_counts[i]) for i in range(len(hist_counts))}

# =====================================================
# Step 8: Per-token-ID analysis (which tokens cost most in total)
# =====================================================
print('\nStep 8: Most expensive tokens by total cost...')
token_total_cost = defaultdict(float)
token_count = defaultdict(int)
for i in range(total_tokens):
    tid = int(token_ids[i])
    token_total_cost[tid] += losses[i]
    token_count[tid] += 1

top_tokens = sorted(token_total_cost.items(), key=lambda x: x[1], reverse=True)[:30]
token_cost_data = []
for tid, total_cost in top_tokens:
    avg = total_cost / token_count[tid]
    tok_str = token_strings.get(tid, f'[{tid}]')
    token_cost_data.append({
        'token': tok_str, 'id': tid,
        'total_cost': float(total_cost), 'count': token_count[tid], 'avg_loss': float(avg)
    })
    print(f'  {tok_str:15s} (id={tid:4d}): total={total_cost:.0f} count={token_count[tid]} avg={avg:.2f}')

# =====================================================
# Save results
# =====================================================
print('\nSaving results...')
results = {
    'summary': {
        'total_tokens': int(total_tokens),
        'windows': int(windows),
        'avg_loss': float(avg_loss),
        'avg_bpb_approx': float(avg_loss / math.log(2) * 0.4104),
    },
    'category_breakdown': {
        cat_names[c]: {
            'pct_tokens': float((categories == c).sum() / total_tokens * 100),
            'pct_bits': float(losses[categories == c].sum() / losses.sum() * 100) if (categories == c).any() else 0,
            'avg_loss': float(losses[categories == c].mean()) if (categories == c).any() else 0,
            'count': int((categories == c).sum()),
        }
        for c in range(4)
    },
    'position_ranges': pos_ranges,
    'position_fine': {str(i): float(pos_avg[i]) for i in range(0, SEQ_LEN, 16)},
    'top_bigrams': bigram_data,
    'entropy_quadrants': {k: int(v) for k, v in quadrants.items()},
    'entropy_quadrants_pct': {k: float(v / total_tokens * 100) for k, v in quadrants.items()},
    'text_samples': text_samples,
    'loss_histogram': loss_histogram,
    'top_costly_tokens': token_cost_data,
    'juncture_analysis': {
        'after_juncture_avg_loss': float(losses[np.isin(prev_ids, list(JUNCTURE_IDS))].mean()),
        'not_after_juncture_avg_loss': float(losses[~np.isin(prev_ids, list(JUNCTURE_IDS))].mean()),
        'after_juncture_pct': float(np.isin(prev_ids, list(JUNCTURE_IDS)).sum() / total_tokens * 100),
        'word_initial_avg_loss': float(losses[np.isin(token_ids, list(WORD_INITIAL_IDS))].mean()),
        'not_word_initial_avg_loss': float(losses[~np.isin(token_ids, list(WORD_INITIAL_IDS))].mean()),
    },
}

with open('/runpod-volume/bits_budget_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to /runpod-volume/bits_budget_results.json')

print('\n=== Analysis Complete ===')
