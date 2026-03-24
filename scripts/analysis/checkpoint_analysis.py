"""
Comprehensive checkpoint analysis. Runs all 12 sections, outputs JSON.

Usage:
  Pod:   python checkpoint_analysis.py --name submission
  Local: python checkpoint_analysis.py --local --name step2000

Output: /runpod-volume/analysis_<name>.json (pod) or dashboard/analysis_<name>.json (local)
"""
import torch, io, os, sys, time, glob, math, json, argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='checkpoint', help='Name for this analysis')
parser.add_argument('--local', action='store_true', help='Run locally on CPU')
parser.add_argument('--windows', type=int, default=500, help='Eval windows for per-token analysis')
parser.add_argument('--ablation_windows', type=int, default=200, help='Windows for ablation')
args = parser.parse_args()

LOCAL = args.local
if LOCAL:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    DEVICE = 'cpu'
    SEQ_LEN = 512
    OUT_DIR = 'dashboard'
else:
    os.environ['MLP_MULT'] = '3'
    os.environ['NUM_LAYERS'] = '11'
    sys.path.insert(0, '/runpod-volume/parameter-golf')
    os.chdir('/runpod-volume/parameter-golf')
    DEVICE = 'cuda'
    SEQ_LEN = 2048
    OUT_DIR = '/runpod-volume'

import sentencepiece as spm

print(f'=== Checkpoint Analysis: {args.name} ===')
print(f'Mode: {"LOCAL" if LOCAL else "POD"}, Device: {DEVICE}, SeqLen: {SEQ_LEN}')

# =====================================================
# Load model + data
# =====================================================
print('\nLoading model...')
if LOCAL:
    from dashboard.layer_xray import GPT
    model = GPT()
    sd = torch.load('dashboard/checkpoints/model_step_2000.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=False)
    NUM_LAYERS = 9
else:
    import zstandard
    from train_gpt import GPT, load_data_shard, build_sentencepiece_luts
    with open('final_model.mixed.ptz', 'rb') as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    qs = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    qs.pop('__quant_format__', None)
    recovered = {}
    seen = set()
    for key in list(qs.keys()):
        if key.endswith('.__q'):
            name = key[:-4]
            if name in seen: continue
            seen.add(name)
            q, scale = qs[name+'.__q'], qs[name+'.__scale']
            dtype = getattr(torch, qs[name+'.__dtype'].split('.')[-1])
            if q.ndim == 2 and scale.ndim == 1:
                recovered[name] = (q.float() * scale.float()[:, None]).to(dtype)
            else:
                recovered[name] = (q.float() * scale.float()).to(dtype)
        elif not any(key.endswith(s) for s in ('.__scale','.__dtype')):
            recovered[key] = qs[key]
    model = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
                mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
                logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
    model.load_state_dict(recovered, strict=False)
    NUM_LAYERS = 11

model = model.to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params}')

print('Loading val tokens...')
if LOCAL:
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens_np = np.fromfile('data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin',
                            dtype="<u2", offset=header_bytes)
    val_tokens = torch.from_numpy(tokens_np.astype(np.int64)).to(DEVICE)
else:
    val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(DEVICE) for s in val_shards]).long()

sp = spm.SentencePieceProcessor()
sp.Load('./data/tokenizers/fineweb_1024_bpe.model' if not LOCAL else 'data/tokenizers/fineweb_1024_bpe.model')
print(f'Val tokens: {val_tokens.numel()}')

# Token string lookup
token_strings = {}
for tid in range(1024):
    try: token_strings[tid] = sp.IdToPiece(tid)
    except: token_strings[tid] = f'[{tid}]'

JUNCTURE_IDS = set()
for tid, s in token_strings.items():
    if s in ['\u2581,', ',', '\u2581.', '.', '\u2581the', '\u2581and', '\u2581to', '\u2581of', '\u2581in', '\u2581a', '\u2581is']:
        JUNCTURE_IDS.add(tid)

WORD_INITIAL_IDS = set(tid for tid, s in token_strings.items() if s.startswith('\u2581') and len(s) > 1)

results = {
    'name': args.name,
    'params': n_params,
    'num_layers': NUM_LAYERS,
    'seq_len': SEQ_LEN,
    'device': DEVICE,
    'val_tokens': val_tokens.numel(),
}


def forward_logits(mdl, x):
    if LOCAL:
        r = mdl.forward_full(x)
        return r["logits"]
    else:
        xemb = mdl.tok_emb(x)
        if hasattr(mdl, 'smear'): xemb = mdl.smear(xemb)
        xemb = F.rms_norm(xemb, (xemb.size(-1),))
        x0 = xemb
        skips = []
        for i in range(mdl.num_encoder_layers):
            xemb = mdl.blocks[i](xemb, x0); skips.append(xemb)
        for i in range(mdl.num_decoder_layers):
            if skips: xemb = xemb + mdl.skip_weights[i].to(dtype=xemb.dtype)[None,None,:] * skips.pop()
            xemb = mdl.blocks[mdl.num_encoder_layers+i](xemb, x0)
        xemb = mdl.final_norm(xemb)
        logits = F.linear(xemb, mdl.tok_emb.weight)
        return mdl.logit_softcap * torch.tanh(logits / mdl.logit_softcap)


# =====================================================
# Section 1-5, 10-11: Per-token analysis (one forward pass)
# =====================================================
print(f'\n--- Per-token analysis ({args.windows} windows) ---')
all_losses = []
all_positions = []
all_token_ids = []
all_prev_ids = []
all_entropies = []
all_top1_correct = []

windows = min(args.windows, (val_tokens.numel() - SEQ_LEN) // SEQ_LEN)
t0 = time.time()
with torch.no_grad():
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()
    with ctx:
        for w in range(windows):
            start = w * SEQ_LEN
            x = val_tokens[start:start+SEQ_LEN].unsqueeze(0)
            y = val_tokens[start+1:start+SEQ_LEN+1].unsqueeze(0)
            logits = forward_logits(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            probs = F.softmax(logits[0].float(), dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            top1 = logits[0].argmax(dim=-1)

            all_losses.append(per_tok.cpu().numpy())
            all_positions.append(np.arange(SEQ_LEN))
            all_token_ids.append(y[0].cpu().numpy())
            all_prev_ids.append(x[0].cpu().numpy())
            all_entropies.append(entropy.cpu().numpy())
            all_top1_correct.append((top1 == y[0]).cpu().numpy())

            if (w+1) % 100 == 0:
                print(f'  {w+1}/{windows} windows, {time.time()-t0:.0f}s')

losses = np.concatenate(all_losses)
positions = np.concatenate(all_positions)
token_ids = np.concatenate(all_token_ids)
prev_ids = np.concatenate(all_prev_ids)
entropies = np.concatenate(all_entropies)
top1_correct = np.concatenate(all_top1_correct)
total_tokens = len(losses)
print(f'Tokens: {total_tokens}, avg_loss: {losses.mean():.4f}, time: {time.time()-t0:.0f}s')

# Section 2: Loss distribution
print('\n[2] Loss distribution')
buckets = {'easy(<1)': 0, 'medium(1-3)': 0, 'hard(3-5)': 0, 'very_hard(5+)': 0}
for l in losses:
    if l < 1: buckets['easy(<1)'] += 1
    elif l < 3: buckets['medium(1-3)'] += 1
    elif l < 5: buckets['hard(3-5)'] += 1
    else: buckets['very_hard(5+)'] += 1
results['loss_distribution'] = {k: {'count': v, 'pct': v/total_tokens*100,
    'avg_loss': float(losses[(losses < 1) if 'easy' in k else (losses >= 1) & (losses < 3) if 'medium' in k else (losses >= 3) & (losses < 5) if 'hard(3' in k else (losses >= 5)].mean())}
    for k, v in buckets.items()}
for k, v in results['loss_distribution'].items():
    print(f'  {k:15s}: {v["pct"]:.1f}% avg={v["avg_loss"]:.2f}')

# Section 3: Hard token breakdown
print('\n[3] Hard token breakdown')
hard_mask = (losses >= 3) & (losses < 5)
hard_losses = losses[hard_mask]
hard_tids = token_ids[hard_mask]
hard_pids = prev_ids[hard_mask]

word_initial_1char = sum(1 for t in hard_tids if len(token_strings.get(int(t), '')) == 2 and token_strings.get(int(t), '').startswith('\u2581'))
func_words = sum(1 for t in hard_tids if token_strings.get(int(t), '') in ['\u2581in', '\u2581for', '\u2581that', '\u2581I', '\u2581A', '\u2581a'])
after_period = sum(1 for p in hard_pids if token_strings.get(int(p), '') in ['.', '\u2581.'])
after_the = sum(1 for p in hard_pids if token_strings.get(int(p), '') == '\u2581the')

results['hard_token_breakdown'] = {
    'total_hard': int(hard_mask.sum()),
    'word_initial_1char': word_initial_1char,
    'function_words': func_words,
    'after_period': after_period,
    'after_the': after_the,
}
for k, v in results['hard_token_breakdown'].items():
    print(f'  {k}: {v}')

# Section 4: Position analysis
print('\n[4] Position analysis')
pos_loss = np.zeros(SEQ_LEN)
pos_count = np.zeros(SEQ_LEN)
for i in range(total_tokens):
    p = positions[i]
    pos_loss[p] += losses[i]
    pos_count[p] += 1
pos_avg = pos_loss / np.maximum(pos_count, 1)
results['position'] = {
    'first_64': float(pos_avg[:64].mean()),
    'last_64': float(pos_avg[-64:].mean()),
    'context_benefit': float(pos_avg[:64].mean() - pos_avg[-64:].mean()),
    'ranges': {f'{s}-{min(s+128,SEQ_LEN)}': float(pos_avg[s:min(s+128,SEQ_LEN)].mean()) for s in range(0, SEQ_LEN, 128)},
}
print(f'  context_benefit: {results["position"]["context_benefit"]:.4f}')

# Section 5: Document analysis
print('\n[5] Document analysis')
bos_pos = (val_tokens.cpu() == 1).nonzero(as_tuple=True)[0].numpy()
doc_lengths = np.diff(bos_pos)
results['documents'] = {
    'count': len(bos_pos),
    'mean_length': float(doc_lengths.mean()),
    'median_length': float(np.median(doc_lengths)),
}
print(f'  {len(bos_pos)} docs, median={np.median(doc_lengths):.0f}')

# Section 10: Entropy quadrants
print('\n[10] Entropy quadrants')
loss_thresh = 2.0
ent_thresh = float(np.median(entropies))
results['entropy_quadrants'] = {
    'confident_right': float(((losses < loss_thresh) & (entropies < ent_thresh)).sum() / total_tokens * 100),
    'uncertain_right': float(((losses < loss_thresh) & (entropies >= ent_thresh)).sum() / total_tokens * 100),
    'confident_wrong': float(((losses >= loss_thresh) & (entropies < ent_thresh)).sum() / total_tokens * 100),
    'uncertain_wrong': float(((losses >= loss_thresh) & (entropies >= ent_thresh)).sum() / total_tokens * 100),
}
for k, v in results['entropy_quadrants'].items():
    print(f'  {k}: {v:.1f}%')

# Section 11: Top bigrams
print('\n[11] Top bigrams')
bigram_cost = defaultdict(float)
bigram_count = defaultdict(int)
for i in range(total_tokens):
    bg = (int(prev_ids[i]), int(token_ids[i]))
    bigram_cost[bg] += losses[i]
    bigram_count[bg] += 1
top_bg = sorted(bigram_cost.items(), key=lambda x: x[1], reverse=True)[:20]
results['top_bigrams'] = [
    {'prev': token_strings.get(p, f'[{p}]'), 'cur': token_strings.get(c, f'[{c}]'),
     'total_cost': float(cost), 'count': bigram_count[(p,c)], 'avg': float(cost/bigram_count[(p,c)])}
    for (p, c), cost in top_bg
]

# Section 11b: Top costly tokens
print('\n[11b] Top costly tokens')
tok_cost = defaultdict(float)
tok_count = defaultdict(int)
for i in range(total_tokens):
    t = int(token_ids[i])
    tok_cost[t] += losses[i]
    tok_count[t] += 1
top_tok = sorted(tok_cost.items(), key=lambda x: x[1], reverse=True)[:15]
results['top_costly_tokens'] = [
    {'token': token_strings.get(t, f'[{t}]'), 'id': t,
     'total_cost': float(cost), 'count': tok_count[t], 'avg': float(cost/tok_count[t])}
    for t, cost in top_tok
]

# Section 11c: Juncture analysis
print('\n[11c] Juncture analysis')
after_juncture = np.isin(prev_ids, list(JUNCTURE_IDS))
word_initial = np.isin(token_ids, list(WORD_INITIAL_IDS))
results['juncture'] = {
    'after_juncture_avg': float(losses[after_juncture].mean()),
    'not_after_juncture_avg': float(losses[~after_juncture].mean()),
    'word_initial_avg': float(losses[word_initial].mean()),
    'not_word_initial_avg': float(losses[~word_initial].mean()),
    'after_juncture_pct': float(after_juncture.sum() / total_tokens * 100),
}
print(f'  after_juncture: {results["juncture"]["after_juncture_avg"]:.3f} vs {results["juncture"]["not_after_juncture_avg"]:.3f}')

# =====================================================
# Section 6: Layer ablation
# =====================================================
print(f'\n[6] Layer ablation ({args.ablation_windows} windows)')
abl_windows = min(args.ablation_windows, windows)
with torch.no_grad():
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()
    with ctx:
        base_nll = 0.0
        for w in range(abl_windows):
            start = w * SEQ_LEN
            x = val_tokens[start:start+SEQ_LEN].unsqueeze(0)
            y = val_tokens[start+1:start+SEQ_LEN+1].unsqueeze(0)
            loss = model(x, y) if not LOCAL else F.cross_entropy(forward_logits(model, x)[0].float(), y[0])
            base_nll += loss.item() * SEQ_LEN
        base_loss = base_nll / (abl_windows * SEQ_LEN)

        layer_impacts = {}
        for li in range(NUM_LAYERS):
            sa = model.blocks[li].attn_scale.data.clone()
            sm = model.blocks[li].mlp_scale.data.clone()
            model.blocks[li].attn_scale.data.zero_()
            model.blocks[li].mlp_scale.data.zero_()
            abl_nll = 0.0
            for w in range(abl_windows):
                start = w * SEQ_LEN
                x = val_tokens[start:start+SEQ_LEN].unsqueeze(0)
                y = val_tokens[start+1:start+SEQ_LEN+1].unsqueeze(0)
                loss = model(x, y) if not LOCAL else F.cross_entropy(forward_logits(model, x)[0].float(), y[0])
                abl_nll += loss.item() * SEQ_LEN
            impact = abl_nll / (abl_windows * SEQ_LEN) - base_loss
            layer_impacts[li] = float(impact)
            model.blocks[li].attn_scale.data = sa
            model.blocks[li].mlp_scale.data = sm
            print(f'  L{li}: {impact:+.4f}')

results['layer_ablation'] = {'base_loss': float(base_loss), 'impacts': layer_impacts}

# =====================================================
# Section 7: Head ablation
# =====================================================
print(f'\n[7] Head ablation ({abl_windows} windows)')
head_impacts = {}
with torch.no_grad():
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()
    with ctx:
        for li in range(NUM_LAYERS):
            for hi in range(8):
                saved = model.blocks[li].attn.q_gain.data[hi].clone()
                model.blocks[li].attn.q_gain.data[hi] = 0.0
                abl_nll = 0.0
                for w in range(min(100, abl_windows)):
                    start = w * SEQ_LEN
                    x = val_tokens[start:start+SEQ_LEN].unsqueeze(0)
                    y = val_tokens[start+1:start+SEQ_LEN+1].unsqueeze(0)
                    loss = model(x, y) if not LOCAL else F.cross_entropy(forward_logits(model, x)[0].float(), y[0])
                    abl_nll += loss.item() * SEQ_LEN
                impact = abl_nll / (min(100, abl_windows) * SEQ_LEN) - base_loss
                head_impacts[f'L{li}H{hi}'] = float(impact)
                model.blocks[li].attn.q_gain.data[hi] = saved
            print(f'  L{li}: done')

results['head_ablation'] = head_impacts

# =====================================================
# Section 9: Component ablation
# =====================================================
print(f'\n[9] Component ablation')

# SmearGate
if hasattr(model, 'smear'):
    saved_gate = model.smear.gate.data.clone()
    model.smear.gate.data.fill_(-20.0)
    smear_nll = 0.0
    with torch.no_grad():
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()
        with ctx:
            for w in range(min(100, abl_windows)):
                start = w * SEQ_LEN
                x = val_tokens[start:start+SEQ_LEN].unsqueeze(0)
                y = val_tokens[start+1:start+SEQ_LEN+1].unsqueeze(0)
                loss = model(x, y) if not LOCAL else F.cross_entropy(forward_logits(model, x)[0].float(), y[0])
                smear_nll += loss.item() * SEQ_LEN
    smear_impact = smear_nll / (min(100, abl_windows) * SEQ_LEN) - base_loss
    model.smear.gate.data = saved_gate
    results['component_ablation'] = {'smeargate_removal': float(smear_impact)}
    print(f'  SmearGate removal: {smear_impact:+.4f}')

# MLP per layer
print('  MLP ablation:')
mlp_impacts = {}
with torch.no_grad():
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else torch.no_grad()
    with ctx:
        for li in range(NUM_LAYERS):
            saved = model.blocks[li].mlp_scale.data.clone()
            model.blocks[li].mlp_scale.data.zero_()
            mlp_nll = 0.0
            for w in range(min(100, abl_windows)):
                start = w * SEQ_LEN
                x = val_tokens[start:start+SEQ_LEN].unsqueeze(0)
                y = val_tokens[start+1:start+SEQ_LEN+1].unsqueeze(0)
                loss = model(x, y) if not LOCAL else F.cross_entropy(forward_logits(model, x)[0].float(), y[0])
                mlp_nll += loss.item() * SEQ_LEN
            impact = mlp_nll / (min(100, abl_windows) * SEQ_LEN) - base_loss
            mlp_impacts[li] = float(impact)
            model.blocks[li].mlp_scale.data = saved
            print(f'    L{li} MLP: {impact:+.4f}')
results['mlp_ablation'] = mlp_impacts

# =====================================================
# Save
# =====================================================
out_path = f'{OUT_DIR}/analysis_{args.name}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
print(f'Total time: {time.time()-t0:.0f}s')
print('=== Done ===')
