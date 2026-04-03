"""Compare XSA-last4 (submission) vs XSA-all (exp26) per-token log-probs.
Outputs JSON for visualization."""
import torch, os, sys, io, glob, math, time, json, numpy as np
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'

import importlib.util
_s = importlib.util.spec_from_file_location('t', os.path.join(os.getcwd(),
    'records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100/train_gpt.py'))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
GPT = _m.GPT; load_data_shard = _m.load_data_shard; dequantize_state_dict_mixed = _m.dequantize_state_dict_mixed
import sentencepiece as spm; import zstandard

device = torch.device('cuda')
N_DOCS = int(os.environ.get('N_DOCS', '100'))
TOP_K = int(os.environ.get('TOP_K', '50'))

# Tokenizer
sp = spm.SentencePieceProcessor(model_file='./data/tokenizers/fineweb_1024_bpe.model')

# Val data
vs = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
vt = torch.cat([load_data_shard(Path(s)).to(device) for s in vs]).long()
vn = vt.cpu().numpy()
bos = np.where(vn == 1)[0]

def load_model(checkpoint_path, xsa_last_n):
    mdl = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
              mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02, logit_softcap=30.0,
              rope_base=10000.0, qk_gain_init=1.5, xsa_last_n=xsa_last_n).to(device)
    blob = open(checkpoint_path, 'rb').read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    ms = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    mdl.load_state_dict(dequantize_state_dict_mixed(ms), strict=False)
    mdl.eval()
    for p in mdl.parameters(): p.requires_grad = False
    return mdl

def get_logprobs(mdl, x):
    """Get per-token log-probabilities for target tokens."""
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Manual forward to get logits
            xe = mdl.tok_emb(x); xe = F.rms_norm(xe, (xe.size(-1),)); xe = mdl.smear(xe); x0 = xe
            skips = []
            for i in range(mdl.num_encoder_layers): xe = mdl.blocks[i](xe, x0); skips.append(xe)
            for i in range(mdl.num_decoder_layers):
                if skips: xe = xe + mdl.skip_weights[i].to(dtype=xe.dtype)[None, None, :] * skips.pop()
                xe = mdl.blocks[mdl.num_encoder_layers + i](xe, x0)
            xe = mdl.final_norm(xe)
            logits = F.linear(xe, mdl.tok_emb.weight)
            logits = mdl.logit_softcap * torch.tanh(logits / mdl.logit_softcap)
        # Log-softmax in float32
        lp = F.log_softmax(logits[0].float(), dim=-1)
    return lp

def decode_token(token_id):
    """Decode a single token to text."""
    if token_id <= 2:
        return ['<unk>', '<s>', '</s>'][token_id]
    return sp.IdToPiece(int(token_id)).replace('\u2581', ' ')

def find_previous_occurrences(token_ids, pos, target_id):
    """Find all positions before pos where target_id appeared."""
    positions = []
    for i in range(pos):
        if token_ids[i] == target_id:
            positions.append(i)
    return positions

print(f'=== XSA Comparison: {N_DOCS} docs ===', flush=True)
print(f'Loading submission model (XSA last 4)...', flush=True)
mdl_sub = load_model('final_model.submission.ptz', xsa_last_n=4)
print(f'Loading exp26 model (XSA all 11)...', flush=True)
mdl_exp = load_model('final_model.mixed.ptz', xsa_last_n=11)

# Collect per-token comparisons
all_improvements = []  # (delta, doc_id, pos, token_id, lp_sub, lp_exp, entropy_sub, doc_len)

# Per-bucket stats
bucket_stats = {
    'prev_far': {'count': 0, 'better': 0, 'total_delta': 0.0},    # >200 tokens back
    'prev_mid': {'count': 0, 'better': 0, 'total_delta': 0.0},    # 50-200 tokens back
    'prev_near': {'count': 0, 'better': 0, 'total_delta': 0.0},   # <50 tokens back
    'prev_none': {'count': 0, 'better': 0, 'total_delta': 0.0},   # never seen
}
pos_stats = {
    'early': {'count': 0, 'better': 0, 'total_delta': 0.0},    # 0-256
    'mid': {'count': 0, 'better': 0, 'total_delta': 0.0},      # 256-1024
    'late': {'count': 0, 'better': 0, 'total_delta': 0.0},     # 1024+
}
ent_stats = {
    'low': {'count': 0, 'better': 0, 'total_delta': 0.0},      # <2.0
    'mid': {'count': 0, 'better': 0, 'total_delta': 0.0},      # 2.0-4.0
    'high': {'count': 0, 'better': 0, 'total_delta': 0.0},     # >4.0
}

# Store doc data for visualization
doc_data = []

t0 = time.perf_counter()
for d in range(N_DOCS):
    ds = int(bos[d]); de = int(bos[d + 1]) if d + 1 < len(bos) else len(vn)
    dl = min(de - ds, 2048)
    if dl < 10: continue
    doc = vt[ds:ds + dl].to(device=device, dtype=torch.int64)
    doc_ids = doc.cpu().numpy().tolist()
    sl = min(2048, dl - 1)
    x = doc[:sl].unsqueeze(0)

    lp_sub = get_logprobs(mdl_sub, x)  # [sl, vocab]
    lp_exp = get_logprobs(mdl_exp, x)  # [sl, vocab]

    # Per-token analysis
    doc_tokens = []
    for t in range(sl):
        tgt = doc_ids[t + 1]
        lp_s = float(lp_sub[t, tgt].item())
        lp_e = float(lp_exp[t, tgt].item())
        delta = lp_e - lp_s  # positive = exp26 better

        # Entropy of submission model at this position
        ent = -float((torch.exp(lp_sub[t]) * lp_sub[t]).sum().item())
        ent = max(ent, 0.0)

        # Previous occurrences
        prev_positions = find_previous_occurrences(doc_ids, t + 1, tgt)
        if prev_positions:
            nearest_dist = (t + 1) - prev_positions[-1]
        else:
            nearest_dist = -1  # never seen

        # Bucket stats
        if nearest_dist > 200:
            b = 'prev_far'
        elif nearest_dist > 50:
            b = 'prev_mid'
        elif nearest_dist > 0:
            b = 'prev_near'
        else:
            b = 'prev_none'
        bucket_stats[b]['count'] += 1
        bucket_stats[b]['total_delta'] += delta
        if delta > 0: bucket_stats[b]['better'] += 1

        if t < 256: pb = 'early'
        elif t < 1024: pb = 'mid'
        else: pb = 'late'
        pos_stats[pb]['count'] += 1
        pos_stats[pb]['total_delta'] += delta
        if delta > 0: pos_stats[pb]['better'] += 1

        if ent < 2.0: eb = 'low'
        elif ent < 4.0: eb = 'mid'
        else: eb = 'high'
        ent_stats[eb]['count'] += 1
        ent_stats[eb]['total_delta'] += delta
        if delta > 0: ent_stats[eb]['better'] += 1

        all_improvements.append((delta, d, t, tgt, lp_s, lp_e, ent, dl, nearest_dist))

        doc_tokens.append({
            'text': decode_token(tgt),
            'delta': round(delta, 4),
            'lp_sub': round(lp_s, 4),
            'lp_exp': round(lp_e, 4),
            'entropy': round(ent, 2),
            'prev_dist': nearest_dist,
        })

    doc_data.append({
        'doc_id': d,
        'doc_len': dl,
        'tokens': doc_tokens,
        'avg_delta': round(sum(t['delta'] for t in doc_tokens) / len(doc_tokens), 4),
    })

    if (d + 1) % 10 == 0:
        elapsed = time.perf_counter() - t0
        print(f'  [{d+1}/{N_DOCS}] {elapsed:.0f}s', flush=True)

elapsed = time.perf_counter() - t0
print(f'\nDone in {elapsed:.0f}s', flush=True)

# Sort for top improvements and top regressions
all_improvements.sort(key=lambda x: -x[0])
top_better = all_improvements[:TOP_K]
top_worse = all_improvements[-TOP_K:]

# Build top improvement details
def build_detail(item):
    delta, doc_id, pos, tgt, lp_s, lp_e, ent, dl, nearest_dist = item
    doc_entry = doc_data[doc_id]
    # Context: 10 tokens before and after
    start = max(0, pos - 10)
    end = min(len(doc_entry['tokens']), pos + 11)
    context_before = [doc_entry['tokens'][i]['text'] for i in range(start, pos)]
    context_after = [doc_entry['tokens'][i]['text'] for i in range(pos + 1, end)]
    target_text = decode_token(tgt)

    # Previous occurrence context
    prev_context = None
    if nearest_dist > 0:
        prev_pos = pos + 1 - nearest_dist
        if prev_pos >= 0:
            ps = max(0, prev_pos - 5)
            pe = min(len(doc_entry['tokens']), prev_pos + 6)
            # prev_pos is in doc_ids space, doc_tokens is offset by 1
            if prev_pos - 1 >= 0 and prev_pos - 1 < len(doc_entry['tokens']):
                prev_ctx_before = [doc_entry['tokens'][i]['text'] for i in range(max(0, prev_pos - 1 - 5), prev_pos - 1)]
                prev_ctx_after = [doc_entry['tokens'][i]['text'] for i in range(prev_pos, min(len(doc_entry['tokens']), prev_pos + 5))]
                prev_context = {
                    'before': ''.join(prev_ctx_before),
                    'after': ''.join(prev_ctx_after),
                    'position': prev_pos,
                }

    return {
        'delta': round(delta, 4),
        'doc_id': doc_id,
        'position': pos,
        'doc_len': dl,
        'target': target_text,
        'prob_sub': round(math.exp(lp_s), 6),
        'prob_exp': round(math.exp(lp_e), 6),
        'lp_sub': round(lp_s, 4),
        'lp_exp': round(lp_e, 4),
        'entropy': round(ent, 2),
        'prev_dist': nearest_dist,
        'context_before': ''.join(context_before),
        'context_after': ''.join(context_after),
        'prev_context': prev_context,
    }

top_better_details = [build_detail(x) for x in top_better]
top_worse_details = [build_detail(x) for x in top_worse]

# Compute bucket summaries
def summarize_bucket(stats):
    result = {}
    for k, v in stats.items():
        c = v['count']
        if c == 0: continue
        result[k] = {
            'count': c,
            'pct_better': round(100 * v['better'] / c, 1),
            'avg_delta': round(v['total_delta'] / c, 4),
        }
    return result

output = {
    'n_docs': N_DOCS,
    'total_tokens': len(all_improvements),
    'overall_pct_better': round(100 * sum(1 for x in all_improvements if x[0] > 0) / len(all_improvements), 1),
    'overall_avg_delta': round(sum(x[0] for x in all_improvements) / len(all_improvements), 4),
    'buckets_prev_occurrence': summarize_bucket(bucket_stats),
    'buckets_position': summarize_bucket(pos_stats),
    'buckets_entropy': summarize_bucket(ent_stats),
    'top_better': top_better_details,
    'top_worse': top_worse_details,
    'docs': doc_data,
}

out_path = os.environ.get('OUTPUT_PATH', 'xsa_comparison.json')
with open(out_path, 'w') as f:
    json.dump(output, f)
print(f'Saved to {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)', flush=True)
