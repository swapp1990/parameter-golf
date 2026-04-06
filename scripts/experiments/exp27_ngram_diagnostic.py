"""Exp 27: N-gram diagnostic — understand WHY n-gram hurts with full-context TTT.

Tests:
1. Per-token comparison: how often does n-gram predict better than model?
2. Alpha sweep: is there a sweet spot where n-gram helps?
3. Entropy-conditional analysis: does n-gram help in specific entropy ranges?
"""
import torch, os, sys, io, glob, math, time, numpy as np
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'

import importlib.util
_s = importlib.util.spec_from_file_location('t', os.path.join(os.getcwd(),
    'records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100/train_gpt.py'))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
GPT = _m.GPT; load_data_shard = _m.load_data_shard; dequantize_state_dict_mixed = _m.dequantize_state_dict_mixed
import sentencepiece as spm; import zstandard

device = torch.device('cuda')

sp = spm.SentencePieceProcessor(model_file='./data/tokenizers/fineweb_1024_bpe.model')
bbl = torch.zeros(1024, dtype=torch.int64, device=device)
hls = torch.zeros(1024, dtype=torch.bool, device=device)
ibt = torch.zeros(1024, dtype=torch.bool, device=device)
for t in range(1024):
    p = sp.IdToPiece(t) if t > 0 else ''; r = p.replace('\u2581', ' ')
    bbl[t] = len(r.encode('utf-8'))
    if p.startswith('\u2581'): hls[t] = True; bbl[t] -= 1
    if t <= 2: ibt[t] = True

vs = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
vt = torch.cat([load_data_shard(Path(s)).to(device) for s in vs]).long()
vn = vt.cpu().numpy(); bos = np.where(vn == 1)[0]

NGRAM_ORDERS = list(range(2, 8))
NGRAM_BUCKETS = 4 * 1024 * 1024
NGRAM_PRIMES = [36313, 27191, 51647, 81929, 131071, 65537, 104729]
NGRAM_MIN_COUNT = 2
N_DOCS = 200

def _get_logits(mdl, x):
    xe = mdl.tok_emb(x); xe = F.rms_norm(xe, (xe.size(-1),)); xe = mdl.smear(xe); x0 = xe
    skips = []
    for i in range(mdl.num_encoder_layers): xe = mdl.blocks[i](xe, x0); skips.append(xe)
    for i in range(mdl.num_decoder_layers):
        if skips: xe = xe + mdl.skip_weights[i].to(dtype=xe.dtype)[None, None, :] * skips.pop()
        xe = mdl.blocks[mdl.num_encoder_layers + i](xe, x0)
    xe = mdl.final_norm(xe); lp = F.linear(xe, mdl.tok_emb.weight)
    return mdl.logit_softcap * torch.tanh(lp / mdl.logit_softcap)

def _ngram_hash(tokens_np, pos, order):
    if pos < order - 1: return -1
    h = 0
    for k in range(order - 1): h ^= int(tokens_np[pos - order + 1 + k]) * NGRAM_PRIMES[k]
    return h & (NGRAM_BUCKETS - 1)

class LR(nn.Module):
    def __init__(self, o, r=8):
        super().__init__()
        self.o = o; ind = o.weight.shape[1]; od = o.weight.shape[0]
        self.lA = nn.Parameter(torch.randn(r, ind, device=device) * 0.01)
        self.lB = nn.Parameter(torch.randn(od, r, device=device) * 0.001)
        self.sc = 1.0 / r
        for p in self.o.parameters(): p.requires_grad = False
    def forward(self, x):
        b = F.linear(x, self.o.weight.to(x.dtype), self.o.bias.to(x.dtype) if self.o.bias is not None else None)
        return b + (x @ self.lA.to(x.dtype).T @ self.lB.to(x.dtype).T) * self.sc
    def reset(self):
        nn.init.normal_(self.lA, std=0.01); nn.init.normal_(self.lB, std=0.001)

def make_model():
    mdl = GPT(vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
              mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02, logit_softcap=30.0,
              rope_base=10000.0, qk_gain_init=1.5).to(device)
    blob = open('final_model.mixed.ptz', 'rb').read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    ms = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    mdl.load_state_dict(dequantize_state_dict_mixed(ms), strict=False)
    for p in mdl.parameters(): p.requires_grad = False
    lm = []
    for blk in mdl.blocks:
        lq = LR(blk.attn.c_q); blk.attn.c_q = lq; lm.append(lq)
        lv = LR(blk.attn.c_v); blk.attn.c_v = lv; lm.append(lv)
    lp_list = [p for m in lm for p in [m.lA, m.lB]]
    return mdl, lm, lp_list


def run_diagnostic():
    """Full-context train-then-score with per-token n-gram analysis."""
    mdl, lm, lp_list = make_model()

    # Per-token stats
    ngram_wins = 0        # n-gram prob > model prob for correct token
    model_wins = 0        # model prob > n-gram prob
    ngram_total = 0       # tokens where n-gram had a prediction
    total_tokens = 0

    # Entropy-binned stats: [0-1), [1-2), [2-3), [3-4), [4-5), [5-6), [6+)
    ENT_BINS = 7
    bin_ngram_wins = [0] * ENT_BINS
    bin_model_wins = [0] * ENT_BINS
    bin_count = [0] * ENT_BINS
    bin_model_nll_sum = [0.0] * ENT_BINS  # sum of -log(model_prob) for n-gram tokens
    bin_ngram_nll_sum = [0.0] * ENT_BINS  # sum of -log(ngram_prob) for n-gram tokens

    # Alpha sweep: test multiple fixed alpha values
    ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60]
    alpha_nll = {a: 0.0 for a in ALPHAS}  # total NLL per alpha
    baseline_nll = 0.0  # model-only NLL

    # N-gram order stats
    order_wins = {o: 0 for o in NGRAM_ORDERS}
    order_total = {o: 0 for o in NGRAM_ORDERS}

    nll_total = 0.0
    byte_total = 0.0

    for d in range(N_DOCS):
        ds = int(bos[d]); de = int(bos[d + 1]) if d + 1 < len(bos) else len(vn)
        dl = min(de - ds, 2048)
        if dl < 5: continue
        doc = vt[ds:ds + dl].to(device=device, dtype=torch.int64)
        doc_np = doc.cpu().numpy()

        # Reset LoRA and train on doc
        for m in lm: m.reset()
        opt = torch.optim.Adam(lp_list, lr=0.05)
        mdl.train()
        for cs in range(0, dl - 1, 1024):
            ce = min(cs + 1024, dl - 1)
            if ce - cs < 2: continue
            tx = doc[cs:ce].unsqueeze(0); ty = doc[cs + 1:ce + 1].unsqueeze(0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): tl = mdl(tx, ty)
            tl.backward(); opt.step(); opt.zero_grad()

        # Score full doc with n-gram analysis
        mdl.eval()
        ng_tables = [{} for _ in range(len(NGRAM_ORDERS))]
        sl = min(2048, dl - 1)
        sx = doc[:sl].unsqueeze(0); sy = doc[1:sl + 1].unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = _get_logits(mdl, sx)
            lp = F.log_softmax(logits[0].float(), dim=-1)
            tgts = sy[0]

            for t in range(tgts.size(0)):
                tgt = int(tgts[t].item())
                mlp = lp[t]
                dp = t + 1  # position in doc
                model_prob = math.exp(float(mlp[tgt].item()))
                model_nll = -float(mlp[tgt].item())

                # Compute entropy
                ent = -float((torch.exp(mlp) * mlp).sum().item())
                ent = max(ent, 0.0)
                ent_bin = min(int(ent), ENT_BINS - 1)

                # Baseline NLL (no n-gram)
                baseline_nll += model_nll
                total_tokens += 1

                # Byte count for BPB
                ti = tgt; pi = int(sx[0, t].item())
                byte_total += float(bbl[ti].item())
                if hls[ti] and not ibt[pi]: byte_total += 1.0

                # Find n-gram prediction
                ng_p = None
                matched_order = None
                for oi in range(len(NGRAM_ORDERS) - 1, -1, -1):
                    o = NGRAM_ORDERS[oi]; h = _ngram_hash(doc_np, dp, o)
                    if h < 0: continue
                    b = ng_tables[oi].get(h)
                    if b is None: continue
                    tc = sum(b.values())
                    if tc < NGRAM_MIN_COUNT: continue
                    ng_p = b.get(tgt, 0) / tc
                    matched_order = o
                    break

                if ng_p is not None:
                    ngram_total += 1
                    order_total[matched_order] += 1
                    ngram_nll = -math.log(max(ng_p, 1e-10))

                    if ng_p > model_prob:
                        ngram_wins += 1
                        order_wins[matched_order] += 1
                        bin_ngram_wins[ent_bin] += 1
                    else:
                        model_wins += 1
                        bin_model_wins[ent_bin] += 1

                    bin_count[ent_bin] += 1
                    bin_model_nll_sum[ent_bin] += model_nll
                    bin_ngram_nll_sum[ent_bin] += ngram_nll

                    # Alpha sweep
                    for a in ALPHAS:
                        mp = (1 - a) * model_prob + a * ng_p
                        alpha_nll[a] += -math.log(max(mp, 1e-10))
                    # Also add model-only NLL for non-ngram tokens later

                else:
                    # No n-gram prediction — all alphas get model NLL
                    for a in ALPHAS:
                        alpha_nll[a] += model_nll

                # Update n-gram tables
                for oi, o in enumerate(NGRAM_ORDERS):
                    h = _ngram_hash(doc_np, dp, o)
                    if h < 0: continue
                    if h not in ng_tables[oi]: ng_tables[oi][h] = {}
                    ng_tables[oi][h][tgt] = ng_tables[oi][h].get(tgt, 0) + 1

        if (d + 1) % 20 == 0:
            bpb_so_far = (baseline_nll / math.log(2)) / byte_total if byte_total > 0 else 0
            print(f'  [{d+1}/{N_DOCS}] bpb={bpb_so_far:.4f} ngram_wins={ngram_wins}/{ngram_total}', flush=True)

    return {
        'total_tokens': total_tokens,
        'ngram_total': ngram_total,
        'ngram_wins': ngram_wins,
        'model_wins': model_wins,
        'baseline_nll': baseline_nll,
        'byte_total': byte_total,
        'alpha_nll': alpha_nll,
        'bin_ngram_wins': bin_ngram_wins,
        'bin_model_wins': bin_model_wins,
        'bin_count': bin_count,
        'bin_model_nll_sum': bin_model_nll_sum,
        'bin_ngram_nll_sum': bin_ngram_nll_sum,
        'order_wins': order_wins,
        'order_total': order_total,
        'alphas': ALPHAS,
    }


def print_results(r):
    ln2 = math.log(2)
    baseline_bpb = (r['baseline_nll'] / ln2) / r['byte_total']

    print(f'\n{"="*60}')
    print(f'N-GRAM DIAGNOSTIC RESULTS ({N_DOCS} docs)')
    print(f'{"="*60}')

    print(f'\n--- Overall ---')
    print(f'Total tokens scored: {r["total_tokens"]:,}')
    print(f'Tokens with n-gram prediction: {r["ngram_total"]:,} ({100*r["ngram_total"]/r["total_tokens"]:.1f}%)')
    print(f'N-gram wins (higher prob): {r["ngram_wins"]:,} ({100*r["ngram_wins"]/max(r["ngram_total"],1):.1f}%)')
    print(f'Model wins (higher prob):  {r["model_wins"]:,} ({100*r["model_wins"]/max(r["ngram_total"],1):.1f}%)')
    print(f'Baseline BPB (no n-gram): {baseline_bpb:.4f}')

    print(f'\n--- Alpha Sweep (BPB) ---')
    print(f'{"Alpha":>8} {"BPB":>8} {"Delta":>8}')
    for a in r['alphas']:
        bpb = (r['alpha_nll'][a] / ln2) / r['byte_total']
        delta = bpb - baseline_bpb
        marker = ' <-- best' if delta == min((r['alpha_nll'][a2] / ln2) / r['byte_total'] - baseline_bpb for a2 in r['alphas']) else ''
        print(f'{a:>8.3f} {bpb:>8.4f} {delta:>+8.4f}{marker}')

    print(f'\n--- Entropy Bins (tokens with n-gram predictions) ---')
    labels = ['[0-1)', '[1-2)', '[2-3)', '[3-4)', '[4-5)', '[5-6)', '[6+)']
    print(f'{"Bin":>8} {"Count":>8} {"NG wins":>8} {"M wins":>8} {"NG win%":>8} {"Avg M NLL":>10} {"Avg NG NLL":>10} {"NG better?":>10}')
    for i in range(len(labels)):
        c = r['bin_count'][i]
        if c == 0: continue
        nw = r['bin_ngram_wins'][i]; mw = r['bin_model_wins'][i]
        avg_m = r['bin_model_nll_sum'][i] / c
        avg_ng = r['bin_ngram_nll_sum'][i] / c
        better = 'YES' if avg_ng < avg_m else 'no'
        print(f'{labels[i]:>8} {c:>8} {nw:>8} {mw:>8} {100*nw/c:>7.1f}% {avg_m:>10.4f} {avg_ng:>10.4f} {better:>10}')

    print(f'\n--- N-gram Order Analysis ---')
    print(f'{"Order":>6} {"Total":>8} {"Wins":>8} {"Win%":>8}')
    for o in NGRAM_ORDERS:
        t = r['order_total'][o]
        if t == 0: continue
        w = r['order_wins'][o]
        print(f'{o:>6} {t:>8} {w:>8} {100*w/t:>7.1f}%')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    print(f'=== Exp 27: N-gram Diagnostic ({N_DOCS} docs, full-context TTT) ===', flush=True)
    t0 = time.perf_counter()
    results = run_diagnostic()
    elapsed = time.perf_counter() - t0
    print(f'\nCompleted in {elapsed:.0f}s')
    print_results(results)
