"""
Local n-gram cache eval using pre-computed log-probs.
Pure CPU numpy — no GPU needed.

Usage: py ngram_eval_local.py
"""
import numpy as np
import math, time, os, sys

NGRAM_ORDERS = list(range(2, 8))
NGRAM_BUCKETS = 4 * 1024 * 1024
NGRAM_PRIMES = [36313, 27191, 51647, 81929, 131071, 65537, 104729]
NGRAM_MIN_COUNT = 2
VOCAB_SIZE = 1024
ENT_BASE = 0.05
ENT_RANGE = 0.55

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..", "..")
INPUT_PATH = os.path.join(PROJECT_DIR, "dashboard", "val_logprobs.npz")
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "data", "tokenizers", "fineweb_1024_bpe.model")

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)


def hash_context(tokens, pos, order, primes, mask):
    if pos < order - 1:
        return -1
    h = 0
    for k in range(order - 1):
        h ^= int(tokens[pos - order + 1 + k]) * primes[k]
    return h & mask


def token_bytes(tid):
    if tid <= 0:
        return 0
    piece = sp.IdToPiece(tid)
    return len(piece.replace("\u2581", " ").encode("utf-8"))


def main():
    print("=== Local N-gram Cache Eval ===")
    print(f"Orders: {NGRAM_ORDERS}, Buckets: {NGRAM_BUCKETS}")
    print()

    print("Loading pre-computed log-probs...")
    t0 = time.time()
    data = np.load(INPUT_PATH)
    log_probs = data["log_probs"].astype(np.float32)
    tokens = data["tokens"]
    doc_lengths = data["doc_lengths"]
    print(f"  Log probs: {log_probs.shape}, Tokens: {tokens.shape}, Docs: {len(doc_lengths)}")
    print(f"  Load time: {time.time()-t0:.0f}s")

    doc_starts = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(doc_lengths, out=doc_starts[1:])
    lp_lengths = doc_lengths - 1
    lp_starts = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(lp_lengths, out=lp_starts[1:])

    mask = NGRAM_BUCKETS - 1
    num_orders = len(NGRAM_ORDERS)

    # N-gram tables — use dict-based sparse storage to save RAM
    # Full arrays would be 6 * 4M * 1024 * 2 bytes = 48 GB
    # Sparse: only store buckets we actually see
    ctx_tables = [{} for _ in range(num_orders)]  # order -> {bucket: {token: count}}

    # Model-only baseline
    print("\nComputing model-only baseline...")
    t0 = time.time()
    total_nll_base = 0.0
    total_bytes = 0
    total_tokens = 0

    for doc_idx in range(len(doc_lengths)):
        ds = int(doc_starts[doc_idx])
        dl = int(doc_lengths[doc_idx])
        lps = int(lp_starts[doc_idx])
        for pos in range(1, dl):
            target = int(tokens[ds + pos])
            total_nll_base -= log_probs[lps + pos - 1, target]
            total_bytes += token_bytes(target)
            total_tokens += 1

    base_bpb = total_nll_base / (total_bytes * math.log(2))
    print(f"  Model-only BPB: {base_bpb:.4f} ({total_tokens} tokens, {total_bytes} bytes, {time.time()-t0:.0f}s)")

    # N-gram cache eval
    print("\nComputing model + n-gram cache...")
    t0 = time.time()
    total_nll_ngram = 0.0
    total_bytes_ngram = 0
    total_tokens_ngram = 0
    ngram_hits = 0

    for doc_idx in range(len(doc_lengths)):
        ds = int(doc_starts[doc_idx])
        dl = int(doc_lengths[doc_idx])
        lps = int(lp_starts[doc_idx])
        doc_toks = tokens[ds:ds + dl]

        for pos in range(1, dl):
            target = int(doc_toks[pos])
            model_lp = log_probs[lps + pos - 1]
            nb = token_bytes(target)

            # N-gram lookup (highest order first)
            ng_p = None
            for oi in range(num_orders - 1, -1, -1):
                order = NGRAM_ORDERS[oi]
                if pos < order - 1:
                    continue
                h = hash_context(doc_toks, pos, order, NGRAM_PRIMES, mask)
                if h < 0:
                    continue
                bucket = ctx_tables[oi].get(h)
                if bucket is None:
                    continue
                total = sum(bucket.values())
                if total < NGRAM_MIN_COUNT:
                    continue
                ng_p = bucket.get(target, 0) / total
                break

            if ng_p is not None:
                ngram_hits += 1
                model_probs = np.exp(model_lp)
                entropy = -np.sum(model_probs * model_lp)
                entropy = max(entropy, 0.0)
                alpha = ENT_BASE + ENT_RANGE / (1.0 + math.exp(-2.0 * (entropy - 4.0)))
                mixed_p = (1 - alpha) * math.exp(model_lp[target]) + alpha * ng_p
                nll = -math.log(max(mixed_p, 1e-10))
            else:
                nll = -model_lp[target]

            total_nll_ngram += nll
            total_bytes_ngram += nb
            total_tokens_ngram += 1

            # Update cache (score-first)
            for oi, order in enumerate(NGRAM_ORDERS):
                if pos < order - 1:
                    continue
                h = hash_context(doc_toks, pos, order, NGRAM_PRIMES, mask)
                if h < 0:
                    continue
                if h not in ctx_tables[oi]:
                    ctx_tables[oi][h] = {}
                bucket = ctx_tables[oi][h]
                bucket[target] = bucket.get(target, 0) + 1

        if (doc_idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            bpb = total_nll_ngram / (total_bytes_ngram * math.log(2)) if total_bytes_ngram > 0 else 0
            hit_rate = ngram_hits / max(total_tokens_ngram, 1) * 100
            print(f"  Docs: {doc_idx+1}/{len(doc_lengths)}, tokens: {total_tokens_ngram}, "
                  f"bpb: {bpb:.4f}, hits: {hit_rate:.1f}%, {elapsed:.0f}s", flush=True)

    ngram_bpb = total_nll_ngram / (total_bytes_ngram * math.log(2))
    hit_rate = ngram_hits / max(total_tokens_ngram, 1) * 100

    print(f"\n{'='*60}")
    print(f"RESULTS ({len(doc_lengths)} docs, {total_tokens_ngram} tokens)")
    print(f"{'='*60}")
    print(f"  Model-only BPB:       {base_bpb:.4f}")
    print(f"  Model + n-gram BPB:   {ngram_bpb:.4f}")
    print(f"  Delta:                {ngram_bpb - base_bpb:+.4f}")
    print(f"  N-gram hit rate:      {hit_rate:.1f}%")
    print(f"  Time:                 {time.time()-t0:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
