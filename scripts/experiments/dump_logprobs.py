"""
Dump per-token log-probs from model for the entire val set.
Saves as npz: token_ids, log_probs (sparse top-K for size), doc_boundaries.

Usage: python dump_logprobs.py
Output: /runpod-volume/val_logprobs.npz
"""
import torch, os, sys, glob, time, math
import numpy as np
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, "/runpod-volume/parameter-golf")
os.chdir("/runpod-volume/parameter-golf")
os.environ["MLP_MULT"] = "3"
os.environ["NUM_LAYERS"] = "11"
from train_gpt import GPT, load_data_shard
import sentencepiece as spm

VOCAB_SIZE = 1024
SEQ_LEN = 2048
MODEL_PATH = "/runpod-volume/final_model_1024.pt"
OUTPUT_PATH = "/runpod-volume/val_logprobs.npz"


def get_logits(model, x):
    x_emb = model.tok_emb(x)
    if model.ngram is not None:
        x_emb = x_emb + model.ngram(x)
    x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))
    x_emb = model.smear(x_emb)
    x0 = x_emb
    skips = []
    for i in range(model.num_encoder_layers):
        x_emb = model.blocks[i](x_emb, x0)
        skips.append(x_emb)
    for i in range(model.num_decoder_layers):
        if skips:
            x_emb = x_emb + model.skip_weights[i].to(dtype=x_emb.dtype)[None, None, :] * skips.pop()
        x_emb = model.blocks[model.num_encoder_layers + i](x_emb, x0)
    x_emb = model.final_norm(x_emb)
    logits_proj = F.linear(x_emb, model.tok_emb.weight)
    logits = model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)
    return logits


def main():
    print("=== Dump Val Log-Probs ===")

    # Load model
    print("Loading model...")
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=11, model_dim=512, num_heads=8,
        num_kv_heads=4, mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
        logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    )
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model = model.cuda().eval().bfloat16()
    print(f"  Params: {sum(p.numel() for p in model.parameters())}")

    # Load val tokens
    print("Loading val tokens...")
    val_shards = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))
    val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
    val_np = val_tokens.cpu().numpy().astype(np.uint16)
    num_tokens = len(val_np)
    print(f"  Val tokens: {num_tokens}")

    # Find document boundaries
    bos_positions = np.where(val_np == 1)[0]
    print(f"  Documents: {len(bos_positions)}")

    # Pre-compute full log-prob distribution per document
    # Store as float16 to save space: [num_tokens, vocab_size]
    # 62M * 1024 * 2 bytes = 127 GB — too big!
    # Instead store per-token NLL (cross-entropy with true target) + full log_probs for just the predicted position
    # Actually, for n-gram interpolation we need the full distribution.
    # Compromise: store full log_softmax as float16, but only for first N docs.

    # Better approach: store full log_probs per document, process docs sequentially
    # Save per-doc: log_probs as float16 [doc_len, vocab_size]
    # For 5000 docs, avg 1240 tokens: 5000 * 1240 * 1024 * 2 = ~12 GB — still too big

    # Best approach: just store the full log_softmax as a memory-mapped file
    # Or: save only what we need — for each token, save the full 1024-dim log_prob vector
    # Limit to first 10000 docs (~12M tokens): 12M * 1024 * 2 bytes = 24 GB — still big

    # Practical approach: save per-token log_probs for first 2000 docs
    # ~2000 * 1240 = 2.5M tokens * 1024 * 2 bytes = 5 GB — manageable but big download

    # Even better: save per-token log_probs as float16 in chunks, compressed
    # Or just save the sparse top-K probs + a "rest" mass

    # Simplest that works: for each token, store full log_softmax as float16
    # Limit to 500 docs for speed and size
    MAX_DOCS = 2000
    doc_ends = np.append(bos_positions, num_tokens)

    all_log_probs = []  # list of numpy arrays [doc_len-1, vocab_size] float16
    all_tokens = []     # list of numpy arrays [doc_len] uint16
    doc_lengths = []
    t0 = time.time()

    with torch.no_grad():
        for doc_idx in range(min(MAX_DOCS, len(bos_positions))):
            doc_start = int(doc_ends[doc_idx])
            doc_end = int(doc_ends[doc_idx + 1])
            doc_toks = val_tokens[doc_start:doc_end]
            if len(doc_toks) < 4:
                continue
            if len(doc_toks) > SEQ_LEN:
                doc_toks = doc_toks[:SEQ_LEN]

            x = doc_toks.unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = get_logits(model, x)
            # log_probs for predicting token[i+1] from position i
            lp = F.log_softmax(logits[0, :-1].float(), dim=-1)  # [T-1, V]
            all_log_probs.append(lp.cpu().half().numpy())
            all_tokens.append(doc_toks.cpu().numpy().astype(np.uint16))
            doc_lengths.append(len(doc_toks))

            if (doc_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                total_toks = sum(doc_lengths)
                print(f"  Docs: {doc_idx+1}, tokens: {total_toks}, {elapsed:.0f}s", flush=True)

    # Concatenate and save
    print("Saving...")
    cat_log_probs = np.concatenate(all_log_probs, axis=0)  # [total_pred_tokens, 1024]
    cat_tokens = np.concatenate(all_tokens, axis=0)          # [total_tokens]
    doc_lengths = np.array(doc_lengths, dtype=np.int32)

    print(f"  Log probs shape: {cat_log_probs.shape}, dtype: {cat_log_probs.dtype}")
    print(f"  Tokens shape: {cat_tokens.shape}")
    print(f"  Estimated file size: {cat_log_probs.nbytes / 1e9:.1f} GB")

    np.savez_compressed(
        OUTPUT_PATH,
        log_probs=cat_log_probs,
        tokens=cat_tokens,
        doc_lengths=doc_lengths,
    )
    fsize = os.path.getsize(OUTPUT_PATH)
    print(f"  Saved to {OUTPUT_PATH}: {fsize / 1e6:.0f} MB")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
