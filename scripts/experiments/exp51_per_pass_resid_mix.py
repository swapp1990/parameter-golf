"""
Exp 51: Per-Pass resid_mix — monkey-patch the model before training.

Instead of each block having one resid_mix used for all passes,
shared blocks get one resid_mix per pass. This lets the block
blend x and x0 differently on each recurrence pass.

Usage: Run this script which patches the model class then launches training.
"""
import os, sys, types, math
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir("/runpod-volume/parameter-golf")

# Set env vars before importing train_gpt
for k, v in {
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024", "NUM_LAYERS": "7", "MODEL_DIM": "624", "MLP_MULT": "3",
    "LAYER_SCHEDULE": "0,1,2,3,4,3,4,3,4,5,6,5,6",
    "XSA_LAST_N": "7", "TRAIN_SEQ_LEN": "2048",
    "TRAIN_BATCH_TOKENS": "524288",
    "MAX_WALLCLOCK_SECONDS": "700",
    "WARMDOWN_ITERS": "150",
    "VAL_LOSS_EVERY": "100",
    "EMA_ENABLED": "0",
    "TTT_ENABLED": "0",
    "RUN_ID": "exp51_per_pass_resid_mix",
}.items():
    os.environ.setdefault(k, v)

# Import training module
import importlib.util
RECORDS_DIR = "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100"
_spec = importlib.util.spec_from_file_location("tgp", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)

# Patch Block.forward BEFORE loading the module
_original_block_init = None
_original_gpt_init = None

_spec.loader.exec_module(_mod)

Block = _mod.Block
GPT = _mod.GPT

# Store original methods
_orig_block_init = Block.__init__
_orig_block_forward = Block.forward
_orig_gpt_forward = GPT.forward

SCHEDULE = [0,1,2,3,4,3,4,3,4,5,6,5,6]
# Count how many times each block appears
from collections import Counter
block_pass_counts = Counter(SCHEDULE)
# Blocks with >1 pass need per-pass resid_mix
shared_blocks = {b: c for b, c in block_pass_counts.items() if c > 1}
print(f"Shared blocks needing per-pass resid_mix: {shared_blocks}")


def patched_block_init(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, use_xsa=False):
    _orig_block_init(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, use_xsa)
    # We'll add per-pass resid_mix later in GPT init when we know the block index


def patched_gpt_init(self, **kwargs):
    _orig_gpt_init(self, **kwargs)
    # Add per-pass resid_mix to shared blocks
    for block_idx, n_passes in shared_blocks.items():
        if block_idx < len(self.blocks):
            block = self.blocks[block_idx]
            # Create per-pass resid_mix: [n_passes, 2, dim]
            # Initialize from the existing resid_mix
            per_pass = block.resid_mix.data.unsqueeze(0).expand(n_passes, -1, -1).clone()
            block.resid_mix_per_pass = nn.Parameter(per_pass)
            print(f"  Block {block_idx}: added per-pass resid_mix [{n_passes}, 2, {per_pass.shape[-1]}]")


# Track which pass each block is on during forward
_block_call_counts = {}

def patched_gpt_forward(self, input_ids, target_ids):
    # Reset pass counters
    for b in shared_blocks:
        _block_call_counts[b] = 0

    x = self.tok_emb(input_ids)
    if self.ngram is not None:
        x = x + self.ngram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = self.smear(x)
    x0 = x
    skips = []

    for i in range(self.num_encoder_layers):
        block_idx = self.layer_schedule[i]
        block = self.blocks[block_idx]

        # Use per-pass resid_mix if available
        if hasattr(block, 'resid_mix_per_pass') and block_idx in shared_blocks:
            pass_idx = _block_call_counts[block_idx]
            mix = block.resid_mix_per_pass[pass_idx].to(dtype=x.dtype)
            _block_call_counts[block_idx] += 1
            # Manual forward with custom resid_mix
            x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out = block.attn(block.attn_norm(x_mixed))
            x_mixed = x_mixed + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x_mixed + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x_mixed))
        else:
            x = block(x, x0)
        skips.append(x)

    for i in range(self.num_decoder_layers):
        if skips:
            x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        block_idx = self.layer_schedule[self.num_encoder_layers + i]
        block = self.blocks[block_idx]

        if hasattr(block, 'resid_mix_per_pass') and block_idx in shared_blocks:
            pass_idx = _block_call_counts[block_idx]
            mix = block.resid_mix_per_pass[pass_idx].to(dtype=x.dtype)
            _block_call_counts[block_idx] += 1
            x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            attn_out = block.attn(block.attn_norm(x_mixed))
            x_mixed = x_mixed + block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            x = x_mixed + block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x_mixed))
        else:
            x = block(x, x0)

    x = self.final_norm(x).reshape(-1, x.size(-1))
    targets = target_ids.reshape(-1)
    if self.tie_embeddings:
        logits_proj = F.linear(x, self.tok_emb.weight)
    else:
        logits_proj = self.lm_head(x)
    logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction="mean")


# Apply patches
GPT_orig_init = GPT.__init__

def new_gpt_init(self, **kwargs):
    GPT_orig_init(self, **kwargs)
    for block_idx, n_passes in shared_blocks.items():
        if block_idx < len(self.blocks):
            block = self.blocks[block_idx]
            per_pass = block.resid_mix.data.unsqueeze(0).expand(n_passes, -1, -1).clone()
            block.resid_mix_per_pass = nn.Parameter(per_pass)

GPT.__init__ = new_gpt_init
GPT.forward = patched_gpt_forward

print("Patches applied. Starting training...")
_mod.main()
