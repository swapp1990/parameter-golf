"""
Exp 36: Depth Recurrence — 300s smoke test.
Train models with shared layers and compare val_loss at equivalent steps.

Variants:
  BASE: 11 unique blocks, dim=512 (current architecture)
  B1: 9 unique blocks (middle 3 shared, looped 2x = 12 effective), dim=512
  B2: 9 unique blocks (middle 3 shared, looped 2x = 12 effective), dim=576 (wider, ~same params)
  A1: 6 unique blocks (all shared, looped 2x = 12 effective), dim=512
  A2: 6 unique blocks (all shared, looped 2x = 12 effective), dim=640 (wider)
"""
import torch, os, sys, glob, time, math, io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = os.environ.get("BASE_DIR", "/runpod-volume/parameter-golf")
RECORDS_DIR = os.path.join(BASE_DIR, "records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100")
os.chdir(BASE_DIR)

os.environ.setdefault("MLP_MULT", "3")
os.environ.setdefault("NUM_LAYERS", "11")
os.environ.setdefault("VOCAB_SIZE", "1024")
os.environ.setdefault("TRAIN_SEQ_LEN", "2048")
os.environ["XSA_LAST_N"] = "11"

import importlib.util
_spec = importlib.util.spec_from_file_location("tgp", os.path.join(RECORDS_DIR, "train_gpt.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Block = _mod.Block
SmearGate = _mod.SmearGate
RMSNorm = _mod.RMSNorm
CastedLinear = _mod.CastedLinear
load_data_shard = _mod.load_data_shard

import sentencepiece as spm

device = torch.device("cuda")
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", "500"))
BATCH_TOKENS = 65536
SEQ_LEN = 2048
VOCAB_SIZE = 1024
EVAL_EVERY = 100


class GPTRecurrent(nn.Module):
    """GPT with depth recurrence support.

    layer_schedule: list of block indices defining the forward pass order.
    E.g., [0,1,2,3,4,5,6,7,8] = normal 9-block model
          [0,1,2,3,4,5,3,4,5,6,7,8] = middle 3 blocks looped 2x (12 effective layers)
          [0,1,2,3,4,5,0,1,2,3,4,5] = all 6 blocks looped 2x
    """
    def __init__(self, vocab_size, num_unique_blocks, model_dim, num_heads, num_kv_heads,
                 mlp_mult, layer_schedule, logit_softcap=30.0, rope_base=10000.0,
                 qk_gain_init=1.5, tied_embed_init_std=0.02):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.layer_schedule = layer_schedule
        effective_layers = len(layer_schedule)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        self.smear = SmearGate(model_dim)

        # U-Net skip setup based on effective layers
        self.num_encoder_layers = effective_layers // 2
        self.num_decoder_layers = effective_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Unique blocks — XSA on all
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, use_xsa=True)
            for _ in range(num_unique_blocks)
        ])

        self.final_norm = RMSNorm()

        # Init
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * effective_layers))

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []

        # Encoder half
        for i in range(self.num_encoder_layers):
            block_idx = self.layer_schedule[i]
            x = self.blocks[block_idx](x, x0)
            skips.append(x)

        # Decoder half
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_idx = self.layer_schedule[self.num_encoder_layers + i]
            x = self.blocks[block_idx](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits, targets)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(name, model, train_shards, val_tokens, steps):
    """Train for N steps, eval every EVAL_EVERY steps."""
    model = model.to(device)
    params = count_params(model)
    print(f"\n{'='*60}")
    print(f"{name} | params={params:,} | schedule={model.layer_schedule}")
    print(f"{'='*60}")

    # Simple optimizer — Muon-like setup simplified to Adam for smoke test
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95))

    # Load all train shards into memory
    shard_paths = sorted(glob.glob(os.path.join(BASE_DIR, "data/datasets/fineweb10B_sp1024/fineweb_train_*.bin")))
    all_train = torch.cat([load_data_shard(Path(s)) for s in shard_paths[:5]]).to(device).long()

    t0 = time.perf_counter()
    results = []

    for step in range(1, steps + 1):
        # Random batch
        batch_seqs = BATCH_TOKENS // SEQ_LEN
        idx = torch.randint(0, all_train.numel() - SEQ_LEN - 1, (batch_seqs,))
        x = torch.stack([all_train[i:i+SEQ_LEN] for i in idx])
        y = torch.stack([all_train[i+1:i+SEQ_LEN+1] for i in idx])

        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % EVAL_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                # Eval on first 100K val tokens
                eval_len = min(100000, val_tokens.numel() - SEQ_LEN - 1)
                eval_seqs = eval_len // SEQ_LEN
                eval_loss = 0
                for i in range(0, eval_seqs * SEQ_LEN, SEQ_LEN):
                    ex = val_tokens[i:i+SEQ_LEN].unsqueeze(0)
                    ey = val_tokens[i+1:i+SEQ_LEN+1].unsqueeze(0)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        el = model(ex, ey)
                    eval_loss += el.item()
                eval_loss /= eval_seqs

            elapsed = time.perf_counter() - t0
            print(f"  step {step:>4}/{steps}: train_loss={loss.item():.4f} val_loss={eval_loss:.4f} time={elapsed:.0f}s", flush=True)
            results.append((step, eval_loss))

    final_val = results[-1][1] if results else float('inf')
    elapsed = time.perf_counter() - t0
    print(f"  [{name}] DONE: final_val_loss={final_val:.4f} time={elapsed:.0f}s params={params:,}")

    del model, all_train, optimizer
    torch.cuda.empty_cache()
    return results, params


if __name__ == "__main__":
    print(f"\n{'#'*60}")
    print(f"# Exp 36: Depth Recurrence Smoke Test ({TRAIN_STEPS} steps)")
    print(f"{'#'*60}")

    # Load val tokens
    val_shards = sorted(glob.glob(os.path.join(BASE_DIR, "data/datasets/fineweb10B_sp1024/fineweb_val_*.bin")))
    val_tokens = torch.cat([load_data_shard(Path(s)).to(device) for s in val_shards]).long()

    all_results = {}

    # BASE: 11 unique blocks, dim=512 (current)
    model = GPTRecurrent(
        vocab_size=VOCAB_SIZE, num_unique_blocks=11, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        layer_schedule=list(range(11)),  # [0,1,2,3,4,5,6,7,8,9,10]
    )
    r, p = train_and_eval("BASE: 11 unique, dim=512", model, None, val_tokens, TRAIN_STEPS)
    all_results["BASE: 11 unique, dim=512"] = (r, p)

    # B1: 9 unique, middle 3 shared 2x = 12 effective, dim=512
    model = GPTRecurrent(
        vocab_size=VOCAB_SIZE, num_unique_blocks=9, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        layer_schedule=[0,1,2, 3,4,5, 3,4,5, 6,7,8],  # middle looped
    )
    r, p = train_and_eval("B1: 9 unique (mid shared), dim=512", model, None, val_tokens, TRAIN_STEPS)
    all_results["B1: 9 unique (mid shared), dim=512"] = (r, p)

    # B2: 9 unique, middle 3 shared 2x, dim=576 (use saved params for width)
    model = GPTRecurrent(
        vocab_size=VOCAB_SIZE, num_unique_blocks=9, model_dim=576,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        layer_schedule=[0,1,2, 3,4,5, 3,4,5, 6,7,8],
    )
    r, p = train_and_eval("B2: 9 unique (mid shared), dim=576", model, None, val_tokens, TRAIN_STEPS)
    all_results["B2: 9 unique (mid shared), dim=576"] = (r, p)

    # A1: 6 unique, all shared 2x = 12 effective, dim=512
    model = GPTRecurrent(
        vocab_size=VOCAB_SIZE, num_unique_blocks=6, model_dim=512,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        layer_schedule=[0,1,2,3,4,5, 0,1,2,3,4,5],  # full loop
    )
    r, p = train_and_eval("A1: 6 unique (all shared), dim=512", model, None, val_tokens, TRAIN_STEPS)
    all_results["A1: 6 unique (all shared), dim=512"] = (r, p)

    # A2: 6 unique, all shared 2x, dim=640 (wider)
    model = GPTRecurrent(
        vocab_size=VOCAB_SIZE, num_unique_blocks=6, model_dim=640,
        num_heads=8, num_kv_heads=4, mlp_mult=3,
        layer_schedule=[0,1,2,3,4,5, 0,1,2,3,4,5],
    )
    r, p = train_and_eval("A2: 6 unique (all shared), dim=640", model, None, val_tokens, TRAIN_STEPS)
    all_results["A2: 6 unique (all shared), dim=640"] = (r, p)

    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({TRAIN_STEPS} steps)")
    print(f"{'='*70}")
    print(f"{'Variant':<45} {'Params':>10} {'Val Loss @{0}':>12} {'Val Loss @{1}':>12}".format(
        EVAL_EVERY, TRAIN_STEPS))
    print(f"{'-'*45} {'-'*10} {'-'*12} {'-'*12}")
    for name, (results, params) in all_results.items():
        first = results[0][1] if results else 0
        last = results[-1][1] if results else 0
        print(f"{name:<45} {params:>10,} {first:>12.4f} {last:>12.4f}")
    print(f"{'='*70}")

    best = min(all_results, key=lambda k: all_results[k][0][-1][1])
    print(f"\nBest: {best} (val_loss={all_results[best][0][-1][1]:.4f})")
