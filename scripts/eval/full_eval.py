"""
Full submission eval: Standard + Sliding Window + TTT, all with proper BPB.
Runs on the mixed-quantized model from this training run.
"""
import torch, io, os, sys, time, glob, math
import zstandard
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard, build_sentencepiece_luts, eval_val
import sentencepiece as spm

print('=== Full Submission Eval ===')

# Load quantized model
print('Loading mixed quantized model...')
with open('final_model.mixed.ptz', 'rb') as f:
    blob = f.read()
print(f'Artifact: {len(blob)} bytes ({len(blob)/1024/1024:.2f} MB)')
raw = zstandard.ZstdDecompressor().decompress(blob)
qs = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
qs.pop('__quant_format__', None)
recovered = {}
seen = set()
for key in list(qs.keys()):
    if key.endswith('.__q'):
        name = key[:-4]
        if name in seen:
            continue
        seen.add(name)
        q, scale = qs[name + '.__q'], qs[name + '.__scale']
        dtype = getattr(torch, qs[name + '.__dtype'].split('.')[-1])
        if q.ndim == 2 and scale.ndim == 1:
            recovered[name] = (q.float() * scale.float()[:, None]).to(dtype)
        else:
            recovered[name] = (q.float() * scale.float()).to(dtype)
    elif not any(key.endswith(s) for s in ('.__scale', '.__dtype')):
        recovered[key] = qs[key]

model = GPT(
    vocab_size=1024, num_layers=11, model_dim=512, num_heads=8, num_kv_heads=4,
    mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)
model.load_state_dict(recovered, strict=False)
model = model.cuda().eval()
print(f'Loaded: {sum(p.numel() for p in model.parameters())} params')

# Load val + tokenizer
val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
sp = spm.SentencePieceProcessor()
sp.Load('./data/tokenizers/fineweb_1024_bpe.model')
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, 1024, 'cuda'
)
print(f'Val tokens: {val_tokens.numel()}')


class Args:
    val_batch_size = 524288
    train_seq_len = 2048


# === 1. Standard eval ===
print('\n--- 1. Standard eval ---')
t0 = time.time()
val_loss, val_bpb = eval_val(
    Args(), model, 0, 1, torch.device('cuda'), 8,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
)
print(f'Standard: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} time={time.time()-t0:.0f}s')

# === 2. Sliding window eval ===
print('\n--- 2. Sliding window (stride=256) ---')
stride = 256
seq_len = 2048
batch_size = 8
starts = list(range(0, val_tokens.numel() - seq_len, stride))
sw_nll = 0.0
sw_bytes = 0.0
sw_tokens = 0
t1 = time.time()
with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for off in range(0, len(starts), batch_size):
        bs = starts[off:off + batch_size]
        x = torch.stack([val_tokens[s:s + seq_len] for s in bs]).to(dtype=torch.int64)
        y = torch.stack([val_tokens[s + 1:s + seq_len + 1] for s in bs]).to(dtype=torch.int64)
        loss = model(x, y).detach()
        sw_nll += loss.item() * y.numel()
        sw_tokens += y.numel()
        prev_ids = x.reshape(-1)
        tgt_ids = y.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.int16)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
        sw_bytes += tb.to(torch.float64).sum().item()
        done = off + batch_size
        if done % 5000 < batch_size:
            print(f'  {done/len(starts)*100:.0f}%')
sw_loss = sw_nll / sw_tokens
sw_bpb = (sw_nll / math.log(2.0)) / sw_bytes
print(f'Sliding: val_loss={sw_loss:.4f} val_bpb={sw_bpb:.4f} time={time.time()-t1:.0f}s')

# === 3. TTT eval ===
print('\n--- 3. TTT (LoRA rank=8, LR=0.05, 1 epoch) ---')


class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_d, device='cuda') * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_d, rank, device='cuda') * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        base = F.linear(
            x, self.original.weight.to(x.dtype),
            self.original.bias.to(x.dtype) if self.original.bias is not None else None,
        )
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale

    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.001)


for p in model.parameters():
    p.requires_grad = False
lora_modules = []
for blk in model.blocks:
    lq = LoRALinear(blk.attn.c_q, rank=8)
    blk.attn.c_q = lq
    lora_modules.append(lq)
    lv = LoRALinear(blk.attn.c_v, rank=8)
    blk.attn.c_v = lv
    lora_modules.append(lv)
lora_params = []
for m in lora_modules:
    lora_params.extend([m.lora_A, m.lora_B])
print(f'LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params')

bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0].cpu().numpy()
n_docs = len(bos_positions)
print(f'Documents: {n_docs}')

ttt_nll = 0.0
ttt_bytes = 0.0
ttt_tokens = 0
t2 = time.time()

for d in range(n_docs):
    doc_start = int(bos_positions[d])
    doc_end = int(bos_positions[d + 1]) if d + 1 < len(bos_positions) else val_tokens.numel()
    doc = val_tokens[doc_start:doc_end].to(dtype=torch.int64)
    if doc.numel() < 5:
        continue

    for m in lora_modules:
        m.reset()
    model.train()
    opt = torch.optim.Adam(lora_params, lr=0.05)
    csz = min(1024, doc.numel() - 1)
    for cs in range(0, doc.numel() - 1, csz):
        ce = min(cs + csz, doc.numel() - 1)
        if ce - cs < 2:
            continue
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(doc[cs:ce].unsqueeze(0), doc[cs + 1:ce + 1].unsqueeze(0))
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.inference_mode():
        slen = min(2048, doc.numel() - 1)
        sx = doc[:slen].unsqueeze(0)
        sy = doc[1:slen + 1].unsqueeze(0)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            sloss = model(sx, sy).detach()
        ttt_nll += sloss.item() * sy.numel()
        ttt_tokens += sy.numel()
        prev_ids = sx.reshape(-1)
        tgt_ids = sy.reshape(-1)
        tb = base_bytes_lut[tgt_ids].to(torch.int16)
        tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
        ttt_bytes += tb.to(torch.float64).sum().item()

    if (d + 1) % 1000 == 0:
        elapsed = time.time() - t2
        eta = elapsed / (d + 1) * (n_docs - d - 1)
        bpb = (ttt_nll / math.log(2.0)) / ttt_bytes
        print(f'  Doc {d+1}/{n_docs}: bpb={bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s')

ttt_loss = ttt_nll / ttt_tokens
ttt_bpb = (ttt_nll / math.log(2.0)) / ttt_bytes
print(f'TTT: val_loss={ttt_loss:.4f} val_bpb={ttt_bpb:.4f} time={time.time()-t2:.0f}s')

# Summary
print(f'\n{"="*50}')
print(f'SUBMISSION RESULTS')
print(f'{"="*50}')
print(f'Artifact: {len(blob)} bytes ({len(blob)/1024/1024:.2f} MB)')
print(f'Standard:       val_loss={val_loss:.4f}  val_bpb={val_bpb:.4f}')
print(f'Sliding(s256):  val_loss={sw_loss:.4f}  val_bpb={sw_bpb:.4f}  delta={sw_bpb-val_bpb:+.4f}')
print(f'TTT(LoRA):      val_loss={ttt_loss:.4f}  val_bpb={ttt_bpb:.4f}  delta={ttt_bpb-val_bpb:+.4f}')
print(f'{"="*50}')
