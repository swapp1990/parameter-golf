"""
LoRA TTT eval on full 50K docs with proper BPB.
No sliding window — just TTT per document with standard scoring.
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

N_DOCS = int(os.environ.get('EVAL_DOCS', '50000'))
SEQ_LEN = 2048
LORA_RANK = 8
LORA_LR = 0.05
BOS_ID = 1

print(f'=== LoRA TTT Full Eval ({N_DOCS} docs) ===')

# Load quantized model
print('Loading model...')
with open('final_model.mixed.ptz', 'rb') as f:
    blob = f.read()
print(f'Artifact: {len(blob)/1024/1024:.2f} MB')
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
model = model.cuda()
print(f'Params: {sum(p.numel() for p in model.parameters())}')

# Val tokens + tokenizer
val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
sp = spm.SentencePieceProcessor()
sp.Load('./data/tokenizers/fineweb_1024_bpe.model')
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, 1024, 'cuda')
print(f'Val tokens: {val_tokens.numel()}')

# Baseline (proper BPB)
print('\n--- Baseline ---')
model.eval()
class A:
    val_batch_size = 524288
    train_seq_len = 2048
t0 = time.time()
std_loss, std_bpb = eval_val(A(), model, 0, 1, torch.device('cuda'), 8,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
print(f'Standard: val_loss={std_loss:.4f} val_bpb={std_bpb:.4f} time={time.time()-t0:.0f}s')

# Inject LoRA
print('\n--- LoRA TTT ---')
for p in model.parameters():
    p.requires_grad = False

class LoRALinear(nn.Module):
    def __init__(self, original, rank=8):
        super().__init__()
        self.original = original
        in_d = original.weight.shape[1]
        out_d = original.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_d, device='cuda') * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_d, rank, device='cuda') * 0.001)
        self.scale = 1.0 / rank
        for p in self.original.parameters(): p.requires_grad = False
    def forward(self, x):
        base = F.linear(x, self.original.weight.to(x.dtype),
                        self.original.bias.to(x.dtype) if self.original.bias is not None else None)
        return base + (x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T) * self.scale
    def reset(self):
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.normal_(self.lora_B, std=0.001)

lora_modules = []
for blk in model.blocks:
    lq = LoRALinear(blk.attn.c_q, rank=LORA_RANK); blk.attn.c_q = lq; lora_modules.append(lq)
    lv = LoRALinear(blk.attn.c_v, rank=LORA_RANK); blk.attn.c_v = lv; lora_modules.append(lv)
lora_params = []
for m in lora_modules: lora_params.extend([m.lora_A, m.lora_B])
print(f'LoRA: {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params')

bos_pos = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
n_docs = min(N_DOCS, len(bos_pos))
print(f'Documents: {n_docs}')

ttt_nll = 0.0
ttt_bytes = 0.0
ttt_tokens = 0
t1 = time.time()

for d in range(n_docs):
    ds = int(bos_pos[d])
    de = int(bos_pos[d+1]) if d+1 < len(bos_pos) else val_tokens.numel()
    doc = val_tokens[ds:de].to(torch.int64)
    if doc.numel() < 5: continue

    # Adapt
    for m in lora_modules: m.reset()
    model.train()
    opt = torch.optim.Adam(lora_params, lr=LORA_LR)
    csz = min(1024, doc.numel()-1)
    for cs in range(0, doc.numel()-1, csz):
        ce = min(cs+csz, doc.numel()-1)
        if ce-cs < 2: continue
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(doc[cs:ce].unsqueeze(0), doc[cs+1:ce+1].unsqueeze(0))
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Score
    model.eval()
    with torch.inference_mode():
        slen = min(SEQ_LEN, doc.numel()-1)
        sx, sy = doc[:slen].unsqueeze(0), doc[1:slen+1].unsqueeze(0)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            sloss = model(sx, sy).detach()
        ttt_nll += sloss.item() * sy.numel()
        ttt_tokens += sy.numel()
        tb = base_bytes_lut[sy.reshape(-1)].to(torch.int16)
        tb += (has_leading_space_lut[sy.reshape(-1)] & ~is_boundary_token_lut[sx.reshape(-1)]).to(torch.int16)
        ttt_bytes += tb.to(torch.float64).sum().item()

    if (d+1) % 1000 == 0:
        elapsed = time.time()-t1
        eta = elapsed/(d+1)*(n_docs-d-1)
        bpb = (ttt_nll/math.log(2.0))/ttt_bytes
        print(f'  Doc {d+1}/{n_docs}: bpb={bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s')

ttt_loss = ttt_nll / ttt_tokens
ttt_bpb = (ttt_nll / math.log(2.0)) / ttt_bytes

print(f'\n{"="*50}')
print(f'RESULTS')
print(f'{"="*50}')
print(f'Standard:     val_bpb={std_bpb:.4f}')
print(f'LoRA TTT:     val_bpb={ttt_bpb:.4f}  delta={ttt_bpb-std_bpb:+.4f}')
print(f'Docs: {n_docs}, Tokens: {ttt_tokens}, Time: {time.time()-t1:.0f}s')
print(f'{"="*50}')
