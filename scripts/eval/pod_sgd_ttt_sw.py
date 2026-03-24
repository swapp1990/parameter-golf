"""
Pod eval: Combined SGD TTT + Sliding Window. ONE loop.
For each document: save weights → SGD adapt → sliding window score → restore.
"""
import torch, io, os, sys, time, glob, math
import zstandard
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
STRIDE = 256
TTT_LR = 3e-4
BOS_ID = 1

print(f'=== SGD TTT + Sliding Window ({N_DOCS} docs) ===')

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

bos_pos = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
n_docs = min(N_DOCS, len(bos_pos))
print(f'Documents: {n_docs}')

# Save original weights
print('Saving original weights...')
original_sd = {k: v.clone() for k, v in model.state_dict().items()}

# Standard baseline (proper BPB)
print('\n--- Baseline (standard eval) ---')
model.eval()
class A:
    val_batch_size = 524288
    train_seq_len = 2048
t0 = time.time()
std_loss, std_bpb = eval_val(A(), model, 0, 1, torch.device('cuda'), 8,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
print(f'Standard: val_loss={std_loss:.4f} val_bpb={std_bpb:.4f} time={time.time()-t0:.0f}s')

# Combined SGD TTT + Sliding Window
print(f'\n--- Combined SGD TTT + Sliding Window (stride={STRIDE}) ---')
total_nll = 0.0
total_bytes = 0.0
total_tokens = 0
t1 = time.time()

for d in range(n_docs):
    ds = int(bos_pos[d])
    de = int(bos_pos[d+1]) if d+1 < len(bos_pos) else val_tokens.numel()
    doc = val_tokens[ds:de]
    if doc.numel() < 5: continue

    # 1. Restore original weights
    model.load_state_dict(original_sd, strict=True)

    # 2. SGD adapt (1 forward+backward, tiny LR)
    model.train()
    slen = min(1024, doc.numel()-1)
    x = doc[:slen].unsqueeze(0)
    y = doc[1:slen+1].unsqueeze(0)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= TTT_LR * p.grad
    model.zero_grad()

    # 3. Score with sliding window (only last stride tokens)
    model.eval()
    with torch.inference_mode():
        if doc.numel() <= SEQ_LEN + 1:
            # Short doc: score all
            sx = doc[:-1].unsqueeze(0)
            sy = doc[1:].unsqueeze(0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model.tok_emb(sx)
                if hasattr(model, 'smear'): logits = model.smear(logits)
                logits = F.rms_norm(logits, (logits.size(-1),))
                x0 = logits
                skips = []
                for i in range(model.num_encoder_layers):
                    logits = model.blocks[i](logits, x0); skips.append(logits)
                for i in range(model.num_decoder_layers):
                    if skips: logits = logits + model.skip_weights[i].to(dtype=logits.dtype)[None,None,:] * skips.pop()
                    logits = model.blocks[model.num_encoder_layers+i](logits, x0)
                logits = model.final_norm(logits)
                logits = F.linear(logits, model.tok_emb.weight)
                logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
            per_tok = F.cross_entropy(logits[0].float(), sy[0], reduction='none')
            total_nll += per_tok.sum().item()
            total_tokens += per_tok.numel()
            # BPB bytes
            tb = base_bytes_lut[sy[0]].to(torch.int16)
            tb += (has_leading_space_lut[sy[0]] & ~is_boundary_token_lut[sx[0]]).to(torch.int16)
            total_bytes += tb.to(torch.float64).sum().item()
        else:
            # Sliding window
            starts = list(range(0, doc.numel()-SEQ_LEN, STRIDE))
            if starts[-1] + SEQ_LEN < doc.numel()-1:
                starts.append(doc.numel()-SEQ_LEN-1)
            scored = set()
            for s in starts:
                sx = doc[s:s+SEQ_LEN].unsqueeze(0)
                sy = doc[s+1:s+SEQ_LEN+1].unsqueeze(0)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model.tok_emb(sx)
                    if hasattr(model, 'smear'): logits = model.smear(logits)
                    logits = F.rms_norm(logits, (logits.size(-1),))
                    x0 = logits
                    skips = []
                    for i in range(model.num_encoder_layers):
                        logits = model.blocks[i](logits, x0); skips.append(logits)
                    for i in range(model.num_decoder_layers):
                        if skips: logits = logits + model.skip_weights[i].to(dtype=logits.dtype)[None,None,:] * skips.pop()
                        logits = model.blocks[model.num_encoder_layers+i](logits, x0)
                    logits = model.final_norm(logits)
                    logits = F.linear(logits, model.tok_emb.weight)
                    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
                # Score last stride tokens
                score_logits = logits[0, -STRIDE:, :]
                score_targets = sy[0, -STRIDE:]
                per_tok = F.cross_entropy(score_logits.float(), score_targets, reduction='none')
                score_prev = sx[0, -(STRIDE+1):-1]
                abs_start = s + SEQ_LEN - STRIDE + 1
                for i in range(per_tok.numel()):
                    ap = abs_start + i
                    if ap not in scored and ap < doc.numel():
                        scored.add(ap)
                        total_nll += per_tok[i].item()
                        total_tokens += 1
                        tid = score_targets[i].item()
                        pid = score_prev[i].item()
                        b = base_bytes_lut[tid].item() + (has_leading_space_lut[tid].item() & ~is_boundary_token_lut[pid].item())
                        total_bytes += b

    if (d+1) % 1000 == 0:
        elapsed = time.time()-t1
        eta = elapsed/(d+1)*(n_docs-d-1)
        bpb = (total_nll/math.log(2.0))/max(total_bytes,1)
        print(f'  Doc {d+1}/{n_docs}: bpb={bpb:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s')

combined_loss = total_nll / max(total_tokens, 1)
combined_bpb = (total_nll / math.log(2.0)) / max(total_bytes, 1)

print(f'\n{"="*50}')
print(f'RESULTS')
print(f'{"="*50}')
print(f'Artifact:               {len(blob)/1024/1024:.2f} MB')
print(f'Standard:               val_bpb={std_bpb:.4f}')
print(f'SGD TTT + SW(s{STRIDE}):    val_bpb={combined_bpb:.4f}  delta={combined_bpb-std_bpb:+.4f}')
print(f'Docs: {n_docs}, Tokens: {total_tokens}, Time: {time.time()-t1:.0f}s')
print(f'{"="*50}')
