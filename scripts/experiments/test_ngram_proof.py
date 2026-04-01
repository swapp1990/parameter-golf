"""Prove n-gram helps chunked but hurts full-context scoring."""
import torch, os, sys, io, glob, math, time, numpy as np
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
os.environ['MLP_MULT']='3'; os.environ['NUM_LAYERS']='11'
import importlib.util
_s = importlib.util.spec_from_file_location('t', os.path.join(os.getcwd(), 'records/track_non_record_16mb/2026-03-24_11L_XSA_SwiGLU_LoRATTT_1xH100/train_gpt.py'))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
GPT=_m.GPT; load_data_shard=_m.load_data_shard; dequantize_state_dict_mixed=_m.dequantize_state_dict_mixed
import sentencepiece as spm; import zstandard
device = torch.device('cuda')

sp = spm.SentencePieceProcessor(model_file='./data/tokenizers/fineweb_1024_bpe.model')
bbl=torch.zeros(1024,dtype=torch.int64,device=device)
hls=torch.zeros(1024,dtype=torch.bool,device=device)
ibt=torch.zeros(1024,dtype=torch.bool,device=device)
for t in range(1024):
    p=sp.IdToPiece(t) if t>0 else ''; r=p.replace('\u2581',' ')
    bbl[t]=len(r.encode('utf-8'))
    if p.startswith('\u2581'): hls[t]=True; bbl[t]-=1
    if t<=2: ibt[t]=True

vs = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
vt = torch.cat([load_data_shard(Path(s)).to(device) for s in vs]).long()
vn = vt.cpu().numpy(); bos = np.where(vn==1)[0]

NGRAM_ORDERS = list(range(2,8))
NGRAM_BUCKETS = 4*1024*1024
NGRAM_PRIMES = [36313,27191,51647,81929,131071,65537,104729]
NGRAM_MIN_COUNT = 2
ENT_BASE = 0.05; ENT_RANGE = 0.55
N_DOCS = 200

def _get_logits(mdl, x):
    xe=mdl.tok_emb(x); xe=F.rms_norm(xe,(xe.size(-1),)); xe=mdl.smear(xe); x0=xe
    skips=[]
    for i in range(mdl.num_encoder_layers): xe=mdl.blocks[i](xe,x0); skips.append(xe)
    for i in range(mdl.num_decoder_layers):
        if skips: xe=xe+mdl.skip_weights[i].to(dtype=xe.dtype)[None,None,:]*skips.pop()
        xe=mdl.blocks[mdl.num_encoder_layers+i](xe,x0)
    xe=mdl.final_norm(xe); lp=F.linear(xe,mdl.tok_emb.weight)
    return mdl.logit_softcap*torch.tanh(lp/mdl.logit_softcap)

def _ngram_hash(tokens_np, pos, order):
    if pos < order-1: return -1
    h=0
    for k in range(order-1): h^=int(tokens_np[pos-order+1+k])*NGRAM_PRIMES[k]
    return h & (NGRAM_BUCKETS-1)

def score_with_ngram(mdl, x, y, tables, doc_np, offset):
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        logits=_get_logits(mdl,x)
    lp=F.log_softmax(logits[0].float(),dim=-1); tgts=y[0]
    total=0.0
    for t in range(tgts.size(0)):
        tgt=int(tgts[t].item()); mlp=lp[t]; dp=offset+t+1
        ng_p=None
        for oi in range(len(NGRAM_ORDERS)-1,-1,-1):
            o=NGRAM_ORDERS[oi]; h=_ngram_hash(doc_np,dp,o)
            if h<0: continue
            b=tables[oi].get(h)
            if b is None: continue
            tc=sum(b.values())
            if tc<NGRAM_MIN_COUNT: continue
            ng_p=b.get(tgt,0)/tc; break
        if ng_p is not None:
            ent=-float((torch.exp(mlp)*mlp).sum().item()); ent=max(ent,0.0)
            alpha=ENT_BASE+ENT_RANGE/(1.0+math.exp(-2.0*(ent-4.0)))
            mp=(1-alpha)*math.exp(float(mlp[tgt].item()))+alpha*ng_p
            total+=-math.log(max(mp,1e-10))
        else:
            total+=-float(mlp[tgt].item())
        for oi,o in enumerate(NGRAM_ORDERS):
            h=_ngram_hash(doc_np,dp,o)
            if h<0: continue
            if h not in tables[oi]: tables[oi][h]={}
            tables[oi][h][tgt]=tables[oi][h].get(tgt,0)+1
    return total

class LR(nn.Module):
    def __init__(self, o, r=8):
        super().__init__()
        self.o=o; ind=o.weight.shape[1]; od=o.weight.shape[0]
        self.lA=nn.Parameter(torch.randn(r,ind,device=device)*0.01)
        self.lB=nn.Parameter(torch.randn(od,r,device=device)*0.001)
        self.sc=1.0/r
        for p in self.o.parameters(): p.requires_grad=False
    def forward(self, x):
        b=F.linear(x,self.o.weight.to(x.dtype),self.o.bias.to(x.dtype) if self.o.bias is not None else None)
        return b+(x@self.lA.to(x.dtype).T@self.lB.to(x.dtype).T)*self.sc
    def reset(self):
        nn.init.normal_(self.lA,std=0.01); nn.init.normal_(self.lB,std=0.001)

def make_model():
    mdl = GPT(vocab_size=1024,num_layers=11,model_dim=512,num_heads=8,num_kv_heads=4,
              mlp_mult=3,tie_embeddings=True,tied_embed_init_std=0.02,logit_softcap=30.0,
              rope_base=10000.0,qk_gain_init=1.5).to(device)
    blob=open('final_model.mixed.ptz','rb').read()
    raw=zstandard.ZstdDecompressor().decompress(blob)
    ms=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    mdl.load_state_dict(dequantize_state_dict_mixed(ms),strict=False)
    for p in mdl.parameters(): p.requires_grad=False
    lm=[]
    for blk in mdl.blocks:
        lq=LR(blk.attn.c_q); blk.attn.c_q=lq; lm.append(lq)
        lv=LR(blk.attn.c_v); blk.attn.c_v=lv; lm.append(lv)
    lp_list=[p for m in lm for p in [m.lA,m.lB]]
    return mdl, lm, lp_list

def run_eval(mode, use_ngram):
    mdl, lm, lp_list = make_model()
    ng_tables=[{} for _ in range(len(NGRAM_ORDERS))] if use_ngram else None
    nll=torch.zeros((),device=device,dtype=torch.float64)
    tbs=torch.zeros((),device=device,dtype=torch.float64)
    tks=torch.zeros((),device=device,dtype=torch.float64)

    for d in range(N_DOCS):
        ds=int(bos[d]); de=int(bos[d+1]) if d+1<len(bos) else len(vn)
        dl=min(de-ds,2048)
        if dl<5: continue
        doc=vt[ds:ds+dl].to(device=device,dtype=torch.int64)
        doc_np=doc.cpu().numpy() if use_ngram else None

        for m in lm: m.reset()
        opt=torch.optim.Adam(lp_list,lr=0.05)

        if mode=='full':
            # Train on 1024-token chunks, then score full doc
            mdl.train()
            for cs in range(0,dl-1,1024):
                ce=min(cs+1024,dl-1)
                if ce-cs<2: continue
                tx=doc[cs:ce].unsqueeze(0); ty=doc[cs+1:ce+1].unsqueeze(0)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16): tl=mdl(tx,ty)
                tl.backward(); opt.step(); opt.zero_grad()
            mdl.eval()
            sl=min(2048,dl-1)
            sx=doc[:sl].unsqueeze(0); sy=doc[1:sl+1].unsqueeze(0)
            with torch.no_grad():
                if use_ngram:
                    cn=score_with_ngram(mdl,sx,sy,ng_tables,doc_np,0)
                    nll+=cn
                else:
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16): loss=mdl(sx,sy).detach()
                    nll+=loss.to(torch.float64)*sl
                tks+=sl
                pi=sx.reshape(-1); ti=sy.reshape(-1)
                tb=bbl[ti].to(torch.float64); tb+=(hls[ti]&~ibt[pi]).to(torch.float64); tbs+=tb.sum()
        else:
            # Score-then-train on 256-token chunks
            pl=dl-1
            for cs in range(0,pl,256):
                ce=min(cs+256,pl); cl=ce-cs
                if cl<2: continue
                x=doc[cs:ce].unsqueeze(0); y=doc[cs+1:ce+1].unsqueeze(0)
                il=(ce>=pl)
                mdl.eval()
                with torch.no_grad():
                    if use_ngram:
                        cn=score_with_ngram(mdl,x,y,ng_tables,doc_np,cs)
                        nll+=cn
                    else:
                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16): loss=mdl(x,y).detach()
                        nll+=loss.to(torch.float64)*cl
                    tks+=cl
                    pi=x.reshape(-1); ti=y.reshape(-1)
                    tb=bbl[ti].to(torch.float64); tb+=(hls[ti]&~ibt[pi]).to(torch.float64); tbs+=tb.sum()
                if not il:
                    mdl.train()
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16): loss=mdl(x,y)
                    loss.backward(); opt.step(); opt.zero_grad()

    return (nll.item()/math.log(2))/tbs.item()

print(f'=== N-gram Proof Test ({N_DOCS} docs) ===', flush=True)
t0=time.perf_counter()
r1=run_eval('chunked',False); print(f'1. Chunked, no ngram:    {r1:.4f}  [{time.perf_counter()-t0:.0f}s]',flush=True)
r2=run_eval('chunked',True);  print(f'2. Chunked + ngram:      {r2:.4f}  [{time.perf_counter()-t0:.0f}s]',flush=True)
r3=run_eval('full',False);    print(f'3. Full-ctx, no ngram:   {r3:.4f}  [{time.perf_counter()-t0:.0f}s]',flush=True)
r4=run_eval('full',True);     print(f'4. Full-ctx + ngram:     {r4:.4f}  [{time.perf_counter()-t0:.0f}s]',flush=True)
print(f'\nN-gram effect (chunked): {r2-r1:+.4f}  {"HELPS" if r2<r1 else "HURTS"}')
print(f'N-gram effect (full):    {r4-r3:+.4f}  {"HELPS" if r4<r3 else "HURTS"}')
print(f'Full-ctx improvement:    {r3-r1:+.4f}')
