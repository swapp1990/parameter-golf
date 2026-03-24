"""Deep evaluation for short experiments. Upload to pod and run:
    python deep_eval_short.py <checkpoint_path> <experiment_name>
"""
import torch, os, time, glob, math, sys
import torch.nn.functional as F
from pathlib import Path

os.environ['MLP_MULT'] = '3'
sys.path.insert(0, '/runpod-volume/parameter-golf')
os.chdir('/runpod-volume/parameter-golf')
from train_gpt import GPT, load_data_shard


def load_model(checkpoint_path):
    model = GPT(vocab_size=1024, num_layers=9, model_dim=512, num_heads=8, num_kv_heads=4,
                mlp_mult=3, tie_embeddings=True, tied_embed_init_std=0.02,
                logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5)
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


def forward_manual(model, x):
    """Manual forward pass returning logits (no loss). Works with autocast."""
    xemb = model.tok_emb(x)
    if hasattr(model, 'bigram'):
        xemb = xemb + model.bigram(x)
    xemb = F.rms_norm(xemb, (xemb.size(-1),))
    if hasattr(model, 'smear'):
        xemb = model.smear(xemb)
    x0 = xemb
    skips = []
    for i in range(model.num_encoder_layers):
        xemb = model.blocks[i](xemb, x0)
        skips.append(xemb)
    for i in range(model.num_decoder_layers):
        if skips:
            xemb = xemb + model.skip_weights[i].to(dtype=xemb.dtype)[None, None, :] * skips.pop()
        xemb = model.blocks[model.num_encoder_layers + i](xemb, x0)
    xemb = model.final_norm(xemb)
    logits = F.linear(xemb, model.tok_emb.weight)
    logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
    return logits


def deep_eval(model, val_tokens, seq_len=1024, max_windows=500):
    """Loss distribution + position degradation."""
    pos_loss_sum = torch.zeros(seq_len, device='cuda')
    pos_count = 0
    buckets = {'easy(<1)': 0, 'medium(1-3)': 0, 'hard(3-5)': 0, 'very_hard(>5)': 0}
    total_nll = 0.0
    total_tokens = 0
    windows = min(max_windows, (val_tokens.numel() - seq_len) // seq_len)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for w in range(windows):
            start = w * seq_len
            x = val_tokens[start:start + seq_len].unsqueeze(0)
            y = val_tokens[start + 1:start + seq_len + 1].unsqueeze(0)
            logits = forward_manual(model, x)
            per_tok = F.cross_entropy(logits[0].float(), y[0], reduction='none')
            total_nll += per_tok.sum().item()
            total_tokens += seq_len
            pos_loss_sum += per_tok
            pos_count += 1
            for lv in per_tok:
                v = lv.item()
                if v < 1: buckets['easy(<1)'] += 1
                elif v < 3: buckets['medium(1-3)'] += 1
                elif v < 5: buckets['hard(3-5)'] += 1
                else: buckets['very_hard(>5)'] += 1

    avg_loss = total_nll / total_tokens
    avg_bpb = avg_loss / math.log(2) * 0.3584
    pos_avg = (pos_loss_sum / pos_count).cpu().numpy()
    first_64 = float(pos_avg[:64].mean())
    last_64 = float(pos_avg[-64:].mean())

    print(f'val_loss: {avg_loss:.4f}  val_bpb: {avg_bpb:.4f}  ({windows} windows)')
    print()
    print('Loss distribution:')
    total_b = sum(buckets.values())
    for k, v in buckets.items():
        print(f'  {k:15s}: {v/total_b*100:.1f}%')
    print()
    print('Position degradation:')
    print(f'  context_benefit: {first_64 - last_64:.4f}')
    print(f'  first_64_loss:   {first_64:.4f}')
    print(f'  last_64_loss:    {last_64:.4f}')
    print()
    print('Position ranges:')
    for s in range(0, seq_len, 64):
        e = min(s + 64, seq_len)
        print(f'  {s:4d}-{e:4d}: {float(pos_avg[s:e].mean()):.4f}')

    return avg_loss, avg_bpb


def layer_ablation(model, val_tokens, seq_len=1024, max_windows=200):
    """Measure impact of zeroing each layer's output scales."""
    windows = min(max_windows, (val_tokens.numel() - seq_len) // seq_len)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Baseline
        base_nll = 0.0
        for w in range(windows):
            start = w * seq_len
            x = val_tokens[start:start + seq_len].unsqueeze(0)
            y = val_tokens[start + 1:start + seq_len + 1].unsqueeze(0)
            loss = model(x, y)
            base_nll += loss.item() * seq_len
        base_loss = base_nll / (windows * seq_len)

        print(f'Layer ablation (base_loss={base_loss:.4f}, {windows} windows):')
        for layer_idx in range(9):
            saved_a = model.blocks[layer_idx].attn_scale.data.clone()
            saved_m = model.blocks[layer_idx].mlp_scale.data.clone()
            model.blocks[layer_idx].attn_scale.data.zero_()
            model.blocks[layer_idx].mlp_scale.data.zero_()

            abl_nll = 0.0
            for w in range(windows):
                start = w * seq_len
                x = val_tokens[start:start + seq_len].unsqueeze(0)
                y = val_tokens[start + 1:start + seq_len + 1].unsqueeze(0)
                loss = model(x, y)
                abl_nll += loss.item() * seq_len
            abl_loss = abl_nll / (windows * seq_len)
            impact = abl_loss - base_loss

            model.blocks[layer_idx].attn_scale.data = saved_a
            model.blocks[layer_idx].mlp_scale.data = saved_m

            role = 'encoder' if layer_idx < 4 else ('bottleneck' if layer_idx == 4 else 'decoder')
            bar = '#' * min(int(impact * 10), 40)
            print(f'  L{layer_idx} ({role:10s}): impact={impact:+.4f} {bar}')


if __name__ == '__main__':
    checkpoint = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else 'experiment'

    print(f'{"="*60}')
    print(f'Deep Eval: {name}')
    print(f'Checkpoint: {checkpoint}')
    print(f'{"="*60}')

    model = load_model(checkpoint)
    print('Model loaded')

    val_shards = sorted(glob.glob('./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin'))
    val_tokens = torch.cat([load_data_shard(Path(s)).cuda() for s in val_shards]).long()
    print(f'Val tokens: {val_tokens.numel()}')
    print()

    print('--- Eval (500 windows) ---')
    deep_eval(model, val_tokens, max_windows=500)
    print()

    print('--- Layer Ablation (200 windows) ---')
    layer_ablation(model, val_tokens, max_windows=200)
    print()
    print('=== Done ===')
