"""
Test full-model SGD TTT locally on CPU. 3 docs.
Verifies: save weights → SGD step → score → restore → loss improves.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import copy

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dashboard.layer_xray import GPT

SEQ_LEN = 512
BOS_ID = 1

print('=== SGD TTT Local Test ===')

print('Loading model...')
model = GPT()
sd = torch.load('dashboard/checkpoints/model_step_2000.pt', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Params: {sum(p.numel() for p in model.parameters())}')

print('Loading val tokens...')
header_bytes = 256 * np.dtype("<i4").itemsize
tokens_np = np.fromfile('data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin',
                        dtype="<u2", offset=header_bytes)
val_tokens = torch.from_numpy(tokens_np.astype(np.int64))
print(f'Val tokens: {val_tokens.numel()}')

bos_pos = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
print(f'Documents: {len(bos_pos)}')


def forward_logits(mdl, x):
    result = mdl.forward_full(x)
    return result["logits"]


def score_doc(mdl, doc):
    """Score a doc, return mean loss."""
    slen = min(SEQ_LEN, doc.numel() - 1)
    x = doc[:slen].unsqueeze(0)
    y = doc[1:slen + 1].unsqueeze(0)
    with torch.no_grad():
        logits = forward_logits(mdl, x)
        return F.cross_entropy(logits[0].float(), y[0]).item()


# Save original weights once
print('\nSaving original weights...')
original_sd = {k: v.clone() for k, v in model.state_dict().items()}


def restore_weights(mdl, orig_sd):
    mdl.load_state_dict(orig_sd, strict=True)


print('\n--- Testing 3 documents ---')
for d in range(3):
    ds = int(bos_pos[d])
    de = int(bos_pos[d + 1]) if d + 1 < len(bos_pos) else val_tokens.numel()
    doc = val_tokens[ds:de]
    print(f'\nDoc {d}: length={doc.numel()} tokens')

    # Score without TTT
    restore_weights(model, original_sd)
    model.eval()
    no_ttt_loss = score_doc(model, doc)
    print(f'  No TTT:  loss={no_ttt_loss:.4f}')

    # SGD TTT: one gradient step on the document
    restore_weights(model, original_sd)
    model.train()
    t0 = time.time()

    slen = min(SEQ_LEN, doc.numel() - 1)
    x = doc[:slen].unsqueeze(0)
    y = doc[1:slen + 1].unsqueeze(0)
    logits = forward_logits(model, x)
    loss = F.cross_entropy(logits.reshape(-1, 1024).float(), y.reshape(-1))
    loss.backward()

    # SGD update (no optimizer needed — just subtract lr * grad)
    lr = 3e-4
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
    model.zero_grad()

    adapt_time = time.time() - t0

    # Score after TTT
    model.eval()
    ttt_loss = score_doc(model, doc)
    delta = ttt_loss - no_ttt_loss
    print(f'  SGD TTT: loss={ttt_loss:.4f}  delta={delta:+.4f}  time={adapt_time:.2f}s')
    print(f'  {"PASS" if delta < 0 else "FAIL"}: TTT {"improved" if delta < 0 else "worsened"} loss')

# Timing estimate
print(f'\n--- Timing estimate ---')
# Average adapt time per doc on CPU
avg_cpu_time = adapt_time  # last doc's time
gpu_speedup = 50  # H100 is roughly 50x faster than CPU for this
est_gpu_time = avg_cpu_time / gpu_speedup
print(f'CPU time per doc: ~{avg_cpu_time:.2f}s')
print(f'Est GPU time per doc: ~{est_gpu_time:.4f}s')
print(f'Est 50K docs on 1xH100: ~{est_gpu_time * 50000:.0f}s ({est_gpu_time * 50000 / 60:.0f} min)')
print(f'Est 50K docs on 8xH100: ~{est_gpu_time * 50000 / 8:.0f}s ({est_gpu_time * 50000 / 8 / 60:.0f} min)')

print('\n=== Done ===')
