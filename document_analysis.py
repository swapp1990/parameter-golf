"""
Document-level analysis: find BOS tokens (doc boundaries), measure cold-start penalty.
Runs locally on CPU using Exp 17 checkpoint + val shard.

Output: dashboard/document_analysis_results.json
"""
import torch, os, sys, time, json, math
import numpy as np
import torch.nn.functional as F

# Model setup
os.environ['MLP_MULT'] = '3'
os.environ['NUM_LAYERS'] = '11'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We need model classes but not the full train_gpt.py imports
# Use the layer_xray model definition instead, extended for 11L + SwiGLU + XSA

print('=== Document-Level Analysis (CPU) ===')

# Load val tokens
print('Loading val tokens...')
header_bytes = 256 * np.dtype("<i4").itemsize
val_path = 'data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin'
header = np.fromfile(val_path, dtype="<i4", count=256)
num_tokens = int(header[2])
tokens = np.fromfile(val_path, dtype="<u2", count=num_tokens, offset=header_bytes)
print(f'Val tokens: {len(tokens)}')

# Find document boundaries (BOS token = id 1)
BOS_ID = 1
bos_positions = np.where(tokens == BOS_ID)[0]
print(f'BOS tokens found: {len(bos_positions)}')

if len(bos_positions) == 0:
    print('No BOS tokens found! Cannot do document analysis.')
    sys.exit(1)

# Compute document lengths
doc_lengths = np.diff(bos_positions)
# Last document goes to end of file
doc_lengths = np.append(doc_lengths, len(tokens) - bos_positions[-1])

print(f'Documents: {len(bos_positions)}')
print(f'Avg doc length: {doc_lengths.mean():.0f} tokens')
print(f'Median doc length: {np.median(doc_lengths):.0f} tokens')
print(f'Min/Max: {doc_lengths.min()}/{doc_lengths.max()}')
print(f'< 50 tokens: {(doc_lengths < 50).sum()} ({(doc_lengths < 50).sum()/len(doc_lengths)*100:.1f}%)')
print(f'< 100 tokens: {(doc_lengths < 100).sum()} ({(doc_lengths < 100).sum()/len(doc_lengths)*100:.1f}%)')
print(f'< 500 tokens: {(doc_lengths < 500).sum()} ({(doc_lengths < 500).sum()/len(doc_lengths)*100:.1f}%)')
print(f'> 2048 tokens: {(doc_lengths > 2048).sum()} ({(doc_lengths > 2048).sum()/len(doc_lengths)*100:.1f}%)')

# Document length distribution
bins = [0, 50, 100, 200, 500, 1000, 2048, 5000, 10000, 50000, max(doc_lengths.max()+1, 50001)]
hist, _ = np.histogram(doc_lengths, bins=bins)
print('\nDocument length distribution:')
for i in range(len(hist)):
    pct = hist[i] / len(doc_lengths) * 100
    bar = '#' * int(pct)
    print(f'  {bins[i]:>6d}-{bins[i+1]:>6d}: {hist[i]:>5d} ({pct:.1f}%) {bar}')

# Now we need per-token losses to measure cold-start
# Load the bits_budget_results which has per-window loss data
# But those are 2048-window losses, not aligned to documents

# Instead, let's compute what fraction of tokens are "cold start"
# (within first N tokens of a document)
print('\n=== Cold Start Analysis ===')
cold_start_sizes = [10, 25, 50, 100, 200]
for cs in cold_start_sizes:
    cold_tokens = 0
    for doc_start in bos_positions:
        doc_end = min(doc_start + cs, len(tokens))
        cold_tokens += (doc_end - doc_start)
    pct = cold_tokens / len(tokens) * 100
    print(f'  First {cs:>3d} tokens of each doc: {cold_tokens:>8d} tokens ({pct:.1f}% of all)')

# How many 2048-windows contain a document boundary?
print('\n=== Document Boundaries in 2048-windows ===')
n_windows = len(tokens) // 2048
windows_with_boundary = 0
boundaries_per_window = []
for w in range(n_windows):
    start = w * 2048
    end = start + 2048
    n_bos = np.sum((bos_positions >= start) & (bos_positions < end))
    if n_bos > 0:
        windows_with_boundary += 1
    boundaries_per_window.append(n_bos)

boundaries_per_window = np.array(boundaries_per_window)
print(f'Total 2048-windows: {n_windows}')
print(f'Windows with >=1 doc boundary: {windows_with_boundary} ({windows_with_boundary/n_windows*100:.1f}%)')
print(f'Avg boundaries per window: {boundaries_per_window.mean():.2f}')
print(f'Max boundaries in one window: {boundaries_per_window.max()}')

# Save results
results = {
    'total_tokens': int(len(tokens)),
    'num_documents': int(len(bos_positions)),
    'doc_length_stats': {
        'mean': float(doc_lengths.mean()),
        'median': float(np.median(doc_lengths)),
        'min': int(doc_lengths.min()),
        'max': int(doc_lengths.max()),
        'std': float(doc_lengths.std()),
    },
    'doc_length_distribution': {
        f'{bins[i]}-{bins[i+1]}': int(hist[i]) for i in range(len(hist))
    },
    'cold_start_token_pct': {
        str(cs): float(sum(min(cs, dl) for dl in doc_lengths) / len(tokens) * 100)
        for cs in cold_start_sizes
    },
    'windows_2048': {
        'total': int(n_windows),
        'with_boundary': int(windows_with_boundary),
        'pct_with_boundary': float(windows_with_boundary / n_windows * 100),
        'avg_boundaries_per_window': float(boundaries_per_window.mean()),
    },
    'short_doc_pct': {
        'under_50': float((doc_lengths < 50).sum() / len(doc_lengths) * 100),
        'under_100': float((doc_lengths < 100).sum() / len(doc_lengths) * 100),
        'under_500': float((doc_lengths < 500).sum() / len(doc_lengths) * 100),
    },
}

out_path = 'dashboard/document_analysis_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out_path}')
print('=== Done ===')
