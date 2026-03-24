"""
Generate markdown report from checkpoint analysis JSON.
Usage: python generate_report.py dashboard/analysis_step2000.json
"""
import json, sys, os

if len(sys.argv) < 2:
    print("Usage: python generate_report.py <analysis.json>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    d = json.load(f)

name = d['name']
out_path = sys.argv[1].replace('.json', '_report.md')

lines = []
def w(s=''):
    lines.append(s)

w(f'# Checkpoint Analysis: {name}')
w()
w('| Metric | Value |')
w('|--------|-------|')
w(f'| Parameters | {d["params"]:,} |')
w(f'| Layers | {d["num_layers"]} |')
w(f'| Seq Length | {d["seq_len"]} |')
w(f'| Val Tokens Analyzed | {d.get("val_tokens", "?")} |')
w()

# Section 2: Loss Distribution
w('## Loss Distribution')
w()
w('| Category | % of Tokens | Avg Loss |')
w('|----------|------------|----------|')
for k, v in d['loss_distribution'].items():
    bar = '\u2588' * int(v['pct'] / 2)
    w(f'| {k} | {v["pct"]:.1f}% {bar} | {v["avg_loss"]:.2f} |')
w()

# Section 3: Hard Token Breakdown
w('## Hard Token Breakdown (loss 3-5)')
w()
ht = d['hard_token_breakdown']
total = ht['total_hard']
w(f'Total hard tokens: {total}')
w()
w('| Sub-category | Count | % of Hard |')
w('|-------------|-------|-----------|')
w(f'| Word-initial letters | {ht["word_initial_1char"]} | {ht["word_initial_1char"]/max(total,1)*100:.1f}% |')
w(f'| Function words | {ht["function_words"]} | {ht["function_words"]/max(total,1)*100:.1f}% |')
w(f'| After period | {ht["after_period"]} | {ht["after_period"]/max(total,1)*100:.1f}% |')
w(f'| After "the" | {ht["after_the"]} | {ht["after_the"]/max(total,1)*100:.1f}% |')
w()

# Section 4: Position Analysis
w('## Position Analysis')
w()
pos = d['position']
w(f'- First 64 tokens avg loss: **{pos["first_64"]:.4f}**')
w(f'- Last 64 tokens avg loss: **{pos["last_64"]:.4f}**')
w(f'- Context benefit: **{pos["context_benefit"]:.4f}**')
w()
w('| Range | Avg Loss |')
w('|-------|----------|')
for k, v in sorted(pos['ranges'].items(), key=lambda x: int(x[0].split('-')[0])):
    w(f'| {k} | {v:.4f} |')
w()

# Section 5: Document Analysis
w('## Document Analysis')
w()
doc = d['documents']
w(f'- Documents: {doc["count"]}')
w(f'- Mean length: {doc["mean_length"]:.0f} tokens')
w(f'- Median length: {doc["median_length"]:.0f} tokens')
w()

# Section 6: Layer Ablation
w('## Layer Ablation')
w()
la = d['layer_ablation']
w(f'Base loss: {la["base_loss"]:.4f}')
w()
w('| Layer | Impact | |')
w('|-------|--------|---|')
for li, impact in sorted(la['impacts'].items(), key=lambda x: int(x[0])):
    bar = '#' * min(int(float(impact) * 10), 40)
    role = 'encoder' if int(li) < d['num_layers'] // 2 else ('bottleneck' if int(li) == d['num_layers'] // 2 else 'decoder')
    w(f'| L{li} ({role}) | {float(impact):+.4f} | `{bar}` |')
w()

# Section 7: Head Ablation
w('## Head Ablation')
w()
ha = d['head_ablation']
# Find notable heads (impact > 0.01)
notable = [(k, v) for k, v in ha.items() if abs(v) > 0.01]
if notable:
    w(f'Notable heads (impact > 0.01): {len(notable)}')
    w()
    w('| Head | Impact |')
    w('|------|--------|')
    for k, v in sorted(notable, key=lambda x: -abs(x[1]))[:20]:
        w(f'| {k} | {v:+.4f} |')
else:
    w('No heads with impact > 0.01. All heads contribute equally.')
w()

# Section 9: Component Ablation
w('## Component Ablation')
w()
if 'component_ablation' in d:
    ca = d['component_ablation']
    if 'smeargate_removal' in ca:
        w(f'- SmearGate removal: **{ca["smeargate_removal"]:+.4f}**')
w()
w('### MLP Ablation (per layer)')
w()
w('| Layer | MLP Impact |')
w('|-------|-----------|')
for li, impact in sorted(d['mlp_ablation'].items(), key=lambda x: int(x[0])):
    w(f'| L{li} | {float(impact):+.4f} |')
w()

# Section 10: Entropy Quadrants
w('## Entropy Quadrants')
w()
eq = d['entropy_quadrants']
w('```')
w(f'  Confident Right:  {eq["confident_right"]:.1f}%   Uncertain Right: {eq["uncertain_right"]:.1f}%')
w(f'  Confident Wrong:  {eq["confident_wrong"]:.1f}%   Uncertain Wrong: {eq["uncertain_wrong"]:.1f}%')
w('```')
w()

# Section 11: Juncture Analysis
w('## Juncture Analysis')
w()
ja = d['juncture']
w(f'- After juncture: **{ja["after_juncture_avg"]:.3f}** ({ja["after_juncture_pct"]:.1f}% of tokens)')
w(f'- Not after juncture: **{ja["not_after_juncture_avg"]:.3f}**')
w(f'- Word-initial: **{ja["word_initial_avg"]:.3f}**')
w(f'- Not word-initial: **{ja["not_word_initial_avg"]:.3f}**')
w()

# Section 11: Top Bigrams
w('## Most Expensive Bigrams')
w()
w('| Prev | Cur | Total Cost | Count | Avg |')
w('|------|-----|-----------|-------|-----|')
for b in d['top_bigrams'][:15]:
    w(f'| {b["prev"]} | {b["cur"]} | {b["total_cost"]:.0f} | {b["count"]} | {b["avg"]:.2f} |')
w()

# Top Costly Tokens
w('## Most Expensive Tokens')
w()
w('| Token | Total Cost | Count | Avg |')
w('|-------|-----------|-------|-----|')
for t in d['top_costly_tokens'][:10]:
    w(f'| {t["token"]} | {t["total_cost"]:.0f} | {t["count"]} | {t["avg"]:.2f} |')

report = '\n'.join(lines)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f'Report saved: {out_path}')
