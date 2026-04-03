"""Build XSA comparison HTML from JSON data."""
import json, html, math, sys

d = json.load(open('dashboard/xsa_comparison.json'))

def esc(s):
    return html.escape(s).replace('\n', ' ')

def color_for_delta(delta):
    if delta > 0.5: return '#065f46'
    if delta > 0.1: return '#14532d'
    if delta < -0.5: return '#7f1d1d'
    if delta < -0.1: return '#531d1d'
    return 'transparent'

def render_doc(doc, max_tokens=200):
    tokens = doc['tokens'][:max_tokens]
    parts = []
    for t in tokens:
        bg = color_for_delta(t['delta'])
        txt = esc(t['text'])
        prev_info = ''
        if t['prev_dist'] > 0:
            prev_info = f'<div class="prev-occurrence">Seen {t["prev_dist"]} tokens back</div>'
        elif t['prev_dist'] == -1:
            prev_info = '<div class="prev-occurrence" style="color:#888">First occurrence</div>'

        delta_class = 'better' if t['delta'] > 0 else ('worse' if t['delta'] < 0 else 'same')
        delta_sign = '+' if t['delta'] > 0 else ''
        p_sub = max(0, min(1, math.exp(t['lp_sub'])))
        p_exp = max(0, min(1, math.exp(t['lp_exp'])))

        parts.append(f'<span class="token" style="background:{bg};">{txt}'
            f'<span class="tooltip">'
            f'<div class="row"><span class="label">Token:</span> <b>{txt}</b></div>'
            f'<div class="row"><span class="label">XSA-last4:</span> {t["lp_sub"]:.3f} nats (p={p_sub:.4f})</div>'
            f'<div class="row"><span class="label">XSA-all:</span> {t["lp_exp"]:.3f} nats (p={p_exp:.4f})</div>'
            f'<div class="row {delta_class}">Delta: {delta_sign}{t["delta"]:.4f} nats</div>'
            f'<div class="row"><span class="label">Entropy:</span> {t["entropy"]:.1f}</div>'
            f'{prev_info}'
            f'</span></span>')
    if len(doc['tokens']) > max_tokens:
        parts.append(f'<span style="color:#555;"> ... [{len(doc["tokens"])-max_tokens} more tokens]</span>')
    return '\n'.join(parts)

def render_improvement(item):
    t = esc(item['target'])
    cb = esc(item['context_before'][-50:])
    ca = esc(item['context_after'][:40])
    delta_sign = '+' if item['delta'] > 0 else ''
    delta_color = '#4ade80' if item['delta'] > 0 else '#f87171'
    border_color = '#065f46' if item['delta'] > 0 else '#7f1d1d'
    target_bg = '#065f46' if item['delta'] > 0 else '#7f1d1d'
    target_color = '#4ade80' if item['delta'] > 0 else '#f87171'

    prev_html = ''
    if item.get('prev_context') and item['prev_dist'] > 0:
        pb = esc(item['prev_context']['before'][-40:])
        pa = esc(item['prev_context']['after'][:40])
        prev_html = (
            f'<div style="color:#888; margin-bottom:4px; font-size:12px;">Earlier in document (position {item["prev_context"]["position"]}):</div>'
            f'<div>...{pb}<span class="imp-prev">{t}</span>{pa}...</div>'
            f'<div class="imp-arrow">&#8595; {item["prev_dist"]} tokens later &#8595;</div>'
        )
    elif item['prev_dist'] == -1:
        prev_html = '<div style="color:#888; font-size:11px; margin-bottom:6px;">Token not seen earlier in document</div>'
    elif item['prev_dist'] > 0:
        prev_html = f'<div style="color:#facc15; font-size:11px; margin-bottom:6px;">Seen {item["prev_dist"]} tokens back</div>'

    p_sub = item['prob_sub']
    p_exp = item['prob_exp']
    bar_sub = min(100, p_sub * 100)
    bar_exp = min(100, p_exp * 100)

    return (
        f'<div class="improvement" style="border-left: 3px solid {border_color};">'
        f'  <div class="imp-header">'
        f'    <div class="imp-delta" style="color:{delta_color};">{delta_sign}{item["delta"]:.3f} nats</div>'
        f'    <div class="imp-meta">Doc {item["doc_id"]} &bull; Position {item["position"]}/{item["doc_len"]} &bull; Entropy: {item["entropy"]:.1f}</div>'
        f'  </div>'
        f'  <div class="imp-context">'
        f'    {prev_html}'
        f'    <div style="color:#888; margin-bottom:4px; font-size:12px;">Prediction point (position {item["position"]}):</div>'
        f'    <div>...{cb}<span class="imp-target" style="background:{target_bg}; color:{target_color};">{t}</span>{ca}...</div>'
        f'  </div>'
        f'  <div class="imp-probs">'
        f'    <div class="model-col">'
        f'      <div class="model-name">XSA-last4: p = {p_sub:.4f}</div>'
        f'      <div class="prob-bar-container"><div class="prob-bar" style="width:{bar_sub}%; background:#f87171;"></div></div>'
        f'    </div>'
        f'    <div class="model-col">'
        f'      <div class="model-name">XSA-all: p = {p_exp:.4f}</div>'
        f'      <div class="prob-bar-container"><div class="prob-bar" style="width:{bar_exp}%; background:#4ade80;"></div></div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )

def bucket_table(buckets, labels):
    rows = ''
    for key, label in labels:
        if key not in buckets: continue
        v = buckets[key]
        delta_color = '#4ade80' if v['avg_delta'] > 0.001 else ('#f87171' if v['avg_delta'] < -0.001 else '#888')
        pct_color = '#4ade80' if v['pct_better'] > 52 else ('#f87171' if v['pct_better'] < 48 else '#888')
        bar_w = min(100, abs(v['avg_delta']) * 5000)
        bar_color = '#065f46' if v['avg_delta'] > 0 else '#7f1d1d'
        rows += (
            f'<tr><td>{label}</td><td>{v["count"]:,}</td>'
            f'<td style="color:{pct_color}">{v["pct_better"]}%</td>'
            f'<td style="color:{delta_color}">{v["avg_delta"]:+.4f}</td>'
            f'<td class="bar-cell"><span class="stat-bar" style="width:{bar_w}px; background:{bar_color};"></span></td></tr>'
        )
    return rows

prev_labels = [('prev_far', 'Token seen before, >200 tokens back'), ('prev_mid', 'Token seen before, 50-200 back'), ('prev_near', 'Token seen before, <50 back'), ('prev_none', 'Token never seen in doc')]
pos_labels = [('early', 'First 256 tokens'), ('mid', 'Tokens 256-1024'), ('late', 'Tokens 1024-2048')]
ent_labels = [('low', 'Low entropy (<2.0)'), ('mid', 'Medium entropy (2.0-4.0)'), ('high', 'High entropy (>4.0)')]

# Pick docs
docs_sorted_best = sorted(d['docs'], key=lambda x: -x['avg_delta'])
docs_sorted_worst = sorted(d['docs'], key=lambda x: x['avg_delta'])
show_docs = [docs_sorted_best[0], docs_sorted_best[4], docs_sorted_worst[0], docs_sorted_worst[4]]

doc_html = ''
for doc in show_docs:
    delta_color = '#4ade80' if doc['avg_delta'] > 0 else '#f87171'
    pct_better = sum(1 for t in doc['tokens'] if t['delta'] > 0) / len(doc['tokens']) * 100
    doc_html += (
        f'<div class="doc">'
        f'  <div class="doc-header">Doc {doc["doc_id"]} &mdash; {doc["doc_len"]} tokens &mdash; '
        f'    avg delta: <span style="color:{delta_color}">{doc["avg_delta"]:+.4f}</span> nats/token &mdash; '
        f'    XSA-all better on {pct_better:.0f}% of tokens</div>'
        f'  <div class="tokens">{render_doc(doc, 200)}</div>'
        f'</div>'
    )

top_better_html = ''.join(render_improvement(x) for x in d['top_better'][:15])
top_worse_html = ''.join(render_improvement(x) for x in d['top_worse'][:10])

CSS = '''
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 30px; max-width: 1100px; margin: auto; }
  h1 { color: #fff; font-size: 22px; }
  h2 { color: #ccc; font-size: 16px; margin-top: 35px; border-bottom: 1px solid #333; padding-bottom: 6px; }
  .summary { background: #16213e; padding: 15px; border-radius: 8px; margin: 15px 0; }
  .summary table { width: 100%; border-collapse: collapse; }
  .summary td, .summary th { padding: 6px 12px; text-align: left; }
  .summary th { color: #888; font-weight: normal; font-size: 13px; }
  .legend { display: flex; gap: 20px; margin: 15px 0; font-size: 13px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 40px; height: 16px; border-radius: 3px; }
  .doc { background: #0f3460; border-radius: 8px; padding: 20px; margin: 20px 0; }
  .doc-header { font-size: 13px; color: #888; margin-bottom: 12px; }
  .tokens { line-height: 2.4; font-family: 'Consolas', monospace; font-size: 14px; }
  .token { padding: 2px 1px; border-radius: 3px; cursor: pointer; position: relative; display: inline; }
  .token:hover { outline: 2px solid #fff; z-index: 5; }
  .tooltip { display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
    background: #1a1a2e; border: 1px solid #555; border-radius: 6px; padding: 12px; z-index: 10;
    white-space: nowrap; font-size: 12px; min-width: 280px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); text-align: left; }
  .token:hover .tooltip { display: block; }
  .tooltip .row { display: flex; justify-content: space-between; margin: 3px 0; }
  .tooltip .label { color: #888; }
  .tooltip .better { color: #4ade80; }
  .tooltip .worse { color: #f87171; }
  .tooltip .same { color: #666; }
  .prev-occurrence { margin-top: 6px; padding-top: 6px; border-top: 1px solid #333; color: #facc15; font-size: 11px; }
  .improvement { background: #16213e; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .imp-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .imp-delta { color: #4ade80; font-weight: bold; font-size: 18px; }
  .imp-meta { color: #888; font-size: 12px; }
  .imp-context { font-family: 'Consolas', monospace; font-size: 13px; line-height: 1.8; background: #0f3460; padding: 12px; border-radius: 6px; }
  .imp-target { padding: 2px 5px; border-radius: 3px; font-weight: bold; }
  .imp-prev { background: #422006; padding: 2px 5px; border-radius: 3px; color: #facc15; }
  .imp-arrow { color: #facc15; font-size: 12px; margin: 6px 0; }
  .imp-probs { display: flex; gap: 20px; margin-top: 10px; font-size: 12px; }
  .imp-probs .model-col { flex: 1; }
  .imp-probs .model-name { color: #888; margin-bottom: 4px; }
  .prob-bar-container { height: 8px; background: #1a1a2e; border-radius: 4px; overflow: hidden; margin-top: 3px; }
  .prob-bar { height: 100%; border-radius: 4px; }
  .stats-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
  .stats-table td, .stats-table th { padding: 8px 12px; text-align: left; font-size: 13px; }
  .stats-table th { color: #888; font-weight: normal; border-bottom: 1px solid #333; }
  .stats-table tr:hover { background: #1a1a3e; }
  .bar-cell { min-width: 120px; }
  .stat-bar { height: 14px; border-radius: 3px; display: inline-block; vertical-align: middle; }
'''

full_html = f'''<!DOCTYPE html>
<html>
<head><style>{CSS}</style></head>
<body>

<h1>XSA Impact Analysis: XSA-last4 vs XSA-all (Real Data)</h1>
<p style="color:#888; font-size: 14px;">Per-token log-probability comparison on {d['n_docs']} validation documents ({d['total_tokens']:,} tokens). Pre-TTT, no LoRA. Hover tokens to see details.</p>

<div class="summary">
  <table>
    <tr><th></th><th>Value</th></tr>
    <tr><td>Total tokens compared</td><td>{d['total_tokens']:,}</td></tr>
    <tr><td>XSA-all better on</td><td>{d['overall_pct_better']}% of tokens</td></tr>
    <tr><td>Average delta</td><td style="color:#4ade80">{d['overall_avg_delta']:+.4f} nats/token</td></tr>
  </table>
</div>

<h2>By Previous Occurrence Distance</h2>
<div class="summary">
  <table class="stats-table">
    <tr><th>Bucket</th><th>Tokens</th><th>XSA-all better</th><th>Avg delta</th><th class="bar-cell"></th></tr>
    {bucket_table(d['buckets_prev_occurrence'], prev_labels)}
  </table>
</div>

<h2>By Position in Document</h2>
<div class="summary">
  <table class="stats-table">
    <tr><th>Bucket</th><th>Tokens</th><th>XSA-all better</th><th>Avg delta</th><th class="bar-cell"></th></tr>
    {bucket_table(d['buckets_position'], pos_labels)}
  </table>
</div>

<h2>By Model Uncertainty (Entropy)</h2>
<div class="summary">
  <table class="stats-table">
    <tr><th>Bucket</th><th>Tokens</th><th>XSA-all better</th><th>Avg delta</th><th class="bar-cell"></th></tr>
    {bucket_table(d['buckets_entropy'], ent_labels)}
  </table>
</div>

<h2>Top 15 Tokens Where XSA-all Helped Most</h2>
{top_better_html}

<h2>Top 10 Tokens Where XSA-last4 Was Better</h2>
{top_worse_html}

<h2>Document Views (hover tokens for details)</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background: #065f46;"></div> XSA-all much better (&gt;0.5 nats)</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #14532d;"></div> Slightly better (0.1-0.5)</div>
  <div class="legend-item"><div class="legend-swatch" style="background: transparent; border: 1px solid #333;"></div> Similar</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #531d1d;"></div> Slightly worse</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #7f1d1d;"></div> XSA-last4 much better</div>
</div>
{doc_html}

</body>
</html>'''

with open('dashboard/xsa_analysis.html', 'w', encoding='utf-8') as f:
    f.write(full_html)
print(f'Written: dashboard/xsa_analysis.html ({len(full_html)//1024} KB)')
