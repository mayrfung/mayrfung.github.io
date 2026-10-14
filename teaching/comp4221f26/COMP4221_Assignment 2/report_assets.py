"""Create a loss plot and comparison table from your saved outputs; no analysis is written for you."""
import argparse
import csv
import html
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoint');p.add_argument('--base',default='results/base')
    p.add_argument('--sft',default='results/sft');p.add_argument('--output',default='report_assets')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (Path(a.checkpoint)/'loss.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise SystemExit('loss.csv has no updates.')
    xs=[int(r['optimizer_step']) for r in rows];ys=[float(r['loss']) for r in rows]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained')
    ax.plot(xs,ys,color='#214b7a',linewidth=1.6)
    ax.set(xlabel='Optimizer update',ylabel='Mean smoothed loss',title='Training loss per optimizer update')
    ax.grid(alpha=.2)
    if len(xs)>1:ax.set_xlim(min(xs),max(xs))
    fig.savefig(out/'loss.png',dpi=180)
    fig.savefig(out/'loss.svg')
    plt.close(fig)
    base_path=Path(a.base);sft_path=Path(a.sft)
    base=json.loads((base_path/'responses.json').read_text());sft=json.loads((sft_path/'responses.json').read_text())
    summaries=[json.loads((x/'summary.json').read_text()) for x in [base_path,sft_path]]
    if len(base)!=20 or [r['id'] for r in base]!=[r['id'] for r in sft]:
        raise SystemExit('Expected matching 20-prompt outputs for both models.')
    if summaries[0]['evaluation_data_sha256']!=summaries[1]['evaluation_data_sha256']:
        raise SystemExit('Evaluations used different data. Regenerate a matched comparison.')
    parts=['<!doctype html><meta charset="utf-8"><title>Your saved experiment outputs</title>',
           '<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:12px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
           '<h1>Your saved experiment outputs</h1><p>Use this page to inspect evidence. Write your own interpretation and select three paired examples for the report.</p>',
           '<img src="loss.png" alt="Training loss versus optimizer update"><h2>Metrics</h2><table><tr><th>Model</th><th>Format</th><th>Grounded format</th></tr>']
    for name,s in zip(['Base','Fine tuned'],summaries):
        parts.append(f'<tr><td>{name}</td><td>{s["format_pass_count"]}/20 ({s["format_pass_rate"]:.0%})</td><td>{s["grounded_pass_count"]}/20 ({s["grounded_pass_rate"]:.0%})</td></tr>')
    parts.append('</table><h2>All paired responses</h2>')
    for b,s in zip(base,sft):
        parts.append(f'<h3>{html.escape(b["id"])}</h3><table><tr><th>Base</th><th>Fine tuned</th></tr><tr>')
        for row in [b,s]:
            parts.append(f'<td><pre>{html.escape(row["response"])}</pre><p>Format: {row["format_pass"]}; grounded format: {row["grounded_pass"]}</p></td>')
        parts.append('</tr></table>')
    (out/'comparison.html').write_text('\n'.join(parts),encoding='utf-8')
    print(f'Open {out}/comparison.html in a browser. Loss plots: {out}/loss.png and loss.svg.')


if __name__=='__main__':main()
