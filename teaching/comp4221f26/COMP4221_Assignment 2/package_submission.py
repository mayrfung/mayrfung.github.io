"""Collect the required code and evidence; excludes all model weights and recovery files."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoint');p.add_argument('--base',default='results/base')
    p.add_argument('--sft',default='results/sft');p.add_argument('--output',default='submission.zip')
    a=p.parse_args()
    files={}
    for f in sorted(Path('.').glob('*.py')):files[f.name]=f
    for f in ['requirements.txt','raw_faq.jsonl','data/counts.json','data/clean_faq.json']:
        files[f]=Path(f)
    for f in ['loss.csv','run_config.json']:files['experiment/'+f]=Path(a.checkpoint)/f
    for label,folder in [('base',a.base),('sft',a.sft)]:
        for f in ['summary.json','responses.json']:files[f'results/{label}/{f}']=Path(folder)/f
    missing=[str(path) for path in files.values() if not path.is_file()]
    if missing:raise SystemExit('Missing required files: '+', '.join(missing))
    config=json.loads(files['experiment/run_config.json'].read_text())
    if config.get('completed_steps')!=100 or not config.get('final_checkpoint_saved'):
        print('NOTICE: this is not a completed 100-update run. Include an explanation of any incomplete work or course-approved accommodation.')
    output=Path(a.output)
    if output.exists():raise SystemExit('Output ZIP already exists. Choose a new --output name to keep the previous submission.')
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in files.items():z.write(path,name)
    print(f'Wrote {output} ({output.stat().st_size/1024:.1f} KiB). Upload it with your separate report.pdf.')
    print('Review the ZIP and include any additional helper code you wrote. This collector is not a grading check.')


if __name__=='__main__':main()
