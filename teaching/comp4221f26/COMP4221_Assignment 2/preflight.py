"""Optional environment check; does not download a model or call any TODO."""
import importlib
import platform
import shutil
import sys
from pathlib import Path


def main():
    print('Python:', sys.version.split()[0], '| platform:', platform.platform())
    if not (3,10) <= sys.version_info[:2] <= (3,12):
        print('Recommended course environment: Python 3.10-3.12; another version may need troubleshooting.')
    failed=False
    for name,wanted in [('torch','2.8.0'),('transformers','4.57.1'),('matplotlib','3.10.8')]:
        try:module=importlib.import_module(name)
        except ImportError:
            print(f'MISSING {name}: run python -m pip install -r requirements.txt');failed=True;continue
        version=module.__version__
        print(f'{name}: {version} (package target {wanted})')
        if version.split('+')[0]!=wanted:print('  Version differs: use the supplied versions before reporting a reproducibility issue.')
        if name=='torch':
            print('CUDA GPU:', module.cuda.get_device_name(0) if module.cuda.is_available() else 'not available; CPU is supported')
    if not failed:
        try:
            from transformers import AutoTokenizer, LlamaForCausalLM
        except Exception as exc:
            print(f'MODEL IMPORT FAILED: {type(exc).__name__}: {exc}')
            print('Use a clean environment with requirements.txt or the course environment; installed optional packages can conflict.')
            failed=True
    free=shutil.disk_usage('.').free/1024**3
    print(f'Free disk space here: {free:.1f} GiB. Allow about 8 GiB for model cache, recovery saves and final weights.')
    if free<8:print('  Free more space or choose an output folder on a larger disk before training.')
    if not Path('raw_faq.jsonl').is_file():
        print('WRONG FOLDER: change into the extracted assignment folder before running the commands.');failed=True
    if failed:raise SystemExit(1)
    print('Environment imports passed. Next: implement a part and run python checks.py --part A (or B/C1/C2).')


if __name__=='__main__':main()
