"""Provided runtime helpers. Paths are relative to the current working folder."""
import hashlib
import json
from pathlib import Path
import torch

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path, expected_count):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f'Missing {path}. Run from the extracted assignment folder and run python prepare_data.py first.')
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise SystemExit(f'{path} must contain {expected_count} records. Check Part A and regenerate the data.')
    return rows


def select_device(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA was requested but is unavailable. Check your GPU runtime/PyTorch installation, or use --device cpu.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    if device == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f'Device: {device} ({name}); torch {torch.__version__}', flush=True)
    return device, name


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
