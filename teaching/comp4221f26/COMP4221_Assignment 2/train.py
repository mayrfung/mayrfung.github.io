"""Provided trainer. The graded run is 100 updates; smaller runs are for debugging."""
import argparse
import csv
import random
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from student_tasks import final_response_features, conversation_balanced_loss
from faq_pipeline import encode_messages
from runtime_utils import MODEL_ID, file_hash, load_data, select_device, write_json


def write_losses(output, losses):
    with (output / 'loss.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['optimizer_step', 'loss'])
        writer.writerows(losses)


def save_state(output, model, optimizer, config, losses, elapsed):
    # Atomic replacement keeps the previous recovery file usable if writing is interrupted.
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                 losses=losses, elapsed=elapsed, python_rng=random.getstate(),
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temp = output / 'training_state.tmp'
    torch.save(state, temp)
    temp.replace(output / 'training_state.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--steps', type=int, default=100, help='Total updates, not additional updates when resuming')
    parser.add_argument('--output', default='checkpoint')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resume', action='store_true', help='Resume the latest complete recovery save in --output')
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error('steps must be between 1 and 100')
    examples = load_data('data/train.json', 300)
    output = Path(args.output)
    if args.resume and not (output / 'training_state.pt').is_file():
        parser.error('No recovery file found in --output. Start a new run in a new folder.')
    if not args.resume and output.exists() and any(output.iterdir()):
        parser.error('Output folder is not empty. Use --resume for the same run or choose a new --output folder.')
    seed = 4221
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device, hardware = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise SystemExit('The required tokenizer must provide an EOS token.')
    tokenizer.pad_token = tokenizer.eos_token
    features = []
    for example in examples:
        ids, spans = encode_messages(tokenizer, example['messages'])
        feature = final_response_features(ids, spans, 256, tokenizer.pad_token_id)
        if (feature['labels'][1:] != -100).any():
            features.append(feature)
    if len(features) < 8:
        raise SystemExit('Too few supervised examples. Run python checks.py --part C1 and check truncation.')
    print(f'Usable training examples: {len(features)}/{len(examples)}', flush=True)
    config = dict(model=args.model, seed=seed, device=device, hardware=hardware,
                  steps=args.steps, completed_steps=0, raw_training_examples=len(examples),
                  usable_training_examples=len(features), sequence_length=256,
                  effective_batch_size=8, micro_batch_size=1, learning_rate=2e-5,
                  epsilon=0.1, torch_version=str(torch.__version__),
                  transformers_version=transformers.__version__,
                  training_data_sha256=file_hash('data/train.json'),
                  student_tasks_sha256=file_hash(Path(__file__).with_name('student_tasks.py')))
    state = None
    losses = []
    previous_elapsed = 0.0
    if args.resume:
        state = torch.load(output / 'training_state.pt', map_location='cpu', weights_only=True)
        for key in ['model', 'seed', 'sequence_length', 'effective_batch_size', 'learning_rate',
                    'epsilon', 'training_data_sha256', 'student_tasks_sha256']:
            if state['config'][key] != config[key]:
                raise SystemExit(f'Cannot resume after changing {key}. Use a new output folder for a fresh run.')
        if state['config']['device'] != device:
            raise SystemExit('Resume on the same device type (CPU or CUDA), or start a fresh run.')
        losses = state['losses']
        previous_elapsed = state['elapsed']
        if args.steps < len(losses):
            parser.error(f'Recovery save already has {len(losses)} updates; --steps must be at least that total.')
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    if state is not None:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
        print(f'Resuming after update {len(losses)}; target total = {args.steps}', flush=True)
    output.mkdir(exist_ok=True, parents=True)
    write_losses(output, losses)
    config['completed_steps'] = len(losses)
    write_json(output / 'run_config.json', config)
    model.train()
    start = time.perf_counter()
    for step in range(len(losses), args.steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for index in random.sample(range(len(features)), 8):
            example = features[index]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in example.items() if k != 'labels'}
            labels = example['labels'].unsqueeze(0).to(device)
            loss = conversation_balanced_loss(model(**inputs).logits, labels, epsilon=.1) / 8
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss. Check Part C2 before starting a fresh run.')
            loss.backward()
            total += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append([step + 1, total])
        write_losses(output, losses)
        config['completed_steps'] = step + 1
        config['elapsed_training_seconds'] = previous_elapsed + time.perf_counter() - start
        write_json(output / 'run_config.json', config)
        print(f'update {step + 1}/{args.steps} loss={total:.6f}', flush=True)
        if (step + 1) % 25 == 0 or step + 1 == args.steps:
            save_state(output, model, optimizer, config, losses, config['elapsed_training_seconds'])
            print(f'Recovery saved after update {step + 1}', flush=True)
    model.config.use_cache = True
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    config['final_checkpoint_saved'] = True
    write_json(output / 'run_config.json', config)
    print('Saved final checkpoint, run_config.json and loss.csv', flush=True)


if __name__ == '__main__':
    main()
