"""Provided fixed-prompt greedy evaluation. Use unchanged for both models."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from faq_pipeline import encode_messages, check_response
from runtime_utils import file_hash, load_data, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    data = load_data('data/eval.json', 20)
    output = Path(args.output)
    if (output / 'summary.json').exists():
        parser.error('This output already has a completed evaluation. Choose a new --output folder.')
    torch.manual_seed(4221)
    device, hardware = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    output.mkdir(exist_ok=True, parents=True)
    results = []
    with torch.inference_mode():
        for index, row in enumerate(data, 1):
            ids, _ = encode_messages(tokenizer, row['messages'], append_assistant_header=True)
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     max_new_tokens=96, do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0, len(ids):], skip_special_tokens=True)
            scores = check_response(response, row['expected_answer'], row['expected_source'])
            results.append(dict(id=row['id'], response=response, **scores))
            write_json(output / 'responses.partial.json', results)
            print(f'prompt {index}/20: {row["id"]}', flush=True)
    summary = dict(model=args.model, prompts=20, max_new_tokens=96, do_sample=False,
                   device=device, hardware=hardware, evaluation_data_sha256=file_hash('data/eval.json'))
    for key in ['format_pass', 'grounded_pass']:
        count = sum(r[key] for r in results)
        summary[key + '_count'] = count
        summary[key + '_rate'] = count / 20
    write_json(output / 'responses.json', results)
    write_json(output / 'summary.json', summary)
    (output / 'responses.partial.json').unlink(missing_ok=True)
    print(f'Saved {output}/summary.json and responses.json', flush=True)
    print(f'Format: {summary["format_pass_count"]}/20; grounded format: {summary["grounded_pass_count"]}/20')


if __name__ == '__main__':
    main()
