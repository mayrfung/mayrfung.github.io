# Start here

Read the assignment handout for what is assessed. This guide explains how to run the supplied files. Run every command from your copy of `COMP4221_Assignment` folder, where `student_tasks.py` and `raw_faq.jsonl` are located. 
## 1 Set up once

Use Python 3.10–3.12 in an isolated environment. Python 3.11 is a suitable default. On Linux or macOS, open a terminal in the assignment folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python preflight.py
```

On Windows, create the environment with `py -3.11 -m venv .venv`, activate it with `.venv\Scripts\activate` in Command Prompt, and use the same `python` commands above. These activation commands are platform-specific; do not paste them into Python's `>>>` prompt.

Use the supplied package versions before requesting help with reproducibility. The default PyTorch installation and CUDA availability depend on the machine. A course-managed GPU environment with the matching PyTorch build is preferable when available. If your course provides an installation recipe, follow it, then run `preflight.py`. No paid API or API key is used. The initial public model/tokenizer download needs internet; the dataset is already included.

A CUDA GPU is recommended, but CPU training and evaluation are supported. Runners print the selected device. Use `--device cuda` to require a CUDA GPU (and get a clear error if unavailable), or `--device cpu` to force CPU. On a Mac, the supplied runner uses CPU; MPS is not a required route. Leave about 8 GiB free for model cache, recovery files, and final weights, in addition to the Python environment. Runtime and memory use depend on your machine; start early.

If using a notebook or Colab, select the desired runtime first, and change the notebook working directory with `%cd /path/to/COMP4221_Assignment`. Use `%pip install -r requirements.txt` and restart the Python runtime if installation requests it. Shell commands such as `python checks.py` need a leading `!` in notebook cells. Re-enter the assignment folder after a runtime restart. Back up work and recovery files to persistent storage before the session ends; notebook local storage can disappear.

## 2 Implement and check one part at a time

Edit only the five TODO functions in `student_tasks.py`. Their arguments and return values are specified in the handout and docstrings. Standard-library modules, PyTorch tensor operations, `softmax`, and `log_softmax` are allowed. Part C2 must explicitly form the loss; it cannot call a cross-entropy API.

```bash
python checks.py --part A
python prepare_data.py
python checks.py --part B
python checks.py --part C1
python checks.py --part C2
python checks.py
```

Run each check after implementing that part. An untouched starter reports `TODO not implemented yet`; this is expected. A traceback for a failed assertion identifies the check to investigate. The public checks are examples of the published contracts, not the complete grading tests. Checks run on CPU and do not download a model. The `scorer: PASS` line checks the provided scorer examples only; it does not mean your Part D experiment is complete. You can save their terminal output as evidence; a screenshot is optional.

`prepare_data.py` writes `data/clean_faq.json`, `data/counts.json`, `data/train.json`, and `data/eval.json`. It stops if Part A does not retain the expected 100 facts. Correct preparation produces 300 training conversations and 20 evaluation prompts. Regenerate after changing Part A. Never fix generated data manually to conceal a Part A bug.

## 3 Run the experiment

After all function checks pass, this optional short run catches integration or resource problems:

```bash
python train.py --steps 2 --output smoke_checkpoint
```

It uses the real required model and downloads it on first use. This two-update run is only a debugging check. It is not the graded experiment, and its checkpoint is not used in the comparison. After verifying that it completes, you may delete only the `smoke_checkpoint` folder to recover disk space.

Run the required experiment once:

```bash
python train.py --steps 100 --output checkpoint
```

One optimizer update processes eight micro-batches of one conversation each. The script prints update numbers and writes `checkpoint/loss.csv` and `checkpoint/run_config.json`. It saves a recovery file every 25 completed updates and at the end. The final model/tokenizer are saved in `checkpoint/` when training finishes. No fixed final loss or score is expected.

If interrupted after a recovery save, keep the folder and run:

```bash
python train.py --steps 100 --output checkpoint --resume
```

Here `100` means **100 total updates**, not 100 more. Work after the latest recovery save may be repeated; the loss CSV is restored to match that save before continuing. If interrupted before the first save, start fresh in a new output folder. Resume on the same device type with the same code and data. After changing your implementation or data, start a fresh run in a new output folder. A run that fails halfway is incomplete, not evidence of poor model performance; save its logs and seek help early if blocked.

The runner refuses to overwrite a nonempty output folder unless you explicitly resume. If you use another folder name, pass it consistently to evaluation, plotting, and submission commands.

## 4 Evaluate and inspect

```bash
python evaluate.py --model HuggingFaceTB/SmolLM2-135M --output results/base
python evaluate.py --model checkpoint --output results/sft
python report_assets.py
```

Each evaluator prints progress through 20 prompts and saves `summary.json` and `responses.json`. While running, a `responses.partial.json` file records progress; it is not a completed evaluation. If interrupted, rerun the same command to restart those 20 prompts. A completed output folder is protected from accidental overwrite; use a new folder if a rerun is needed after correcting a bug.

Open `report_assets/comparison.html` in a browser to see your loss plot, metric table, and all paired responses. `report_assets/loss.png` and `report_assets/loss.svg` are also available separately. Insert the PNG directly into your report; a readable screenshot is also acceptable. The helper does not write your explanations or choose your three examples. Do not submit the entire HTML page as your report.

The two evaluations must use the same generated `data/eval.json`. Their summaries include its SHA-256 digest; the reporting helper detects a mismatch. Scores can stay flat or fall after fine-tuning without losing score-based marks: there are no score-based marks. Explain the observed outputs honestly.

## 5 Submit

Write `report.pdf` using `REPORT_CHECKLIST.md`, then run:

```bash
python checks.py
python package_submission.py
```

Upload **both** `report.pdf` and `submission.zip` to Canvas. The ZIP contains code and small evidence files, not weights. The collector maps `checkpoint/run_config.json` and `checkpoint/loss.csv` to `experiment/` inside the ZIP. You do not need to copy those files manually. If you used custom output paths:

```bash
python report_assets.py --checkpoint my_run --base results/base2 --sft results/sft2
python package_submission.py --checkpoint my_run --base results/base2 --sft results/sft2 --output submission_v2.zip
```

The collector warns about incomplete runs but still allows an honest partial submission. That warning does not grant an accommodation or full experimental credit. Review the ZIP and include any extra helper code you wrote; the collector includes root-level Python files only. If missing outputs prevent the collector from running, manually ZIP the work you do have and explain the missing evidence in your report; do not fabricate files.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `No module named torch` or `transformers` | Activate the environment used for installation; use `python -m pip`, not a different Python's pip. |
| TODO message | Implement the named function; use `--part` to check completed parts independently. |
| Missing raw data or `data/train.json` | Check the working folder and run Part A preparation first. |
| CUDA requested but unavailable | Check the runtime/GPU PyTorch build, or remove `--device cuda` to use CPU. |
| CUDA out of memory | End other processes using your GPU or restart the runtime. Keep the prescribed batch/length settings. CPU is a supported alternative; contact staff if resources remain a blocker. |
| Model download or connection error | Check internet access and retry. This is not a TODO failure. The required model is public. |
| Output folder already exists | Resume the same training run or choose a new folder; do not accidentally evaluate an old checkpoint. |
| Very low or zero evaluation scores | Inspect raw outputs and the exact checker. This can be a legitimate result; do not change the scorer or keep tuning for a higher score. |
| A paraphrase is marked wrong | Expected: grounded-format success uses normalized exact answers, not semantic similarity. Discuss the limitation. |
| Training loss is not zero | Expected: language-model uncertainty and label smoothing both affect the objective. No loss threshold is required. |

When asking for help, include the command, the error text, `preflight.py` output, the relevant checker output, and whether you are on CPU or CUDA. Do not post credentials or another student's code.
