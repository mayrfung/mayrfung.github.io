# COMP4221 Programming Assignment

- **Instructor:** Yi R. (May) Fung
- **Term:** Fall 2026
- **Total:** 100 marks
- **Due date:** Nov 7th

Build a small assistant that answers questions from supplied facts and returns each answer with its source ID, such as `F001`.

You will:

- Clean structured FAQ records.
- Implement nucleus probability filtering.
- Select training targets and compute a smoothed loss.
- Run one short fine-tuning experiment and compare the results.

**Note:** The campus facts are fictional teaching examples, not HKUST policy. All data and runners are included in this package.

## What you will complete

| Part | Task | Marks |
| --- | --- | ---: |
| A | Normalize and validate FAQ records | 15 |
| B | Implement nucleus probability filtering | 20 |
| C | Final-response masking and balanced smoothed loss | 30 |
| D | One fine-tuning run and matched evaluation | 35 |

- **Your code:** Implement the **five TODO functions** in `student_tasks.py`.
- **Provided code:** The data builder, encoder, training loop, evaluator, plotting helper, and submission collector are ready to use.
  
## Start here

### Set up and run your first check

- **Read first:** `START_HERE.md` covers installation, the full command sequence, recovery, and troubleshooting.
- **Python:** Use version 3.10–3.12.
- **Hardware:** A CUDA GPU is recommended for the experiment. CPU is also supported.
- **Model download:** Local function checks do not require one.
- **Working folder:** Run commands from the extracted folder containing `student_tasks.py` and `raw_faq.jsonl`.

```bash
python -m pip install -r requirements.txt
python preflight.py
# Implement Part A, then check just that part
python checks.py --part A
python prepare_data.py
```

### Check your progress

- You can check **B, C1, and C2 separately** while other TODOs remain unfinished.
- After completing all five functions, run `python checks.py`.
- An untouched starter is expected to report TODOs. This does **not** indicate an installation failure.

### Keep the experiment consistent

- Keep the supplied data, prompt builder, encoder, scoring rules, and experiment settings fixed.
- Implement your own solution for this assignment.
- Any reuse or AI assistance must follow the published COMP4221 course policy.

## Part A: Normalize and validate FAQ records

**15 marks**

- **Input file:** `raw_faq.jsonl` contains 120 records.
- **Required fields:** Each accepted record must have string fields `id`, `question`, and `answer`.
- **Paths:** All paths below are relative to the extracted assignment folder.

### A1: Implement two functions (10 marks)

**Function 1: `normalize_text(text)`**

The input is a string. Apply these steps **in order**:

1. Decode HTML entities with `html.unescape`.
2. Apply Unicode NFKC normalization.
3. Collapse every whitespace run to one ASCII space and trim the ends.

**Important details:**

- Do not lowercase the text or remove punctuation.
- NFKC may change compatibility characters, including full-width punctuation. These changes are required.

**Function 2: `deduplicate_records(records)`**

1. Reject non-dictionary entries and entries with missing or non-string required fields.
2. Normalize all three required fields.
3. Reject a record if its normalized ID does not exactly match uppercase `F` followed by three ASCII digits, or if its normalized question or answer is empty.
4. Deduplicate using the key `(normalized id, normalized question.casefold())`.
5. Keep the **first normalized record** for each valid key and preserve input order.
6. Return dictionaries containing only `id`, `question`, and `answer`. Ignore extra fields.

**Important details:**

- Use `casefold` only for the comparison key. Preserve the normalized question’s original case in the output.
- Different IDs are different keys, even when their questions match.

### A2: Report and interpret (5 marks)

**Run:** `python prepare_data.py`

**Expected output from a correct implementation:**

- 100 retained facts.
- 300 training conversations.
- 20 evaluation prompts.

**Include in your report:**

- The raw count, valid count before deduplication, and unique count from `data/counts.json`.
- Two actual before/after normalization examples, with record IDs.
- One risk when duplicate questions have conflicting answers.
- One way to detect that conflict. **No conflict-detector implementation is required.**

## Part B: Nucleus probability filtering

**20 marks**

**Function:** `nucleus_distribution(logits, temperature, top_p)`

- **Input logits:** Finite, nonempty, one-dimensional floating-point values.
- **Return value:** A probability vector in the original vocabulary order, **not a sampled token ID**.

### B1: Implement filtering (15 marks)

**Steps:**

1. Divide logits by `temperature` and compute a stable softmax.
2. Sort the probabilities in descending order.
3. Retain the smallest prefix whose cumulative probability is **greater than or equal to `top_p`**, including the crossing token.
4. Set the remaining probabilities to zero.
5. Renormalize the retained probabilities and restore the original vocabulary order.

**Edge cases and requirements:**

- At `top_p=1`, return the full softmax distribution.
- Keep at least one token.
- Reject a nonfinite or nonpositive `temperature` using `ValueError` or `TypeError`.
- Reject a nonfinite `top_p` or a value outside `(0,1]` using `ValueError` or `TypeError`.
- Any consistent ordering of ties is acceptable.
- Preserve device and shape. Any floating output dtype is allowed.

### B2: Work through an example (5 marks)

Use logits `ln([0.50,0.30,0.15,0.05])` and temperature 1.

**Include in your report:**

- The zero-based retained indices for `top_p=0.70` and `top_p=0.90`.
- The four-element output vector for each threshold, rounded to three decimals.
- An explanation of crossing-token inclusion and why the retained count is not fixed.
- Your B checker result.

**Threshold equality:** If the cumulative probability exactly equals the threshold, the prefix already reaches it. No extra token is required.

## Part C: Final response supervision and smoothed loss

**30 marks**

The provided encoder returns token IDs and ordered, valid, nonoverlapping assistant spans.

- A span `[start,end)` covers positions `start` through `end−1`.
- It **includes** that response’s EOS token and **excludes** its role header.
- You do not need to infer spans or change a chat template.

### C1: Supervise the last original response (15 marks)

**Function:** `final_response_features(input_ids, assistant_spans, max_length, pad_token_id)`

**Inputs and return value:**

- `input_ids` and `assistant_spans` are Python lists of integer token IDs and span pairs, respectively.
- `max_length` is a positive integer.
- Return a dictionary with `input_ids`, `labels`, and `attention_mask`.
- Each returned value must be a one-dimensional `torch.long` tensor of length `max_length`.

**Steps:**

1. Select the **last assistant span in the original conversation**.
2. Copy input IDs into labels only at positions in this span. Set every other label to `-100`.
3. Right-truncate all arrays consistently, then right-pad to `max_length`.

**Masking and padding rules:**

- Earlier assistant turns, system and user messages, and all role headers remain input context but are not supervised targets.
- Real tokens have attention 1.
- Padding uses the supplied pad ID, attention 0, and label `-100`.
- Identify padding by **position**, even when pad and EOS have the same ID.
- If there is no assistant span, all labels are `-100`.
- If the last original response is wholly truncated, all labels are `-100`. **Never fall back to an earlier response.**

**Include in your report:**

- A small example containing two assistant spans.
- Input IDs, spans, labels, and attention for the example **without truncation**.
- The same information with truncation that **removes the final response**.
- An explanation of the difference between useful context and supervised targets.

**Trainer behavior:** The trainer skips examples with no valid shifted targets.

### C2: Conversation-balanced label smoothing (15 marks)

**Function:** `conversation_balanced_loss(logits, labels, epsilon=0.1)`

**Inputs:**

- `logits` has shape `[batch, length, vocabulary]`.
- `labels` is a `torch.long` tensor with shape `[batch, length]` on the same device as `logits`.
- Labels are valid vocabulary indices or `-100`.
- Inputs are finite. Batch, length, and vocabulary sizes are positive.

**Steps:**

1. Align `logits[:, :-1, :]` with `labels[:, 1:]` for next-token prediction.
2. Ignore targets equal to `-100`.
3. Compute stable log-softmax over the vocabulary to obtain `log_p`.
4. For each valid target `y`, compute:

   ```text
   (1−epsilon) × (−log_p[y]) + epsilon × mean_v(−log_p[v])
   ```

5. Average valid token losses **within each sequence**.
6. Average those sequence means over **nonempty sequences**.

**Edge cases and requirements:**

- The uniform smoothing term includes the correct token.
- Exclude sequences with no valid shifted targets.
- If no valid shifted targets remain, including when `length=1`, return a **differentiable scalar zero** on the logits device.
- Preserve gradients.
- Reject `epsilon` outside `[0,1]` or nonfinite `epsilon`.
- Derive the objective using tensor operations and log-softmax.
- Do **not** call `CrossEntropyLoss` or `functional.cross_entropy` in your implementation. The public checks may use them as an independent oracle.
- Never gather a vocabulary element using label `-100`.

### Explain and verify

**Include in your report:**

- The loss formula and the meaning of uniform smoothing.
- An arithmetic example with two conversations of unequal target lengths, showing how conversation averaging differs from a global valid-token mean.
- Your C1 and C2 checker results.

**Grading scope:** Public checks cover representative cases. Grading may use other inputs within these documented contracts.

## Part D: Fine-tuning and grounded-format evaluation

**35 marks**

- **Model:** `HuggingFaceTB/SmolLM2-135M`.
- **Data:** Use the prepared FAQ data, with three training phrasings per fact.
- **Evaluation:** The 20 evaluation prompts use a different question framing. Their facts appear in both the training data and the evaluation prompts.
- **What this tests:** Responses to supplied evidence, not knowledge of unseen facts.

### D1: Complete one run (15 marks)

**Run:**

```bash
python train.py --steps 100 --output checkpoint
```

**Required run:**

- Complete **100 optimizer updates** and use the **final checkpoint**.
- Each update accumulates gradients from eight one-conversation micro-batches.
- An update is not one example or one epoch.
- Smaller `--steps` values are for debugging. They do not count as a completed graded experiment.
- Do not tune the settings or select an intermediate checkpoint.

**Fixed settings:**

| Setting | Value |
| --- | --- |
| Seed | 4221 |
| Sequence length | 256 |
| Learning rate | 2e-5 |
| Epsilon | 0.1 |
| Effective batch size | 8 |

**Include in your report:**

- Actual hardware, completed updates, usable examples after truncation, and initialization from `checkpoint/run_config.json`.
- Batch, sequence-length, learning-rate, and smoothing settings from the same file.
- A plot of `checkpoint/loss.csv`.
- An explanation of how the encoder and your C1 mask preserve earlier dialogue while supervising the final response.

**Recovery and resource issues:**

- The runner prints progress and saves recovery state every 25 updates.
- `START_HERE.md` explains how to resume to a total of 100 updates.
- If resources block completion, contact the course staff early and preserve logs.
- Any alternative run length must follow a course-approved accommodation.

### D2: Compare base and final models (15 marks)

**Run:**

```bash
python evaluate.py --model HuggingFaceTB/SmolLM2-135M --output results/base
python evaluate.py --model checkpoint --output results/sft
python report_assets.py
```

**Use identical evaluation settings for both models:**

- The same 20 prompts.
- The provided encoding.
- Greedy decoding.
- A 96-new-token cap.

**Outputs:**

- Each evaluation folder receives `summary.json` and `responses.json`.
- The helper builds a loss plot and a browser-readable comparison.
- Write your **own interpretation** of the results.

**Format pass:** After trimming outer whitespace, a response must contain exactly two lines:

- The first starts exactly with `Answer:`, followed by a space and nonempty text.
- The second matches `Source:`, followed by a space and an ID such as `F001`.

**Format failures:**

- Internal extra lines, wrong label case, or commentary fail.
- Outer whitespace alone is ignored.

**Grounded-format pass:** A response must pass the format check and also contain:

- The correct source ID.
- The expected answer after casefolding and whitespace collapse.

**Scoring details:**

- Punctuation is significant, so a paraphrase can fail.
- These are classroom checks, not IFEval. `FAQ.md` gives exact examples.
- Do not edit, postprocess, or hand-correct generated responses to raise scores.

**Include in your report:**

- Format-pass and grounded-format-pass counts **out of 20**, and their rates, for both models.
- Three paired raw responses, with their evaluation IDs.
- An explanation of why each response passes or fails and what changed between models.
- For long outputs, quote the relevant lines and label any omission. Keep the complete response in the ZIP.

### D3: Interpret limitations (5 marks)

**Include in your report:**

- Why lower training loss need not improve the two discrete rates.
- One evaluation limitation.
- One proposed follow-up. You do not need to run it.

## Submission and completion checklist

### Prepare the report

- **Submit to Canvas:** `report.pdf` and one code ZIP.
- **Presentation:** Do not paste full code or all 20 outputs into the report - concise tables and figures, along with elaboration/explanation, are sufficient. 

### Collect code and experiment evidence

**Run:**

```bash
python checks.py
python package_submission.py
```

**What the collector does:**

- Creates `submission.zip`.
- Maps the two small files from `checkpoint/` into `experiment/`.
- Excludes model weights, `training_state.pt`, model caches, and temporary outputs.

**What you must do:**

- Keep your checkpoint locally until grading is complete.
- Upload `report.pdf` separately.
- Include any additional code dependencies you introduced.

**Required ZIP contents:**

- `student_tasks.py`.
- All supplied Python runners and any helper code you changed or added.
- `requirements.txt` and `raw_faq.jsonl`.
- `data/counts.json` and `data/clean_faq.json`.
- `experiment/run_config.json` and `experiment/loss.csv`.
- `results/base/summary.json` and `results/base/responses.json`.
- `results/sft/summary.json` and `results/sft/responses.json`.

**Additional notes:**

- Training and evaluation JSON files can be regenerated with `prepare_data.py` and need not be uploaded.
- If you used different output folder names, pass `--checkpoint`, `--base`, and `--sft` to the collector and plotting helper.
- Neither helper completes TODOs or validates correctness.

### Course policy and questions

- Work individually and follow the published course policy. Briefly disclose any permitted AI assistance.
