# Clarifications and scoring examples

## What counts as completing the assignment

There are five functions, one 100-update training run, two evaluations on the same 20 prompts, and a report of at most 5 pages excluding references. The optional smoke test is for debugging. Part B does not feed into the experiment: the comparison uses the supplied greedy decoder. You do not need another course's notebook, a pretraining run, IFEval, a second dataset, or a hyperparameter sweep.

## Which code may I change

Implement `student_tasks.py`. Keep the supplied data, prompts, encoder, evaluator, model, and experiment settings unchanged. Runtime accommodations must follow course instructions. If you fix a genuine runner problem, submit the modified file and explain the change; do not change the experiment or scoring contract to improve results. Additional grading tests use the documented function contracts, not unstated requirements on arbitrary malformed inputs.

## Normalization and duplicate keys

Normalization preserves case except for using `casefold()` in the duplicate-comparison key. NFKC changes compatibility characters; for example, full-width `Ｆ` becomes `F`. That is consistent with preserving punctuation and case *after the specified normalization*, rather than implementing extra punctuation removal or lowercasing. Leading/trailing whitespace in an ID is removed before validating it. Lowercase `f001` is invalid. IDs with a different valid number create different duplicate keys. Extra record fields are discarded. Do not mutate the input records.

## What if the final assistant answer disappears on truncation

Select the last assistant span **before** truncation. If no part of that span remains, all labels are `-100`, even when an earlier assistant response remains visible. Padding and a real EOS can have the same token ID: a real EOS in the last span is still a target, whereas padding is not. Causal shifting happens in the loss function, not in the feature builder.

## How does smoothing interact with conversation averaging

Use the uniform distribution across **all** vocabulary entries, including the target. Calculate token losses, average them within each nonempty sequence, then average those sequence means. Ignore empty sequences. A sequence of length one has no next-token targets after shifting and therefore contributes nothing. An entirely empty-target batch still needs a scalar zero connected to the logits so `.backward()` works. `epsilon=0` turns off smoothing; it does not turn off conversation balancing.

## Exactly how are responses scored

For illustration only, suppose the expected answer is `3 days.` and the expected source is `F001`. These are scorer examples, not answers to the experiment.

| Generated response (`\n` denotes an actual newline) | Format pass | Grounded-format pass | Reason |
| --- | --- | --- | --- |
| `Answer: 3 days.\nSource: F001` | Yes | Yes | Both schema and expected content match. |
| `Answer: 3 DAYS.\nSource: F001` | Yes | Yes | Answer comparison ignores letter case. |
| `Answer:  3   days.\nSource: F001` | Yes | Yes | Answer comparison collapses whitespace. |
| `Answer: 3 days.\nSource: F002` | Yes | No | A valid-looking source ID is not necessarily the correct source. |
| `Answer: 3 days\nSource: F001` | Yes | No | Missing punctuation changes the exact answer. |
| `Answer: three days.\nSource: F001` | Yes | No | A paraphrase may be semantically valid but fails exact matching. |
| `answer: 3 days.\nSource: F001` | No | No | The label must be exactly `Answer: `. |
| `Answer: 3 days.\nSource:F001` | No | No | The source label needs a space after the colon. |
| `Answer: 3 days.\n\nSource: F001` | No | No | An internal blank line creates a third line. |
| `Answer: 3 days.\nSource: F001\nThanks!` | No | No | Extra commentary is a third line. |

The checker first applies `response.strip()`, so outer whitespace and blank lines are ignored. It then uses `splitlines()`. There must be exactly two lines. The first must start with `Answer: ` and have nonempty answer text. The second must exactly match `Source: F[0-9]{3}`. For grounded-format success, the source ID must equal the expected ID and the answer is compared using `' '.join(answer.casefold().split())`. Punctuation is not removed. The script scores only newly generated text, after decoding with special tokens skipped; it does not score the prompt.

A format failure is always a grounded-format failure. Both rates use the denominator 20, never only the number of format passes. For example, 7 passes means `7/20 = 0.35 = 35%`. Keep the original responses; do not strip unwanted content yourself, replace outputs, or substitute human judgments for the automatic scores. You may discuss where human judgment would differ.

## Is this a held-out factual test

No. Facts intentionally appear in training and are supplied in evaluation prompts. Only the question framing changes. Describe results as adherence to a structured format and use of supplied evidence on this small set. Do not claim unseen-fact generalization or broad assistant competence from these results.

## Will poor scores lose marks

There is no target accuracy, final loss, or required improvement. Marks depend on implementing the functions, completing the prescribed experiment, reporting its actual results, and interpreting the evidence. Incomplete or mismatched experiments still affect their relevant criteria. More runs or higher scores do not earn bonus marks.

## Must all outputs fit in the report

No. Show three paired examples, selecting useful cases with their IDs. Quote relevant portions if an output is long, clearly marking omissions. Keep the complete unedited outputs for both models in the code ZIP. The helper HTML is for inspecting your evidence; it is not another deliverable.
