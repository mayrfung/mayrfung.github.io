# Report checklist

Write your own report.pdf, at most 5 pages excluding references. No extra experiment is implied by this checklist.

- Identity: name, student ID, UST email, and any disclosure required by course policy.
- A: raw/valid/unique counts, two actual normalization examples, one conflicting-answer risk and one proposed detector (no detector code).
- B: two requested top-p distributions and zero-based supports; crossing-token/variable-support explanation; checker result.
- C1: one small two-response trace showing IDs, spans, labels, and attention before truncation and when the final response is removed; context-versus-target explanation; checker result.
- C2: formula, smoothing meaning, unequal-length arithmetic example comparing conversation and token averages; checker result.
- D1: actual model/hardware/100 completed updates/usable examples/settings; loss plot; role/content/EOS/header explanation. Use run_config.json and loss.csv.
- D2: table with base and SFT counts out of 20 and rates for both metrics; three identified paired raw outputs or clearly marked excerpts; explanations tied to the exact checks.
- D3: why loss and metric changes can differ; one concrete limitation and one proposed follow-up (do not run it).
- Evidence: report numbers match the submitted JSON and CSV, and both evaluations use identical prompt IDs and data hash.
- Upload: report.pdf plus the code ZIP produced by package_submission.py; inspect the ZIP contents before uploading.

Suggested layout: A+B about 1 page; C about 1 page; D settings and loss about 1 page; D metrics, examples, and interpretation about 1–2 pages. You need not fill all 5 pages.
