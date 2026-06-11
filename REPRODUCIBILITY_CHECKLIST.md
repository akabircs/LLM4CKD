# Reproducibility checklist

Use this checklist before submitting code with the paper.

- [ ] Dataset-1 and Dataset-2 paths are documented.
- [ ] Data access restrictions are stated.
- [ ] Target-label encoding is documented.
- [ ] Leakage variables are excluded: eGFR/uACR for Dataset-1 and serum creatinine for Dataset-2.
- [ ] Selected-feature set is documented.
- [ ] Random seeds are fixed: 0, 1, 32, 42, 1024.
- [ ] Stratified 80/20 splits are used for few-shot and supervised baselines.
- [ ] Low-data settings are 4, 8, 16, 32.
- [ ] Prompt style and serialization style are recorded for every LLM run.
- [ ] Exact LLM model IDs, revisions, quantization settings, and GPU are recorded.
- [ ] API model version and date are recorded for GPT-4o-mini.
- [ ] Output-token log-probability extraction is logged.
- [ ] Raw predictions are saved before aggregation.
- [ ] Metrics include balanced accuracy, AUROC, macro-F1, sensitivity, Brier loss, and ECE.
- [ ] Paired statistical tests use per-sample Brier loss differences.
