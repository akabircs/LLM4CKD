# LLM4CKD

This repository is for **LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening**.
It implements the experimental pipeline described in the paper:

1. harmonize tabular CKD features;
2. serialize patient records as list-style or text-style prompts;
3. run zero-shot and few-shot LLM inference;
4. train low-data ML baselines with the same train/test splits;
5. compute balanced accuracy, AUROC, macro-F1, sensitivity, Brier loss, and expected calibration error;
6. run paired permutation tests on per-sample Brier loss;
7. generate paper-style result tables and plots.

The package does **not** include the study datasets or proprietary API keys. Exact numeric reproduction requires the same Dataset-1 CSV, Dataset-2/UCI CKD CSV, preprocessing choices, and model versions used by the authors.

## What this code reproduces

The paper reports:

- Dataset-1: a Bangladeshi community cohort with 284 complete records, 112 early-stage CKD cases and 172 non-CKD controls.
- Dataset-2: the UCI CKD dataset with 400 records, 250 CKD and 150 non-CKD records.
- Selected-feature prompting based on clinically meaningful predictors including hypertension, age, urinary RBC, sleep duration, anemia, diabetes, obesity/BMI, family history of hypertension, and gender.
- LLMs: Gemma-2-9B, Llama-3-8B, Qwen-3-8B, Mistral-7B, and GPT-4o-mini.
- Seeds: `0, 1, 32, 42, 1024`.
- Low-data settings: `4, 8, 16, 32` in-context examples or training samples.

## Repository layout

```text
config/default.yaml              Main experiment configuration
data/README.md                   Dataset placement and expected schema
scripts/run_ml_baselines.py      Train/evaluate scikit-learn baselines
scripts/run_llm_zero_few_shot.py Generate prompts or run LLM inference
scripts/evaluate_predictions.py  Summarize prediction CSVs into metrics
scripts/feature_ranking.py       SHAP feature-ranking analysis for ML baseline
scripts/plot_results.py          Paper-style figures from prediction outputs
src/llm4ckd/                     Reusable package modules
tests/test_serialization.py      Minimal sanity tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional packages:

- `xgboost` and `lightgbm` for XGB/LGB baselines.
- `tabpfn` for TabPFN.
- `transformers`, `accelerate`, and `torch` for local open-weight LLM inference.
- `openai` for GPT-4o-mini inference.
- `shap` for feature-ranking analysis.

## Data placement

Place the datasets as CSV files:

```text
data/dataset1.csv
data/dataset2.csv
```

The code uses flexible column-name aliases, but the safest option is to harmonize your CSV columns to the canonical names listed in `data/README.md`.

The target column should be one of:

```text
ckd, CKD, class, classification, target, label, y
```

Target encoding should be binary: `1/ckd/yes/true` for CKD and `0/notckd/no/false/non-ckd` for non-CKD.

## Quick start: ML baselines

```bash
python scripts/run_ml_baselines.py \
  --dataset dataset1 \
  --feature-set selected \
  --config config/default.yaml \
  --output outputs/predictions/ml_dataset1_selected.csv
```

Summarize metrics:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/predictions/ml_dataset1_selected.csv \
  --output outputs/tables/ml_dataset1_selected_metrics.csv
```

## Quick start: LLM prompts only

The following command creates prompts without calling any model. This is useful for auditing serialization and few-shot example selection.

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset1 \
  --feature-set selected \
  --prompt-style instruction \
  --serialization list \
  --shots 4 \
  --backend prompt_only \
  --output outputs/predictions/prompts_dataset1_selected.csv
```

## Local Hugging Face LLM inference

Example for Qwen-3-8B:

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset1 \
  --feature-set selected \
  --prompt-style instruction \
  --serialization list \
  --shots 4 \
  --backend hf \
  --model-id Qwen/Qwen3-8B \
  --output outputs/predictions/qwen_dataset1_4shot.csv
```

Open-weight models tested in the paper:

```text
google/gemma-2-9b-it
meta-llama/Meta-Llama-3-8B-Instruct
Qwen/Qwen3-8B
mistralai/Mistral-7B-Instruct-v0.3
```

## OpenAI inference

Set your key:

```bash
export OPENAI_API_KEY="..."
```

Run:

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset1 \
  --feature-set selected \
  --prompt-style chat \
  --serialization text \
  --shots 4 \
  --backend openai \
  --model-id gpt-4o-mini \
  --output outputs/predictions/gpt4omini_dataset1_4shot.csv
```

## Feature ranking

```bash
python scripts/feature_ranking.py \
  --dataset dataset1 \
  --feature-set selected \
  --model mlp \
  --output outputs/tables/mlp_shap_feature_ranking.csv
```

## Recreating plots

After prediction CSVs are generated:

```bash
python scripts/plot_results.py \
  --predictions outputs/predictions/ml_dataset1_selected.csv \
  --figure outputs/figures/ml_balanced_accuracy.png
```

## Important reproducibility notes

- The paper evaluates LLM predictions using token-level log-probabilities over labels `0` and `1`. The implementation here normalizes the two label likelihoods into a CKD probability.
- Different model checkpoints, tokenizer revisions, quantization, GPU kernels, and API versions can change results.
- The paper excludes direct diagnostic leakage variables such as eGFR/uACR in Dataset-1 and serum creatinine in Dataset-2. Keep those columns out of the feature set.
- For few-shot LLM and ML/DL baselines, the same stratified 80/20 split and same sampled training examples should be used per seed.

