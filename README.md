# LLM4CKD: Large Language Models for Early-Stage Chronic Kidney Disease Screening

This repository contains the code for **LLM4CKD**, a framework for evaluating large language models for low-resource chronic kidney disease screening using structured tabular clinical features.

The code supports:

- Zero-shot and few-shot LLM-based CKD screening
- Instruction-style and chat-style prompt construction
- List-style and text-style tabular feature serialization
- All-feature and selected-feature settings
- Conventional machine-learning baselines
- Tabular deep-learning and foundation-model baselines
- Per-sample prediction export for downstream statistical analysis
- Reproducibility utilities for metrics, plotting, and feature-ranking analysis

---

## Datasets

The study uses two datasets.

### Dataset-1: Private Bangladesh Cohort

Dataset-1 is a private community-based Bangladeshi cohort used for the main early-stage CKD screening experiments.

This repository does **not** include Dataset-1 because it contains private participant-level data.

Expected local path:

```text
data/dataset1.csv
```

The file is ignored by Git and should not be committed.

### [Dataset-2](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease): Public UCI CKD Dataset

[Dataset-2](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) is the public UCI Chronic Kidney Disease dataset used for independent/cross-dataset evaluation.

To download Dataset-2 locally:

```bash
python -m pip install ucimlrepo pandas
python scripts/download_dataset2_uci.py
```

On Windows Command Prompt:

```cmd
python -m pip install ucimlrepo pandas
python scripts\download_dataset2_uci.py
```

This creates:

```text
data/dataset2.csv
```

The generated CSV is ignored by Git so that the repository remains code-focused and does not redistribute local data files.

---

## Repository Structure

```text
LLM4CKD/
├── config/
│   └── default.yaml
├── data/
│   └── README.md
├── scripts/
│   ├── download_dataset2_uci.py
│   ├── run_llm_zero_few_shot.py
│   ├── run_ml_baselines.py
│   └── run_dl_baselines.py
├── src/
│   └── llm4ckd/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── prompts.py
│       ├── llm_inference.py
│       ├── ml_baselines.py
│       ├── dl_baselines.py
│       ├── node.py
│       ├── saint.py
│       ├── metrics.py
│       ├── splits.py
│       └── stats.py
├── tests/
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── REPRODUCIBILITY_CHECKLIST.md
└── README.md
```

---

## Installation

### Option 1: Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Option 2: Conda environment

```bash
conda env create -f environment.yml
conda activate llm4ckd
```

---

## Configuration

The main configuration file is:

```text
config/default.yaml
```

It defines:

- Dataset paths
- Random seeds
- Train/test split settings
- All-feature and selected-feature configurations
- Leakage-variable exclusion rules
- LLM model identifiers
- ML and DL training sizes

The default repeated-seed setting is:

```text
0, 1, 32, 42, 1024
```

The low-data training sizes are:

```text
4, 8, 16, 32
```

---

## Feature Harmonization and Serialization

Feature harmonization is implemented mainly in:

```text
src/llm4ckd/data.py
src/llm4ckd/features.py
config/default.yaml
```

The code performs:

- Column-name harmonization
- Dataset-specific feature selection
- Leakage-variable exclusion
- Missing-value handling
- List-style serialization
- Text-style sentence serialization

Important serialization rule:

```text
Missing value  -> omit the feature from the prompt
False / 0 / No -> write an explicit negative statement
True / 1 / Yes -> write an explicit positive statement
```

For example, if `anemia` is missing, no anemia-related sentence is added. If `anemia = 0`, the prompt may include:

```text
The patient does not have anemia.
```

If `anemia = 1`, the prompt may include:

```text
The patient has anemia.
```

---

## Pretrained LLMs

The LLM experiments use five instruction-tuned models:

- [Gemma-2-9B](https://huggingface.co/google/gemma-2-9b-it)
- [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Qwen-3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [GPT-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini)

For open-weight Hugging Face models, pretrained weights are loaded in:

```text
src/llm4ckd/llm_inference.py
```

The main weight-loading call is:

```python
AutoModelForCausalLM.from_pretrained(model_id, ...)
```

For GPT-4o-mini, no local model weights are loaded. The model is accessed through the OpenAI API.

---

## Running LLM Experiments

### Prompt-only dry run

Use `prompt_only` to verify data loading, feature serialization, and prompt construction without loading an actual LLM:

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset2 \
  --feature-set selected \
  --prompt-style instruction \
  --serialization list \
  --shots 0 \
  --backend prompt_only \
  --output outputs/predictions/dataset2_prompt_only.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_llm_zero_few_shot.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --prompt-style instruction ^
  --serialization list ^
  --shots 0 ^
  --backend prompt_only ^
  --output outputs\predictions\dataset2_prompt_only.csv
```

### Hugging Face LLM inference

Example using Qwen-3-8B:

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset2 \
  --feature-set selected \
  --prompt-style instruction \
  --serialization list \
  --shots 0 \
  --backend hf \
  --model-id Qwen/Qwen3-8B \
  --output outputs/predictions/dataset2_qwen3_zero_shot.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_llm_zero_few_shot.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --prompt-style instruction ^
  --serialization list ^
  --shots 0 ^
  --backend hf ^
  --model-id Qwen/Qwen3-8B ^
  --output outputs\predictions\dataset2_qwen3_zero_shot.csv
```

### OpenAI GPT-4o-mini inference

Set your OpenAI API key first:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

On Windows Command Prompt:

```cmd
set OPENAI_API_KEY=your_api_key_here
```

Then run:

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset2 \
  --feature-set selected \
  --prompt-style chat \
  --serialization text \
  --shots 0 \
  --backend openai \
  --model-id gpt-4o-mini \
  --output outputs/predictions/dataset2_gpt4omini_zero_shot.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_llm_zero_few_shot.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --prompt-style chat ^
  --serialization text ^
  --shots 0 ^
  --backend openai ^
  --model-id gpt-4o-mini ^
  --output outputs\predictions\dataset2_gpt4omini_zero_shot.csv
```

---

## Conventional ML Baselines

Conventional ML baselines are implemented in:

```text
src/llm4ckd/ml_baselines.py
scripts/run_ml_baselines.py
```

The ML baselines include:

```text
RF, GB, ET, LR, AB, DT, XGB, MLP, LGB
```

To run ML baselines on Dataset-2:

```bash
python scripts/run_ml_baselines.py \
  --dataset dataset2 \
  --feature-set selected \
  --output outputs/predictions/dataset2_ml_selected.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_ml_baselines.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --output outputs\predictions\dataset2_ml_selected.csv
```

---

## Tabular DL and Foundation Baselines

Tabular deep-learning and foundation-model baselines are implemented separately from conventional ML baselines.

The relevant files are:

```text
src/llm4ckd/dl_baselines.py
src/llm4ckd/node.py
src/llm4ckd/saint.py
scripts/run_dl_baselines.py
```

The DL/foundation baselines are:

- [TabPFN](https://github.com/PriorLabs/TabPFN)
- [TabNet](https://github.com/dreamquark-ai/tabnet)
- [NODE](https://github.com/Qwicen/node)
- [SAINT](https://github.com/somepago/saint)

This separation follows the experimental grouping in the paper:

```text
ML baselines: RF, GB, ET, LR, AB, DT, XGB, MLP, LGB
DL/foundation baselines: TabPFN, TabNet, NODE, SAINT
```

The DL baseline implementation is organized as follows:

```text
src/llm4ckd/dl_baselines.py  -> shared preprocessing, TabPFN, TabNet, and DL model registry
src/llm4ckd/node.py          -> official Qwicen/node ODST-based NODE wrapper
src/llm4ckd/saint.py         -> official somepago/saint TabAttention-based SAINT wrapper
```

NODE and SAINT are not implemented through generic tabular-DL approximations. They use their official repositories when those repositories are available locally.

### Installing optional DL dependencies

The DL/foundation baselines require optional packages that are not installed by the default lightweight setup.

For CPU execution:

```bash
python -m pip install torch pytorch-tabnet tabpfn
```

For GPU execution, install the PyTorch build matching your CUDA version from the official PyTorch installation guide, then install:

```bash
python -m pip install pytorch-tabnet tabpfn
```

NODE and SAINT additionally require local clones of their official repositories:

```bash
mkdir -p external
git clone https://github.com/Qwicen/node.git external/node
git clone https://github.com/somepago/saint.git external/saint
```

On Windows Command Prompt:

```cmd
mkdir external
git clone https://github.com/Qwicen/node.git external\node
git clone https://github.com/somepago/saint.git external\saint
```

If either official repository has its own environment or package requirements, install those requirements in the same Python environment before running the corresponding baseline.

### Running DL/foundation baselines

To run TabPFN, TabNet, NODE, and SAINT on Dataset-2 with selected features:

```bash
python scripts/run_dl_baselines.py \
  --dataset dataset2 \
  --feature-set selected \
  --training-sizes 4 8 16 32 \
  --device cpu \
  --output outputs/predictions/dataset2_dl_selected.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_dl_baselines.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --training-sizes 4 8 16 32 ^
  --device cpu ^
  --output outputs\predictions\dataset2_dl_selected.csv
```

To explicitly provide the official NODE and SAINT repository paths:

```bash
python scripts/run_dl_baselines.py \
  --dataset dataset2 \
  --feature-set selected \
  --training-sizes 4 8 16 32 \
  --node-repo-dir external/node \
  --saint-repo-dir external/saint \
  --device cpu \
  --output outputs/predictions/dataset2_dl_selected.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_dl_baselines.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --training-sizes 4 8 16 32 ^
  --node-repo-dir external\node ^
  --saint-repo-dir external\saint ^
  --device cpu ^
  --output outputs\predictions\dataset2_dl_selected.csv
```

To run only NODE and SAINT:

```bash
python scripts/run_dl_baselines.py \
  --dataset dataset2 \
  --feature-set selected \
  --training-sizes 4 8 16 32 \
  --models NODE SAINT \
  --node-repo-dir external/node \
  --saint-repo-dir external/saint \
  --device cpu \
  --output outputs/predictions/dataset2_node_saint_selected.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_dl_baselines.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --training-sizes 4 8 16 32 ^
  --models NODE SAINT ^
  --node-repo-dir external\node ^
  --saint-repo-dir external\saint ^
  --device cpu ^
  --output outputs\predictions\dataset2_node_saint_selected.csv
```

For GPU execution, replace:

```text
--device cpu
```

with:

```text
--device cuda
```

### Notes on TabPFN

TabPFN is a tabular foundation model. In this repository it is grouped with DL/foundation baselines rather than conventional ML baselines.

TabPFN is usually easy to run through the Python package:

```bash
python -m pip install tabpfn
```

If GPU memory is limited, run it with:

```text
--device cpu
```

### Notes on TabNet

TabNet is handled through `pytorch-tabnet`.

Install it with:

```bash
python -m pip install pytorch-tabnet
```

TabNet can run on CPU or GPU depending on the installed PyTorch environment.

### Notes on NODE

NODE is implemented through an official [Qwicen/node](https://github.com/Qwicen/node) wrapper in:

```text
src/llm4ckd/node.py
```

The repository wrapper imports the official ODST layer from the local NODE clone and trains a compact NODE classifier for low-data settings. The official NODE repository should be cloned locally before running NODE:

```bash
git clone https://github.com/Qwicen/node.git external/node
```

On Windows Command Prompt:

```cmd
git clone https://github.com/Qwicen/node.git external\node
```

The script searches for the NODE repository in the following order:

```text
--node-repo-dir argument
NODE_REPO_DIR environment variable
external/node
node
node_official
~/node
~/node_official
```

NODE can be memory-sensitive, especially with larger feature spaces, deeper trees, or larger batches. If NODE fails because of memory constraints:

1. Try CPU execution.
2. Reduce the NODE batch size or model size in `src/llm4ckd/node.py`.
3. Reduce `num_trees`, `depth`, or `num_layers`.
4. Run only NODE first using `--models NODE`.

### Notes on SAINT

SAINT is implemented through an official [somepago/saint](https://github.com/somepago/saint) wrapper in:

```text
src/llm4ckd/saint.py
```

The repository wrapper imports the official `TabAttention` class from the local SAINT clone and follows the official SAINT forward contract using continuous-feature encodings. In this pipeline, categorical variables are first converted to dense one-hot numeric features by the shared DL preprocessor, so SAINT receives all features as continuous inputs.

The official SAINT repository should be cloned locally before running SAINT:

```bash
git clone https://github.com/somepago/saint.git external/saint
```

On Windows Command Prompt:

```cmd
git clone https://github.com/somepago/saint.git external\saint
```

The script searches for the SAINT repository in the following order:

```text
--saint-repo-dir argument
SAINT_REPO_DIR environment variable
external/saint
saint
~/saint
```

For exact SAINT environment reproduction, follow the official SAINT repository setup. A practical workflow is:

```bash
git clone https://github.com/somepago/saint.git external/saint
cd external/saint
conda env create -f saint_environment.yml
conda activate saint_env
```

If using a separate SAINT conda environment, make sure the LLM4CKD package and required dependencies are also available in that environment, or export the preprocessed train/test splits and run SAINT there.

---

## Output Format

Prediction scripts write per-sample outputs such as:

```text
dataset, feature_set, family, model, seed, training_size, shots,
sample_id, y_true, y_prob, y_pred
```

Example output files:

```text
outputs/predictions/dataset2_qwen3_zero_shot.csv
outputs/predictions/dataset2_ml_selected.csv
outputs/predictions/dataset2_dl_selected.csv
outputs/predictions/dataset2_node_saint_selected.csv
```

Generated outputs are ignored by Git by default.

---

## Reproducibility Checklist

Before running experiments, verify:

1. Dataset-1 is available locally if reproducing private-cohort results.
2. [Dataset-2](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) has been downloaded using `scripts/download_dataset2_uci.py`.
3. Private data are not committed.
4. API keys are stored locally and not committed.
5. Optional DL dependencies are installed only when needed.
6. The official NODE repository has been cloned if running NODE.
7. The official SAINT repository has been cloned if running SAINT.
8. GPU/CUDA versions are compatible with the installed PyTorch build.
9. The same random seeds and training sizes are used as in `config/default.yaml`.

---

## Git Hygiene

The repository should not include:

```text
data/dataset1.csv
data/dataset2.csv
.env
outputs/
results/
cache/
models/
checkpoints/
external/
*.pt
*.pth
*.safetensors
```

These should remain ignored through `.gitignore`.

Before committing, check:

```bash
git status
```

On Windows Command Prompt:

```cmd
git status
```

---

## Typical Workflow

### 1. Clone the repository

```bash
git clone https://github.com/akabircs/LLM4CKD.git
cd LLM4CKD
```

### 2. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 3. Download Dataset-2

```bash
python scripts/download_dataset2_uci.py
```

On Windows Command Prompt:

```cmd
python scripts\download_dataset2_uci.py
```

### 4. Run a prompt-only test

```bash
python scripts/run_llm_zero_few_shot.py \
  --dataset dataset2 \
  --feature-set selected \
  --prompt-style instruction \
  --serialization list \
  --shots 0 \
  --backend prompt_only \
  --output outputs/predictions/test_prompt_only.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_llm_zero_few_shot.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --prompt-style instruction ^
  --serialization list ^
  --shots 0 ^
  --backend prompt_only ^
  --output outputs\predictions\test_prompt_only.csv
```

### 5. Run ML baselines

```bash
python scripts/run_ml_baselines.py \
  --dataset dataset2 \
  --feature-set selected \
  --output outputs/predictions/dataset2_ml_selected.csv
```

### 6. Prepare official NODE and SAINT repositories

```bash
mkdir -p external
git clone https://github.com/Qwicen/node.git external/node
git clone https://github.com/somepago/saint.git external/saint
```

On Windows Command Prompt:

```cmd
mkdir external
git clone https://github.com/Qwicen/node.git external\node
git clone https://github.com/somepago/saint.git external\saint
```

### 7. Run DL/foundation baselines

```bash
python scripts/run_dl_baselines.py \
  --dataset dataset2 \
  --feature-set selected \
  --training-sizes 4 8 16 32 \
  --node-repo-dir external/node \
  --saint-repo-dir external/saint \
  --device cpu \
  --output outputs/predictions/dataset2_dl_selected.csv
```

On Windows Command Prompt:

```cmd
python scripts\run_dl_baselines.py ^
  --dataset dataset2 ^
  --feature-set selected ^
  --training-sizes 4 8 16 32 ^
  --node-repo-dir external\node ^
  --saint-repo-dir external\saint ^
  --device cpu ^
  --output outputs\predictions\dataset2_dl_selected.csv
```

---

## Notes on Clinical Use

This repository is intended for research reproducibility only.

The code and models are not intended for clinical deployment, diagnosis, or treatment decisions without prospective validation, clinical governance, calibration assessment, and safety evaluation.

---

## License

See `LICENSE`.

---

## Citation

If you use this repository, please cite the corresponding LLM4CKD paper.

```bibtex
@article{llm4ckd,
  title   = {LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening},
  author  = {Kabir, Ashad and Munira, Sirajam},
  year    = {2025}
}
```
