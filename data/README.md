# Data directory

Place CSV files here:

```text
data/dataset1.csv
data/dataset2.csv
```

## Canonical feature names

The loader normalizes column names to lowercase snake_case. For maximum compatibility, use these names.

## Dataset-1: Private Bangladesh Cohort

Dataset-1 is a private community-based Bangladeshi cohort used for the main early-stage CKD screening experiments.

This repository does **not** include Dataset-1 because it contains private participant-level data.

Expected local path:

```text
data/dataset1.csv
```

This file is ignored by Git and should not be committed.

### Dataset-1 all features

- age
- gender
- illiterate
- occupation
- marital_status
- sleeping_duration
- tobacco_smoker
- smokeless_tobacco
- history_of_hypertension
- history_of_diabetes
- heart_disease
- stroke
- family_history_of_diabetes
- family_history_of_hypertension
- family_history_of_ckd
- body_mass_index
- abdominal_obesity
- undernutrition
- anemia
- presence_of_red_blood_cells_in_urine
- serum_albumin
- hypercholesterolemia
- hdl_cholesterol
- hypertriglyceridemia

### Dataset-1 selected features

- history_of_hypertension
- age
- presence_of_red_blood_cells_in_urine
- sleeping_duration
- anemia
- history_of_diabetes
- body_mass_index
- family_history_of_hypertension
- gender

## Dataset-2: Public UCI CKD Dataset

Dataset-2 is the public [UCI Chronic Kidney Disease dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease), used for independent/cross-dataset evaluation.

To download Dataset-2 locally, run:

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

### Dataset-2 Feature Name Mapping

The first column below shows the short feature name used in the downloaded UCI Dataset-2 file. The second column shows the corresponding full clinical feature name. The third column shows the harmonized feature name used internally by this repository.

| UCI downloaded column name | Full feature name | Harmonized repository feature name |
|---|---|---|
| `age` | age | `age` |
| `bp` | blood pressure | `blood_pressure` |
| `sg` | specific gravity | `specific_gravity` |
| `al` | albumin | `albumin` |
| `su` | sugar | `sugar` |
| `rbc` | red blood cells | `presence_of_red_blood_cells_in_urine` |
| `pc` | pus cell | `pus_cell` |
| `pcc` | pus cell clumps | `pus_cell_clumps` |
| `ba` | bacteria | `bacteria` |
| `bgr` | blood glucose random | `blood_glucose_random` |
| `bu` | blood urea | `blood_urea` |
| `sc` | serum creatinine | `serum_creatinine` |
| `sod` | sodium | `sodium` |
| `pot` | potassium | `potassium` |
| `hemo` | hemoglobin | `hemoglobin` |
| `pcv` | packed cell volume | `packed_cell_volume` |
| `wc` | white blood cell count | `white_blood_cell_count` |
| `rc` | red blood cell count | `red_blood_cell_count` |
| `htn` | hypertension | `history_of_hypertension` |
| `dm` | diabetes mellitus | `history_of_diabetes` |
| `cad` | coronary artery disease | `coronary_artery_disease` |
| `appet` | appetite | `appetite_condition` |
| `pe` | pedal edema | `pedal_edema` |
| `ane` | anemia | `anemia` |
| `class` | class | `target` |

### Dataset-2 selected features

The paper maps the selected Dataset-1 concepts to Dataset-2 where available. This scaffold uses:

- age
- history_of_hypertension
- history_of_diabetes
- presence_of_red_blood_cells_in_urine
- anemia

## Leakage exclusions

Do not use direct diagnostic markers as model inputs because they are direct kidney-function markers strongly related to CKD diagnosis:

- Dataset-1: eGFR and uACR/ACR columns.
- Dataset-2: serum creatinine.

The loader removes common aliases for these leakage variables automatically when present.

## Do Not Commit Local Data

The following files should remain local and should not be committed:

```text
data/dataset1.csv
data/dataset2.csv
data/private/
data/raw_private/
```
