# Data directory

Place CSV files here:

```text
data/dataset1.csv
data/dataset2.csv
```

## Expected target column

The loader searches for the first available target column among:

```text
ckd, CKD, class, classification, target, label, y
```

Positive CKD values accepted by default:

```text
1, ckd, yes, true, positive, case
```

Negative non-CKD values accepted by default:

```text
0, notckd, non-ckd, non_ckd, no, false, negative, control
```

## Canonical feature names

The loader normalizes column names to lowercase snake_case. For maximum compatibility, use these names.

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

### Dataset-2 all features

- age
- blood_pressure
- specific_gravity
- albumin
- sugar
- presence_of_red_blood_cells_in_urine
- pus_cell
- pus_cell_clumps
- bacteria
- blood_glucose_random
- blood_urea
- sodium
- potassium
- hemoglobin
- packed_cell_volume
- white_blood_cell_count
- red_blood_cell_count
- history_of_hypertension
- history_of_diabetes
- coronary_artery_disease
- appetite_condition
- pedal_edema
- anemia

### Dataset-2 selected features

The paper maps the selected Dataset-1 concepts to Dataset-2 where available. This scaffold uses:

- age
- history_of_hypertension
- history_of_diabetes
- presence_of_red_blood_cells_in_urine
- anemia

## Leakage exclusions

Do not use direct diagnostic markers as model inputs:

- Dataset-1: eGFR and uACR/ACR columns.
- Dataset-2: serum creatinine.

The loader removes common aliases for these leakage variables automatically when present.
