# Klebsiella-pneumoniae-patient-outcomes-ML
Repository hosting machine learning models and scripts for predicting the clinical outcomes of _Klebsiella pneumoniae_ infections. 


## Installation
```bash
# Copy repo
git clone https://github.com/bananabenana/Klebsiella-pneumoniae-patient-outcomes-ML

# Move into downloaded directory
cd Klebsiella-pneumoniae-patient-outcomes-ML

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate catboost_ML_env

# Test software
python deployable_prediction_models/patient_early_predictions.py --help

```

## Quickstart

### Patient predictions using gradient-boosting models
If you would like to use these models to run test patient data, see below. Note that this is not approved for clinical use and is for research use only.

#### Run patient predictions using the Clinical+Genomic models
1. Fill appropriate columns in: `Input/Clinical+Genomic_new_patients.tsv`
    - Leave any missing data as a blank cell
    - This includes 5 rows of simulated patient data as example input
2. Run the following:
```bash
# Change to directory
cd Klebsiella-pneumoniae-patient-outcomes-ML/deployable_prediction_models

# Activate environment
conda activate catboost_ML

# Set variables for Clinical+Genomic
model="Clinical+Genomic"
model_path="Models/${model}"
outdir="Output/${model}_predictions"

# Make outdir
mkdir "${outdir}"

# Run script
python patient_early_predictions.py \
    --patient_data "Input/${model}_new_patients.tsv" \
    --model_dir "${model_path}" \
    --config "${model_path}/outcome_config.json" \
    --output "${outdir}/patient_predictions.tsv"
```

#### Run patient predictions using the 30-Minute-Clinical models
1. Fill appropriate columns in: `Input/30-Minute-Clinical_new_patients.tsv`.
    - Leave any missing data as a blank cell
    - This includes 5 rows of simulated patient data as example input
2. Run the following:
```bash
# Change to directory
cd Klebsiella-pneumoniae-patient-outcomes-ML/deployable_prediction_models

# Activate environment
conda activate catboost_ML

# Set variables for 30-Minute-Clinical
model="30-Minute-Clinical"
model_path="Models/${model}"
outdir="Output/${model}_predictions"

# Make outdir
mkdir "${outdir}"

# Run script
python patient_early_predictions.py \
    --patient_data "Input/${model}_new_patients.tsv" \
    --model_dir "${model_path}" \
    --config "${model_path}/outcome_config.json" \
    --output "${outdir}/patient_predictions.tsv"
```

#### Expected outputs

You will get a `patient_predictions.tsv` file that looks like this:

| Patient_ID     | Clinical_outcome | Risk_label                                          | Predictive_value | Model    | Optimal_threshold_mean | Optimal_threshold_ci_low | Optimal_threshold_ci_high | Optimal_threshold_delta | Number_features_present_in_input_data | Ideal_number_features | Feature_coverage_percent | Notes                       |
| -------------- | ---------------- | --------------------------------------------------- | ---------------- | -------- | ---------------------- | ------------------------ | ------------------------- | ----------------------- | ------------------------------------- | --------------------- | ------------------------ | --------------------------- |
| fake_patient_1 | Mortality        | Low risk                                            | 0.034348         | CatBoost | 0.2399                 | 0.167                    | 0.3128                    | 0.0729                  | 10                                    | 10                    | 100                      | Sufficient feature coverage |
| fake_patient_1 | Poor prognosis   | Low risk                                            | 0.045612         | CatBoost | 0.2769                 | 0.1899                   | 0.364                     | 0.08705                 | 50                                    | 50                    | 100                      | Sufficient feature coverage |
| fake_patient_1 | Sepsis           | High risk                                           | 0.984461         | LightGBM | 0.4084                 | 0.347                    | 0.4698                    | 0.0614                  | 20                                    | 20                    | 100                      | Sufficient feature coverage |
| fake_patient_2 | Mortality        | Intermediate risk - interpret with clinical context | 0.267525         | CatBoost | 0.2399                 | 0.167                    | 0.3128                    | 0.0729                  | 10                                    | 10                    | 100                      | Sufficient feature coverage |
| fake_patient_2 | Poor prognosis   | Intermediate risk - interpret with clinical context | 0.342903         | CatBoost | 0.2769                 | 0.1899                   | 0.364                     | 0.08705                 | 50                                    | 50                    | 100                      | Sufficient feature coverage |


The `Risk_label` column is the risk category for a particular patient and `Clinical_outcome`. We use guard-banding to prevent alarm fatigue caused by false-positives. Briefly, if the `Predictive_value` is within the 95% confidence interval, it will be called as "Intermediate risk" instead of "High risk".

| Outcome label                                       | Predictive probability is                                |
| --------------------------------------------------- | -------------------------------------------------------- |
| High risk                                           | Above optimal threshold. Outside 95% confidence interval |
| Intermediate risk - interpret with clinical context | Above optimal threshold. Within 95% confidence interval  |
| Low risk                                            | Below optimal threshold                                  |


### Model training used in manuscript
For this article, the following scripts were used for model training. Due to ethics, clinical data used as input matrix is available upon request. Please refer to data accessibility section of {doi}

#### First round of model training to identify important features
```bash
# Activate environment
conda activate catboost_ML

# Set variables
round=v1
train_script="${round}_training.py"

# Create dirs
mkdir -p $round/input $round/output; cd $round

# Run script
python -u $train_script

cd ..
```

#### Second round of model training to identify top n features for feature selection
```bash
# Set variables
round=v2
train_script="${round}_training.py"

# Create dirs
mkdir -p $round/input $round/output; cd $round

# Run script
python -u $train_script

cd ..
```

#### Third round of model training for hyperparameter optimisation using selected features
```bash
# Set variables
round=v3
train_script="${round}_training.py"

# Create dirs
mkdir -p $round/input $round/output; cd $round

# Generate new v3 config.json and a v3 matrix file to optimise loading times
python v3_ML_hyperparam_opt_prep.py \
  --matrix ../v2/input/v2_30_mins_patient_mdata.tsv \
  --config ../v2/input/outcome_config_v2.json \
  --top_n_features 50 \
  --top_n_exceptions input/all_targets_top_n_median_exceptions.tsv \
  --outdir input

# Run script
python -u $train_script

cd ..
```

#### Final round of model training to fully train hyperparameter-optimised models
```bash
# Set variables
round=v4
train_script="${round}_training.py"

# Create dirs
mkdir -p $round/input $round/output; cd $round

# Run script
python -u $train_script

cd ..
```


## Reference

- To update: {doi}


## Authors

- Ben Vezina 1,4#*
- Pengcheng Du 2#
- Yunfei Tang 3
- Nenad Macesic 1,4,5
- Hoai-An Nguyen 1,4
- Kelly L. Wyres 1,4,6
- Gianluca Morroni 7,8
- Chao Liu 3,+*
- Margaret M. C. Lam 1,4,+*
```
1 Department of Infectious Diseases, School of Translational Medicine, Monash University, Melbourne, Victoria, Australia
2 Medical Research Center, Beijing Institute of Respiratory Medicine and Beijing Chao-Yang Hospital, Capital Medical University, Beijing, China
3 Department of Infectious Disease, Peking University Third Hospital, Beijing, China
4 Centre to Impact AMR, Monash University, Clayton, Victoria, Australia
5 Infection Prevention & Healthcare Epidemiology, Alfred Health, Melbourne, Australia.
6 Department of Infection Biology, London School of Hygiene and Tropical Medicine, London, UK
7 Microbiology Unit, Department of Biomedical Sciences & Public Health, Polytechnic University of Marche, Ancona, Italy
8 SOS Microbiologia, SOD Medicina di Laboratorio, Azienda Ospedaliero Universitaria delle Marche, Ancona, Italy
# These authors contributed equally
+ These authors contributed equally
```

## Disclaimer

For research use only.
