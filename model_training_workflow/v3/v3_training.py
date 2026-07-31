import os
import random
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import optuna
# from optuna.integration import CatBoostPruningCallback


# === Options ===

# Paths and config
matrix_path = "input/merged_patient_genome_feature_matrix_v3.tsv"
outcome_config_path = "input/outcome_config_v3.json"
output_dir = "output"

# Columns that should not be used as model features
metadata_cols = {"Infecting_isolate"}

# Model options
model_classifier_list = ["CatBoost", "LightGBM"]
standard_model_training_parameters = "Yes"  # or "No"
features_selected = "Yes"  # or "No"
threads = 128  # used for n threads for model training
cv_folds = 5
cv_seed = random.randint(1, 10_000_000)

# Load libraries conditionally
for model in model_classifier_list:
    if model == "CatBoost":
        from catboost import CatBoostClassifier, Pool as CatPool
    if model == "LightGBM":
        import lightgbm as lgb


# === Helper functions ===


# K-fold function
def make_poppunk_stratify_labels(y, poppunk_cluster_label, cv_folds):
    poppunk_cluster_label = np.asarray(
        poppunk_cluster_label
    ).astype(str)

    composite_labels = np.array([
        f"{y_i}_{c}"
        for y_i, c in zip(y, poppunk_cluster_label)
    ])

    composite_counts = pd.Series(
        composite_labels
    ).value_counts()

    stratify_labels = np.array([
        cl if composite_counts[cl] >= cv_folds
        else f"{y_i}_RARE_CLUSTER"
        for y_i, cl in zip(y, composite_labels)
    ])

    return stratify_labels


# CatBoost function
def build_catboost_model(best_params, seed):
    return CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="Logloss",
        custom_loss=["Logloss", "CrossEntropy"],
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=False,
        task_type="CPU",
        thread_count=threads,
        allow_writing_files=False,
    )


# LightGBM function
def build_lgb_model(best_params, seed):
    return lgb.LGBMClassifier(
        **best_params,
        objective="binary",
        random_state=seed,
        n_jobs=threads,
    )


def catboost_objective(trial, X, y, cat_idx, clusters, cv_folds, cv_seed):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli"]
        ),
    }

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    stratify_labels = make_poppunk_stratify_labels(
        y,
        clusters,
        cv_folds
    )

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=cv_seed,
    )

    aucs = []

    for fold, (tr, va) in enumerate(skf.split(X, stratify_labels)):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        y_tr, y_va = y[tr], y[va]

        # Convert ALL categorical features to string and fill NaN for both train and val
        for i in cat_idx:
            col = X.columns[i]
            X_tr[col] = X_tr[col].astype(str).fillna("Missing")
            X_va[col] = X_va[col].astype(str).fillna("Missing")

        train_pool = CatPool(X_tr, y_tr, cat_features=cat_idx)
        val_pool = CatPool(X_va, y_va, cat_features=cat_idx)

        model = build_catboost_model(params, seed=trial.number)
        model.fit(train_pool, eval_set=val_pool)

        probs = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, probs))

        trial.report(np.mean(aucs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(aucs))


def lgb_objective(trial, X, y, cat_cols, clusters, cv_folds, cv_seed):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    stratify_labels = make_poppunk_stratify_labels(
        y,
        clusters,
        cv_folds
    )

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=cv_seed,
    )

    aucs = []

    for fold, (tr, va) in enumerate(skf.split(X, stratify_labels)):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        y_tr, y_va = y[tr], y[va]

        # Factorize categorical features in train and apply same mapping to val
        for c in cat_cols:
            X_tr[c], uniques = pd.factorize(X_tr[c])
            X_va[c] = pd.Categorical(X_va[c], categories=uniques).codes

        model = build_lgb_model(params, seed=trial.number)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        probs = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, probs))

        trial.report(np.mean(aucs), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(aucs))


# === Load data ===
print("=== Loading matrix ===")
df = pd.read_csv(matrix_path, sep="\t")

print("=== Loading config ===")
with open(outcome_config_path, "r", encoding="utf-8") as f:
    outcome_config = json.load(f)

outcomes = list(outcome_config.keys())

# Create outputs
os.makedirs(output_dir, exist_ok=True)
outcome_config_v4 = json.loads(json.dumps(outcome_config))


# === Training ===

metrics = []

# Run training loop
for outcome in outcomes:
    print(f"\n=== Optimising: {outcome} ===")

    cfg = outcome_config[outcome]

    not_na = ~df[outcome].isna()
    y = df.loc[not_na, outcome].values

    # Obtain exclude list from config.json
    base_exclude = set(cfg.get("exclude_predictors", []))
    base_exclude.add(outcome)
    base_exclude.update(metadata_cols)

    for model_name in model_classifier_list:
        print(f"Model: {model_name}")

        model_cfg = cfg["model"][model_name]
        feature_types_full = model_cfg["features"]

        # Ensure no excluded features are kept
        features = [
            f for f in feature_types_full.keys()
            if f not in base_exclude
        ]

        feature_types = {
            f: feature_types_full[f]
            for f in features
        }

        cat_features = [
            f for f, t in feature_types.items()
            if t == "categorical"
        ]

        # Feature matrix (now safe)
        X = df.loc[not_na, features].copy()

        # Prepare categorical features
        for c in cat_features:
            X[c] = X[c].astype(str).fillna("")

        # Objective wiring
        if model_name == "CatBoost":
            cat_idx = [X.columns.get_loc(c) for c in cat_features]
            objective = lambda t: catboost_objective(
                t, X, y, cat_idx,
                df.loc[not_na, "PopPUNK_cluster"].values,
                cv_folds, cv_seed
            )

        elif model_name == "LightGBM":
            objective = lambda t: lgb_objective(
                t, X, y, cat_features, df.loc[not_na, "PopPUNK_cluster"].values,
                cv_folds, cv_seed
            )

        else:
            raise ValueError(model_name)

        # Optuna study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=2
            ),
        )

        study.optimize(
            objective,
            n_trials=150,
            timeout=1800,
            n_jobs=1,  # keep Optuna sequential
        )

        best = study.best_trial

        outcome_config_v4[outcome]["model"][model_name][
            "best_params"
        ] = best.params

        metrics.append({
            "outcome": outcome,
            "model": model_name,
            "best_auc": best.value,
            **best.params,
        })


# === Outputs ===

print("=== Writing output json and metrics ===")
# Updated json with optimal hyperparameters
with open(os.path.join(output_dir, "outcome_config_v4.json", encoding="utf-8"), "w") as f:
    json.dump(outcome_config_v4, f, indent=4)

# Output of hyperparameters as tsv
pd.DataFrame(metrics).to_csv(
    os.path.join(output_dir, "hyperparam_optimisation_summary.tsv"),
    sep="\t",
    index=False,
)
