import os
import json
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss


# === Options ===

# Paths and config
matrix_path = "input/merged_patient_genome_feature_matrix_v2.tsv"
outcome_config_path = "input/outcome_config_v2.json"
output_dir = "output"

feature_n_ranges = list(range(50, 301, 50))

# Model options
model_classifier_list = ["CatBoost", "LightGBM"]
standard_model_training_parameters = "Yes"
features_selected = "Yes"  # Or "No"
threads = 128


metadata_cols = {"Infecting_isolate"}
cv_folds = 5
cv_seed = random.randint(1, 10_000_000)

os.makedirs(output_dir, exist_ok=True)


# === Helper functions ===

# Load libraries conditionally
if "CatBoost" in model_classifier_list:
    from catboost import CatBoostClassifier, Pool as CatPool
if "LightGBM" in model_classifier_list:
    import lightgbm as lgb


# CatBoost standard model
def build_standard_catboost_model(seed):
    """
    Build a CatBoost model with standard parameters.
    """
    # from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=False,
        thread_count=threads,
        allow_writing_files=False,
    )


# LigthGBM standard model
def build_standard_lgb_model(seed):
    """
    Build a LightGBM model with standard parameters.
    """
    # import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=1,
        objective="binary",
        random_state=seed,
        n_jobs=threads,
    )


# === Load data ===

df = pd.read_csv(matrix_path, sep="\t", low_memory=False)

with open(outcome_config_path, encoding="utf-8") as f:
    outcome_config = json.load(f)

outcomes = list(outcome_config.keys())


# === Model training ===

all_metrics = []

# Training loop
for outcome in outcomes:
    print(f"\n=== Outcome: {outcome} ===")

    cfg = outcome_config[outcome]

    not_na = ~df[outcome].isna()
    y = df.loc[not_na, outcome].values

    poppunk_cluster_label = (
        df.loc[not_na, "PopPUNK_cluster"]
        .fillna("UNKNOWN")
        .astype(str)
        .values
    )

    # Initial composite labels (outcome + cluster)
    composite_labels = np.array([
        f"{y_i}_{c}"
        for y_i, c in zip(y, poppunk_cluster_label)
    ])

    # Count composite strata
    composite_counts = pd.Series(composite_labels).value_counts()

    # Collapse rare composite strata
    stratify_labels = np.array([
        cl if composite_counts[cl] >= cv_folds
        else f"{y_i}_RARE_CLUSTER"
        for y_i, cl in zip(y, composite_labels)
    ])

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=cv_seed,
    )

    # Obtain exclude list from config.json
    base_exclude = set(cfg.get("exclude_predictors", []))
    base_exclude.add(outcome)
    base_exclude.update(metadata_cols)

    for model_name in model_classifier_list:
        print(f"Model: {model_name}")

        model_cfg = cfg["model"][model_name]
        feature_types_full = model_cfg["features"]  # SHAP-ranked

        # Ensure no excluded features within features prior to top-n slice
        ordered_features = [
            f for f in feature_types_full.keys()
            if f not in base_exclude
        ]

        for n_features in feature_n_ranges:
            print(f"    Top N features: {n_features}")

            # Top-N slicing on VALID feature space
            selected_features = ordered_features[:n_features]

            feature_types = {
                f: feature_types_full[f]
                for f in selected_features
            }

            categorical_features = [
                f for f, t in feature_types.items()
                if t == "categorical"
            ]

            # Feature matrix (already safe)
            X = df.loc[not_na, selected_features].copy()

            # Prepare categorical features
            for c in categorical_features:
                X[c] = X[c].astype(str).fillna("")

            for fold, (train_idx, val_idx) in enumerate(
                skf.split(X, stratify_labels)
            ):
                fold_seed = random.randint(1, 10_000_000)

                # CatBoost
                if model_name == "CatBoost":
                    model = build_standard_catboost_model(fold_seed)

                    cat_idx = [
                        X.columns.get_loc(c)
                        for c in categorical_features
                    ]

                    train_pool = CatPool(
                        X.iloc[train_idx],
                        y[train_idx],
                        cat_features=cat_idx,
                    )
                    val_pool = CatPool(
                        X.iloc[val_idx],
                        y[val_idx],
                        cat_features=cat_idx,
                    )

                    model.fit(
                        train_pool,
                        eval_set=val_pool,
                        use_best_model=True,
                    )

                    probs = model.predict_proba(
                        X.iloc[val_idx]
                    )[:, 1]

                # LightGBM
                elif model_name == "LightGBM":
                    model = build_standard_lgb_model(fold_seed)

                    X_train = X.iloc[train_idx].copy()
                    X_val = X.iloc[val_idx].copy()

                    cat_lgb = []
                    for c in categorical_features:
                        X_train[c], uniques = pd.factorize(X_train[c])
                        X_val[c] = pd.Categorical(
                            X_val[c], categories=uniques
                        ).codes
                        cat_lgb.append(c)

                    model.fit(
                        X_train,
                        y[train_idx],
                        eval_set=[(X_val, y[val_idx])],
                        categorical_feature=cat_lgb,
                        callbacks=[
                            lgb.early_stopping(100, verbose=False)
                        ],
                    )

                    probs = model.predict_proba(X_val)[:, 1]

                else:
                    raise ValueError(
                        f"Unsupported model: {model_name}"
                    )

                # Minimal metrics for feature selection
                all_metrics.append({
                    "outcome": outcome,
                    "model": model_name,
                    "fold": fold,
                    "top_n_features": n_features,
                    "auc": roc_auc_score(y[val_idx], probs),
                    "logloss": log_loss(y[val_idx], probs),
                })


# === Metric aggregation ===

print("=== Outputting metrics ===")

# Model metrics
metrics_df = pd.DataFrame(all_metrics)

for metric in ["auc", "logloss"]:
    metrics_df[f"{metric}_mean"] = (
        metrics_df
        .groupby(["outcome", "model", "top_n_features"])[metric]
        .transform("mean")
    )
    metrics_df[f"{metric}_sd"] = (
        metrics_df
        .groupby(["outcome", "model", "top_n_features"])[metric]
        .transform("std")
    )

metrics_df.to_csv(
    os.path.join(output_dir, "v2_feature_selection_metrics.tsv"),
    sep="\t",
    index=False
)

print("\nV2 feature selection evaluation complete.")
