import os
import json
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    log_loss,
    roc_curve
)

# === Options ===

# Paths and config
matrix_path = "input/merged_patient_genome_feature_matrix_v1.tsv"
outcome_config_path = "input/outcome_config.json"
categorical_cols_list_path = "input/categorical_cols_list.txt"
output_dir = "output"

# Model options
model_classifier_list = ["CatBoost", "LightGBM"]
standard_model_training_parameters = "Yes"  # or "No"
features_selected = "No"  # or "Yes"
threads = 128

# Columns that should not be used as model features
metadata_cols = {"Infecting_isolate"}

cv_folds = 5
cv_seed = random.randint(1, 10_000_000)


# === Helper functions ===

# Load libraries conditionally
if "CatBoost" in model_classifier_list:
    from catboost import CatBoostClassifier, Pool as CatPool
if "LightGBM" in model_classifier_list:
    import lightgbm as lgb
    import shap


# CatBoost (best params)
def build_catboost_model(best_params, seed):
    """
    Build a CatBoost model with the given best parameters.
    """
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


# CatBoost (standard params)
def build_standard_catboost_model(seed):
    """
    Build a CatBoost model with standard parameters.
    """
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        loss_function="Logloss",
        eval_metric="Logloss",
        custom_loss=["Logloss", "CrossEntropy"],
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=100,
        bootstrap_type="Bernoulli",
        l2_leaf_reg=3,
        grow_policy="Depthwise",
        border_count=254,
        max_ctr_complexity=2,
        task_type="CPU",
        thread_count=threads,
        allow_writing_files=False,
    )


# LightGBM (best params)
def build_lgb_model(best_params, seed):
    """
    Build a LightGBM model with the given best parameters.
    """
    return lgb.LGBMClassifier(
        **best_params,
        objective="binary",
        random_state=seed,
        n_jobs=threads,
    )


# LightGBM (standard params)
def build_standard_lgb_model(seed):
    """
    Build a LightGBM model with standard parameters.
    """
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        min_data_in_leaf=1,
        colsample_bytree=0.8,
        objective="binary",
        random_state=seed,
        n_jobs=threads,
    )


# Write config file for v2 feature selection
def write_top_features_config(
    mean_abs_shap_df,
    output_path,
    GLOBAL_CATEGORICAL_COLS,
    original_config,
    top_n=None,
):
    """
    Generate a config JSON with top features per outcome and model
    based on mean absolute SHAP values.

    Assumes mean_abs_shap_df is already sorted by:
        ['outcome', 'model', 'mean_abs_SHAP'] descending.
    """
    config_v2 = {}

    for outcome in mean_abs_shap_df["outcome"].unique():
        # Subset ONCE per outcome  ← FIX
        outcome_df = mean_abs_shap_df[
            mean_abs_shap_df["outcome"] == outcome
        ]

        config_v2[outcome] = {
            "model": {},
            "exclude_predictors": (
                original_config
                .get(outcome, {})
                .get("exclude_predictors", [])
            ),
        }

        for model in outcome_df["model"].unique():
            model_df = outcome_df[
                outcome_df["model"] == model
            ]

            features = model_df["feature"].tolist()
            if top_n is not None:
                features = features[:top_n]

            feature_types = {
                f: (
                    "categorical"
                    if f in GLOBAL_CATEGORICAL_COLS
                    else "numeric"
                )
                for f in features
            }

            config_v2[outcome]["model"][model] = {
                "best_params": {},
                "features": feature_types,
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_v2, f, indent=4)

    print(f"Updated config JSON written to {output_path}")


# Load data

os.makedirs(output_dir, exist_ok=True)

print("=== Loading categorical col data ===")
with open(categorical_cols_list_path, encoding="utf-8") as f:
    GLOBAL_CATEGORICAL_COLS = {
        line.strip() for line in f if line.strip()
    }

print("=== Loading input matrix ===")
df = pd.read_csv(matrix_path, sep="\t", low_memory=False)

print("=== Loading contig ===")
with open(outcome_config_path, encoding="utf-8") as f:
    outcome_config = json.load(f)

outcomes = list(outcome_config.keys())


# === Model training ===

# Global containers
all_metrics = []
all_mean_abs_shap = []

# Main training loop
for target in outcomes:
    print(f"\n=== Processing outcome: {target} ===")

    cfg = outcome_config[target]

    # fallback for v1 config file lacking model information
    if "model" not in cfg:
        cfg["model"] = {}
        for mn in model_classifier_list:
            cfg["model"][mn] = {"best_params": {}, "features": {}}

    not_na = ~df[target].isna()
    y_target = df.loc[not_na, target].values

    infecting_isolate = df.loc[not_na, "Infecting_isolate"].values
    poppunk_cluster_label = (
        df.loc[not_na, "PopPUNK_cluster"]
        .fillna("UNKNOWN")
        .astype(str)
        .values
    )

    # Initial composite labels
    composite_labels = np.array([
        f"{y}_{c}"
        for y, c in zip(y_target, poppunk_cluster_label)
    ])

    # Count composite strata
    composite_counts = pd.Series(
        composite_labels
    ).value_counts()

    # Collapse rare composite strata
    stratify_labels = np.array([
        cl if composite_counts[cl] >= cv_folds
        else f"{y}_RARE_CLUSTER"
        for y, cl in zip(y_target, composite_labels)
    ])

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=cv_seed,
    )

    shap_dfs = []

    # Obtain exclude list from config.json
    base_exclude = set(cfg.get("exclude_predictors", []))
    base_exclude.add(target)
    base_exclude.update(metadata_cols)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df.loc[not_na], stratify_labels), start=1
    ):
        fold_seed = random.randint(1, 10_000_000)

        for model_name in model_classifier_list:
            model_cfg = cfg["model"][model_name]

            if features_selected == "Yes":
                # SHAP-selected features, BUT still enforce exclusions
                feature_types = model_cfg["features"]

                target_features = [
                    f for f in feature_types.keys()
                    if f not in base_exclude
                ]

                categorical_features = [
                    f for f, t in feature_types.items()
                    if t == "categorical" and f in target_features
                ]
            else:
                # Full feature space, excluding forbidden predictors
                target_features = [
                    c for c in df.columns
                    if c not in base_exclude
                ]

                categorical_features = [
                    c for c in target_features
                    if c in GLOBAL_CATEGORICAL_COLS
                ]

            X_target = df.loc[not_na, target_features].copy()

            # Prepare categorical features
            for c in categorical_features:
                X_target[c] = X_target[c].astype(str).fillna("")

            cat_feature_indices = [
                X_target.columns.get_loc(c)
                for c in categorical_features
            ]

            # CatBoost
            if model_name == "CatBoost":
                model = (
                    build_standard_catboost_model(fold_seed)
                    if standard_model_training_parameters == "Yes"
                    else build_catboost_model(
                        model_cfg["best_params"], fold_seed
                    )
                )

                train_pool = CatPool(
                    X_target.iloc[train_idx],
                    y_target[train_idx],
                    cat_features=cat_feature_indices,
                )
                val_pool = CatPool(
                    X_target.iloc[val_idx],
                    y_target[val_idx],
                    cat_features=cat_feature_indices,
                )

                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    use_best_model=True,
                )

                probs = model.predict_proba(
                    X_target.iloc[val_idx]
                )[:, 1]

                fpr, tpr, thresholds = roc_curve(
                    y_target[val_idx], probs
                )
                youden_j = tpr - fpr
                best_idx = youden_j.argmax()

                best_threshold = thresholds[best_idx]
                best_j = youden_j[best_idx]

                preds = (probs >= best_threshold).astype(int)

                shap_vals = model.get_feature_importance(
                    val_pool, type="ShapValues"
                )[:, :-1]

            # LightGBM
            elif model_name == "LightGBM":
                model = (
                    build_standard_lgb_model(fold_seed)
                    if standard_model_training_parameters == "Yes"
                    else build_lgb_model(
                        model_cfg["best_params"], fold_seed
                    )
                )

                X_train = X_target.iloc[train_idx].copy()
                X_val = X_target.iloc[val_idx].copy()

                cat_features_lgb = []
                for c in categorical_features:
                    X_train[c], uniques = pd.factorize(X_train[c])
                    X_val[c] = pd.Categorical(
                        X_val[c], categories=uniques
                    ).codes
                    cat_features_lgb.append(c)

                model.fit(
                    X_train,
                    y_target[train_idx],
                    eval_set=[(X_val, y_target[val_idx])],
                    categorical_feature=cat_features_lgb,
                    callbacks=[
                        lgb.early_stopping(100, verbose=False)
                    ],
                )

                probs = model.predict_proba(X_val)[:, 1]

                fpr, tpr, thresholds = roc_curve(
                    y_target[val_idx], probs
                )
                youden_j = tpr - fpr
                best_idx = youden_j.argmax()

                best_threshold = thresholds[best_idx]
                best_j = youden_j[best_idx]

                preds = (probs >= best_threshold).astype(int)

                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_val)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

            # SHAP dataframe
            shap_df = pd.DataFrame(
                shap_vals, columns=target_features
            )
            shap_df.insert(0, "model", model_name)
            shap_df.insert(0, "outcome", target)
            shap_df.insert(
                0,
                "Infecting_isolate",
                infecting_isolate[val_idx],
            )
            shap_df.insert(
                0,
                "PopPUNK_cluster_label",
                poppunk_cluster_label[val_idx],
            )
            shap_dfs.append(shap_df)

            # Metrics
            tn, fp, fn, tp = confusion_matrix(
                y_target[val_idx], preds, labels=[0, 1]
            ).ravel()

            # Sensitivity and Specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            balanced_accuracy = (sensitivity + specificity) / 2

            all_metrics.append({
                "outcome": target,
                "model": model_name,
                "fold": fold,
                "auc": roc_auc_score(
                    y_target[val_idx], probs
                ),
                "balanced_accuracy": balanced_accuracy,
                "logloss": log_loss(
                    y_target[val_idx], probs
                ),
                "crossentropy": log_loss(
                    y_target[val_idx], probs
                ),
                "youden_j": best_j,
                "optimal_threshold": best_threshold,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": accuracy_score(
                    y_target[val_idx], preds
                ),
                "f1": f1_score(
                    y_target[val_idx], preds
                ),
                "seed": fold_seed,
            })

    print("=== Aggregating SHAP ===")
    # SHAP aggregation
    cv_shap_df = pd.concat(shap_dfs, ignore_index=True)

    cv_shap_df.to_csv(
        os.path.join(
            output_dir, f"CV_aggregated_shap_{target}.tsv"
        ),
        sep="\t",
        index=False,
    )

    mean_abs_shap = (
        cv_shap_df
        .groupby("model")[target_features]
        .apply(lambda x: x.abs().mean())
        .reset_index()
        .melt(
            id_vars="model",
            var_name="feature",
            value_name="mean_abs_SHAP",
        )
    )
    mean_abs_shap.insert(0, "outcome", target)
    all_mean_abs_shap.append(mean_abs_shap)

    # clone_mean_abs_shap = (
    #     cv_shap_df
    #     .groupby(
    #         ["model", "PopPUNK_cluster_label"]
    #     )[target_features]
    #     .apply(lambda x: x.abs().mean())
    #     .reset_index()
    # )

    # clone_mean_abs_shap_long = clone_mean_abs_shap.melt(
    #     id_vars=["model", "PopPUNK_cluster_label"],
    #     var_name="feature",
    #     value_name="mean_abs_SHAP",
    # )
    # clone_mean_abs_shap_long.insert(0, "outcome", target)
    #
    # clone_mean_abs_shap_long.to_csv(
    #     os.path.join(
    #         output_dir,
    #         f"clone_mean_abs_shap_{target}.tsv",
    #     ),
    #     sep="\t",
    #     index=False,
    # )

# === Metric aggregation ===

print("=== Outputting metrics ===")

# Model metrics
metrics_df = pd.DataFrame(all_metrics)

for col in ["auc", "balanced_accuracy", "accuracy", "logloss", "crossentropy"]:
    metrics_df[f"{col}_mean"] = (
        metrics_df
        .groupby(["outcome", "model"])[col]
        .transform("mean")
    )
    metrics_df[f"{col}_sd"] = (
        metrics_df
        .groupby(["outcome", "model"])[col]
        .transform("std")
    )

metrics_df.to_csv(
    os.path.join(output_dir, "cv_metrics_all_outcomes.tsv"),
    sep="\t",
    index=False,
)

print("=== Saving mean absolute SHAP ===")

# Mean |SHAP|
combined_mean_abs_shap = pd.concat(
    all_mean_abs_shap, ignore_index=True
)

# Sort combined mean |SHAP|
combined_mean_abs_shap_sorted = combined_mean_abs_shap.sort_values(
    ["outcome", "model", "mean_abs_SHAP"], ascending=[True, True, False]
)

# Save sorted TSV
combined_mean_abs_shap_sorted.to_csv(
    os.path.join(output_dir, "mean_abs_shap_all_outcomes.tsv"),
    sep="\t",
    index=False
)

# Generate updated JSON with top features
write_top_features_config(
    combined_mean_abs_shap_sorted,
    os.path.join(output_dir, "outcome_config_v2.json"),
    GLOBAL_CATEGORICAL_COLS,
    outcome_config,
    top_n=301
)
