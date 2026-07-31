import os
import json
import argparse
import pandas as pd

from catboost import CatBoostClassifier, Pool as CatPool
import lightgbm as lgb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--patient_data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def validate_headers(input_df, expected_features, target):
    missing = set(expected_features) - set(input_df.columns)

    if missing:
        print(f"WARNING: Missing columns for {target}: {missing}")

    return missing


def prepare_features(input_df, feature_types):

    target_features = list(feature_types.keys())
    categorical_features = [
        f for f, t in feature_types.items()
        if t == "categorical"
    ]

    X = input_df.reindex(columns=target_features).copy()

    for f in target_features:
        if f in categorical_features:
            X[f] = X[f].astype(str).fillna("")
        else:
            X[f] = pd.to_numeric(X[f], errors="coerce")

    present_feature_count = X.notna().sum(axis=1)

    return X, categorical_features, present_feature_count


def load_model(model_name, model_dir, target):

    if model_name == "CatBoost":
        model_path = os.path.join(
            model_dir, f"catboost_full_{target}.cbm"
        )
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model

    elif model_name == "LightGBM":
        model_path = os.path.join(
            model_dir, f"lightgbm_full_{target}.txt"
        )
        booster = lgb.Booster(model_file=model_path)
        return booster

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def run_prediction(model_name, model, X, categorical_features):

    if model_name == "CatBoost":
        cat_feature_indices = [
            X.columns.get_loc(c)
            for c in categorical_features
        ]
        pool = CatPool(
            X,
            cat_features=cat_feature_indices
        )
        probs = model.predict_proba(pool)[:, 1]

    elif model_name == "LightGBM":
        X_lgb = X.copy()
        for c in categorical_features:
            X_lgb[c], _ = pd.factorize(X_lgb[c])
        probs = model.predict(X_lgb, raw_score=False)

    return probs


def apply_guard_band(probs, T, T_low, T_high):

    delta = (T_high - T_low) / 2

    labels = []

    for p in probs:
        if p >= T + delta:
            labels.append("High risk")
        elif p <= T - delta:
            labels.append("Low risk")
        else:
            labels.append(
                "Intermediate risk - interpret with clinical context"
            )

    return labels, delta


def build_output_df(
    patient_ids,
    target,
    labels,
    probs,
    model_name,
    T,
    T_low,
    T_high,
    delta,
    present_feature_count,
    ideal_feature_count
):
    present_feature_count = present_feature_count.astype(int)
    ideal_feature_count = int(ideal_feature_count)

    coverage_percent = (
        present_feature_count / ideal_feature_count * 100
    ).round(1)

    notes = []
    for pct in coverage_percent:
        if pct < 20:
            notes.append("CRITICAL: Extremely low feature coverage")
        elif pct < 50:
            notes.append("WARNING: Low feature coverage")
        else:
            notes.append("Sufficient feature coverage")

    # Explicit column order
    df_out = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Clinical_outcome": target,
        "Risk_label": labels,
        "Predictive_value": probs,
        "Model": model_name,
        "Optimal_threshold_mean": T,
        "Optimal_threshold_ci_low": T_low,
        "Optimal_threshold_ci_high": T_high,
        "Optimal_threshold_delta": delta,
        "Number_features_present_in_input_data": present_feature_count,
        "Ideal_number_features": ideal_feature_count,
        "Feature_coverage_percent": coverage_percent,
        "Notes": notes
    })[
        [
            "Patient_ID",
            "Clinical_outcome",
            "Risk_label",
            "Predictive_value",
            "Model",
            "Optimal_threshold_mean",
            "Optimal_threshold_ci_low",
            "Optimal_threshold_ci_high",
            "Optimal_threshold_delta",
            "Number_features_present_in_input_data",
            "Ideal_number_features",
            "Feature_coverage_percent",
            "Notes",
        ]
    ]

    return df_out


# Main
def main():

    args = parse_arguments()

    input_df = pd.read_csv(args.patient_data, sep="\t",)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    if "patient_id" not in input_df.columns:
        input_df["patient_id"] = range(1, len(input_df) + 1)

    results = []

    for target in config.keys():

        print(f"\n=== Processing outcome: {target} ===")
        cfg = config[target]

        for model_name in cfg["model"].keys():

            # Features and ideal count
            feature_types = cfg["model"][model_name]["features"]
            ideal_feature_count = len(feature_types)

            # Validate input headers
            validate_headers(input_df, feature_types.keys(), target)

            # Prepare input features
            X, categorical_features, present_feature_count = prepare_features(
                input_df, feature_types
            )

            # Load trained model
            model = load_model(model_name, args.model_dir, target)

            # Predict probabilities
            probs = run_prediction(model_name, model, X, categorical_features)

            # Pull thresholds from JSON for this model
            thresholds = cfg["model"][model_name]["optimal_thresholds"]
            T = thresholds["optimal_threshold_mean"]
            T_low = thresholds["optimal_threshold_ci_low"]
            T_high = thresholds["optimal_threshold_ci_high"]

            # Apply guard-band
            labels, delta = apply_guard_band(probs, T, T_low, T_high)

            # Format clinical outcome name
            target_clean = target.replace("_", " ")

            # Build output dataframe
            df_out = build_output_df(
                input_df["patient_id"],
                target_clean,
                labels,
                probs,
                model_name,
                T,
                T_low,
                T_high,
                delta,
                present_feature_count,
                ideal_feature_count
            )

            results.append(df_out)

    # Combine all outcomes and models
    final_df = pd.concat(results, ignore_index=True)

    # Sort df
    final_df = final_df.sort_values(
        by=["Patient_ID", "Clinical_outcome"],
        kind="mergesort"
    ).reset_index(drop=True)

    # Save
    final_df.to_csv(args.output, sep="\t", index=False)

    print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
