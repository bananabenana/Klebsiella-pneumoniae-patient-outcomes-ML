#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd


# === Helper functions ===


# Options
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare reduced matrix + config for v3 hyperparameter optimisation"
    )
    parser.add_argument("--matrix", required=True, help="Input feature matrix TSV")
    parser.add_argument("--config", required=True, help="Outcome config JSON (v2)")
    parser.add_argument("--top_n_features", type=int, required=True,
                        help="Default top-N features per outcome/model")
    parser.add_argument("--top_n_exceptions", required=False,
                        help="TSV with columns: Top_n_features,Model,Outcome")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def load_exceptions(path):
    """
    Returns dict keyed by (outcome, model) -> top_n
    """
    if path is None:
        return {}

    df = pd.read_csv(path, sep="\t")
    required = {"Top_n_features", "Model", "Outcome"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"exceptions.tsv must contain columns: {required}"
        )

    exceptions = {}
    for _, row in df.iterrows():
        key = (row["Outcome"], row["Model"])
        exceptions[key] = int(row["Top_n_features"])

    return exceptions


def subset_config(config, default_top_n, exceptions):
    """
    Returns:
      - new_config (features truncated per outcome/model)
      - features_to_keep (set of all feature names needed)
    """
    new_config = {}
    features_to_keep = set()

    for outcome, outcome_cfg in config.items():
        if "model" not in outcome_cfg:
            raise ValueError(
                f"Config missing 'model' block for outcome: {outcome}"
            )

        # Preserve exclude_predictors verbatim
        exclude_predictors = outcome_cfg.get("exclude_predictors", [])

        new_config[outcome] = {
            "model": {},
            "exclude_predictors": exclude_predictors,
        }

        for model_name, model_cfg in outcome_cfg["model"].items():
            if "features" not in model_cfg:
                raise ValueError(
                    f"Missing features for outcome={outcome}, model={model_name}"
                )

            feature_types = model_cfg["features"]
            ordered_features = list(feature_types.keys())

            n = exceptions.get(
                (outcome, model_name),
                default_top_n
            )

            selected_features = ordered_features[:n]

            # Safety: never re-include excluded predictors
            selected_features = [
                f for f in selected_features
                if f not in exclude_predictors
            ]

            new_feature_types = {
                f: feature_types[f]
                for f in selected_features
            }

            new_config[outcome]["model"][model_name] = {
                "best_params": {},   # intentionally empty for v3
                "features": new_feature_types,
            }

            features_to_keep.update(selected_features)

    return new_config, features_to_keep


# Main
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load inputs
    df = pd.read_csv(args.matrix, sep="\t")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    exceptions = load_exceptions(args.top_n_exceptions)

    # Subset config + determine needed features
    new_config, features_to_keep = subset_config(
        config=config,
        default_top_n=args.top_n_features,
        exceptions=exceptions,
    )

    # Search for missing features that are in the config but absent in the matrix (can be due to naming mismatches)
    missing_features = sorted(
        f for f in features_to_keep
        if f not in df.columns
    )

    if missing_features:
        raise ValueError(
            "The following features are present in the config "
            "but missing from the matrix:\n"
            + "\n".join(missing_features[:20])
            + (
                f"\n... ({len(missing_features)} total)"
                if len(missing_features) > 20 else ""
            )
        )

    # Ensure all outcomes and also row label (Infecting_isolate) are retained
    outcome_cols = set(new_config.keys())
    mandatory_cols = outcome_cols.union({"Infecting_isolate", "PopPUNK_cluster"})

    matrix_cols_to_keep = [
        c for c in df.columns
        if c in features_to_keep or c in mandatory_cols
    ]

    reduced_df = df[matrix_cols_to_keep].copy()

    # Write outputs
    matrix_out = os.path.join(
        args.outdir,
        "merged_patient_genome_feature_matrix_v3.tsv",
    )
    config_out = os.path.join(
        args.outdir,
        "outcome_config_v3.json",
    )

    reduced_df.to_csv(matrix_out, sep="\t", index=False)

    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=4)

    print("v3 prep complete")
    print(f"Reduced matrix: {matrix_out}")
    print(f"Reduced config: {config_out}")


if __name__ == "__main__":
    main()
