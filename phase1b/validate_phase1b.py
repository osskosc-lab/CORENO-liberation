#!/usr/bin/env python3
"""Independent structural validator for frozen CORENO Phase 1B outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_ENVS = {
    "correlation_reversal",
    "minority_correct",
    "high_confidence_trap",
    "repeated_regime",
    "no_shift_null",
}
EXPECTED_CONDITIONS = {
    "single", "majority", "strong", "coreno_full", "plurality_only",
    "cloned_agents", "static_id_permute", "dynamic_id_shuffle", "profile_shuffle",
    "collapsed_plurality", "no_dissent", "no_stop", "no_reconstruct",
    "random_dissent", "random_stop", "random_reconstruct",
}


def fail(msg: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--mode", choices=["smoke", "confirmatory"], required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    dec = json.loads(Path(args.decision).read_text(encoding="utf-8"))

    expected_seeds = 200 if args.mode == "confirmatory" else int(dec["seeds"])
    if set(df["environment"].unique()) != EXPECTED_ENVS:
        fail("environment set differs from preregistration")
    if set(df["condition"].unique()) != EXPECTED_CONDITIONS:
        fail("condition set differs from preregistration")
    if df["seed"].nunique() != expected_seeds:
        fail(f"expected {expected_seeds} seeds, found {df['seed'].nunique()}")

    expected_rows = expected_seeds * len(EXPECTED_ENVS) * len(EXPECTED_CONDITIONS)
    if len(df) != expected_rows:
        fail(f"expected {expected_rows} rows, found {len(df)}")
    if df.duplicated(["seed", "environment", "condition"]).any():
        fail("duplicate seed/environment/condition rows")
    if df[["primary_loss", "stop_rate", "high_conf_wrong"]].isna().any().any():
        fail("NaN in primary outputs")
    for col in ["primary_loss", "stop_rate", "high_conf_wrong"]:
        if ((df[col] < 0) | (df[col] > 1)).any():
            fail(f"{col} outside [0,1]")

    if df["agent_forward_budget"].nunique() != 1 or int(df["agent_forward_budget"].iloc[0]) != 7500:
        fail("agent forward budget mismatch")
    if df["agent_update_budget"].nunique() != 1 or int(df["agent_update_budget"].iloc[0]) != 7500:
        fail("agent update budget mismatch")

    required_decision_keys = {
        "mechanism_status", "global_status", "global_Q_full_over_strong",
        "identity_Q_full_over_dynamic_shuffle", "static_id_Q_over_full",
        "static_id_equivalence", "safety_pass",
    }
    if not required_decision_keys.issubset(dec):
        fail("decision.json missing required keys")
    if args.mode == "confirmatory" and int(dec.get("seeds", -1)) != 200:
        fail("confirmatory decision does not record 200 seeds")

    # Exact relabeling invariance is expected up to floating-point noise because the
    # static permutation is applied from t=0 while all controller operations are symmetric.
    full = df[df.condition == "coreno_full"].sort_values(["seed", "environment"])["primary_loss"].to_numpy()
    static = df[df.condition == "static_id_permute"].sort_values(["seed", "environment"])["primary_loss"].to_numpy()
    max_abs = float(abs(full - static).max())
    if max_abs > 1e-12:
        fail(f"static ID relabeling changed primary loss (max abs diff={max_abs})")

    print(json.dumps({
        "validation": "PASS",
        "mode": args.mode,
        "rows": len(df),
        "seeds": expected_seeds,
        "static_id_max_abs_loss_diff": max_abs,
        "mechanism_status": dec["mechanism_status"],
        "global_status": dec["global_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
