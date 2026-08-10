#!/usr/bin/env python3
"""CORENO Multi-Agent Phase 1B: frozen mechanism-isolation falsification.

Synthetic engineering experiment only. It does not test ontology, consciousness, or liberation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

TOTAL_STEPS = 1500
PRIMARY_START = 1000
PRIMARY_END = 1300  # exclusive
ABSTAIN_COST = 0.25
HC_THRESH = 0.90
ALPHA = 0.05
CONFIRMATORY_SEEDS = 200
SMOKE_SEEDS = 32
BOOTSTRAPS = 10_000

ENVIRONMENTS = [
    "correlation_reversal",
    "minority_correct",
    "high_confidence_trap",
    "repeated_regime",
    "no_shift_null",
]

CONDITIONS = [
    "single",
    "majority",
    "strong",
    "coreno_full",
    "plurality_only",
    "cloned_agents",
    "static_id_permute",
    "dynamic_id_shuffle",
    "profile_shuffle",
    "collapsed_plurality",
    "no_dissent",
    "no_stop",
    "no_reconstruct",
    "random_dissent",
    "random_stop",
    "random_reconstruct",
]

MASKS = [
    (0, 1),       # mixed causal/spurious
    (1, 4),       # strongly spurious/trap sensitive
    (0, 2),       # robust specialist
    (0, 3),       # alternative robust specialist
    (0, 1, 2, 3, 4),  # broad agent
]
CLONE_MASK = (0, 1)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def stable_int(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2**32 - 1)


def feature_accuracy(env: str, t: int) -> Tuple[float, float, float, float, float]:
    """Accuracy of five informative binary features relative to y."""
    if env == "correlation_reversal":
        return (0.70, 0.90 if t < 1000 else 0.10, 0.65, 0.62, 0.72)
    if env == "minority_correct":
        if t < 1000:
            return (0.70, 0.85, 0.68, 0.60, 0.75)
        return (0.68, 0.20, 0.82, 0.78, 0.15)
    if env == "high_confidence_trap":
        if t < 1000:
            return (0.72, 0.92, 0.66, 0.64, 0.90)
        return (0.70, 0.05, 0.68, 0.66, 0.05)
    if env == "repeated_regime":
        if t < 1000:
            spur, trap = 0.90, 0.86
        elif t < 1150:
            spur, trap = 0.10, 0.12
        elif t < 1300:
            spur, trap = 0.88, 0.84
        else:
            spur, trap = 0.12, 0.15
        return (0.70, spur, 0.68, 0.64, trap)
    if env == "no_shift_null":
        return (0.70, 0.88, 0.66, 0.63, 0.78)
    raise ValueError(env)


def generate_environment(seed: int, env: str) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_int("environment", seed, env))
    y = rng.integers(0, 2, size=TOTAL_STEPS, dtype=np.int8)
    x = np.empty((TOTAL_STEPS, 6), dtype=np.float64)
    for t in range(TOTAL_STEPS):
        accs = feature_accuracy(env, t)
        for j, acc in enumerate(accs):
            correct = rng.random() < acc
            bit = int(y[t]) if correct else 1 - int(y[t])
            x[t, j] = 1.0 if bit else -1.0
        x[t, 5] = 1.0 if rng.random() < 0.5 else -1.0
    return x, y.astype(np.float64)


@dataclass
class OnlineLogistic:
    mask: Tuple[int, ...]
    lr: float = 0.055
    l2: float = 0.001

    def __post_init__(self) -> None:
        self.w = np.zeros(len(self.mask) + 1, dtype=np.float64)

    def predict(self, x: np.ndarray) -> float:
        z = self.w[0] + float(np.dot(self.w[1:], x[list(self.mask)]))
        return float(sigmoid(z))

    def update(self, x: np.ndarray, y: float) -> None:
        p = self.predict(x)
        err = p - y
        self.w[0] -= self.lr * err
        self.w[1:] -= self.lr * (err * x[list(self.mask)] + self.l2 * self.w[1:])


def precompute_agent_stream(x: np.ndarray, y: np.ndarray, masks: List[Tuple[int, ...]]) -> np.ndarray:
    agents = [OnlineLogistic(mask=m) for m in masks]
    stream = np.empty((TOTAL_STEPS, len(agents)), dtype=np.float64)
    for t in range(TOTAL_STEPS):
        for j, a in enumerate(agents):
            stream[t, j] = a.predict(x[t])
        for a in agents:
            a.update(x[t], y[t])
    return stream


def bounded_normalize(w: np.ndarray, lo: float = 0.08, hi: float = 0.45) -> np.ndarray:
    w = np.maximum(w, 1e-12)
    w = w / w.sum()
    for _ in range(8):
        w = np.clip(w, lo, hi)
        w = w / w.sum()
    return w


class Integrator:
    def __init__(self, condition: str, seed: int, env: str):
        self.condition = condition
        self.rng = np.random.default_rng(stable_int("controller", seed, env, condition))
        self.w = np.ones(5, dtype=np.float64) / 5.0
        self.err_ewma = 0.20
        self.ref_err = 0.20
        self.eta = 0.42
        self.static_perm = np.random.default_rng(stable_int("static", seed, env)).permutation(5)
        self.profile_perm = np.random.default_rng(stable_int("profile", seed, env)).permutation(5)
        self.random_dissent_rate = 0.10
        self.random_stop_rate = 0.025
        self.random_reconstruct_rate = 0.05

    def transform(self, p: np.ndarray, t: int) -> np.ndarray:
        c = self.condition
        if c == "static_id_permute":
            return p[self.static_perm]
        if c == "dynamic_id_shuffle" and t >= PRIMARY_START:
            return p[self.rng.permutation(5)]
        if c == "profile_shuffle" and t >= PRIMARY_START:
            return p[self.profile_perm]
        return p.copy()

    def flags(self) -> Tuple[bool, bool, bool]:
        c = self.condition
        dissent = c not in {"plurality_only", "no_dissent", "random_dissent"}
        stop = c not in {"plurality_only", "no_stop", "random_stop"}
        reconstruct = c not in {"plurality_only", "no_reconstruct", "random_reconstruct"}
        return dissent, stop, reconstruct

    def predict(self, raw_p: np.ndarray, t: int) -> Tuple[float, bool, Dict[str, float], np.ndarray]:
        c = self.condition
        if c == "single":
            q = raw_p.copy()
            return float(q[0]), False, {"dissent": 0.0, "reconstruct": 0.0}, q
        if c == "majority":
            q = raw_p.copy()
            pred = float(np.mean(q >= 0.5))
            return pred, False, {"dissent": 0.0, "reconstruct": 0.0}, q

        q = self.transform(raw_p, t)

        if c == "collapsed_plurality":
            p = float(np.mean(q))
            return p, False, {"dissent": 0.0, "reconstruct": 0.0}, q

        if c == "strong":
            ww = self.w / self.w.sum()
            p = float(np.dot(ww, q))
            disagreement = float(np.std(q))
            confidence = max(p, 1.0 - p)
            stop = bool((disagreement > 0.30 and confidence < 0.70) or confidence < 0.56)
            return p, stop, {"dissent": 0.0, "reconstruct": 0.0}, q

        dissent_on, stop_on, reconstruct_on = self.flags()
        ww = bounded_normalize(self.w)
        p0 = float(np.dot(ww, q))
        side = q >= 0.5
        majority_side = p0 >= 0.5
        dissent_mask = side != majority_side
        dissent_applied = False

        if c == "random_dissent":
            dissent_on = self.rng.random() < self.random_dissent_rate
        if dissent_on and np.any(dissent_mask) and np.any(~dissent_mask):
            d_rel = float(np.mean(ww[dissent_mask]))
            m_rel = float(np.mean(ww[~dissent_mask]))
            d_conf = float(np.mean(np.abs(q[dissent_mask] - 0.5)))
            if d_rel > m_rel + 0.025 and d_conf > 0.16:
                p_d = float(np.average(q[dissent_mask], weights=ww[dissent_mask]))
                p = 0.65 * p0 + 0.35 * p_d
                dissent_applied = True
            else:
                p = p0
        else:
            p = p0

        disagreement = float(np.std(q))
        confidence = max(p, 1.0 - p)
        stop = False
        if c == "random_stop":
            stop = bool(self.rng.random() < self.random_stop_rate)
        elif stop_on:
            stop = bool((disagreement > 0.33 and confidence < 0.72) or confidence < 0.55)

        reconstruct_now = False
        if c == "random_reconstruct":
            reconstruct_now = bool(self.rng.random() < self.random_reconstruct_rate)
        elif reconstruct_on:
            reconstruct_now = bool(t >= PRIMARY_START and self.err_ewma > self.ref_err + 0.16)

        return p, stop, {"dissent": float(dissent_applied), "reconstruct": float(reconstruct_now)}, q

    def update(self, presented_p: np.ndarray, y: float, pred_p: float, stopped: bool, meta: Dict[str, float], t: int) -> None:
        if self.condition in {"single", "majority", "collapsed_plurality"}:
            return
        losses = ((presented_p >= 0.5) != bool(y)).astype(np.float64)
        self.w *= np.exp(-self.eta * losses)
        self.w = np.maximum(self.w, 1e-10)
        self.w /= self.w.sum()

        wrong = float((pred_p >= 0.5) != bool(y))
        self.err_ewma = 0.94 * self.err_ewma + 0.06 * wrong
        if t < 800:
            self.ref_err = 0.995 * self.ref_err + 0.005 * wrong

        if self.condition == "strong":
            if t >= PRIMARY_START and self.err_ewma > self.ref_err + 0.18:
                self.w = 0.70 * self.w + 0.30 * (np.ones(5) / 5.0)
        elif meta.get("reconstruct", 0.0) > 0.5:
            self.w = 0.75 * self.w + 0.25 * (np.ones(5) / 5.0)
        self.w /= self.w.sum()


def evaluate_condition(
    condition: str,
    base_stream: np.ndarray,
    clone_stream: np.ndarray,
    y: np.ndarray,
    seed: int,
    env: str,
) -> Dict[str, float]:
    stream = clone_stream if condition == "cloned_agents" else base_stream
    controller_condition = "coreno_full" if condition == "cloned_agents" else condition
    integ = Integrator(controller_condition, seed, env)

    losses: List[float] = []
    stops: List[float] = []
    hcw: List[float] = []
    dissents: List[float] = []
    reconstructs: List[float] = []

    for t in range(TOTAL_STEPS):
        raw_p = stream[t]
        p, stop, meta, presented = integ.predict(raw_p, t)
        wrong = float((p >= 0.5) != bool(y[t]))
        conf = max(p, 1.0 - p)
        step_loss = ABSTAIN_COST if stop else wrong
        if PRIMARY_START <= t < PRIMARY_END:
            losses.append(step_loss)
            stops.append(float(stop))
            hcw.append(float((not stop) and wrong > 0.5 and conf >= HC_THRESH))
            dissents.append(meta.get("dissent", 0.0))
            reconstructs.append(meta.get("reconstruct", 0.0))
        integ.update(presented, y[t], p, stop, meta, t)

    return {
        "seed": seed,
        "environment": env,
        "condition": condition,
        "primary_loss": float(np.mean(losses)),
        "stop_rate": float(np.mean(stops)),
        "high_conf_wrong": float(np.mean(hcw)),
        "dissent_rate": float(np.mean(dissents)),
        "reconstruct_rate": float(np.mean(reconstructs)),
        "agent_forward_budget": int(TOTAL_STEPS * 5),
        "agent_update_budget": int(TOTAL_STEPS * 5),
    }


def run_seed_env(seed: int, env: str) -> List[Dict[str, float]]:
    x, y = generate_environment(seed, env)
    base_stream = precompute_agent_stream(x, y, MASKS)
    clone_stream = precompute_agent_stream(x, y, [CLONE_MASK] * 5)
    return [evaluate_condition(c, base_stream, clone_stream, y, seed, env) for c in CONDITIONS]


def cluster_bootstrap_ratio(df: pd.DataFrame, num: str, den: str, n_boot: int, rng: np.random.Generator) -> Dict[str, float]:
    piv = df.pivot_table(index=["seed", "environment"], columns="condition", values="primary_loss").reset_index()
    seeds = np.array(sorted(piv["seed"].unique()))
    point = float(piv[num].mean() / piv[den].mean())
    vals = np.empty(n_boot, dtype=np.float64)
    grouped = {s: piv[piv["seed"] == s] for s in seeds}
    for b in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        n_sum = 0.0
        d_sum = 0.0
        count = 0
        for s in sampled:
            g = grouped[int(s)]
            n_sum += float(g[num].sum())
            d_sum += float(g[den].sum())
            count += len(g)
        vals[b] = (n_sum / count) / (d_sum / count)
    lo, hi = np.quantile(vals, [ALPHA / 2, 1 - ALPHA / 2])
    return {"point": point, "lci95": float(lo), "uci95": float(hi)}


def cluster_bootstrap_delta(df: pd.DataFrame, metric: str, a: str, b: str, n_boot: int, rng: np.random.Generator) -> Dict[str, float]:
    piv = df.pivot_table(index=["seed", "environment"], columns="condition", values=metric).reset_index()
    seeds = np.array(sorted(piv["seed"].unique()))
    point = float(piv[a].mean() - piv[b].mean())
    vals = np.empty(n_boot, dtype=np.float64)
    grouped = {s: piv[piv["seed"] == s] for s in seeds}
    for k in range(n_boot):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        ds = []
        for s in sampled:
            g = grouped[int(s)]
            ds.extend((g[a] - g[b]).tolist())
        vals[k] = float(np.mean(ds))
    lo, hi = np.quantile(vals, [ALPHA / 2, 1 - ALPHA / 2])
    return {"point": point, "lci95": float(lo), "uci95": float(hi)}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["environment", "condition"], as_index=False)
        .agg(
            primary_loss=("primary_loss", "mean"),
            primary_loss_sd=("primary_loss", "std"),
            stop_rate=("stop_rate", "mean"),
            high_conf_wrong=("high_conf_wrong", "mean"),
            dissent_rate=("dissent_rate", "mean"),
            reconstruct_rate=("reconstruct_rate", "mean"),
        )
    )


def decision(df: pd.DataFrame, n_boot: int) -> Dict[str, object]:
    rng = np.random.default_rng(20260811)
    global_q = cluster_bootstrap_ratio(df, "coreno_full", "strong", n_boot, rng)
    identity_q = cluster_bootstrap_ratio(df, "coreno_full", "dynamic_id_shuffle", n_boot, rng)
    static_q = cluster_bootstrap_ratio(df, "static_id_permute", "coreno_full", n_boot, rng)
    hcw_delta = cluster_bootstrap_delta(df, "high_conf_wrong", "coreno_full", "strong", n_boot, rng)
    stop_delta = cluster_bootstrap_delta(df, "stop_rate", "coreno_full", "strong", n_boot, rng)

    static_equiv = static_q["lci95"] >= 0.98 and static_q["uci95"] <= 1.02
    safety = hcw_delta["uci95"] <= 0.005 and stop_delta["uci95"] <= 0.02
    if not static_equiv:
        mechanism = "INVALID_OR_ID_LABEL_SENSITIVE"
    elif identity_q["uci95"] <= 0.95 and safety:
        mechanism = "SUPPORTED"
    elif identity_q["lci95"] >= 1.00:
        mechanism = "FALSIFIED"
    else:
        mechanism = "INCONCLUSIVE"

    if global_q["uci95"] <= 0.90 and safety:
        global_status = "STRONG_SUPPORT"
    elif global_q["uci95"] < 1.00 and safety:
        global_status = "REPLICATED_SUPERIORITY"
    elif global_q["lci95"] >= 1.00:
        global_status = "FALSIFIED"
    else:
        global_status = "INCONCLUSIVE"

    env_secondary: Dict[str, object] = {}
    contrasts = {
        "minority_correct": ("coreno_full", "no_dissent"),
        "high_confidence_trap": ("coreno_full", "no_stop"),
        "repeated_regime": ("coreno_full", "no_reconstruct"),
        "no_shift_null": ("coreno_full", "strong"),
    }
    for env, (a, b) in contrasts.items():
        sub = df[df["environment"] == env]
        env_secondary[env] = cluster_bootstrap_ratio(sub, a, b, n_boot, rng)

    return {
        "phase": "CORENO Multi-Agent Phase 1B",
        "mechanism_status": mechanism,
        "global_status": global_status,
        "global_Q_full_over_strong": global_q,
        "identity_Q_full_over_dynamic_shuffle": identity_q,
        "static_id_Q_over_full": static_q,
        "static_id_equivalence": static_equiv,
        "safety_pass": safety,
        "high_conf_wrong_delta_full_minus_strong": hcw_delta,
        "stop_rate_delta_full_minus_strong": stop_delta,
        "secondary_environment_contrasts": env_secondary,
        "rules": {
            "identity_support": "UCI95(full/dynamic_id_shuffle) <= 0.95",
            "identity_falsified": "LCI95(full/dynamic_id_shuffle) >= 1.00",
            "static_id_equivalence": "entire 95% CI within [0.98, 1.02]",
            "global_strong_support": "UCI95(full/strong) <= 0.90 plus safety",
            "global_replicated_superiority": "UCI95(full/strong) < 1.00 plus safety",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "confirmatory"], required=True)
    ap.add_argument("--outdir", default="results/phase1b")
    ap.add_argument("--smoke-seeds", type=int, default=SMOKE_SEEDS)
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "confirmatory":
        seeds = CONFIRMATORY_SEEDS
        n_boot = BOOTSTRAPS if args.bootstrap is None else args.bootstrap
        if seeds != 200:
            raise RuntimeError("Frozen confirmatory seed count changed")
    else:
        seeds = args.smoke_seeds
        n_boot = 1000 if args.bootstrap is None else args.bootstrap

    rows: List[Dict[str, float]] = []
    for seed in range(seeds):
        for env in ENVIRONMENTS:
            rows.extend(run_seed_env(seed, env))

    df = pd.DataFrame(rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"phase1b_{args.mode}_raw.csv"
    summary_path = out / f"phase1b_{args.mode}_summary.csv"
    decision_path = out / f"phase1b_{args.mode}_decision.json"
    df.to_csv(raw_path, index=False)
    summarize(df).to_csv(summary_path, index=False)
    dec = decision(df, n_boot)
    dec["mode"] = args.mode
    dec["seeds"] = seeds
    dec["bootstrap_resamples"] = n_boot
    decision_path.write_text(json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
