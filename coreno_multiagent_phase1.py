"""CORENO Multi-Agent Phase 1 Falsification.

Frozen Phase-1 simulation.  The treatment is a controller, not a better agent set.
Run `python coreno_multiagent_phase1.py --help` for options.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


N_AGENTS = 5
T = 3000
PRIMARY_START, PRIMARY_END = 1000, 1300
STOP_COST = 0.25
WRONG_COST = 1.0
FEATURE_MASKS = np.array(
    [
        [1, 1, 1, 1],  # generalist
        [1, 0, 1, 1],  # causal specialist; disadvantaged before the shift
        [0, 1, 1, 1],  # spurious specialist
        [1, 1, 0, 1],  # mixed specialist
        [1, 0, 1, 0],  # causal/context specialist
    ],
    dtype=float,
)
CONDITIONS = [
    "single",
    "majority",
    "strong",
    "coreno_full",
    "coreno_no_plurality",
    "coreno_no_dissent",
    "coreno_no_stop",
    "coreno_no_reconstruct",
    "coreno_history_shuffle",
    "coreno_agent_id_shuffle",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_episode(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create one frozen environment with two spurious-correlation reversals."""
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array([-1.0, 1.0]), size=T)
    spurious_acc = np.full(T, 0.90)
    spurious_acc[1000:1500] = 0.10
    spurious_acc[2500:] = 0.10
    def noisy_copy(acc: np.ndarray | float) -> np.ndarray:
        a = np.full(T, acc) if np.isscalar(acc) else acc
        return y * np.where(rng.random(T) < a, 1.0, -1.0)
    z = noisy_copy(0.70)
    s = noisy_copy(spurious_acc)
    # Context is weakly useful but signals the first and second regime boundary.
    c = noisy_copy(0.58)
    c[1000:1035] = 1.0
    c[2500:2535] = 1.0
    n = rng.choice(np.array([-1.0, 1.0]), size=T)
    return np.column_stack([z, s, c, n]), y


@dataclass
class Agents:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def new(cls) -> "Agents":
        return cls(weights=np.zeros((N_AGENTS, 4)), bias=np.zeros(N_AGENTS))

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = (self.weights * FEATURE_MASKS) @ x + self.bias
        return sigmoid(logits)

    def update(self, x: np.ndarray, y: float) -> None:
        target = (y + 1.0) / 2.0
        p = self.predict(x)
        grad = p - target
        # Identical update count and learning rate for every condition.
        self.weights -= 0.075 * grad[:, None] * (FEATURE_MASKS * x)
        self.bias -= 0.075 * grad


def normalized(v: np.ndarray, floor: float = 0.0) -> np.ndarray:
    v = np.maximum(v, 1e-12)
    v = v / v.sum()
    if floor <= 0:
        return v
    if floor * len(v) >= 1:
        raise ValueError("invalid plural-weight floor")
    return floor + (1.0 - floor * len(v)) * v


def entropy(p: np.ndarray) -> float:
    q = np.clip(p, 1e-8, 1 - 1e-8)
    return float(np.mean(-(q * np.log(q) + (1 - q) * np.log(1 - q))) / np.log(2))


def bootstrap_ratio(full: np.ndarray, baseline: np.ndarray, b: int, seed: int = 20260809) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(full)
    idx = rng.integers(0, n, size=(b, n))
    ratios = full[idx].mean(axis=1) / baseline[idx].mean(axis=1)
    return float(full.mean() / baseline.mean()), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def config_for(condition: str) -> Dict[str, bool]:
    return {
        "plurality": condition != "coreno_no_plurality",
        "dissent": condition != "coreno_no_dissent",
        "stop": condition != "coreno_no_stop",
        "reconstruct": condition != "coreno_no_reconstruct",
        "history_shuffle": condition == "coreno_history_shuffle",
        "id_shuffle": condition == "coreno_agent_id_shuffle",
    }


def run_condition(seed: int, condition: str) -> Dict[str, float]:
    x_all, y_all = make_episode(seed)
    rng = np.random.default_rng(seed + 99173 + CONDITIONS.index(condition))
    agents = Agents.new()
    ensemble_w = np.full(N_AGENTS, 1 / N_AGENTS)
    rolling_error: List[float] = []
    agent_loss_history: List[np.ndarray] = []
    losses, primary_losses = [], []
    stop_count = highconf_wrong = highconf_total = reconstruct_count = 0
    cooldown = 0
    cfg = config_for(condition)

    for t, (x, y) in enumerate(zip(x_all, y_all)):
        probs = agents.predict(x)
        confs = np.maximum(probs, 1 - probs)
        votes = np.where(probs >= 0.5, 1.0, -1.0)
        p_ens = float(np.dot(ensemble_w, probs))
        pred = 1.0 if p_ens >= 0.5 else -1.0
        unc = entropy(probs)
        recent_error = float(np.mean(rolling_error[-20:])) if rolling_error else 0.0
        surprise = recent_error > 0.42
        majority = 1.0 if np.mean(votes) >= 0 else -1.0
        minority = votes != majority
        dissent = bool(minority.any() and np.max(confs[minority]) >= 0.67)

        if condition == "single":
            pred, p_ens, dissent, surprise = votes[0], float(probs[0]), False, False
        elif condition == "majority":
            pred, p_ens, dissent, surprise = majority, float((votes.mean() + 1) / 2), False, False

        if condition in ("strong",):
            # Strong alternative: confident-weighted Hedge + detector + costed abstention.
            stop = bool((unc > 0.55 or surprise) and (abs(p_ens - 0.5) < 0.23 or surprise))
        elif condition.startswith("coreno"):
            active_dissent = dissent if cfg["dissent"] else False
            stop = bool(cfg["stop"] and ((active_dissent and surprise) or (unc > 0.62 and surprise)))
        else:
            stop = False

        if stop:
            loss = STOP_COST
            stop_count += 1
            action_wrong = 0.0
        else:
            action_wrong = float(pred != y)
            loss = WRONG_COST * action_wrong
            if abs(p_ens - 0.5) >= 0.35:
                highconf_total += 1
                highconf_wrong += int(action_wrong)
        losses.append(loss)
        if PRIMARY_START <= t < PRIMARY_END:
            primary_losses.append(loss)
        rolling_error.append(action_wrong if not stop else 0.0)

        target = (y + 1.0) / 2.0
        agent_loss = -(target * np.log(np.clip(probs, 1e-8, 1)) + (1 - target) * np.log(np.clip(1 - probs, 1e-8, 1)))
        agent_loss_history.append(agent_loss)
        update_loss = rng.permutation(agent_loss) if cfg["id_shuffle"] and condition.startswith("coreno") else agent_loss

        # Baseline and CORENO use exactly the same online-Hedge update.  Rebuild is the
        # only controller-level difference after surprise.
        ensemble_w = normalized(ensemble_w * np.exp(-0.85 * update_loss), 0.025 if condition.startswith("coreno") and cfg["plurality"] else 0.0)
        should_reconstruct = condition in ("strong",) or (condition.startswith("coreno") and cfg["reconstruct"])
        if should_reconstruct and surprise and cooldown == 0:
            if condition == "strong":
                ensemble_w = np.full(N_AGENTS, 1 / N_AGENTS)
            else:
                hist = np.vstack(agent_loss_history[-80:])
                scores = hist.mean(axis=0)
                if cfg["history_shuffle"]:
                    scores = rng.permutation(scores)
                proposed = normalized(np.exp(-1.2 * scores), 0.025 if cfg["plurality"] else 0.0)
                # Dissolve without erasing prior information.
                ensemble_w = normalized(0.55 * ensemble_w + 0.45 * proposed, 0.025 if cfg["plurality"] else 0.0)
            reconstruct_count += 1
            cooldown = 28
        cooldown = max(cooldown - 1, 0)
        agents.update(x, y)

    return {
        "seed": seed,
        "condition": condition,
        "primary_loss": float(np.mean(primary_losses)),
        "full_loss": float(np.mean(losses)),
        "stop_rate": stop_count / T,
        "highconf_wrong_rate": highconf_wrong / max(highconf_total, 1),
        "reconstruct_count": reconstruct_count,
    }


def run_job(job: tuple[int, str]) -> Dict[str, float]:
    """Pickle-safe wrapper for independent seed/condition evaluations."""
    return run_condition(*job)


def write_figures(data: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    ordered = [c for c in CONDITIONS if c in summary.condition.values]
    s = summary.set_index("condition").loc[ordered].reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(s)), s.primary_loss_mean, yerr=[s.primary_loss_mean-s.primary_loss_ci_low, s.primary_loss_ci_high-s.primary_loss_mean], capsize=3, color=["#85929e" if not c.startswith("coreno") else "#2e86c1" for c in s.condition])
    ax.set_xticks(range(len(s)), [c.replace("coreno_", "C3-") for c in s.condition], rotation=35, ha="right")
    ax.set_ylabel("Primary post-shift loss (mean, seed CI)")
    ax.set_title("CORENO Multi-Agent Phase 1: frozen primary window")
    fig.tight_layout(); fig.savefig(outdir / "primary_loss_by_condition.png", dpi=180); plt.close(fig)
    paired = data.pivot(index="seed", columns="condition", values="primary_loss")
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(paired["strong"], paired["coreno_full"], alpha=0.65, s=20, color="#2e86c1")
    lim = [0, max(paired["strong"].max(), paired["coreno_full"].max()) * 1.05]
    ax.plot(lim, lim, "--", color="#566573", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Strong baseline primary loss"); ax.set_ylabel("CORENO full primary loss")
    ax.set_title("Paired seed outcome")
    fig.tight_layout(); fig.savefig(outdir / "paired_primary_loss.png", dpi=180); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--workers", type=int, default=1, help="Independent seed workers; does not alter results.")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, condition) for seed in range(args.seeds) for condition in CONDITIONS]
    if args.workers == 1:
        rows = [run_condition(seed, condition) for seed, condition in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(run_job, jobs, chunksize=4))
    data = pd.DataFrame(rows); data.to_csv(outdir / "condition_seed_results.csv", index=False)
    summary_rows = []
    rng = np.random.default_rng(20260809)
    for condition, grp in data.groupby("condition", sort=False):
        values = grp.primary_loss.to_numpy()
        sample = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
        summary_rows.append({
            "condition": condition, "n_seeds": len(values),
            "primary_loss_mean": values.mean(), "primary_loss_ci_low": np.quantile(sample, .025), "primary_loss_ci_high": np.quantile(sample, .975),
            "full_loss_mean": grp.full_loss.mean(), "stop_rate_mean": grp.stop_rate.mean(),
            "highconf_wrong_rate_mean": grp.highconf_wrong_rate.mean(), "reconstruct_mean": grp.reconstruct_count.mean(),
        })
    summary = pd.DataFrame(summary_rows); summary.to_csv(outdir / "condition_summary.csv", index=False)
    paired = data.pivot(index="seed", columns="condition", values="primary_loss")
    q, q_low, q_high = bootstrap_ratio(paired.coreno_full.to_numpy(), paired.strong.to_numpy(), args.bootstrap)
    safety_delta = float((data[data.condition == "coreno_full"].highconf_wrong_rate.mean() - data[data.condition == "strong"].highconf_wrong_rate.mean()))
    stop_rate = float(data[data.condition == "coreno_full"].stop_rate.mean())
    if q_high <= 0.90 and safety_delta <= 0.01 and stop_rate <= 0.35:
        verdict = "SUPPORTED"
    elif q_low >= 1.00:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"
    decision = {
        "protocol": "CORENO Multi-Agent Phase 1 Falsification v1.0",
        "seeds": args.seeds, "bootstrap_replicates": args.bootstrap,
        "primary_window": [PRIMARY_START, PRIMARY_END - 1], "stop_cost": STOP_COST,
        "q_coreno_over_strong": q, "q_ci95_low": q_low, "q_ci95_high": q_high,
        "highconf_wrong_rate_delta": safety_delta, "coreno_stop_rate": stop_rate,
        "verdict": verdict,
        "support_rule": "UCI95(Q)<=0.90 AND highconf_delta<=0.01 AND stop_rate<=0.35",
        "falsification_rule": "LCI95(Q)>=1.00",
    }
    with open(outdir / "decision.json", "w", encoding="utf-8") as f: json.dump(decision, f, indent=2)
    write_figures(data, summary, outdir)
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nPrimary comparison:", json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
