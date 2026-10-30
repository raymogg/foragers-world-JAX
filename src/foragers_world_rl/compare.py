"""Compare one band across several experiments, as a CSV.

    uv run foragers-compare results/floor_control/comparison.csv --band floor \
        baseline=results/band_sweep \
        growth=results/floor_growth_obs \
        band_scaled=results/floor_band_scaled_pop

Reads each experiment's episodes.csv and raw/ arrays and writes one row per
(experiment, policy) with the metrics the floor-control log reports: returns,
extinctions split by when they happen, and how the fatal step looked. Every
number in results/floor_control_experiments.md's tables should come from this
file, so the log and any website charts are regenerated from stored results
rather than one-off scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Extinctions are bucketed by step: the start-up crash happens in the first ~20.
BUCKETS = [(0, 20), (20, 100), (100, 500)]
POLICIES = ["ppo_greedy", "ppo_sampled", "best_fixed", "random"]


def compare_row(exp_dir: Path, band: str, policy: str) -> dict:
    meta = json.loads((exp_dir / "meta.json").read_text())
    rows = [r for r in csv.DictReader((exp_dir / "episodes.csv").open())
            if r["band"] == band and r["policy"] == policy]
    ret = np.array([float(r["return"]) for r in rows])
    ext = np.array([r["extinct"] == "True" for r in rows])
    es = np.array([int(r["extinct_step"]) for r in rows])

    out = {
        "policy": policy,
        "policy_detail": rows[0]["policy_detail"],
        "obs_features": " ".join(meta["train"]["ppo"].get(
            "obs_features", ["food", "population", "energy"])),
        "worlds": len(rows),
        "mean_return": round(ret.mean(), 1),
        "median_return": float(np.median(ret)),
        "std_return": round(ret.std(), 1),
        "extinct": int(ext.sum()),
        "extinct_frac": round(ext.mean(), 3),
        **{f"extinct_steps_{a}_{b - 1}": int((ext & (es >= a) & (es < b)).sum())
           for a, b in BUCKETS},
        "median_extinct_step": float(np.median(es[ext])) if ext.any() else None,
        "mean_return_survivors": round(ret[~ext].mean(), 1) if (~ext).any() else None,
        "mean_return_extinct": round(ret[ext].mean(), 1) if ext.any() else None,
        "mean_max_overshoot": round(np.nanmean([float(r["max_overshoot"]) for r in rows]), 1),
        "mean_cost": round(np.mean([float(r["mean_cost"]) for r in rows]), 2),
        "mean_action_switches": round(np.mean([float(r["action_switches"]) for r in rows]), 1),
    }

    # How the extinctions after the start-up looked, from the per-step arrays.
    raw_path = exp_dir / "raw" / f"{band}__{policy}.npz"
    if raw_path.exists():
        z = np.load(raw_path)
        names = list(z["obs_feature_names"]) if "obs_feature_names" in z.files \
            else ["food", "population", "energy"]
        pops = z["pops"]
        dead = (pops == 0).any(1)
        step = np.where(dead, (pops == 0).argmax(1), -1)
        late = np.where(dead & (step >= 20))[0]
        if len(late):
            before = np.array([pops[i, step[i] - 1] for i in late])
            fatal_cost = np.array([z["move_costs"][z["actions"][i, step[i]]] for i in late])
            e = names.index("energy")
            energy = np.array([z["obs"][i, step[i], e] for i in late]) * 4.0  # / BREED_ENERGY
            out.update({
                "late_extinctions": len(late),
                "late_from_10_or_more": int((before >= 10).sum()),
                "late_median_pop_before": float(np.median(before)),
                "late_fatal_cost_2_5": int(np.isclose(fatal_cost, 2.5).sum()),
                "late_median_energy_seen": round(float(np.median(energy)), 2),
            })
        alive = z["live"] & (pops > 0)
        out["mean_energy_live_steps"] = round(
            float(z["obs"][..., names.index("energy")][alive].mean()) * 4.0, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("experiments", nargs="+", metavar="LABEL=DIR")
    ap.add_argument("--band", required=True)
    a = ap.parse_args()

    rows = []
    for spec in a.experiments:
        label, d = spec.split("=", 1)
        for policy in POLICIES:
            rows.append({"experiment": label, "results_dir": d, "band": a.band,
                         **compare_row(Path(d), a.band, policy)})
    # Not every row has the late-extinction columns; keep one consistent header.
    fields = list(dict.fromkeys(k for r in rows for k in r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
