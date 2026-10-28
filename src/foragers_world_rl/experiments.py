"""Run a config-driven experiment: train PPO per target band, evaluate, dump raw data.

    uv run foragers-experiment experiments/band_sweep.toml
    uv run foragers-experiment experiments/band_sweep.toml --skip-train   # reuse saved params
    uv run foragers-experiment experiments/band_sweep.toml --force        # overwrite output

Everything lands in the config's `output_dir`:

    config.toml               the exact config file that was run
    meta.json                 resolved PPO config, env constants, git commit, versions
    git.diff                  uncommitted changes, if the working tree was dirty
    summary.csv               one row per (band, policy), every fixed action included
    episodes.csv              one row per (band, policy, world): the per-episode metrics
    training/<band>.csv       PPO learning curve, one row per update
    params/<band>.pkl         trained weights, {"params": ..., "band": ...}
    raw/<band>__<policy>.npz  per-step arrays for every evaluated world (see RAW DATA)
    episodes/*.json           selected episodes in the web viewer's format

RAW DATA. Each .npz holds arrays of shape (worlds, steps):

    pops[w, t]     population AFTER step t (the initial population is 16)
    actions[w, t]  action index chosen at step t; move cost = move_costs[action]
    rewards[w, t]  reward for step t (1 if pops[w, t] is in band)
    dones[w, t]    episode ended at step t
    live[w, t]     step t is part of the episode (False after extinction)
    obs[w, t, :]   observation the action at step t was chosen from

plus `band`, `move_costs`, and `world_seed`. Steps after an episode ends are
still in the arrays (scan has a fixed length) but `live` is False for them, and
every metric ignores them.

Env constants (grid size, breeding, regrowth...) are module constants in
world.py, not config: they are snapshotted into meta.json so a result can be
matched to the environment that produced it.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Make the package importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from foragers_world_rl import world as W
from foragers_world_rl.agent import (
    N_ACTIONS,
    batched_rollout,
    fixed_policy,
    random_policy,
    record,
)
from foragers_world_rl.ppo import CONFIG as PPO_DEFAULTS
from foragers_world_rl.ppo import ppo_greedy_policy, ppo_policy, train

MOVE_COSTS = np.asarray(W.MOVE_COSTS)


# ---------------------------------------------------------------------- metrics


def episode_metrics(pops, actions, rewards, dones, band,
                    initial_pop: int = W.INITIAL_POPULATION) -> dict[str, np.ndarray]:
    """Per-episode metrics from (worlds, steps) arrays. All numpy, one value per world.

    Overshoot and undershoot are measured only AFTER the population first
    enters the band, so the initial climb from 16 does not count as an
    undershoot. They are NaN for episodes that never entered the band.

    `pops` holds the population after each step, so the initial state is not
    in it; it is passed separately because a band can contain it (floor band:
    16 is in 0-30). Entry is then step 0, and an early overshoot counts.
    steps_to_enter is 0 if the episode started in band, t + 1 if it entered
    after step t, and -1 if it never did.
    """
    pops = np.asarray(pops, dtype=np.int64)
    actions = np.asarray(actions)
    dones = np.asarray(dones, dtype=bool)
    worlds, steps = pops.shape
    low, high = band
    t = np.arange(steps)[None, :]

    # Episode length: up to and including the first done.
    ended = dones.any(axis=1)
    length = np.where(ended, dones.argmax(axis=1) + 1, steps)
    live = t < length[:, None]

    in_band = (pops >= low) & (pops <= high) & live
    started_in = low <= initial_pop <= high
    entered = in_band.any(axis=1) | started_in
    steps_to_enter = np.where(started_in, 0,
                              np.where(entered, in_band.argmax(axis=1) + 1, -1))
    # pops[:, t] is the state after step t, i.e. state index t + 1.
    after_entry = live & entered[:, None] & (t + 1 >= steps_to_enter[:, None])

    above = after_entry & (pops > high)
    below = after_entry & (pops < low)
    overshoot = np.where(above, pops - high, 0).max(axis=1).astype(float)
    undershoot = np.where(below, low - pops, 0).max(axis=1).astype(float)
    overshoot[~entered] = np.nan
    undershoot[~entered] = np.nan

    extinct = ((pops == 0) & live).any(axis=1)
    extinct_step = np.where(extinct, ((pops == 0) & live).argmax(axis=1), -1)

    costs = MOVE_COSTS[actions]
    n_live = live.sum(axis=1)
    mean_cost = np.where(live, costs, 0).sum(axis=1) / n_live
    # Action changes between consecutive live steps.
    switches = ((actions[:, 1:] != actions[:, :-1]) & live[:, 1:]).sum(axis=1)

    ret = np.where(live, rewards, 0).sum(axis=1)

    return {
        "return": ret,
        "length": length,
        "steps_in_band": in_band.sum(axis=1),
        "steps_out_of_band": n_live - in_band.sum(axis=1),
        "frac_in_band": in_band.sum(axis=1) / steps,
        "steps_to_enter": steps_to_enter,
        "steps_above_after_entry": above.sum(axis=1),
        "steps_below_after_entry": below.sum(axis=1),
        "max_overshoot": overshoot,
        "max_undershoot": undershoot,
        "peak_pop": np.where(live, pops, 0).max(axis=1),
        # -1 if the band was never entered
        "min_pop_after_entry": np.where(
            entered, np.where(after_entry, pops, np.iinfo(np.int64).max).min(axis=1), -1),
        "final_pop": pops[np.arange(worlds), length - 1],
        "extinct": extinct,
        "extinct_step": extinct_step,
        "mean_cost": mean_cost,
        "action_switches": switches,
        **{f"n_action_{a}": (live & (actions == a)).sum(axis=1) for a in range(N_ACTIONS)},
    }


def summarise(m: dict[str, np.ndarray]) -> dict:
    """Aggregate per-episode metrics into one summary row."""
    r = m["return"]
    entered = m["steps_to_enter"] >= 0
    nanmean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
    nanmax = lambda x: float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")
    return {
        "mean_return": float(r.mean()),
        "std_return": float(r.std()),
        "min_return": float(r.min()),
        "median_return": float(np.median(r)),
        "max_return": float(r.max()),
        "mean_frac_in_band": float(m["frac_in_band"].mean()),
        "extinct_frac": float(m["extinct"].mean()),
        "entered_frac": float(entered.mean()),
        "mean_steps_to_enter": nanmean(np.where(entered, m["steps_to_enter"], np.nan)),
        "mean_max_overshoot": nanmean(m["max_overshoot"]),
        "worst_overshoot": nanmax(m["max_overshoot"]),
        "mean_max_undershoot": nanmean(m["max_undershoot"]),
        "worst_undershoot": nanmax(m["max_undershoot"]),
        "mean_steps_above": float(m["steps_above_after_entry"].mean()),
        "mean_steps_below": float(m["steps_below_after_entry"].mean()),
        "mean_cost": float(m["mean_cost"].mean()),
        "mean_action_switches": float(m["action_switches"].mean()),
    }


def pick_world(returns: np.ndarray, which: str) -> int:
    """World index for 'median', 'worst' or 'best' by return (lowest index on ties)."""
    if which == "worst":
        return int(np.argmin(returns))
    if which == "best":
        return int(np.argmax(returns))
    if which == "median":
        return int(np.argmin(np.abs(returns - np.median(returns))))
    raise ValueError(f"unknown episode selector {which!r}")


# ---------------------------------------------------------------------- running


def run_policy(policy_fn, params, band, seed, worlds) -> dict[str, np.ndarray]:
    keys = jax.random.split(jax.random.key(seed), worlds)
    traj = batched_rollout(policy_fn, params, keys, jnp.asarray(band))
    return {k: np.asarray(v) for k, v in traj.items()}


def save_raw(path: Path, traj, band, seed, policy):
    m_live = episode_metrics(traj["pops"], traj["actions"], traj["rewards"], traj["dones"], band)
    t = np.arange(traj["pops"].shape[1])[None, :]
    np.savez_compressed(
        path,
        pops=traj["pops"].astype(np.int16),
        actions=traj["actions"].astype(np.int8),
        rewards=traj["rewards"].astype(np.int8),
        dones=traj["dones"].astype(bool),
        live=t < m_live["length"][:, None],
        obs=traj["obs"].astype(np.float32),
        band=np.asarray(band),
        move_costs=MOVE_COSTS,
        world_seed=seed,
        policy=policy,
    )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "diff": git("diff", "HEAD")}


def env_constants() -> dict:
    return {
        "grid_size": W.GRID_SIZE,
        "max_population": W.MAX_POPULATION,
        "initial_population": W.INITIAL_POPULATION,
        "episode_steps": W.EPISODE_STEPS,
        "breed_energy": W.BREED_ENERGY,
        "max_food_per_tile": W.MAX_FOOD_PER_TILE,
        "food_regrowth_rate": W.FOOD_REGROWTH_RATE,
        "move_costs": [float(c) for c in MOVE_COSTS],
    }


def run(config_path: Path, skip_train: bool, force: bool):
    cfg = tomllib.loads(config_path.read_text())
    out = Path(cfg["output_dir"])
    bands = {name: tuple(b) for name, b in cfg["bands"].items()}
    ppo_cfg_base = {**PPO_DEFAULTS, **cfg.get("train", {}).get("ppo", {})}
    unknown = set(cfg.get("train", {}).get("ppo", {})) - set(PPO_DEFAULTS)
    if unknown:
        raise SystemExit(f"unknown ppo settings in config: {sorted(unknown)}")
    train_seed = cfg.get("train", {}).get("seed", 0)
    ev = cfg.get("eval", {})
    eval_seed, worlds = ev.get("seed", 0), ev.get("worlds", 256)
    policies = ev.get("policies", ["ppo_greedy", "ppo_sampled", "best_fixed", "random"])

    if out.exists() and not skip_train:
        if not force:
            raise SystemExit(f"{out} exists; pass --force to overwrite, or --skip-train "
                             "to re-evaluate with its saved params")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out / "config.toml")

    git = git_info()
    # Uncommitted changes to tracked files are saved, so a run made from a
    # dirty tree can still be reproduced: check out the commit, apply the diff.
    if git["dirty"] and git["diff"]:
        (out / "git.diff").write_text(git["diff"] + "\n")
    meta = {
        "name": cfg.get("name"),
        "description": cfg.get("description", "").strip(),
        "config_file": str(config_path),
        "git": {"commit": git["commit"], "dirty": git["dirty"],
                "diff_file": "git.diff" if git["dirty"] else None},
        "versions": {"jax": jax.__version__, "optax": optax.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0]},
        "env": env_constants(),
        "bands": {k: list(v) for k, v in bands.items()},
        "train": {"seed": train_seed, "skipped": skip_train,
                  "ppo": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in ppo_cfg_base.items() if k != "band"}},
        "eval": {"seed": eval_seed, "worlds": worlds, "policies": policies},
        "best_fixed": {},
        "recorded": [],
    }

    summary_rows, episode_rows = [], []
    for band_name, band in bands.items():
        print(f"\n=== band {band_name} {band[0]}-{band[1]}")

        # --- train (or load) ------------------------------------------------
        params_path = out / "params" / f"{band_name}.pkl"
        if skip_train:
            params = pickle.loads(params_path.read_bytes())["params"]
        else:
            params, history = train({**ppo_cfg_base, "band": band}, train_seed)
            params_path.parent.mkdir(parents=True, exist_ok=True)
            params_path.write_bytes(pickle.dumps({"params": jax.device_get(params),
                                                  "band": band}))
            write_csv(out / "training" / f"{band_name}.csv", [
                {**{k: v for k, v in h.items() if k != "action_hist"},
                 **{f"n_action_{a}": n for a, n in enumerate(h["action_hist"])}}
                for h in history
            ])

        # --- evaluate -------------------------------------------------------
        # Every fixed action goes in the summary; the best one is also kept raw.
        candidates = {f"fixed:{a}": (fixed_policy, a) for a in range(N_ACTIONS)}
        candidates.update({
            "random": (random_policy, None),
            "ppo_greedy": (ppo_greedy_policy, params),
            "ppo_sampled": (ppo_policy, params),
        })
        results = {}
        for name, (fn, p) in candidates.items():
            traj = run_policy(fn, p, band, eval_seed, worlds)
            m = episode_metrics(traj["pops"], traj["actions"], traj["rewards"],
                                traj["dones"], band)
            results[name] = (traj, m)

        best = max((n for n in results if n.startswith("fixed:")),
                   key=lambda n: results[n][1]["return"].mean())
        meta["best_fixed"][band_name] = best
        results["best_fixed"] = results[best]
        print(f"best fixed action: {best} "
              f"(cost {MOVE_COSTS[int(best.split(':')[1])]:g})")

        for name, (traj, m) in results.items():
            if name == "best_fixed":
                continue  # already in the table under its fixed:N name
            s = summarise(m)
            summary_rows.append({"band": band_name, "band_low": band[0], "band_high": band[1],
                                 "policy": name, "is_best_fixed": name == best,
                                 "worlds": worlds, **s})
            if name.startswith("fixed:") and name != best:
                continue
            print(f"  {name:>12}  return {s['mean_return']:6.1f} ± {s['std_return']:5.1f}"
                  f"  in band {100 * s['mean_frac_in_band']:3.0f}%"
                  f"  extinct {100 * s['extinct_frac']:3.0f}%"
                  f"  over {s['mean_max_overshoot']:5.1f}  under {s['mean_max_undershoot']:5.1f}")

        for policy in policies:
            traj, m = results[policy]
            (out / "raw").mkdir(exist_ok=True)
            save_raw(out / "raw" / f"{band_name}__{policy}.npz", traj, band, eval_seed,
                     best if policy == "best_fixed" else policy)
            for w in range(worlds):
                episode_rows.append({
                    "band": band_name, "band_low": band[0], "band_high": band[1],
                    "policy": policy,
                    "policy_detail": best if policy == "best_fixed" else policy,
                    "world": w,
                    **{k: (v[w].item() if hasattr(v[w], "item") else v[w])
                       for k, v in m.items()},
                })

        # --- record selected episodes for the viewer ------------------------
        for spec in cfg.get("record", []):
            policy, which = spec["policy"], spec["which"]
            fn, p = candidates[best if policy == "best_fixed" else policy]
            w = pick_world(results[policy][1]["return"], which)
            path = out / "episodes" / f"{band_name}__{policy}__{which}.json"
            label = f"{band_name} {band[0]}-{band[1]} · {policy} · {which} (world {w})"
            record(fn, p, eval_seed, path, label, band=band, world=w, worlds=worlds)
            meta["recorded"].append({"band": band_name, "policy": policy, "which": which,
                                     "world": w, "file": str(path.relative_to(out))})

    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "episodes.csv", episode_rows)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path, help="experiment TOML, e.g. experiments/band_sweep.toml")
    ap.add_argument("--skip-train", action="store_true",
                    help="load params from the output dir instead of training")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    a = ap.parse_args()
    run(a.config, a.skip_train, a.force)


if __name__ == "__main__":
    main()
