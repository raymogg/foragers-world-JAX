"""Export compact trajectories for the website's side-by-side viewer.

    uv run python src/foragers_world_rl/export_web.py path/to/site/runs
    uv run python src/foragers_world_rl/export_web.py path/to/site/runs \
        --experiment results/band_sweep

The repo's Recorder writes a debug-grade trajectory: per-forager slots, float
energies, full food deltas. That is the right thing for the local viewer, which
needs to show individual foragers and scrub frame by frame, but it is ~20x more
than three small canvases on a web page need.

This writes food as one base64 byte per cell and foragers as flat cell indices,
which takes a 3.3 MB set of runs down to about 570 KB.

--experiment converts the episodes an experiment recorded (its episodes/ dir)
instead of simulating new ones. Converting rather than re-running guarantees the
web file is exactly the scored world. Those files add, on top of the format
below: `actions` and `rewards` (one per step, so len(frames) - 1), `moveCosts`
(action index -> cost), and band/policy/world metadata. `action`/`moveCost` are
null since the action varies per step. An index.json lists every exported
episode with its metrics from episodes.csv.
"""
import argparse
import base64
import csv
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jax, numpy as np
import foragers_world_rl.world as W


def pack(grid):
    """Food grid -> base64 of one byte per cell.

    Food is an integer 0..MAX_FOOD_PER_TILE, so a byte each is plenty. JSON
    numbers cost ~6 bytes per cell; base64 costs ~1.33. The grid is noisy, so
    run-length encoding barely compresses it -- 785 runs for 1024 cells.
    """
    return base64.b64encode(grid.astype(np.uint8).tobytes()).decode("ascii")


def export(action, steps, seed=0, regrowth=None):
    if regrowth is not None:
        W.FOOD_REGROWTH_RATE = regrowth
    key = jax.random.key(seed)
    key, ik = jax.random.split(key)
    state = W.init_env_state(ik)
    frames = []

    def snap(st):
        food = np.asarray(st.world.food).astype(np.int8)
        alive = np.asarray(st.foragers.alive)
        pos = np.asarray(st.foragers.pos)[alive]
        return {
            "f": pack(food.reshape(-1)),
            # flat cell index per living forager
            "p": [int(r) * W.GRID_SIZE + int(c) for r, c in pos],
            "n": int(alive.sum()),
        }

    frames.append(snap(state))
    rewards = 0.0
    for _ in range(steps):
        key, sk = jax.random.split(key)
        state, _obs, r, d = W.env_step(state, action, sk)
        rewards += float(r)
        frames.append(snap(state))
        if bool(d):
            break

    return {
        "grid": W.GRID_SIZE,
        "maxPop": W.MAX_POPULATION,
        "maxFood": int(W.MAX_FOOD_PER_TILE),
        "band": [W.POP_BAND_LOW, W.POP_BAND_HIGH],
        "action": action,
        "moveCost": float(W.MOVE_COSTS[action]),
        "regrowth": W.FOOD_REGROWTH_RATE,
        "episodeSteps": W.EPISODE_STEPS,
        "return": int(rewards),
        "frames": frames,
    }


def convert_recording(rec: dict) -> dict:
    """Viewer-format recording (Recorder.to_dict) -> compact web format."""
    p = rec["params"]
    n = rec["grid_size"]
    frames, actions, rewards = [], [], []
    food = None
    for fr in rec["frames"]:
        # Food is delta-encoded in the recording: full grid on frame 0 only.
        if "food" in fr:
            food = np.asarray(fr["food"], dtype=float)
        else:
            food = food.copy()
            for r, c, v in fr.get("food_delta", []):
                food[r, c] = v
        frames.append({
            "f": pack(food.reshape(-1)),
            "p": [a["row"] * n + a["col"] for a in fr["foragers"]],
            "n": fr["population"],
        })
        if "action" in fr:
            actions.append(fr["action"])
            rewards.append(int(fr["reward"]))
    return {
        "grid": n,
        "maxPop": rec["max_population"],
        "maxFood": int(p["max_food_per_tile"]),
        "band": p["pop_band"],
        "action": None,
        "moveCost": None,
        "moveCosts": p["move_costs"],
        "regrowth": p["food_regrowth"],
        "episodeSteps": p["episode_steps"],
        "return": sum(rewards),
        "actions": actions,
        "rewards": rewards,
        "frames": frames,
    }


def export_experiment(results_dir: Path, out_dir: Path) -> None:
    meta = json.loads((results_dir / "meta.json").read_text())
    with (results_dir / "episodes.csv").open() as f:
        metrics = {(r["band"], r["policy"], int(r["world"])): r for r in csv.DictReader(f)}

    index = []
    for ep in meta["recorded"]:
        rec = json.loads((results_dir / ep["file"]).read_text())
        d = convert_recording(rec)
        name = f"{ep['band']}__{ep['policy']}__{ep['which']}"
        m = metrics[(ep["band"], ep["policy"], ep["world"])]
        d.update({
            "experiment": meta["name"],
            "bandName": ep["band"],
            "policy": ep["policy"],
            "policyDetail": m["policy_detail"],
            "which": ep["which"],
            "world": ep["world"],
        })
        assert d["return"] == float(m["return"]), f"{name}: return mismatch with episodes.csv"
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(d, separators=(",", ":")))
        index.append({"file": path.name, **{k: d[k] for k in
                      ("bandName", "band", "policy", "policyDetail", "which", "world", "return")},
                      "metrics": m})
        print(f"{name:>32}: {len(d['frames']):>4} frames, return {d['return']:>3}, "
              f"{path.stat().st_size / 1024:>6.0f} KB")
    (out_dir / "index.json").write_text(json.dumps(
        {"experiment": meta["name"], "bands": meta["bands"],
         "bestFixed": meta["best_fixed"], "episodes": index}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--experiment", type=Path,
                    help="experiment output dir; convert its recorded episodes")
    a = ap.parse_args()
    out_dir = a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.experiment:
        export_experiment(a.experiment, out_dir)
        sys.exit(0)
    # (name, action, steps, regrowth). Boom-bust needs a leaner world: at the
    # default regrowth the population just sits high, so the crash never comes.
    runs = [
        ("dying", 8, 40, None),
        ("stable", 6, 150, None),
        ("boombust", 4, 150, 0.12),
    ]
    for name, action, steps, regrowth in runs:
        d = export(action, steps, regrowth=regrowth)
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(d, separators=(",", ":")))
        print(f"{name:>9}: {len(d['frames']):>4} frames, return {d['return']:>3}, "
              f"{p.stat().st_size / 1024:>6.0f} KB")
