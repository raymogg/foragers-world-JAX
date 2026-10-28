"""Export compact trajectories for the website's side-by-side viewer.

    uv run python src/foragers_world_rl/export_web.py path/to/site/runs

The repo's Recorder writes a debug-grade trajectory: per-forager slots, float
energies, full food deltas. That is the right thing for the local viewer, which
needs to show individual foragers and scrub frame by frame, but it is ~20x more
than three small canvases on a web page need.

This writes food as one base64 byte per cell and foragers as flat cell indices,
which takes a 3.3 MB set of runs down to about 570 KB.
"""
import base64
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


if __name__ == "__main__":
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
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
