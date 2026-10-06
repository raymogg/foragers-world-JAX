"""Run a simulation and write its trajectory to JSON for the web viewer.

    uv run python src/foragers_world_rl/record_run.py
    uv run python src/foragers_world_rl/record_run.py --steps 200 --seed 3 --out runs/seed3.json

The loop here mirrors world.py's main block: it drives env_step and snapshots
the state it returns. The step order lives in env_step, not here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run directly as a script, so this works
# without a PYTHONPATH prefix.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax

from foragers_world_rl.recorder import Recorder
from foragers_world_rl.world import (
    BREED_ENERGY,
    EPISODE_STEPS,
    FOOD_REGROWTH_RATE,
    GRID_SIZE,
    INITIAL_POPULATION,
    MAX_FOOD_PER_TILE,
    MAX_POPULATION,
    MOVE_COSTS,
    POP_BAND_HIGH,
    POP_BAND_LOW,
    env_step,
    init_env_state,
)


def record(
    steps: int, seed: int, out: Path, stop_when_extinct: bool = True, action: int = 0
) -> Path:
    key = jax.random.key(seed)

    key, init_key = jax.random.split(key)
    state = init_env_state(init_key)

    rec = Recorder(
        grid_size=GRID_SIZE,
        max_population=MAX_POPULATION,
        label=f"seed {seed}",
        params={
            "seed": seed,
            "grid_size": GRID_SIZE,
            "max_population": MAX_POPULATION,
            "initial_population": INITIAL_POPULATION,
            "breed_energy": BREED_ENERGY,
            "food_regrowth": FOOD_REGROWTH_RATE,
            "max_food_per_tile": MAX_FOOD_PER_TILE,
            "move_costs": [float(c) for c in MOVE_COSTS],
            "action": action,
            "move_cost": float(MOVE_COSTS[action]),
            "pop_band": [POP_BAND_LOW, POP_BAND_HIGH],
            "episode_steps": EPISODE_STEPS,
        },
    )
    # frame 0 = the initial state
    rec.snapshot(state.world, state.foragers)

    # No agent yet, so the action is held fixed for the whole run.
    total_reward = 0.0
    for _ in range(steps):
        key, step_key = jax.random.split(key)
        state, _obs, reward, done = env_step(state, action, step_key)
        total_reward += float(reward)

        rec.snapshot(state.world, state.foragers)

        if stop_when_extinct and bool(done):
            break

    in_band_steps = int(total_reward)
    print(
        f"reward: {total_reward:.0f} "
        f"({in_band_steps}/{len(rec.frames) - 1} steps in band "
        f"{POP_BAND_LOW}-{POP_BAND_HIGH})"
    )

    path = rec.save(out)
    s = rec.to_dict()["summary"]
    print(
        f"wrote {path}  "
        f"({s['steps']} frames, peak pop {s['peak_population']}, "
        f"final {s['final_population']}"
        f"{', extinct' if s['extinct'] else ''})"
    )
    print(f"size: {path.stat().st_size / 1024:.1f} KB")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--action", type=int, default=0, help="fixed action index (MOVE_COSTS)"
    )
    ap.add_argument("--out", type=Path, default=Path("runs/run.json"))
    ap.add_argument(
        "--run-to-end",
        action="store_true",
        help="keep going after extinction instead of stopping",
    )
    a = ap.parse_args()
    record(
        a.steps, a.seed, a.out, stop_when_extinct=not a.run_to_end, action=a.action
    )
