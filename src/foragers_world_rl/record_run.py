"""Run a simulation and write its trajectory to JSON for the web viewer.

    .venv/bin/python src/foragers_world_rl/record_run.py
    .venv/bin/python src/foragers_world_rl/record_run.py --steps 200 --seed 3 --out runs/seed3.json

The step order here mirrors world.py's main block: move -> count -> eat ->
clear food -> breed, with the snapshot taken at the end of each step.
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
    GRID_SIZE,
    INITIAL_POPULATION,
    MAX_POPULATION,
    World,
    breed,
    eat_all,
    init_foragers,
    step_all,
    zero_all,
)


def record(steps: int, seed: int, out: Path, stop_when_extinct: bool = True) -> Path:
    key = jax.random.key(seed)

    key, food_key = jax.random.split(key)
    initial_food = jax.random.randint(
        food_key, (GRID_SIZE, GRID_SIZE), 0, 5
    ).astype(float)
    world = World(initial_food)

    key, forager_key = jax.random.split(key)
    foragers = init_foragers(forager_key)

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
            "food_regrowth": False,
        },
    )
    rec.snapshot(world, foragers)  # frame 0 = the initial state

    for _ in range(steps):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)

        per_grid = foragers.per_position_count()
        foragers = eat_all(world, foragers, per_grid)
        world = zero_all(world, per_grid)

        foragers = breed(foragers)

        rec.snapshot(world, foragers)

        if stop_when_extinct and foragers.population == 0:
            break

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
    ap.add_argument("--out", type=Path, default=Path("runs/run.json"))
    ap.add_argument(
        "--run-to-end",
        action="store_true",
        help="keep going after extinction instead of stopping",
    )
    a = ap.parse_args()
    record(a.steps, a.seed, a.out, stop_when_extinct=not a.run_to_end)
