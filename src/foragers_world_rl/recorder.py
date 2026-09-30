"""Record a simulation run to JSON so the browser viewer can replay it.

Deliberately kept OUTSIDE the simulation. Everything here uses Python loops,
lists and dicts, which would break jit/vmap/scan if it lived in the step. That
separation is the point: the sim stays compilable, and recording is ordinary
Python that runs between steps.

    from foragers_world_rl.recorder import Recorder

    rec = Recorder(grid_size=GRID_SIZE, max_population=MAX_POPULATION)
    for i in range(steps):
        ...                          # advance the sim
        rec.snapshot(world, foragers)
    rec.save("runs/my_run.json")

Only ALIVE foragers are written per frame, so the file stays small when the
population is sparse.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class Recorder:
    """Accumulates per-step snapshots of a run, then dumps them to JSON."""

    def __init__(
        self,
        grid_size: int,
        max_population: int,
        params: dict[str, Any] | None = None,
        label: str | None = None,
        food_decimals: int = 2,
    ) -> None:
        self.grid_size = grid_size
        self.max_population = max_population
        self.params = dict(params or {})
        self.label = label
        self.food_decimals = food_decimals
        self.frames: list[dict[str, Any]] = []
        self._prev_food: np.ndarray | None = None

    # --- recording -----------------------------------------------------------
    def snapshot(self, world, foragers) -> None:
        """Append one frame. Call once per simulation step.

        Converts JAX arrays to plain Python via numpy. That forces the
        computation to complete, so recording every step of a long run is
        slower than not recording -- fine for producing a trajectory to look
        at, but do not leave it on when benchmarking steps/sec.
        """
        alive = np.asarray(foragers.alive)
        pos = np.asarray(foragers.pos)
        energy = np.asarray(foragers.energy)
        food = np.asarray(world.food, dtype=float)

        idx = np.flatnonzero(alive)  # only serialise the living
        food = np.round(food, self.food_decimals)

        frame: dict[str, Any] = {
            "step": len(self.frames),
            "population": int(alive.sum()),
            "total_food": round(float(food.sum()), self.food_decimals),
            "mean_energy": round(float(energy[idx].mean()) if idx.size else 0.0, 3),
            # One entry per living forager. `slot` is its index in the
            # fixed-size arrays, which lets the viewer track an individual
            # across frames.
            "foragers": [
                {
                    "slot": int(s),
                    "row": int(pos[s][0]),
                    "col": int(pos[s][1]),
                    "energy": round(float(energy[s]), 3),
                }
                for s in idx
            ],
        }

        # Food is delta-encoded. On a 100x100 grid only a handful of cells change
        # per step, so writing the whole grid every frame is ~96% waste. Frame 0
        # carries the full grid ("food"); later frames carry only the cells that
        # changed ("food_delta" as flat [row, col, value] triples) and the viewer
        # reconstructs by applying them in order.
        if self._prev_food is None:
            # Row-major grid of food amounts: food[row][col]
            frame["food"] = food.tolist()
        else:
            changed = np.argwhere(food != self._prev_food)
            frame["food_delta"] = [
                [int(r), int(c), float(food[r, c])] for r, c in changed
            ]
        self._prev_food = food

        self.frames.append(frame)

    # --- output --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        pops = [f["population"] for f in self.frames]
        return {
            "format": "foragers-world-rl/trajectory",
            "version": 1,
            "label": self.label,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "grid_size": self.grid_size,
            "max_population": self.max_population,
            "params": self.params,
            "summary": {
                "steps": len(self.frames),
                "peak_population": max(pops) if pops else 0,
                "final_population": pops[-1] if pops else 0,
                "extinct": bool(pops and pops[-1] == 0),
            },
            "frames": self.frames,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path
