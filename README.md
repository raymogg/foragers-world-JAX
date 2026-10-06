# Foragers World RL

A small JAX project building an RL environment for a population of foragers on a
confined grid.

Foragers move, compete for food, breed, and (later) evolve. The RL agent's goal is
to keep the population inside a target band using food and poison — while the
population adapts to its interventions, making poison progressively less
effective.

**Built end-to-end in JAX:** pure functional `step()`, `vmap` over the population
and over parallel worlds, `jit`-compiled throughout.

## Setup

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running

```bash
uv run python src/foragers_world_rl/world.py   # run the simulation
uv run python tests/test_world.py              # run the test suite
```

## Viewing a run

Record a trajectory, then replay it in the browser:

```bash
uv run python src/foragers_world_rl/agent.py fixed:6 --record runs/run.json

uv run python -m http.server          # then open localhost:8000/viewer/
```

`--record` replays a single episode and snapshots every step. It is separate
from the batched evaluation: `--worlds` runs N episodes in parallel to get a
score, while `--record` writes one watchable trajectory.

The viewer is a single static HTML file with no build step. It shows the food
grid, foragers sized by energy, a population/food chart, and a scrubber. The
dropdown lists whatever is in `runs/`, or you can drag any trajectory JSON onto
the page — that works without the server too. `runs/sample.json` is checked in
as an example.

Food is delta-encoded in the JSON: frame 0 carries the full grid and later frames
only the cells that changed. On a 100×100 grid roughly a dozen of 10,000 cells
change per step, so this is about 17× smaller than storing every frame in full.

## Current feature set

- Foragers move (random walk, energy cost per step)
- Foragers eat, sharing a tile's food evenly between everyone standing on it
- Foragers starve when energy hits zero
- Foragers breed asexually, splitting their energy with the offspring
- Trajectory recording to JSON, plus a browser viewer
- Property-based test suite (20 tests, no pytest needed)

## Coming soon

- Food regrowth (so the population can reach equilibrium rather than boom-bust)
- Heritable traits: speed and vision, with mutation
- Poison, which selects for vision — the agent's control tool degrades as it uses it
- Paired breeding (needed once traits are heritable, for recombination)
- `vmap` over thousands of parallel worlds; `lax.scan` over episodes
- The RL agent (PPO from scratch)
- Predators?

## AGENTS.txt

This repo ships a second artifact: `AGENTS.txt` is a coaching script that lets an
AI agent guide someone else through building this environment from scratch. It
records the JAX concepts in dependency order and — more usefully — the bugs that
produce no error message.
