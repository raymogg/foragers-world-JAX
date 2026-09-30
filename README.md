# Foragers World RL

A small JAX project building an RL environment for a population of foragers on a
confined grid.

Foragers move, compete for food, breed, and (later) evolve. The RL agent's goal is
to keep the population inside a target band using food and poison — while the
population adapts to its interventions, making poison progressively less
effective.

**Built end-to-end in JAX:** pure functional `step()`, `vmap` over the population
and over parallel worlds, `jit`-compiled throughout.

## Running

```bash
.venv/bin/python src/foragers_world_rl/world.py   # run the simulation
.venv/bin/python tests/test_world.py              # run the test suite
```

## Viewing a run

Record a trajectory, then replay it in the browser:

```bash
PYTHONPATH=src .venv/bin/python src/foragers_world_rl/record_run.py \
    --steps 200 --seed 0 --out runs/run.json

python3 -m http.server          # then open localhost:8000/viewer/
```

The viewer is a single static HTML file with no build step. It shows the food
grid, foragers sized by energy, a population/food chart, and a scrubber — or
drag any trajectory JSON onto the page. `runs/sample.json` is checked in as an
example.

## Current feature set

- Foragers move (random walk, energy cost per step)
- Foragers eat, sharing a tile's food evenly between everyone standing on it
- Foragers starve when energy hits zero
- Foragers breed asexually, splitting their energy with the offspring
- Property-based test suite (17 tests, no pytest needed)

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
