# Foragers World RL

A small JAX project building an RL environment for a population of foragers on a
confined grid.

Foragers move, compete for food, starve and breed. The agent's job is to hold the
population inside a target band for as long as possible, and its only lever is a
global movement cost — raise it and foragers burn energy faster and die off, lower
it and they survive and breed.

**Built end-to-end in JAX:** pure functional `env_step`, `vmap` over the
population and over parallel worlds, `lax.scan` over episode steps, `jit`
throughout. 256 worlds x 500 steps runs in about a second.

## Setup

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running

```bash
uv run python src/foragers_world_rl/world.py   # one episode, fixed action
uv run python tests/test_world.py              # the test suite

uv run foragers-world-rl                       # list policies
uv run foragers-world-rl random                # score a policy over 256 worlds
uv run foragers-world-rl all                   # rank every policy
```

`agent.py` holds the policies and the shared rollout. A policy is a pure
function `(params, obs, key) -> action`, so adding one means writing that
function and registering it in `POLICIES` — the rollout, scoring and recording
are shared.

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
only the cells that changed. `runs/sample.json` is checked in as an example.

`--worlds` and `--record` are separate paths: `--worlds N` runs N episodes in
parallel under `scan`/`vmap` to get a score, while `--record` replays one episode
eagerly to produce a watchable trajectory. Recording cannot go inside `scan`,
since the recorder builds Python lists.

## The environment

- Foragers random-walk on a 32x32 grid, paying energy per step
- They share a tile's food evenly between everyone standing on it
- They starve at zero energy and breed asexually, splitting energy with the child
- Food regrows stochastically (Bernoulli per cell) up to a per-tile cap, which is
  what gives the world a carrying capacity instead of a slow death
- Population oscillates around that capacity, so it drifts out of the target band
  on its own and the agent has to keep pulling it back

**Observation** (3 floats, all scaled to roughly 0-1 so no channel dominates by
magnitude): total food, population, mean energy.
**Action:** one of 12 movement costs, 0.0 to 2.75.
**Reward:** +1 per step the population is inside the band.
**Episode:** 500 steps, ending early on extinction.

## Baselines

Mean return over 256 parallel worlds, 500 steps, out of a possible 500:

| policy | return | extinct |
| --- | --- | --- |
| best fixed action (cost 1.50) | 298 | 0% |
| second-best fixed (cost 1.25) | 247 | 0% |
| uniform random | 38 | 94% |
| anything above cost 2.0 | 0 | 100% |

Reproduce with `uv run foragers-world-rl all`.

A learned policy has to beat **298** to show it is reacting to the observation
rather than settling on one action. The population oscillates under every fixed
action, so there is real headroom between 298 and 500.

## Not built yet

- **PPO.** The policy interface and the batched rollout are in place; the
  learning is not. This is the next piece of work.
- Heritable traits (speed, vision) with mutation
- Spatial actions — poison or plant food in one region rather than a global knob.
  `World.poison` exists as a field and is carried through untouched; nothing
  writes to it yet.
- Per-region observations (the global summary is deliberately the simplest thing
  that works)

## AGENTS.txt

This repo ships a second artifact: `AGENTS.txt` is a coaching script that lets an
AI agent guide someone else through building this environment from scratch. It
records the JAX concepts in dependency order and — more usefully — the bugs that
produce no error message.

## License

MIT
