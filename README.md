# Foragers World RL

A small JAX project building an RL environment for a population of foragers on a
confined grid.

Foragers move, compete for food, starve and breed. The agent's job is to hold the
population inside a target band for as long as possible, and its only lever is a
global movement cost — raise it and foragers burn energy faster and die off, lower
it and they survive and breed.

**Built end-to-end in JAX:** pure functional `env_step`, `vmap` over the
population and over parallel worlds, `lax.scan` over episode steps, `jit`
throughout. 256 worlds x 500 steps runs in about 1.5 seconds on CPU.

## Setup

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Running

```bash
uv run python src/foragers_world_rl/world.py   # one episode, fixed action
uv run python tests/test_world.py              # environment tests
uv run python tests/test_ppo.py                # GAE / policy tests
uv run python tests/test_experiments.py        # experiment metrics

uv run foragers-world-rl                       # list policies
uv run foragers-world-rl random                # score a policy over 256 worlds
uv run foragers-world-rl all                   # rank every policy

uv run foragers-ppo                            # train PPO (~2 min on CPU), then score it
uv run foragers-ppo --load runs/ppo_params.pkl --record runs/ppo.json
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
grid, foragers sized by energy, a population/food chart with the target band,
the move cost the policy chose at each step, and a scrubber. The
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
| **PPO, greedy** (5M steps, ~2 min CPU) | **496** | 0% |
| PPO, sampled | 490 | 0% |
| best fixed action (cost 1.50) | 298 | 0% |
| second-best fixed (cost 1.25) | 247 | 0% |
| uniform random | 38 | 94% |
| anything above cost 2.0 | 0 | 100% |

Reproduce the baselines with `uv run foragers-world-rl all` and PPO with
`uv run foragers-ppo`. Every row is scored by the same `evaluate` (seed 0).

A learned policy has to beat **298** to show it is reacting to the observation
rather than settling on one action. The population oscillates under every fixed
action, so there is real headroom between 298 and 500.

PPO closes nearly all of it: it learns a reactive controller that switches
between costs 1.0 and 1.75 depending on the observation, and the few steps it
misses are the start of the episode, while the population grows from 16 into
the band. In other words, this version of the task is solved, and the
interesting work now is making it harder (below).

## PPO

`ppo.py` follows [PureJaxRL](https://github.com/luchris429/purejaxrl)'s
structure: one jitted `update` collects 128 steps from 256 worlds under
`scan`/`vmap`, computes GAE with a reverse `scan`, and runs 4 epochs of
minibatch PPO, all on device. The network is a hand-written MLP (a dict of
arrays) and the optimiser is optax. Two choices worth knowing about:

- **Truncation is not death.** Extinction is terminal, but hitting the 500-step
  limit bootstraps from the value of the final observation: the observation has
  no clock in it, so the critic cannot see an end coming.
- **Rewards are scaled by `1 - gamma`** during training, so value targets sit in
  [0, 1] and PureJaxRL's value-clipping range of 0.2 is sensible.

## Experiments

Experiments are driven by TOML files in `experiments/`, so each one has a
fixed, versioned config:

```bash
uv run foragers-experiment experiments/band_sweep.toml              # train + evaluate
uv run foragers-experiment experiments/band_sweep.toml --skip-train # re-evaluate saved params
```

To try something new, copy a config rather than editing it, so existing
results keep a matching config. Each run writes to the config's `output_dir`:

| file | contents |
| --- | --- |
| `config.toml` | the exact config that was run |
| `meta.json` | resolved PPO settings, env constants, git commit, library versions |
| `git.diff` | uncommitted changes, if the working tree was dirty |
| `summary.csv` | one row per (band, policy), including every fixed action |
| `episodes.csv` | one row per (band, policy, world): return, steps in/out of band, steps to enter, max overshoot/undershoot after entry, extinction, action stats |
| `training/<band>.csv` | PPO learning curve, one row per update |
| `raw/<band>__<policy>.npz` | per-step population, action, reward, done, live mask and observation for every evaluated world |
| `params/<band>.pkl` | trained weights |
| `episodes/*.json` | selected episodes in the viewer's format |

The recorded episodes are the same worlds that were scored, not lookalikes
(`record(world=i)` reuses `evaluate`'s keys), so an episode shown alongside
the numbers is one of the episodes behind them. To browse them:

```bash
uv run python -m http.server
# localhost:8000/viewer/?dir=../results/band_sweep/episodes/
```

Figures and web exports for a finished experiment:

```bash
uv run --group plots foragers-plots results/band_sweep        # figures/ + figures/data/*.csv
uv run python src/foragers_world_rl/export_web.py results/band_sweep/web \
    --experiment results/band_sweep                            # compact episodes for a website
```

`foragers-plots` writes every figure in light and dark, as PNG and SVG, plus a
CSV of exactly what each one plots. `export_web.py --experiment` converts the
recorded episodes into the compact web format (about 880 KB each, from 3-5 MB)
with the per-step action and reward added, and writes an `index.json` with each
episode's metrics. `episodes/` and `web/` are gitignored, since both can be
regenerated from the committed params.

### Band sweep

`experiments/band_sweep.toml` trains PPO separately on four target bands
(wide 80-140, tight 75-85, floor 0-30, ceiling 120-154) and compares greedy
and sampled PPO against the best fixed action and random play on the same 256
worlds.

## Not built yet

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
