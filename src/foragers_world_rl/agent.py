"""Run a policy against the foraging world, and report what it scored.

    uv run python src/foragers_world_rl/agent.py                    # list policies
    uv run python src/foragers_world_rl/agent.py random
    uv run python src/foragers_world_rl/agent.py fixed:6 --worlds 256
    uv run python src/foragers_world_rl/agent.py random --record runs/random.json

Everything except the policy is shared, so a new policy means one function and
one entry in POLICIES -- not a new rollout loop.

The rollout runs under jax.lax.scan and vmaps over worlds, so a sweep of 256
worlds x 500 steps takes about a second. Recording a trajectory for the viewer
is the one thing that cannot go inside scan (it builds Python lists), so
--record takes the slow eager path over a single world.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

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
    observe,
)

N_ACTIONS = len(MOVE_COSTS)


# ---------------------------------------------------------------------- policies
#
# A policy is a pure function:
#
#     policy(params, obs, key) -> action        # () int32 in [0, N_ACTIONS)
#
# Pure and params-in because that is the only shape jax.grad can differentiate,
# so a learned policy drops in here without the rollout changing. `params` is
# whatever that policy needs (None for the stateless ones).


def random_policy(params, obs: jax.Array, key: jax.Array) -> jax.Array:
    """Uniform over the action space. The floor any real policy must beat."""
    return jax.random.randint(key, (), 0, N_ACTIONS)


def fixed_policy(params, obs: jax.Array, key: jax.Array) -> jax.Array:
    """Always the same action. `params` is the action index.

    The ceiling for a non-reactive agent: the best fixed action is what a
    learned policy has to beat to show it is actually using the observation.
    """
    return jnp.asarray(params, dtype=jnp.int32)


# name -> (policy_fn, params)
POLICIES: dict[str, tuple] = {
    "random": (random_policy, None),
    **{f"fixed:{a}": (fixed_policy, a) for a in range(N_ACTIONS)},
}


# ---------------------------------------------------------------------- rollout


def rollout(policy_fn, params, key: jax.Array):
    """One episode under `policy_fn`. Returns the per-step trajectory.

    Observe, act, then step -- in that order, so the obs stored alongside an
    action is the obs that action was chosen from.
    """

    def body(carry, step_key):
        state = carry
        act_key, env_key = jax.random.split(step_key)

        obs = observe(state)
        action = policy_fn(params, obs, act_key)
        state, _obs_after, reward, done = env_step(state, action, env_key)

        return state, (obs, action, reward, done, state.foragers.population)

    init_key, scan_key = jax.random.split(key)
    state = init_env_state(init_key)
    keys = jax.random.split(scan_key, EPISODE_STEPS)
    _final, traj = jax.lax.scan(body, state, keys)

    obs, actions, rewards, dones, pops = traj
    return {
        "obs": obs,            # (STEPS, 3)
        "actions": actions,    # (STEPS,)
        "rewards": rewards,    # (STEPS,)
        "dones": dones,        # (STEPS,)
        "pops": pops,          # (STEPS,)
    }


def evaluate(policy_fn, params, seed: int = 0, worlds: int = 256) -> dict:
    """Run `worlds` independent episodes in parallel and summarise them."""
    batched = jax.jit(jax.vmap(rollout, in_axes=(None, None, 0)), static_argnums=(0,))
    keys = jax.random.split(jax.random.key(seed), worlds)
    traj = batched(policy_fn, params, keys)

    # Reward is 1 per in-band step, so an episode's return is its in-band count.
    returns = traj["rewards"].sum(axis=1)       # (worlds,)
    # scan has a fixed trip count, so a world that dies keeps getting stepped.
    # Check whether population EVER hit zero, not just where it ended -- a dead
    # population stays dead, but reading only the last step would also count a
    # world that merely happened to end low.
    died = (traj["pops"] == 0).any(axis=1)

    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "extinct_frac": float(died.mean()),
        "mean_final_pop": float(traj["pops"][:, -1].mean()),
        "action_hist": jnp.bincount(
            traj["actions"].reshape(-1), length=N_ACTIONS
        ).tolist(),
        "worlds": worlds,
    }


# ---------------------------------------------------------------------- recording


def record(policy_fn, params, seed: int, out: Path, label: str) -> Path:
    """Replay one episode eagerly, snapshotting each step for the web viewer.

    Deliberately outside scan: the Recorder builds Python lists and dicts, so
    it cannot be traced. Slow, but it is one episode.
    """
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = init_env_state(init_key)

    rec = Recorder(
        grid_size=GRID_SIZE,
        max_population=MAX_POPULATION,
        label=label,
        params={
            "policy": label,
            "seed": seed,
            "grid_size": GRID_SIZE,
            "max_population": MAX_POPULATION,
            "initial_population": INITIAL_POPULATION,
            "breed_energy": BREED_ENERGY,
            "food_regrowth": FOOD_REGROWTH_RATE,
            "max_food_per_tile": MAX_FOOD_PER_TILE,
            "move_costs": [float(c) for c in MOVE_COSTS],
            "pop_band": [POP_BAND_LOW, POP_BAND_HIGH],
            "episode_steps": EPISODE_STEPS,
        },
    )
    rec.snapshot(state.world, state.foragers)  # frame 0 = initial state

    total_reward = 0.0
    for _ in range(EPISODE_STEPS):
        key, step_key = jax.random.split(key)
        act_key, env_key = jax.random.split(step_key)

        obs = observe(state)
        action = policy_fn(params, obs, act_key)
        state, _obs, reward, done = env_step(state, action, env_key)

        total_reward += float(reward)
        rec.snapshot(state.world, state.foragers)
        if bool(done):
            break

    path = rec.save(out)
    s = rec.to_dict()["summary"]
    print(
        f"wrote {path} ({s['steps']} frames, peak pop {s['peak_population']}, "
        f"final {s['final_population']}{', extinct' if s['extinct'] else ''})"
    )
    print(f"  return: {total_reward:.0f}/{EPISODE_STEPS}")
    return path


# ---------------------------------------------------------------------- reporting


def report(name: str, stats: dict) -> None:
    print(f"\n{name}  ({stats['worlds']} worlds, {EPISODE_STEPS} steps)")
    print(
        f"  return    {stats['mean_return']:.1f} +/- {stats['std_return']:.1f}"
        f"   (min {stats['min_return']:.0f}, max {stats['max_return']:.0f})"
    )
    print(f"  of a possible {EPISODE_STEPS}"
          f"  -> {100 * stats['mean_return'] / EPISODE_STEPS:.0f}% of steps in band")
    print(f"  extinct   {100 * stats['extinct_frac']:.0f}% of worlds")
    print(f"  final pop {stats['mean_final_pop']:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy", nargs="?", help="policy name, or 'all' to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=256, help="parallel episodes")
    ap.add_argument("--record", type=Path, help="also save a trajectory JSON")
    a = ap.parse_args()

    if a.policy is None:
        print("policies:", ", ".join(POLICIES))
        print(f"\nband {POP_BAND_LOW}-{POP_BAND_HIGH}, {EPISODE_STEPS} steps, "
              f"{N_ACTIONS} actions (move costs "
              f"{float(MOVE_COSTS[0]):g}..{float(MOVE_COSTS[-1]):g})")
        return

    if a.policy == "all":
        rows = []
        for name, (fn, params) in POLICIES.items():
            stats = evaluate(fn, params, seed=a.seed, worlds=a.worlds)
            rows.append((name, stats))
        rows.sort(key=lambda r: -r[1]["mean_return"])
        print(f"{'policy':>12} {'return':>14} {'extinct':>8} {'final pop':>10}")
        for name, s in rows:
            print(
                f"{name:>12} {s['mean_return']:>7.1f} +/-{s['std_return']:>5.1f} "
                f"{100 * s['extinct_frac']:>7.0f}% {s['mean_final_pop']:>10.0f}"
            )
        best = rows[0]
        print(f"\nbest: {best[0]} at {best[1]['mean_return']:.1f}")
        return

    if a.policy not in POLICIES:
        ap.error(f"unknown policy {a.policy!r}; choose from {', '.join(POLICIES)}")

    fn, params = POLICIES[a.policy]
    report(a.policy, evaluate(fn, params, seed=a.seed, worlds=a.worlds))
    if a.record:
        record(fn, params, a.seed, a.record, a.policy)


if __name__ == "__main__":
    main()
