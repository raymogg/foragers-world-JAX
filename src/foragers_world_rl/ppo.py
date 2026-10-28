"""PPO for the foraging world, in the PureJaxRL style.

    uv run python src/foragers_world_rl/ppo.py
    uv run python src/foragers_world_rl/ppo.py --timesteps 10_000_000 --out runs/ppo.pkl
    uv run python src/foragers_world_rl/ppo.py --record runs/ppo.json

Structure follows PureJaxRL's ppo.py (github.com/luchris429/purejaxrl): one
jitted `update` collects a rollout from every world with scan + vmap, computes
GAE with a reverse scan, then runs a few epochs of minibatch PPO -- all on
device. Differences from PureJaxRL:

  - the network is a hand-written MLP (a dict of arrays) instead of Flax, and
    the action distribution is jax.random.categorical instead of distrax
  - updates are driven from a Python loop rather than one giant scan, so
    progress prints as it trains
  - time-limit truncation bootstraps from the value of the final observation
    instead of being treated as death (see `_env_step`)
  - rewards are scaled by (1 - gamma) so value targets sit in [0, 1]

The trained policy has the same `(params, obs, key) -> action` signature as the
baselines in agent.py, so it is scored by the exact same `evaluate`.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Make the package importable when run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import optax

from foragers_world_rl.agent import N_ACTIONS, evaluate, record, report
from foragers_world_rl.world import (
    DEFAULT_BAND,
    EnvState,
    env_step,
    init_env_state,
    observe,
)

OBS_DIM = 3

CONFIG = dict(
    num_envs=256,
    num_steps=128,            # rollout length per update, per world
    total_timesteps=5_000_000,
    update_epochs=4,
    num_minibatches=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    lr=2.5e-4,
    anneal_lr=True,
    hidden=64,
    band=DEFAULT_BAND,        # target population band the reward is for
)


# ---------------------------------------------------------------------- network
#
# Separate actor and critic MLPs, 2 hidden tanh layers each -- PureJaxRL's
# default. PureJaxRL's init gains: sqrt(2) on hidden layers, 0.01 on the policy
# head (starts near-uniform), 1.0 on the value head. Scaled-normal rather than
# its orthogonal init: orthogonal needs a QR, which on CPU loads scipy's LAPACK,
# and the installed scipy build fails to dlopen on this macOS.


def _dense(key, n_in, n_out, scale):
    w = jax.random.normal(key, (n_in, n_out)) * scale / jnp.sqrt(n_in)
    return {"w": w, "b": jnp.zeros(n_out)}


def init_params(key: jax.Array, hidden: int) -> dict:
    k = jax.random.split(key, 6)
    gain = jnp.sqrt(2.0)
    return {
        "actor": [
            _dense(k[0], OBS_DIM, hidden, gain),
            _dense(k[1], hidden, hidden, gain),
            _dense(k[2], hidden, N_ACTIONS, 0.01),
        ],
        "critic": [
            _dense(k[3], OBS_DIM, hidden, gain),
            _dense(k[4], hidden, hidden, gain),
            _dense(k[5], hidden, 1, 1.0),
        ],
    }


def _mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def forward(params: dict, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """obs (..., OBS_DIM) -> logits (..., N_ACTIONS), value (...)."""
    return _mlp(params["actor"], obs), _mlp(params["critic"], obs)[..., 0]


def _log_prob(logits, action):
    logp = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(logp, action[..., None], axis=-1)[..., 0]


def ppo_policy(params, obs: jax.Array, key: jax.Array) -> jax.Array:
    """Sample from the learned policy -- what it was trained as."""
    logits, _ = forward(params, obs)
    return jax.random.categorical(key, logits)


def ppo_greedy_policy(params, obs: jax.Array, key: jax.Array) -> jax.Array:
    """Most likely action. Usually a little better than sampling once trained."""
    logits, _ = forward(params, obs)
    return jnp.argmax(logits)


# ---------------------------------------------------------------------- training


class Transition(NamedTuple):
    obs: jax.Array          # obs the action was chosen from
    action: jax.Array
    log_prob: jax.Array     # under the policy that collected it (PPO's "old")
    value: jax.Array
    reward: jax.Array
    done: jax.Array         # episode ended on this step (either way)
    # gamma * V(final obs) if the episode was cut off by the step limit, else 0.
    # Used in place of the next value when done.
    trunc_bootstrap: jax.Array
    ep_return: jax.Array    # finished episode's return when done, else 0


class Runner(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: EnvState     # batched over worlds
    obs: jax.Array          # (num_envs, OBS_DIM)
    ep_return: jax.Array    # (num_envs,) running return of the current episode
    key: jax.Array


def compute_gae(traj: Transition, last_value, gamma, lam):
    """Advantages and value targets, scanning backwards over time.

    Backwards because A_t = delta_t + gamma*lam*A_{t+1}: each step needs the
    advantage of the step after it. At an episode boundary the chain is cut,
    and the next value is the truncation bootstrap (0 on extinction).
    """

    def body(carry, t):
        gae, next_value = carry
        not_done = 1.0 - t.done
        next_v = jnp.where(t.done, t.trunc_bootstrap, gamma * next_value)
        delta = t.reward + next_v - t.value
        gae = delta + gamma * lam * not_done * gae
        return (gae, t.value), gae

    init = (jnp.zeros_like(last_value), last_value)
    _, adv = jax.lax.scan(body, init, traj, reverse=True)
    return adv, adv + traj.value


def make_train(config: dict):
    c = config
    num_updates = c["total_timesteps"] // (c["num_steps"] * c["num_envs"])
    batch_size = c["num_steps"] * c["num_envs"]
    minibatch_size = batch_size // c["num_minibatches"]
    reward_scale = 1.0 - c["gamma"]
    band = jnp.asarray(c["band"])

    def lr_schedule(count):
        # count is optimiser steps; linear decay to 0 over training.
        frac = 1.0 - count // (c["num_minibatches"] * c["update_epochs"]) / num_updates
        return c["lr"] * frac

    tx = optax.chain(
        optax.clip_by_global_norm(c["max_grad_norm"]),
        optax.adam(lr_schedule if c["anneal_lr"] else c["lr"], eps=1e-5),
    )

    def init(key: jax.Array) -> Runner:
        key, p_key, env_key = jax.random.split(key, 3)
        params = init_params(p_key, c["hidden"])
        env_state = jax.vmap(init_env_state)(jax.random.split(env_key, c["num_envs"]))
        obs = jax.vmap(observe)(env_state)
        return Runner(params, tx.init(params), env_state, obs,
                      jnp.zeros(c["num_envs"]), key)

    def _env_step(params, carry, _):
        env_state, obs, ep_return, key = carry
        key, act_key, step_key, reset_key = jax.random.split(key, 4)

        logits, value = forward(params, obs)
        action = jax.random.categorical(act_key, logits)
        log_prob = _log_prob(logits, action)

        step_keys = jax.random.split(step_key, c["num_envs"])
        next_state, next_obs, reward, done = jax.vmap(env_step, in_axes=(0, 0, 0, None))(
            env_state, action, step_keys, band
        )
        reward = reward * reward_scale

        # Extinction is a real terminal: no future reward. Hitting the step
        # limit is not -- the world would have carried on, and the obs has no
        # clock in it, so the critic cannot know an end is coming. Bootstrap
        # from the final obs instead of treating it as death.
        extinct = next_state.foragers.population == 0
        truncated = done & ~extinct
        _, final_value = forward(params, next_obs)
        trunc_bootstrap = jnp.where(truncated, c["gamma"] * final_value, 0.0)

        # Auto-reset finished worlds. jnp.where over the whole state pytree,
        # since both branches have to exist under jit anyway.
        reset_state = jax.vmap(init_env_state)(jax.random.split(reset_key, c["num_envs"]))
        next_state = jax.tree.map(
            lambda r, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), r, s),
            reset_state, next_state,
        )
        next_obs = jnp.where(done[:, None], jax.vmap(observe)(reset_state), next_obs)

        # Track true (unscaled) episode returns for logging.
        ep_return = ep_return + reward / reward_scale
        finished = jnp.where(done, ep_return, 0.0)
        ep_return = jnp.where(done, 0.0, ep_return)

        t = Transition(obs, action, log_prob, value, reward, done.astype(jnp.float32),
                       trunc_bootstrap, finished)
        return (next_state, next_obs, ep_return, key), t

    def loss_fn(params, mb: Transition, adv, targets):
        logits, value = forward(params, mb.obs)
        log_prob = _log_prob(logits, mb.action)

        # Value loss, clipped around the old value as in PureJaxRL.
        v_clipped = mb.value + jnp.clip(value - mb.value, -c["clip_eps"], c["clip_eps"])
        v_loss = 0.5 * jnp.maximum((value - targets) ** 2, (v_clipped - targets) ** 2).mean()

        # Clipped surrogate objective.
        ratio = jnp.exp(log_prob - mb.log_prob)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = -jnp.minimum(
            ratio * adv,
            jnp.clip(ratio, 1.0 - c["clip_eps"], 1.0 + c["clip_eps"]) * adv,
        ).mean()

        logp = jax.nn.log_softmax(logits)
        entropy = -(jnp.exp(logp) * logp).sum(-1).mean()

        total = pg_loss + c["vf_coef"] * v_loss - c["ent_coef"] * entropy
        return total, (v_loss, pg_loss, entropy)

    @jax.jit
    def update(runner: Runner) -> tuple[Runner, dict]:
        params, opt_state, env_state, obs, ep_return, key = runner

        # 1. Rollout: (num_steps, num_envs, ...) of transitions.
        (env_state, obs, ep_return, key), traj = jax.lax.scan(
            lambda carry, x: _env_step(params, carry, x),
            (env_state, obs, ep_return, key), None, c["num_steps"],
        )

        # 2. GAE, bootstrapping the unfinished tail from the current value.
        _, last_value = forward(params, obs)
        adv, targets = compute_gae(traj, last_value, c["gamma"], c["gae_lambda"])

        # 3. update_epochs passes over the batch in shuffled minibatches.
        def epoch(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)

            def to_minibatches(x):
                x = x.reshape((batch_size,) + x.shape[2:])[perm]
                return x.reshape((c["num_minibatches"], minibatch_size) + x.shape[1:])

            mbs = jax.tree.map(to_minibatches, (traj, adv, targets))

            def minibatch(carry, mb):
                params, opt_state = carry
                grads, aux = jax.grad(loss_fn, has_aux=True)(params, *mb)
                updates, opt_state = tx.update(grads, opt_state)
                return (optax.apply_updates(params, updates), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), mbs)
            return (params, opt_state, key), aux

        (params, opt_state, key), (v_loss, pg_loss, entropy) = jax.lax.scan(
            epoch, (params, opt_state, key), None, c["update_epochs"]
        )

        n_done = traj.done.sum()
        metrics = {
            "episodes": n_done,
            "mean_return": traj.ep_return.sum() / jnp.maximum(n_done, 1),
            "v_loss": v_loss.mean(),
            "pg_loss": pg_loss.mean(),
            "entropy": entropy.mean(),
            "action_hist": jnp.bincount(traj.action.reshape(-1), length=N_ACTIONS),
        }
        return Runner(params, opt_state, env_state, obs, ep_return, key), metrics

    return init, update, num_updates


def train(config: dict, seed: int, log_every: int = 10) -> tuple[dict, list[dict]]:
    """Train PPO. Returns (params, history), history being one dict per update."""
    init, update, num_updates = make_train(config)
    runner = init(jax.random.key(seed))
    steps_per_update = config["num_steps"] * config["num_envs"]
    print(f"{num_updates} updates x {steps_per_update} steps "
          f"({num_updates * steps_per_update / 1e6:.1f}M env steps)")
    print(f"{'update':>6} {'steps':>8} {'return':>7} {'eps':>5} {'entropy':>8} "
          f"{'v_loss':>8}  top action")

    t0 = time.perf_counter()
    # Episode returns are only reported in updates where some episodes ended
    # (every 500 steps at the earliest), so hold the last one for printing.
    last_return = float("nan")
    history = []
    for i in range(num_updates):
        runner, m = update(runner)
        if int(m["episodes"]) > 0:
            last_return = float(m["mean_return"])
        history.append({
            "update": i,
            "env_steps": (i + 1) * steps_per_update,
            # NaN in updates where no episode finished
            "mean_return": float(m["mean_return"]) if int(m["episodes"]) else float("nan"),
            "episodes": int(m["episodes"]),
            "entropy": float(m["entropy"]),
            "v_loss": float(m["v_loss"]),
            "pg_loss": float(m["pg_loss"]),
            "action_hist": [int(x) for x in m["action_hist"]],
        })
        if i % log_every == 0 or i == num_updates - 1:
            hist = m["action_hist"]
            top = int(jnp.argmax(hist))
            print(f"{i:>6} {(i + 1) * steps_per_update / 1e6:>7.1f}M "
                  f"{last_return:>7.1f} {int(m['episodes']):>5} "
                  f"{float(m['entropy']):>8.3f} {float(m['v_loss']):>8.4f}  "
                  f"{top} ({100 * float(hist[top] / hist.sum()):.0f}%)")
    print(f"trained in {time.perf_counter() - t0:.0f}s")
    return runner.params, history


# ---------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=float, default=CONFIG["total_timesteps"])
    ap.add_argument("--band", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help=f"target population band (default {DEFAULT_BAND[0]} {DEFAULT_BAND[1]}; "
                         "with --load, the band saved alongside the params)")
    ap.add_argument("--out", type=Path, default=Path("runs/ppo_params.pkl"))
    ap.add_argument("--load", type=Path, help="skip training, evaluate saved params")
    ap.add_argument("--record", type=Path, help="also save a trajectory JSON")
    a = ap.parse_args()

    if a.load:
        saved = pickle.loads(a.load.read_bytes())
        # Older files are the bare params dict, trained on the default band.
        if "params" in saved:
            params, band = saved["params"], tuple(saved["band"])
        else:
            params, band = saved, DEFAULT_BAND
        band = tuple(a.band) if a.band else band
    else:
        band = tuple(a.band) if a.band else DEFAULT_BAND
        config = {**CONFIG, "total_timesteps": int(a.timesteps), "band": band}
        print(f"band {band[0]}-{band[1]}")
        params, _history = train(config, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(pickle.dumps({"params": jax.device_get(params), "band": band}))
        print(f"saved {a.out}")

    # Score on the same evaluation as the baselines (seed 0, 256 worlds,
    # full 500-step episodes from a fresh start).
    report(f"ppo (sampled), band {band[0]}-{band[1]}", evaluate(ppo_policy, params, band=band))
    greedy = evaluate(ppo_greedy_policy, params, band=band)
    report(f"ppo (greedy), band {band[0]}-{band[1]}", greedy)
    print(f"  actions   {greedy['action_hist']}")

    if a.record:
        record(ppo_greedy_policy, params, 0, a.record, f"ppo band {band[0]}-{band[1]}", band=band)


if __name__ == "__main__":
    main()
