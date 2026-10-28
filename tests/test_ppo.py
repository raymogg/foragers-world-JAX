"""Tests for the PPO pieces that fail silently: GAE and the policy shapes.

Run with:   .venv/bin/python tests/test_ppo.py

A wrong advantage does not error -- training just learns worse, and nothing
says why. So GAE is checked against values worked out by hand.
"""


#@dev: Tests are written and maintained by Claude

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from foragers_world_rl.agent import N_ACTIONS
from foragers_world_rl.ppo import (
    OBS_DIM,
    Transition,
    compute_gae,
    forward,
    init_params,
    ppo_policy,
)


def _traj(reward, value, done, trunc_bootstrap=None):
    """A (T, 1) trajectory with only the fields GAE reads filled in."""
    col = lambda x: jnp.asarray(x, dtype=jnp.float32)[:, None]
    T = len(reward)
    zeros = jnp.zeros((T, 1))
    return Transition(
        obs=jnp.zeros((T, 1, OBS_DIM)),
        action=jnp.zeros((T, 1), dtype=jnp.int32),
        log_prob=zeros,
        value=col(value),
        reward=col(reward),
        done=col(done),
        trunc_bootstrap=zeros if trunc_bootstrap is None else col(trunc_bootstrap),
        ep_return=zeros,
    )


def test_gae_lambda_one_is_discounted_return():
    # With lam=1 and no episode end, A_t = sum of discounted rewards plus the
    # discounted bootstrap, minus V(s_t).
    g = 0.9
    traj = _traj(reward=[1, 0, 2], value=[0.5, 0.2, 0.1], done=[0, 0, 0])
    last_value = jnp.array([3.0])
    adv, targets = compute_gae(traj, last_value, g, 1.0)

    ret2 = 2 + g * 3.0
    ret1 = 0 + g * ret2
    ret0 = 1 + g * ret1
    expected = jnp.array([ret0 - 0.5, ret1 - 0.2, ret2 - 0.1])
    assert jnp.allclose(adv[:, 0], expected, atol=1e-5), f"{adv[:, 0]} != {expected}"
    assert jnp.allclose(targets[:, 0], expected + traj.value[:, 0], atol=1e-5)


def test_gae_lambda_zero_is_one_step_td():
    g = 0.9
    traj = _traj(reward=[1, 0], value=[0.5, 0.2], done=[0, 0])
    adv, _ = compute_gae(traj, jnp.array([3.0]), g, 0.0)
    expected = jnp.array([1 + g * 0.2 - 0.5, 0 + g * 3.0 - 0.2])
    assert jnp.allclose(adv[:, 0], expected, atol=1e-5), f"{adv[:, 0]} != {expected}"


def test_gae_extinction_cuts_the_chain():
    # Step 1 ends in extinction: its advantage is just r - V, and nothing from
    # the next episode (steps 2+, and last_value) leaks back into step 0 or 1.
    g, lam = 0.9, 0.95
    traj = _traj(reward=[1, 1, 5], value=[0.5, 0.4, 0.3], done=[0, 1, 0])
    adv, _ = compute_gae(traj, jnp.array([100.0]), g, lam)

    a1 = 1 - 0.4
    a0 = (1 + g * 0.4 - 0.5) + g * lam * a1
    assert jnp.isclose(adv[1, 0], a1, atol=1e-5), f"terminal step adv {adv[1, 0]} != {a1}"
    assert jnp.isclose(adv[0, 0], a0, atol=1e-5), f"step 0 adv {adv[0, 0]} != {a0}"


def test_gae_truncation_bootstraps_from_final_obs():
    # Step 0 hit the step limit: its target uses the stored bootstrap (already
    # discounted), not the value of the reset obs that follows it.
    traj = _traj(reward=[1, 0], value=[0.5, 9.0], done=[1, 0], trunc_bootstrap=[0.7, 0])
    adv, _ = compute_gae(traj, jnp.array([0.0]), 0.9, 0.95)
    assert jnp.isclose(adv[0, 0], 1 + 0.7 - 0.5, atol=1e-5), f"got {adv[0, 0]}"


def test_forward_shapes():
    params = init_params(jax.random.key(0), 16)
    logits, value = forward(params, jnp.zeros((5, 7, OBS_DIM)))
    assert logits.shape == (5, 7, N_ACTIONS), logits.shape
    assert value.shape == (5, 7), value.shape


def test_initial_policy_is_near_uniform():
    # The 0.01 gain on the policy head is what makes early exploration broad.
    params = init_params(jax.random.key(0), 64)
    logits, _ = forward(params, jnp.full((OBS_DIM,), 0.5))
    probs = jax.nn.softmax(logits)
    assert jnp.allclose(probs, 1 / N_ACTIONS, atol=0.02), probs


def test_policy_returns_valid_action():
    params = init_params(jax.random.key(0), 16)
    keys = jax.random.split(jax.random.key(1), 200)
    actions = jax.vmap(lambda k: ppo_policy(params, jnp.zeros(OBS_DIM), k))(keys)
    assert actions.shape == (200,)
    assert ((actions >= 0) & (actions < N_ACTIONS)).all(), actions


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}\n        {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}\n        {type(e).__name__}: {e}")

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
