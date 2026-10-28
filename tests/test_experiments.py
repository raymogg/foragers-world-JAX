"""Tests for the experiment metrics and for recording the episodes that were scored.

Run with:   .venv/bin/python tests/test_experiments.py

The write-up reads these numbers directly, so each metric is checked against
an episode small enough to work out by hand.
"""


#@dev: Tests are written and maintained by Claude

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from foragers_world_rl.agent import batched_rollout, fixed_policy, record
from foragers_world_rl.experiments import episode_metrics, pick_world


def _metrics(pops, band, dones=None, actions=None, initial_pop=1):
    pops = np.asarray([pops])
    T = pops.shape[1]
    dones = np.zeros((1, T), bool) if dones is None else np.asarray([dones], bool)
    actions = np.zeros((1, T), int) if actions is None else np.asarray([actions])
    rewards = ((pops >= band[0]) & (pops <= band[1])).astype(float)
    m = episode_metrics(pops, actions, rewards, dones, band, initial_pop)
    return {k: v[0] for k, v in m.items()}


def test_overshoot_and_undershoot_only_count_after_entry():
    # Starts below the band (5 < 10): that climb is NOT an undershoot.
    # Enters after step 1, overshoots to 25 (by 5) and drops to 7 (by 3).
    m = _metrics([5, 12, 25, 18, 7, 15], band=(10, 20))
    assert m["steps_to_enter"] == 2
    assert m["max_overshoot"] == 5, m["max_overshoot"]
    assert m["max_undershoot"] == 3, m["max_undershoot"]
    assert m["steps_above_after_entry"] == 1
    assert m["steps_below_after_entry"] == 1
    assert m["steps_in_band"] == 3 and m["steps_out_of_band"] == 3
    assert m["min_pop_after_entry"] == 7
    assert m["peak_pop"] == 25


def test_never_entering_the_band_gives_nan_deviation():
    m = _metrics([5, 6, 7], band=(10, 20))
    assert m["steps_to_enter"] == -1
    assert np.isnan(m["max_overshoot"]) and np.isnan(m["max_undershoot"])
    assert m["min_pop_after_entry"] == -1
    assert m["return"] == 0


def test_starting_in_band_counts_an_early_overshoot():
    # Floor-band case: the initial population (16) is already in 0-30, so the
    # boom to 55 straight after is an overshoot, not "before entry".
    m = _metrics([32, 55, 20], band=(0, 30), initial_pop=16)
    assert m["steps_to_enter"] == 0
    assert m["max_overshoot"] == 25, m["max_overshoot"]
    assert m["steps_above_after_entry"] == 2


def test_steps_after_extinction_are_ignored():
    # Extinct at step 2. The band includes 0, so the padded steps after the end
    # would score if they were counted -- they must not be.
    pops = [10, 5, 0, 0, 0, 0]
    dones = [0, 0, 1, 0, 0, 0]
    m = _metrics(pops, band=(0, 30), dones=dones)
    assert m["length"] == 3
    assert m["return"] == 3, f"counted padded steps: return {m['return']}"
    assert m["extinct"] and m["extinct_step"] == 2
    assert m["final_pop"] == 0


def test_action_switches_and_cost():
    m = _metrics([10, 10, 10, 10], band=(0, 30), actions=[4, 4, 7, 4])
    assert m["action_switches"] == 2
    assert m["n_action_4"] == 3 and m["n_action_7"] == 1
    # costs 1.0, 1.0, 1.75, 1.0
    assert np.isclose(m["mean_cost"], (1.0 + 1.0 + 1.75 + 1.0) / 4)


def test_pick_world():
    r = np.array([10.0, 50.0, 30.0, 40.0, 20.0])
    assert pick_world(r, "worst") == 0
    assert pick_world(r, "best") == 1
    assert pick_world(r, "median") == 2


def test_recording_replays_the_evaluated_world():
    """record(world=i) must be world i of the batched evaluation, step for step.

    Otherwise the episode shown in the write-up is not one the numbers came from.
    """
    seed, worlds, world, steps = 3, 8, 5, 40
    band = (80, 140)
    keys = jax.random.split(jax.random.key(seed), worlds)
    traj = batched_rollout(fixed_policy, 6, keys, jnp.asarray(band))
    expected = [int(p) for p in traj["pops"][world, :steps]]

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ep.json"
        record(fixed_policy, 6, seed, path, "test", steps=steps, band=band,
               world=world, worlds=worlds)
        frames = json.loads(path.read_text())["frames"]
    # frame 0 is the initial state; frame t+1 is after step t.
    got = [f["population"] for f in frames[1:]]
    assert got == expected, f"recorded {got[:10]}... != evaluated {expected[:10]}..."


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
