"""Regression tests for the foraging world.

Run with:   .venv/bin/python tests/test_world.py

Plain asserts, no pytest needed. Every test is a PROPERTY the simulation must
satisfy, not a fixed expected output -- so they keep working as you tune
parameters, and only fail when the mechanics actually break.
"""


#@dev: Tests are written and maintained by Claude

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax
import jax.numpy as jnp

from foragers_world_rl.recorder import Recorder
from foragers_world_rl.world import (
    BREED_ENERGY,
    GRID_SIZE,
    MAX_POPULATION,
    Foragers,
    World,
    breed,
    eat_all,
    init_foragers,
    step_all,
    zero_eaten_food,
)




def _foragers(positions, energies, alives):
    """Build a full-size Foragers from a short list, padding the unused slots.

    Padding matters: every slot is computed over, so the padding must not
    accidentally look eligible. Dead + zero energy is inert.
    """
    n = len(positions)
    assert n <= MAX_POPULATION
    pos = jnp.zeros((MAX_POPULATION, 2), dtype=int).at[:n].set(jnp.array(positions))
    energy = jnp.zeros((MAX_POPULATION,)).at[:n].set(jnp.array(energies, dtype=float))
    alive = jnp.zeros((MAX_POPULATION,), dtype=bool).at[:n].set(jnp.array(alives))
    return Foragers(pos, energy, alive)


def test_food_shared_evenly():
    """Foragers on the same tile split its food; dead ones neither eat nor dilute.

    This is the scatter-add/gather logic. Never exercised by the main loop --
    foragers rarely collide on a sparse grid -- so it needs a rigged setup.
    """
    SHARED = (4, 4)
    FOOD_THERE = 12.0

    # 3 alive on SHARED, 1 alive alone, 1 DEAD on SHARED.
    # The dead one is the point: if the alive mask is wrong it inflates the
    # divisor to 4 and each sharer gets 3.0 instead of 4.0.
    pos = jnp.array([[4, 4], [4, 4], [4, 4], [7, 1], [4, 4]])
    energy = jnp.full((5,), 10.0)
    alive = jnp.array([True, True, True, True, False])
    foragers = Foragers(pos, energy, alive)

    food = jnp.zeros((GRID_SIZE, GRID_SIZE)).at[SHARED].set(FOOD_THERE)
    food = food.at[7, 1].set(5.0)
    world = World(food)

    per_grid = foragers.per_position_count()
    assert jnp.allclose(per_grid[SHARED], 3.0), "dead forager counted in divisor"
    assert jnp.allclose(per_grid[7, 1], 1.0)

    fed = eat_all(world, foragers, per_grid)
    gained = fed.energy - energy

    assert jnp.allclose(gained[:3], FOOD_THERE / 3.0), "food not split evenly"
    assert jnp.allclose(gained[3], 5.0), "lone forager did not eat whole tile"

    new_world = zero_eaten_food(world, per_grid)
    assert jnp.allclose(new_world.food[SHARED], 0.0), "shared tile not emptied"
    assert jnp.allclose(new_world.food[7, 1], 0.0), "lone tile not emptied"


def test_energy_conservation():
    """Energy gained by the living == food removed from the world.

    Catches double-eating, dead foragers eating, and food vanishing without
    being consumed -- the whole class of scatter/gather bugs at once.
    """
    key = jax.random.key(0)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    key, k = jax.random.split(key)
    foragers = init_foragers(k)

    for _ in range(10):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)

        food_before = world.food.sum()
        energy_before = jnp.where(foragers.alive, foragers.energy, 0.0).sum()

        per_grid = foragers.per_position_count()
        foragers = eat_all(world, foragers, per_grid)
        world = zero_eaten_food(world, per_grid)

        food_after = world.food.sum()
        energy_after = jnp.where(foragers.alive, foragers.energy, 0.0).sum()

        eaten = food_before - food_after
        gained = energy_after - energy_before
        # rtol, not a tight atol: float32 sums over a GRID_SIZE**2 food grid
        # accumulate rounding error that scales with the grid, so an absolute
        # tolerance that passes at 10x10 fails at 100x100 for no real reason.
        assert jnp.allclose(eaten, gained, rtol=1e-4, atol=1e-3), (
            f"energy gained {gained} != food eaten {eaten}"
        )


def test_population_only_declines_without_food():
    """With no food, population declines monotonically to zero.

    The death check. If population ever rises, the alive mask is being
    computed or propagated wrongly.
    """
    key = jax.random.key(0)
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)))
    foragers = init_foragers(key)

    prev = foragers.population
    for _ in range(100):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)
        now = foragers.population
        assert now <= prev, f"population rose from {prev} to {now} with no food"
        prev = now

    assert foragers.population == 0, "foragers survived with no food at all"


def test_dead_foragers_stay_dead():
    """Death is permanent: a dead slot never becomes alive again.

    Nothing in the current design resurrects, so this guards against a future
    birth implementation reusing slots without clearing them properly.
    """
    key = jax.random.key(1)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    foragers = init_foragers(key)

    # Kill slot 0 outright.
    foragers = foragers._replace(
        energy=foragers.energy.at[0].set(0.0),
        alive=foragers.alive.at[0].set(False),
    )

    for _ in range(20):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)
        per_grid = foragers.per_position_count()
        foragers = eat_all(world, foragers, per_grid)
        world = zero_eaten_food(world, per_grid)
        assert not foragers.alive[0], "a dead forager came back to life"


def test_foragers_stay_on_grid():
    """Positions never leave the grid. Catches a broken clip in the move step."""
    key = jax.random.key(2)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    foragers = init_foragers(key)

    for _ in range(50):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)
        assert foragers.pos.min() >= 0, "position went negative"
        assert foragers.pos.max() < GRID_SIZE, "position exceeded grid"


def test_shapes_are_stable():
    """Shapes must not change across steps.

    A vmap that returns a shared input (the world, say) silently grows an
    extra axis every iteration -- which freezes the machine rather than
    raising. This catches that in one cheap assertion.
    """
    key = jax.random.key(3)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    foragers = init_foragers(key)

    expected = jax.tree.map(lambda a: a.shape, foragers)
    food_shape = world.food.shape

    for i in range(10):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)
        per_grid = foragers.per_position_count()
        foragers = eat_all(world, foragers, per_grid)
        world = zero_eaten_food(world, per_grid)

        got = jax.tree.map(lambda a: a.shape, foragers)
        assert got == expected, f"forager shapes changed at step {i}: {got}"
        assert world.food.shape == food_shape, (
            f"food shape changed at step {i}: {world.food.shape}"
        )


def test_no_nans():
    """No NaN or inf anywhere, ever. Usually a divide-by-zero in the sharing."""
    key = jax.random.key(4)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    foragers = init_foragers(key)

    for i in range(50):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)
        per_grid = foragers.per_position_count()
        foragers = eat_all(world, foragers, per_grid)
        world = zero_eaten_food(world, per_grid)

        assert jnp.isfinite(foragers.energy).all(), f"non-finite energy at step {i}"
        assert jnp.isfinite(world.food).all(), f"non-finite food at step {i}"


def test_randomness_actually_advances():
    """Successive steps must use different randomness.

    Reusing a key is silent: the code runs, foragers just replay identical
    moves forever. Over 20 steps with fresh keys, at least one forager must
    have visited more than one position.
    """
    key = jax.random.key(5)
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)))
    foragers = init_foragers(key)
    foragers = foragers._replace(energy=jnp.full((MAX_POPULATION,), 1e6))

    seen = []
    for _ in range(20):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        foragers = step_all(world, foragers, step_keys)
        seen.append(foragers.pos[0].tolist())

    assert len(set(map(tuple, seen))) > 1, (
        "forager 0 never changed position -- key is probably not advancing"
    )


def test_breed_child_gets_half_parent_energy():
    """A breeding forager halves its energy; the child gets the other half.

    Energy is conserved by breeding, which is why the conservation test above
    keeps working unchanged.
    """
    foragers = _foragers(
        positions=[[4, 4]],
        energies=[10.0],
        alives=[True],
    )
    out = breed(foragers)

    assert out.population == 2, f"expected 1 birth, got pop {out.population}"
    assert jnp.allclose(out.energy[0], 5.0), "parent did not halve its energy"
    # The child is in the first free slot, which is slot 1.
    assert jnp.allclose(out.energy[1], 5.0), "child did not get half"
    assert jnp.allclose(out.energy.sum(), foragers.energy.sum()), "breeding changed total energy"


def test_breed_child_starts_at_parent_position():
    """The child appears on the parent's tile."""
    foragers = _foragers(positions=[[3, 7]], energies=[10.0], alives=[True])
    out = breed(foragers)
    assert out.pos[1].tolist() == [3, 7], out.pos[1].tolist()


def test_breed_below_threshold_does_nothing():
    """A forager under BREED_ENERGY does not breed and is not charged."""
    foragers = _foragers(positions=[[4, 4]], energies=[BREED_ENERGY - 0.01], alives=[True])
    out = breed(foragers)
    assert out.population == 1, "bred despite being under the threshold"
    assert jnp.allclose(out.energy, foragers.energy), "charged despite not breeding"


def test_breed_threshold_is_inclusive():
    """Energy exactly at BREED_ENERGY is enough. Pins the >= / > boundary."""
    foragers = _foragers(positions=[[4, 4]], energies=[BREED_ENERGY], alives=[True])
    out = breed(foragers)
    assert out.population == 2, "energy at the threshold should breed"


def test_breed_many_parents_get_distinct_slots():
    """Several parents breed at once and no two children share a slot.

    The whole point of the cumsum allocator. If two parents picked the same
    free slot, one child would be silently overwritten and the population
    would come out short.
    """
    n = 5
    foragers = _foragers(
        positions=[[i, i] for i in range(n)],
        energies=[10.0] * n,
        alives=[True] * n,
    )
    out = breed(foragers)

    assert out.population == 2 * n, f"expected {2*n} foragers, got {out.population}"
    assert jnp.allclose(out.energy.sum(), foragers.energy.sum()), "energy not conserved"
    # Every child sits on its parent's tile, so each tile holds exactly 2 foragers.
    for i in range(n):
        on_tile = ((out.pos[:, 0] == i) & (out.pos[:, 1] == i) & out.alive).sum()
        assert on_tile == 2, f"tile ({i},{i}) holds {on_tile} foragers, expected 2"


def test_breed_respects_capacity():
    """More would-be parents than free slots: only as many births as slots.

    Critically, a parent that misses out must NOT be charged energy -- that is
    what the two-stage is_parent gate is for.
    """
    # Fill every slot but 3, all with breeding energy.
    n_free = 3
    n_alive = MAX_POPULATION - n_free
    foragers = _foragers(
        positions=[[i % GRID_SIZE, 0] for i in range(n_alive)],
        energies=[10.0] * n_alive,
        alives=[True] * n_alive,
    )
    out = breed(foragers)

    assert out.population == MAX_POPULATION, "should fill every free slot"
    charged = ((out.energy < foragers.energy) & foragers.alive).sum()
    assert charged == n_free, f"{charged} foragers charged, expected {n_free}"
    assert jnp.allclose(out.energy.sum(), foragers.energy.sum()), "energy not conserved"


def test_breed_full_population_does_nothing():
    """With no free slots, nothing happens and nobody pays."""
    foragers = _foragers(
        positions=[[i % GRID_SIZE, 0] for i in range(MAX_POPULATION)],
        energies=[10.0] * MAX_POPULATION,
        alives=[True] * MAX_POPULATION,
    )
    out = breed(foragers)

    assert out.population == MAX_POPULATION
    assert jnp.allclose(out.energy, foragers.energy), "charged energy with nowhere to put a child"


def test_breed_dead_foragers_do_not_breed():
    """A dead slot with high energy is not a parent.

    Slot 0 is dead, so it is also the first FREE slot -- the living forager's
    child lands there. So the check is that exactly one birth happened and
    energy was conserved, not that slot 0 is untouched.
    """
    foragers = _foragers(
        positions=[[2, 2], [2, 2]],
        energies=[10.0, 10.0],
        alives=[False, True],
    )
    out = breed(foragers)

    # One living parent -> one child. Total alive goes 1 -> 2.
    assert foragers.population == 1
    assert out.population == 2, "expected exactly one birth"
    # The dead forager's 10.0 was never a parent's energy, so it is not halved and
    # handed on -- the child's 5.0 comes from the LIVING forager only.
    assert jnp.allclose(out.energy[1], 5.0), "living parent did not halve"
    assert jnp.allclose(out.energy[0], 5.0), "child should hold the other half"


def test_breed_dead_forager_energy_is_not_inherited():
    """A dead forager's energy is never passed to a child.

    Here the dead forager is at a HIGHER slot than the parent, so it is not the
    first free slot and does not get overwritten. Its energy must be untouched.
    """
    foragers = _foragers(
        positions=[[2, 2], [2, 2], [2, 2]],
        energies=[10.0, 99.0, 0.0],
        alives=[True, False, False],
    )
    out = breed(foragers)

    assert out.population == 2, "only the living forager should breed"
    assert jnp.allclose(out.energy[0], 5.0), "parent did not halve"
    # Slot 1 is free and is free_rank 0, so the child lands there, replacing the
    # stale 99.0. That is the point: a reused slot must be fully reinitialised.
    assert jnp.allclose(out.energy[1], 5.0), "reused slot kept its stale energy"
def test_recorder_frame_matches_state():
    """A snapshot records exactly the living foragers and the real food grid.

    The viewer trusts population == len(foragers) and that every coordinate is
    inside the grid, so pin both.
    """
    foragers = _foragers(
        positions=[[1, 2], [3, 4], [5, 6]],
        energies=[10.0, 7.5, 3.0],
        alives=[True, False, True],
    )
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)).at[1, 2].set(4.0))

    rec = Recorder(grid_size=GRID_SIZE, max_population=MAX_POPULATION)
    rec.snapshot(world, foragers)
    frame = rec.frames[0]

    assert frame["population"] == 2, "dead forager was counted"
    assert len(frame["foragers"]) == 2, "dead forager was serialised"
    assert {a["slot"] for a in frame["foragers"]} == {0, 2}
    assert frame["total_food"] == 4.0
    # Grid must be grid_size x grid_size, row-major.
    assert len(frame["food"]) == GRID_SIZE
    assert all(len(row) == GRID_SIZE for row in frame["food"])
    assert frame["food"][1][2] == 4.0, "food grid is not row-major [row][col]"
    # mean_energy over the LIVING only: (10.0 + 3.0) / 2
    assert abs(frame["mean_energy"] - 6.5) < 1e-6, frame["mean_energy"]


def test_recorder_handles_extinction():
    """Zero living foragers must not divide by zero in mean_energy."""
    foragers = _foragers(positions=[[0, 0]], energies=[0.0], alives=[False])
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)))

    rec = Recorder(grid_size=GRID_SIZE, max_population=MAX_POPULATION)
    rec.snapshot(world, foragers)

    assert rec.frames[0]["population"] == 0
    assert rec.frames[0]["foragers"] == []
    assert rec.frames[0]["mean_energy"] == 0.0


def test_recorder_round_trips_to_json():
    """The saved file parses back and keeps the frames. Guards the viewer contract."""
    import json
    import tempfile

    foragers = _foragers(positions=[[2, 2]], energies=[8.0], alives=[True])
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)).at[2, 2].set(1.0))

    rec = Recorder(grid_size=GRID_SIZE, max_population=MAX_POPULATION, label="test")
    rec.snapshot(world, foragers)
    rec.snapshot(world, foragers)

    with tempfile.TemporaryDirectory() as d:
        path = rec.save(Path(d) / "t.json")
        loaded = json.loads(path.read_text())

    # Keys the viewer reads.
    for k in ("grid_size", "max_population", "frames", "summary", "params"):
        assert k in loaded, f"viewer needs top-level key {k!r}"
    assert loaded["summary"]["steps"] == 2
    assert loaded["frames"][0]["step"] == 0
    assert loaded["frames"][1]["step"] == 1, "step numbers must increment"
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
