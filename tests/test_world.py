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

from jax_duck_farm.world import (
    GRID_SIZE,
    MAX_POPULATION,
    Creatures,
    World,
    compute_breed_rank,
    eat_all,
    init_creatures,
    step_all,
    zero_eaten_food,
)

BREED_ENERGY = 4.0  # compute_breed_rank's eligibility threshold


def _ducks(positions, energies, alives):
    """Build a full-size Creatures from a short list, padding the unused slots.

    Padding matters: every slot is computed over, so the padding must not
    accidentally look eligible. Dead + zero energy is inert.
    """
    n = len(positions)
    assert n <= MAX_POPULATION
    pos = jnp.zeros((MAX_POPULATION, 2), dtype=int).at[:n].set(jnp.array(positions))
    energy = jnp.zeros((MAX_POPULATION,)).at[:n].set(jnp.array(energies, dtype=float))
    alive = jnp.zeros((MAX_POPULATION,), dtype=bool).at[:n].set(jnp.array(alives))
    return Creatures(pos, energy, alive)


def test_food_shared_evenly():
    """Creatures on the same tile split its food; dead ones neither eat nor dilute.

    This is the scatter-add/gather logic. Never exercised by the main loop --
    creatures rarely collide on a sparse grid -- so it needs a rigged setup.
    """
    SHARED = (4, 4)
    FOOD_THERE = 12.0

    # 3 alive on SHARED, 1 alive alone, 1 DEAD on SHARED.
    # The dead one is the point: if the alive mask is wrong it inflates the
    # divisor to 4 and each sharer gets 3.0 instead of 4.0.
    pos = jnp.array([[4, 4], [4, 4], [4, 4], [7, 1], [4, 4]])
    energy = jnp.full((5,), 10.0)
    alive = jnp.array([True, True, True, True, False])
    creatures = Creatures(pos, energy, alive)

    food = jnp.zeros((GRID_SIZE, GRID_SIZE)).at[SHARED].set(FOOD_THERE)
    food = food.at[7, 1].set(5.0)
    world = World(food)

    per_grid = creatures.per_position_count()
    assert jnp.allclose(per_grid[SHARED], 3.0), "dead creature counted in divisor"
    assert jnp.allclose(per_grid[7, 1], 1.0)

    fed = eat_all(world, creatures, per_grid)
    gained = fed.energy - energy

    assert jnp.allclose(gained[:3], FOOD_THERE / 3.0), "food not split evenly"
    assert jnp.allclose(gained[3], 5.0), "lone creature did not eat whole tile"

    new_world = zero_eaten_food(world, per_grid)
    assert jnp.allclose(new_world.food[SHARED], 0.0), "shared tile not emptied"
    assert jnp.allclose(new_world.food[7, 1], 0.0), "lone tile not emptied"


def test_energy_conservation():
    """Energy gained by the living == food removed from the world.

    Catches double-eating, dead creatures eating, and food vanishing without
    being consumed -- the whole class of scatter/gather bugs at once.
    """
    key = jax.random.key(0)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    key, k = jax.random.split(key)
    creatures = init_creatures(k)

    for _ in range(10):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)

        food_before = world.food.sum()
        energy_before = jnp.where(creatures.alive, creatures.energy, 0.0).sum()

        per_grid = creatures.per_position_count()
        creatures = eat_all(world, creatures, per_grid)
        world = zero_eaten_food(world, per_grid)

        food_after = world.food.sum()
        energy_after = jnp.where(creatures.alive, creatures.energy, 0.0).sum()

        eaten = food_before - food_after
        gained = energy_after - energy_before
        assert jnp.allclose(eaten, gained, atol=1e-4), (
            f"energy gained {gained} != food eaten {eaten}"
        )


def test_population_only_declines_without_food():
    """With no food, population declines monotonically to zero.

    The death check. If population ever rises, the alive mask is being
    computed or propagated wrongly.
    """
    key = jax.random.key(0)
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)))
    creatures = init_creatures(key)

    prev = creatures.population
    for _ in range(100):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        now = creatures.population
        assert now <= prev, f"population rose from {prev} to {now} with no food"
        prev = now

    assert creatures.population == 0, "creatures survived with no food at all"


def test_dead_creatures_stay_dead():
    """Death is permanent: a dead slot never becomes alive again.

    Nothing in the current design resurrects, so this guards against a future
    birth implementation reusing slots without clearing them properly.
    """
    key = jax.random.key(1)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    creatures = init_creatures(key)

    # Kill slot 0 outright.
    creatures = creatures._replace(
        energy=creatures.energy.at[0].set(0.0),
        alive=creatures.alive.at[0].set(False),
    )

    for _ in range(20):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        per_grid = creatures.per_position_count()
        creatures = eat_all(world, creatures, per_grid)
        world = zero_eaten_food(world, per_grid)
        assert not creatures.alive[0], "a dead creature came back to life"


def test_creatures_stay_on_grid():
    """Positions never leave the grid. Catches a broken clip in the move step."""
    key = jax.random.key(2)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    creatures = init_creatures(key)

    for _ in range(50):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        assert creatures.pos.min() >= 0, "position went negative"
        assert creatures.pos.max() < GRID_SIZE, "position exceeded grid"


def test_shapes_are_stable():
    """Shapes must not change across steps.

    A vmap that returns a shared input (the world, say) silently grows an
    extra axis every iteration -- which freezes the machine rather than
    raising. This catches that in one cheap assertion.
    """
    key = jax.random.key(3)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    creatures = init_creatures(key)

    expected = jax.tree.map(lambda a: a.shape, creatures)
    food_shape = world.food.shape

    for i in range(10):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        per_grid = creatures.per_position_count()
        creatures = eat_all(world, creatures, per_grid)
        world = zero_eaten_food(world, per_grid)

        got = jax.tree.map(lambda a: a.shape, creatures)
        assert got == expected, f"creature shapes changed at step {i}: {got}"
        assert world.food.shape == food_shape, (
            f"food shape changed at step {i}: {world.food.shape}"
        )


def test_no_nans():
    """No NaN or inf anywhere, ever. Usually a divide-by-zero in the sharing."""
    key = jax.random.key(4)
    key, k = jax.random.split(key)
    world = World(jax.random.uniform(k, (GRID_SIZE, GRID_SIZE), minval=0.0, maxval=4.0))
    creatures = init_creatures(key)

    for i in range(50):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        per_grid = creatures.per_position_count()
        creatures = eat_all(world, creatures, per_grid)
        world = zero_eaten_food(world, per_grid)

        assert jnp.isfinite(creatures.energy).all(), f"non-finite energy at step {i}"
        assert jnp.isfinite(world.food).all(), f"non-finite food at step {i}"


def test_randomness_actually_advances():
    """Successive steps must use different randomness.

    Reusing a key is silent: the code runs, creatures just replay identical
    moves forever. Over 20 steps with fresh keys, at least one creature must
    have visited more than one position.
    """
    key = jax.random.key(5)
    world = World(jnp.zeros((GRID_SIZE, GRID_SIZE)))
    creatures = init_creatures(key)
    creatures = creatures._replace(energy=jnp.full((MAX_POPULATION,), 1e6))

    seen = []
    for _ in range(20):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        seen.append(creatures.pos[0].tolist())

    assert len(set(map(tuple, seen))) > 1, (
        "creature 0 never changed position -- key is probably not advancing"
    )


def test_breed_rank_paper_case():
    """The hand-worked example: ducks rank 0,1,2.. among eligible peers on their tile.

    4 ducks on tile (4,4) but slot 3 is INELIGIBLE (energy below threshold),
    so the eligible ones there rank 0, 1, 2 -- skipping slot 3 entirely.
    Slot 3 gets -1 even though it sits among them. Slots 2 and 5 are alone on
    their own tiles, so each ranks 0.
    """
    ducks = _ducks(
        positions=[[4, 4], [4, 4], [7, 1], [4, 4], [4, 4], [0, 0]],
        energies=[10.0, 10.0, 10.0, 1.0, 10.0, 10.0],  # slot 3 too poor to breed
        alives=[True] * 6,
    )
    rank = compute_breed_rank(ducks)

    assert rank[:6].tolist() == [0, 1, 0, -1, 2, 0], rank[:6].tolist()
    # Unused padding slots are dead, so all ineligible.
    assert (rank[6:] == -1).all(), "padding slots should be ineligible"


def test_breed_rank_dead_ducks_excluded():
    """A dead duck is never eligible, and never occupies a rank.

    Slot 1 is dead but has plenty of energy and sits on the shared tile. If
    the alive mask were dropped it would take rank 1 and push slot 2 to 2.
    """
    ducks = _ducks(
        positions=[[3, 3], [3, 3], [3, 3]],
        energies=[10.0, 10.0, 10.0],
        alives=[True, False, True],
    )
    rank = compute_breed_rank(ducks)
    assert rank[:3].tolist() == [0, -1, 1], rank[:3].tolist()


def test_breed_rank_nobody_eligible():
    """No eligible ducks -> every rank is -1, and nothing errors."""
    ducks = _ducks(
        positions=[[1, 1], [1, 1], [2, 2]],
        energies=[0.5, 1.0, 2.0],  # all below BREED_ENERGY
        alives=[True, True, True],
    )
    rank = compute_breed_rank(ducks)
    assert (rank == -1).all(), "nobody should be eligible"


def test_breed_rank_all_on_one_tile():
    """N eligible ducks on one tile get exactly the ranks 0..N-1, no repeats.

    A repeat here would mean two ducks pair with the same partner.
    """
    n = 8
    ducks = _ducks(
        positions=[[5, 5]] * n,
        energies=[10.0] * n,
        alives=[True] * n,
    )
    rank = compute_breed_rank(ducks)
    assert sorted(rank[:n].tolist()) == list(range(n)), rank[:n].tolist()


def test_breed_rank_all_on_separate_tiles():
    """Ducks alone on their own tiles all rank 0 -- rank is per-tile, not global."""
    n = 6
    ducks = _ducks(
        positions=[[i, 0] for i in range(n)],
        energies=[10.0] * n,
        alives=[True] * n,
    )
    rank = compute_breed_rank(ducks)
    assert rank[:n].tolist() == [0] * n, rank[:n].tolist()


def test_breed_rank_threshold_is_inclusive():
    """Energy exactly at the threshold is eligible; just under is not.

    Pins the boundary so a >= / > slip shows up as a failure.
    """
    ducks = _ducks(
        positions=[[2, 2], [2, 2]],
        energies=[BREED_ENERGY, BREED_ENERGY - 0.01],
        alives=[True, True],
    )
    rank = compute_breed_rank(ducks)
    assert rank[0] == 0, "energy at the threshold should be eligible"
    assert rank[1] == -1, "energy below the threshold should not be eligible"


def test_breed_rank_ranks_are_contiguous_per_tile():
    """On any tile, the eligible ducks' ranks are exactly 0..k-1.

    A property over random states rather than a fixed case: catches gaps and
    duplicates that a hand-built example might miss.
    """
    key = jax.random.key(7)
    for _ in range(20):
        key, kp, ke, ka = jax.random.split(key, 4)
        pos = jax.random.randint(kp, (MAX_POPULATION, 2), 0, 3)  # small grid -> collisions
        energy = jax.random.uniform(ke, (MAX_POPULATION,), minval=0.0, maxval=8.0)
        alive = jax.random.uniform(ka, (MAX_POPULATION,)) > 0.3
        ducks = Creatures(pos, energy, alive)

        rank = compute_breed_rank(ducks)
        eligible = alive & (energy >= BREED_ENERGY)
        flat = pos[:, 0] * GRID_SIZE + pos[:, 1]

        for tile in jnp.unique(flat).tolist():
            on_tile = (flat == tile) & eligible
            got = sorted(rank[on_tile].tolist())
            assert got == list(range(len(got))), (
                f"tile {tile}: ranks {got} are not contiguous from 0"
            )


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
