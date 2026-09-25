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

from jax_rl_evolutionary.world import (
    GRID_SIZE,
    MAX_POPULATION,
    Creatures,
    World,
    eat_all,
    init_creatures,
    step_all,
    zero_eaten_food,
)


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
