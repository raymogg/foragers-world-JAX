import jax, jax.numpy as jnp
from typing import NamedTuple
import random

GRID_SIZE = 10

class World(NamedTuple):
    food: jax.Array #(X, Y) int

class Creature(NamedTuple):

    pos: jax.Array #(X, Y) float
    energy: float


def step(world: World, creature: Creature, action: jax.Array, key: jax.Array) -> tuple[World, Creature]:

    can_step = jnp.where(creature.energy > 1.0, True, False)
    if not can_step:
        jax.debug.print(f"Creature has no energy to move, its dead")
        return world, creature
    
    # Single step for a single creature
    delta = jax.random.randint(key, (2,), -1, 2)
    new_creature_pos = jnp.clip(creature.pos + delta, 0, GRID_SIZE - 1)
    jax.debug.print(f"Creature moved from {creature.pos[0]} {creature.pos[1]} to {new_creature_pos[0]} {new_creature_pos[1]}")

    # Eat food if there is any there
    food_at_pos = world.food[new_creature_pos[0], new_creature_pos[1]]
    jax.debug.print(f"Food at new pos {food_at_pos}")
    new_food = world.food.at[new_creature_pos[0], new_creature_pos[1]].set(0.0)

    # Cost of 1 step is 1 energy, food gives 1 energy per unit
    new_creature_energy = creature.energy - 1.0 + food_at_pos
    jax.debug.print(f"Creature new energy {new_creature_energy}")

    # Build new NamedTuples to return
    new_creature = Creature(new_creature_pos, new_creature_energy)
    new_world = World(new_food)

    return new_world, new_creature

key = jax.random.key(0)

key, food_rand = jax.random.split(key)
initial_food = jax.random.randint(food_rand, (GRID_SIZE, GRID_SIZE), 0, 5, dtype=int)
world = World(initial_food)

# Create a random creature
key, creature_rand = jax.random.split(key)
creature = Creature(jax.random.randint(creature_rand, (2,), 0, GRID_SIZE), 10.0)

# Simulate N steps
for i in range(50):
    key, step_key = jax.random.split(key)
    world, creature = step(world, creature, [0], step_key)


