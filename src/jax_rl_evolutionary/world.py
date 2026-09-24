import jax, jax.numpy as jnp
from typing import NamedTuple

# keep small for initial testing
MAX_POPULATION = 5
INITIAL_POPULATION = 2
GRID_SIZE = 10

class World(NamedTuple):
    food: jax.Array #(X, Y) int

class Creatures(NamedTuple):
    pos: jax.Array #(MAX_POPULATION, 2) int
    energy: jax.Array #(MAX_POPULATION, ) float
    alive: jax.Array #(MAX_POPULATION, ) bool

    @property
    def population(self):
        return self.alive.sum()

    @property
    def mean_energy(self):
        # where iterates over creates alive, and returns its energy if its alive, or 0. Div by pop.
        return jnp.where(self.alive, self.energy, 0.0).sum() / jnp.maximum(self.population, 1)

def step(world: World, creature: Creatures, key: jax.Array) -> tuple[Creatures]:
    
    # Single step for a single creature
    delta = jax.random.randint(key, (2,), -1, 2)

    # produce can_move array
    can_move = creature.alive & (creature.energy >= 1.0)

    # Avoid creature moving if no energy to do so
    new_creature_pos = jnp.where(can_move, 
                                jnp.clip(creature.pos + delta, 0, GRID_SIZE - 1),
                                creature.pos)

    # Cost of 1 step is 1 energy, food gives 1 energy per unit
    new_creature_energy = jnp.where(creature.alive, creature.energy - 1, 0.0)

    new_alive = creature.alive & (new_creature_energy > 0.0)

    # Build new NamedTuples to return
    new_creature = Creatures(new_creature_pos, new_creature_energy, alive=new_alive)

    return new_creature

def init_creatures(key: jax.Array) -> Creatures:
    key, creatures_rand = jax.random.split(key)
    # Put INITIAL_POPULATION creatures on the grid
    initial_creatures = jax.random.randint(creatures_rand, (MAX_POPULATION, 2), 0, GRID_SIZE)
    initial_creatures_energy = jnp.ones(shape=(MAX_POPULATION, )) * 10
    initial_creatures_alive = jnp.arange(MAX_POPULATION) < INITIAL_POPULATION
    
    creatures = Creatures(initial_creatures, initial_creatures_energy, initial_creatures_alive)
    return creatures

key = jax.random.key(0)

key, food_rand = jax.random.split(key)
initial_food = jax.random.randint(food_rand, (GRID_SIZE, GRID_SIZE), 0, 5, dtype=int)
world = World(initial_food)

# Init some random creatures for using vmap
key, creatures_rand = jax.random.split(key)
# Put INITIAL_POPULATION creatures on the grid
creatures = init_creatures(key)

step_all = jax.vmap(step, in_axes=(None, 0, 0))
# Simulate N steps
for i in range(50):
    keys, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, MAX_POPULATION)
    creatures = step_all(world, creatures, step_keys)
    print(f"creatures mean energy: {creatures.mean_energy}")
    print(f"Population alive {creatures.population}")





