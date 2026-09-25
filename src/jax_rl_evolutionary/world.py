import jax, jax.numpy as jnp
from typing import NamedTuple

# keep small for initial testing
MAX_POPULATION = 5
INITIAL_POPULATION = 2
GRID_SIZE = 10
SIMULATION_STEPS = 3

class World(NamedTuple):
    food: jax.Array #(X, Y) int

    @property
    def total_food(self):
        return jnp.sum(self.food)

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

    # Returns the number of creatures per position
    def per_position_count(self) -> jax.Array:
        is_alive = self.alive.astype(float)
        rows = self.pos[:, 0]
        cols = self.pos[:, 1]
        return jnp.zeros((GRID_SIZE, GRID_SIZE)).at[rows, cols].add(is_alive)


# Single move step of all creatures
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

def eat(world: World, creature: Creatures, per_grid_count: jax.Array) -> tuple[Creatures]:

    food_at_pos = world.food[creature.pos[0], creature.pos[1]]
    # Avoid divide 0 for dead creature on a tile (registers as no creatures here)
    divisor = jnp.maximum(per_grid_count[creature.pos[0], creature.pos[1]], 1.0)
    our_share = food_at_pos / divisor
    # Dead creatures don't get to eat
    new_creature_energy = jnp.where(creature.alive, creature.energy + our_share, 0.0)

    return Creatures(creature.pos, new_creature_energy, creature.alive)

def zero_eaten_food(world: World, per_grid_count: jax.Array) -> tuple[World]:
    new_food = jnp.where(per_grid_count > 0, 0, world.food)
    new_world = World(new_food)
    return new_world

# Vmap functions
step_all = jax.vmap(step, in_axes=(None, 0, 0))
eat_all = jax.vmap(eat, in_axes=(None, 0, None))
zero_all = jax.vmap(zero_eaten_food, in_axes=(0, 0))

if __name__ == "__main__":
    key = jax.random.key(0)

    key, food_rand = jax.random.split(key)
    initial_food = jax.random.randint(food_rand, (GRID_SIZE, GRID_SIZE), 0, 5, dtype=int)
    world = World(initial_food)

    # Init some random creatures for using vmap
    key, creatures_rand = jax.random.split(key)
    # Put INITIAL_POPULATION creatures on the grid
    creatures = init_creatures(key)

    # Simulate N steps
    for i in range(SIMULATION_STEPS):
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, MAX_POPULATION)
        creatures = step_all(world, creatures, step_keys)
        print(f"creatures mean energy pre eat: {creatures.mean_energy}")
        print(f"total food pre eat {world.total_food}")
        per_grid = creatures.per_position_count()
        creatures = eat_all(world, creatures, per_grid)
        world = zero_all(world, per_grid)
        print(f"creatures mean energy post eat: {creatures.mean_energy}")
        print(f"total food post eat {world.total_food}")





