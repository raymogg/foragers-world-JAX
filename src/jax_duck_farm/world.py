import jax, jax.numpy as jnp
from typing import NamedTuple

# keep small for initial testing
MAX_POPULATION = 64
INITIAL_POPULATION = 16
GRID_SIZE = 10
SIMULATION_STEPS = 1000

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

    def per_position_breedable_count(self) -> jax.Array:
        can_breed = self.alive & (self.energy >= 4)
        rows = self.pos[:, 0]
        cols = self.pos[:, 1]
        return jnp.zeros((GRID_SIZE, GRID_SIZE)).at[rows, cols].add(can_breed)

    # Returns the position grid flattened
    def flat_pos(self):
      return self.pos[:, 0] * GRID_SIZE + self.pos[:, 1]


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

# Returns a rank to breed amongst ducks on the same tile
# Used to pair off ducks for breeding
# return (MAX_POPULATION, ) int - rank or -1 if ineligible
def compute_breed_rank(creature: Creatures) -> jax.Array:
    # Minimum 4 energy so parent is always left with atleast 2 energy after breeding
    # (MAX_CREATURES, bool)
    can_breed = creature.alive & (creature.energy >= 4)

    # (MAX_CREATURES, int)
    flattened_pos = creature.flat_pos()

    # Build a matrix of if duck i is on the same tile as duck j using their flattened positions
    # flattened_pos[:, None] is all value stretched across the columns
    # flattened_pos[None, :] is all values stretched across the row
    # i,i = true for all i
    same_tile = flattened_pos[:, None] == flattened_pos[None, :]

    # Construct an index per duck
    idx = jnp.arange(MAX_POPULATION)
    earlier = idx[:, None] > idx[None, :]

    combined_conditions =  same_tile & earlier & can_breed[None, :]
    # raw rank for an individual duck. Counts eligible ducks on my tile with a
    # lower slot index. Ineligible ducks still get a count here -- the
    # expression never asked whether *I* can breed -- so mask them to -1.
    rank_raw = combined_conditions.sum(axis=1)

    # Set ineligble to rank -1 for ease later on
    return jnp.where(can_breed, rank_raw, -1)

# not vmap'd
def breed(breed_ranks: jax.Array, creature: Creatures) -> tuple[Creatures]:


    # # Breeding is possible if rank >= 0 & there is another eligble duck with rank >0 on this tile
    # on_my_grid = creature.per_position_breedable_count()[creature.pos[0], creature.pos[1]]

    # # If there are other ducks here, this duck has a breed rank, then partition by odd and even indices
    # can_breed = jnp.where((breed_ranks >= 0) & (breed_ranks % 2 == 0) & (on_my_grid > 0), 
    #           on_my_grid >= breed_ranks + 1, # index 0, 2, etc - check there is a duck index 1, 3, etc
    #           on_my_grid <= breed_ranks - 1) # otherwise needs to check index before

    # Simpler not using vmap approach

    # Compute the same tile matrix
    flattened_pos = creature.flat_pos()
    same_tile = flattened_pos[:, None] == flattened_pos[None, :]

    # 



    


    return creature

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





