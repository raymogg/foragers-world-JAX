import jax, jax.numpy as jnp
from typing import NamedTuple

# keep small for initial testing
MAX_POPULATION = 256
INITIAL_POPULATION = 16
GRID_SIZE = 100
SIMULATION_STEPS = 1000
# Min energy to breed. A parent pays energy/2 + 1, so at the threshold it keeps 1.
BREED_ENERGY = 4.0

class World(NamedTuple):
    food: jax.Array #(X, Y) int
    poison: jax.Array #(X, Y) int

    @property
    def total_food(self):
        return jnp.sum(self.food)

    @property
    def total_poison(self):
        return jnp.sum(self.poison)

class Foragers(NamedTuple):
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

    # Returns the number of foragers per position
    def per_position_count(self) -> jax.Array:
        is_alive = self.alive.astype(float)
        rows = self.pos[:, 0]
        cols = self.pos[:, 1]
        return jnp.zeros((GRID_SIZE, GRID_SIZE)).at[rows, cols].add(is_alive)

    def per_position_breedable_count(self) -> jax.Array:
        can_breed = self.alive & (self.energy >= BREED_ENERGY)
        rows = self.pos[:, 0]
        cols = self.pos[:, 1]
        return jnp.zeros((GRID_SIZE, GRID_SIZE)).at[rows, cols].add(can_breed)

    # Returns the position grid flattened
    def flat_pos(self):
      return self.pos[:, 0] * GRID_SIZE + self.pos[:, 1]

    # Returns the population eligble to breed
    def can_breed(self):
        return self.alive & (self.energy >= BREED_ENERGY)

class EnvState(NamedTuple):

    foragers: Foragers
    world: World

# Single move step of all foragers
def step(world: World, forager: Foragers, key: jax.Array) -> tuple[Foragers]:
    
    # Single step for a single forager
    delta = jax.random.randint(key, (2,), -1, 2)

    # produce can_move array
    can_move = forager.alive & (forager.energy >= 1.0)

    # Avoid forager moving if no energy to do so
    new_forager_pos = jnp.where(can_move, 
                                jnp.clip(forager.pos + delta, 0, GRID_SIZE - 1),
                                forager.pos)

    # Cost of 1 step is 1 energy, food gives 1 energy per unit
    new_forager_energy = jnp.where(forager.alive, forager.energy - 1, 0.0)

    new_alive = forager.alive & (new_forager_energy > 0.0)

    # Build new NamedTuples to return
    new_forager = Foragers(new_forager_pos, new_forager_energy, alive=new_alive)

    return new_forager

def init_foragers(key: jax.Array) -> Foragers:
    key, foragers_rand = jax.random.split(key)
    # Put INITIAL_POPULATION foragers on the grid
    initial_foragers = jax.random.randint(foragers_rand, (MAX_POPULATION, 2), 0, GRID_SIZE)
    initial_foragers_energy = jnp.ones(shape=(MAX_POPULATION, )) * 10
    initial_foragers_alive = jnp.arange(MAX_POPULATION) < INITIAL_POPULATION
    
    foragers = Foragers(initial_foragers, initial_foragers_energy, initial_foragers_alive)
    return foragers

def eat(world: World, forager: Foragers, per_grid_count: jax.Array) -> tuple[Foragers]:

    food_at_pos = world.food[forager.pos[0], forager.pos[1]]
    # Avoid divide 0 for dead forager on a tile (registers as no foragers here)
    divisor = jnp.maximum(per_grid_count[forager.pos[0], forager.pos[1]], 1.0)
    our_share = food_at_pos / divisor
    # Dead foragers don't get to eat
    new_forager_energy = jnp.where(forager.alive, forager.energy + our_share, 0.0)

    return Foragers(forager.pos, new_forager_energy, forager.alive)

def zero_eaten_food(world: World, per_grid_count: jax.Array) -> tuple[World]:
    new_food = jnp.where(per_grid_count > 0, 0, world.food)
    new_world = World(new_food, world.poison)
    return new_world

# asexual breeding
def breed(forager: Foragers) -> Foragers:
    # (MAX_POPULATION,) bool
    can_breed = forager.can_breed()

    # compute which slots are currently free
    free = ~forager.alive

    # array of free slots up to slot n
    free_rank = jnp.cumsum(free) - 1
    # array of eligble parents up to slot n
    parent_rank = jnp.cumsum(can_breed) - 1 

    # if more parents want to breed than slots available, prevent
    # parents after total free slots from breeding
    # note: since this is based on parents index, parents near the end of the array
    # will build up energy if continously blocked from breeding
    is_parent = can_breed & (parent_rank < free.sum())

    # gets_child[i, j] = does free slot i take forager j's child?
    gets_child = (
        free[:, None] #(N, 1)
        & is_parent[None, :] #(1, N)
        & (free_rank[:, None] == parent_rank[None, :]) #(N, N)
    )

    # Reduce along axis 1 (collapse the parent axis) -> one answer per SLOT.
    # (N, )
    # which slot receives a child
    receives = gets_child.any(axis=1)
    # which parent goes with which slot
    from_parent = jnp.where(receives, jnp.argmax(gets_child, axis=1), -1)

    # any parent that breeds looses half its energy
    new_energy = jnp.where(is_parent, forager.energy / 2, forager.energy)

    # only update slots where they are receiving a child
    new_pos = jnp.where(receives[:, None], forager.pos[from_parent], forager.pos)
    # Children take half the parent's ORIGINAL energy, which is what new_energy
    # already holds at the parent's slot.
    new_energy = jnp.where(receives, new_energy[from_parent], new_energy)
    # any slot that received a child is now alive
    new_alive = forager.alive | receives

    return Foragers(new_pos, new_energy, new_alive)


# Vmap functions
step_all = jax.vmap(step, in_axes=(None, 0, 0))
eat_all = jax.vmap(eat, in_axes=(None, 0, None))
zero_all = jax.vmap(zero_eaten_food, in_axes=(0, 0))

# Initial base for RL -> returns new state, obs, rewards and done.
def env_step(state: EnvState, action, step_key) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:

    # for i in range(SIMULATION_STEPS):
    #key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, MAX_POPULATION)
    foragers = step_all(state.world, state.foragers, step_keys)

    per_grid = foragers.per_position_count()
    foragers = eat_all(state.world, foragers, per_grid)
    world = zero_all(state.world, per_grid)

    foragers = breed(foragers)

    # Update the state
    new_state = EnvState(foragers, world)

    # Finished without population dying
    return new_state, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1) 

if __name__ == "__main__":
    key = jax.random.key(0)

    key, food_rand = jax.random.split(key)
    initial_food = jax.random.randint(food_rand, (GRID_SIZE, GRID_SIZE), 0, 5, dtype=int)
    initial_poison = jnp.zeros((GRID_SIZE, GRID_SIZE))
    world = World(initial_food, initial_poison)

    # Init some random foragers for using vmap
    key, foragers_rand = jax.random.split(key)
    # Put INITIAL_POPULATION foragers on the grid
    foragers = init_foragers(key)

    env_state = EnvState(foragers, world)


    # Simulate N steps. Order: move -> count -> eat -> clear food -> breed.
    # Breeding last so newborns do not dilute the food share on their tile this
    # step; they move and eat on the next one.
    print(f"{'step':>5} {'pop':>5} {'mean_energy':>12} {'food':>10}")
    for i in range(SIMULATION_STEPS):
        key, step_key = jax.random.split(key)
        env_state, obs, rewards, done = env_step(env_state, 0, step_key)

        if i % 10 == 0 or env_state.foragers.population == 0:
            print(f"{i:>5} {int(env_state.foragers.population):>5} "
                    f"{float(env_state.foragers.mean_energy):>12.2f} "
                    f"{float(env_state.world.total_food):>10.1f}")
        if env_state.foragers.population == 0:
            print("extinct")




