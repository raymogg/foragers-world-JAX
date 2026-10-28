import jax, jax.numpy as jnp
from typing import NamedTuple

# keep small for initial testing
MAX_POPULATION = 256
INITIAL_POPULATION = 16
# Default target range for RL. env_step takes the band as an argument so
# experiments can vary it without recompiling; this is what it falls back to.
POP_BAND_LOW, POP_BAND_HIGH = 80, 140
DEFAULT_BAND = (POP_BAND_LOW, POP_BAND_HIGH)

# must be 2^n
GRID_SIZE = 32
# Episode length for RL. scan needs a statically known trip count.
EPISODE_STEPS = 500
MAX_FOOD_PER_TILE = 5.0
FOOD_REGROWTH_RATE = 0.2

# Min energy to breed. A parent pays energy/2 + 1, so at the threshold it keeps 1.
BREED_ENERGY = 4.0
MOVE_COSTS = jnp.arange(0, 3, 0.25)

class World(NamedTuple):
    food: jax.Array #(X, Y) float
    # Not used for now
    poison: jax.Array

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

    def population_in_band(self, band_low, band_high) -> jax.Array:
        # `&`, not `and`/chained comparison: Python's version calls __bool__ on a
        # traced array, which raises under jit.
        return (self.population >= band_low) & (self.population <= band_high)

class EnvState(NamedTuple):
    foragers: Foragers
    world: World
    # Step counter lives in the state so lax.scan threads it for us, rather
    # than the caller passing it in and the two drifting apart.
    step_count: jax.Array  # () int

# Single move step of all foragers
def step(world: World, forager: Foragers, key: jax.Array, move_cost: float) -> tuple[Foragers]:
    
    # Single step for a single forager
    delta = jax.random.randint(key, (2,), -1, 2)

    # produce can_move array
    can_move = forager.alive & (forager.energy >= move_cost)

    # Avoid forager moving if no energy to do so
    new_forager_pos = jnp.where(can_move, 
                                jnp.clip(forager.pos + delta, 0, GRID_SIZE - 1),
                                forager.pos)

    # Cost of 1 step is 1 energy, food gives 1 energy per unit
    new_forager_energy = jnp.where(forager.alive, forager.energy - move_cost, 0.0)

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

def regrow_food(world: World, key: jax.Array) -> tuple[World]:
    # uniform 0, 1 -> bernouli food_regrowth_rate
    additional_food = (jax.random.uniform(key, (GRID_SIZE, GRID_SIZE)) < FOOD_REGROWTH_RATE).astype(float)

    # apply new food with max cap
    new_food = jnp.minimum(additional_food + world.food, MAX_FOOD_PER_TILE)

    new_world = World(new_food, world.poison)
    return new_world

# asexual breeding
def breed(forager: Foragers) -> Foragers:
    # (MAX_POPULATION,) bool
    can_breed = forager.can_breed()

    # compute which slots are currently free
    free = ~forager.alive

    # array of eligble parents up to slot n
    parent_rank = jnp.cumsum(can_breed) - 1

    # if more parents want to breed than slots available, prevent
    # parents after total free slots from breeding
    # note: since this is based on parents index, parents near the end of the array
    # will build up energy if continously blocked from breeding
    is_parent = can_breed & (parent_rank < free.sum())

    # The k-th parent's child goes into the k-th free slot.
    # free_slots[k] = index of the k-th free slot, padded with MAX_POPULATION.
    # `size` fixes the output shape so this works under jit.
    free_slots = jnp.nonzero(free, size=MAX_POPULATION, fill_value=MAX_POPULATION)[0]
    # Non-parents point out of bounds, and mode="drop" discards those writes.
    child_slot = jnp.where(is_parent, free_slots[parent_rank], MAX_POPULATION)

    # any parent that breeds looses half its energy
    new_energy = jnp.where(is_parent, forager.energy / 2, forager.energy)

    # Scatter each parent into its child's slot. Children take half the
    # parent's ORIGINAL energy, which is what new_energy already holds.
    new_pos = forager.pos.at[child_slot].set(forager.pos, mode="drop")
    new_energy = new_energy.at[child_slot].set(new_energy, mode="drop")
    new_alive = forager.alive.at[child_slot].set(True, mode="drop")

    return Foragers(new_pos, new_energy, new_alive)


# Vmap functions
step_all = jax.vmap(step, in_axes=(None, 0, 0, None))
eat_all = jax.vmap(eat, in_axes=(None, 0, None))
zero_all = jax.vmap(zero_eaten_food, in_axes=(0, 0))
regrow_all_food = jax.vmap(regrow_food, in_axes=(0, None))

def observe(state: EnvState) -> jax.Array:
    max_total_food = GRID_SIZE * GRID_SIZE * MAX_FOOD_PER_TILE
    # return observations as proportion of total
    return jnp.array([
        state.world.total_food / max_total_food,
        state.foragers.population / MAX_POPULATION,
        state.foragers.mean_energy / BREED_ENERGY,
    ], dtype=jnp.float32)

def init_env_state(key: jax.Array) -> EnvState:
    food_key, forager_key = jax.random.split(key)
    food = jax.random.randint(
        food_key, (GRID_SIZE, GRID_SIZE), 0, int(MAX_FOOD_PER_TILE) + 1
    ).astype(float)
    world = World(food, jnp.zeros((GRID_SIZE, GRID_SIZE)))
    return EnvState(init_foragers(forager_key), world, jnp.int32(0))


# Initial base for RL -> returns new state, obs, rewards and done.
# `band` is (low, high), inclusive. Pass it as an array to keep it traced, so a
# different band reuses the same compiled step.
def env_step(state: EnvState, action: jax.Array, step_key: jax.Array,
             band=DEFAULT_BAND) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:

    # setup all random keys needed. Each consumer of randomness gets its own
    # branch so the food draw and the movement draw stay independent.
    move_key, food_key = jax.random.split(step_key)
    step_keys = jax.random.split(move_key, MAX_POPULATION)

    # agent action - currently only updates forager movement cost
    new_move_cost = MOVE_COSTS[action]

    foragers = step_all(state.world, state.foragers, step_keys, new_move_cost)

    per_grid = foragers.per_position_count()
    foragers = eat_all(state.world, foragers, per_grid)
    world = zero_all(state.world, per_grid)

    # regrow food using bernouli(FOOD_REGROWTH_RATE)
    world = regrow_food(world, food_key)

    foragers = breed(foragers)

    # Update the state
    new_state = EnvState(foragers, world, state.step_count + 1)

    # Reward -> 1 for each step kept in band. jnp.where, not a Python
    # conditional: both branches are evaluated and selected elementwise.
    in_band = foragers.population_in_band(band[0], band[1])
    reward = jnp.where(in_band, 1.0, 0.0)

    obs = observe(new_state)
    # Extinction is terminal; the step cap is what gives scan a fixed length.
    done = (foragers.population == 0) | (new_state.step_count >= EPISODE_STEPS)

    return new_state, obs, reward, done


if __name__ == "__main__":
    # A bare look at the environment: one episode, one fixed action, no agent.
    # For policies, scoring and recording, use agent.py.
    DEMO_ACTION = 6

    key = jax.random.key(0)
    key, init_key = jax.random.split(key)
    state = init_env_state(init_key)

    print(f"grid {GRID_SIZE}x{GRID_SIZE}, band {POP_BAND_LOW}-{POP_BAND_HIGH}, "
          f"move cost {float(MOVE_COSTS[DEMO_ACTION]):g} (action {DEMO_ACTION})")
    print(f"{'step':>5} {'pop':>5} {'energy':>8} {'food':>8} {'return':>8}")

    total_reward = 0.0
    for i in range(EPISODE_STEPS):
        key, step_key = jax.random.split(key)
        state, obs, reward, done = env_step(state, DEMO_ACTION, step_key)
        total_reward += float(reward)

        if i % 25 == 0 or bool(done):
            print(f"{i:>5} {int(state.foragers.population):>5} "
                  f"{float(state.foragers.mean_energy):>8.2f} "
                  f"{float(state.world.total_food):>8.0f} "
                  f"{total_reward:>8.0f}")
        # done covers extinction and the episode cap.
        if bool(done):
            if state.foragers.population == 0:
                print("extinct")
            break

    print(f"\nreturn {total_reward:.0f}/{EPISODE_STEPS} "
          f"({100 * total_reward / EPISODE_STEPS:.0f}% of steps in band)")
