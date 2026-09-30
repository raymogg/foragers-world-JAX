import jax, jax.numpy as jnp
from typing import NamedTuple

# keep small for initial testing
MAX_POPULATION = 64
INITIAL_POPULATION = 16
GRID_SIZE = 10
SIMULATION_STEPS = 1000
# Min energy to breed. A parent pays energy/2 + 1, so at the threshold it keeps 1.
BREED_ENERGY = 4.0

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
    can_breed = creature.can_breed()

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
    # Compute the same tile matrix
    flattened_pos = creature.flat_pos()

    # Breeding requires min 4 energy
    can_breed = creature.can_breed()

    # This uses new axis to add a column axis and row axis
    # same tile = (len(flattened_pos), 1) == (1, len(flattened_pos)) broadcasts to (N, N)
    # same_tile = NxN where i,j = is duck i on same tile as duck j
    same_tile = flattened_pos[:, None] == flattened_pos[None, :]

    # partner must be on the same tile AND have the next rank in that tile
    rank_match = (breed_ranks[None, :] == breed_ranks[:, None] + 1)

    # elementwise and across the two NxN matrices. 
    is_partner = same_tile & rank_match

    # each row should contain only a single 1 value, argmax returns the index
    # of that value when run across axis 1
    partner_id = jnp.argmax(is_partner, axis=1)
    has_partner = is_partner.any(axis=1)
    # argmax returns 0 when a row has no match, which is a real duck. Make the
    # junk explicit so misusing it fails loudly instead of silently pointing at
    # slot 0.
    partner_id = jnp.where(has_partner, partner_id, -1)
    partner_energy = creature.energy[partner_id]

    # --- Stage 1: provisional mothers -----------------------------------------
    # Cannot include the free-slot capacity check yet: mother_rank is derived
    # from is_mother, so the gate would be circular. Rank the candidates first.
    maybe_mother = can_breed & has_partner & (breed_ranks % 2 == 0)

    # Find the free slots in the alive array
    free = ~creature.alive
    # how many free slots up to index i exist. Only meaningful where free is True.
    free_rank = jnp.cumsum(free) - 1
    # how many candidate mothers up to index i exist
    mother_rank = jnp.cumsum(maybe_mother) - 1

    # --- Stage 2: final mothers ------------------------------------------------
    # A mother beyond the number of free slots has nowhere to put a child, so she
    # must not breed at all -- otherwise she pays the energy for a child that is
    # never created.
    is_mother = maybe_mother & (mother_rank < free.sum())

    # partner ID is only valid for mothers - mothers index their partners as they look for rank + 1.
    # fathers have no index back to mothers.
    # we compute the deduction of the partners energy using the mothers partner index
    # we compute the deduction of the mothers energy using their own index
    mother_energy_cost = jnp.where(is_mother, (creature.energy / 2), 0.0)
    father_energy_cost = jnp.where(is_mother, ((partner_energy) / 2), 0.0 )
    # the father's cost is indexed by MOTHER, so scatter it onto the father's slot
    combined_cost = mother_energy_cost + jnp.zeros(MAX_POPULATION).at[partner_id].add(father_energy_cost)
    new_energy   = creature.energy - combined_cost

    # Child gets the two halves the parents gave up, so energy is conserved.
    # Indexed by MOTHER.
    child_energy = jnp.where(is_mother, (creature.energy + partner_energy) / 2, 0.0)

    # --- Stage 3: place the children ------------------------------------------
    # gets_child[i, j] = does free slot i take mother j's child?
    #   free[:, None]        -- slot i must actually be free
    #   is_mother[None, :]   -- duck j must actually be a mother
    #   free_rank == mother_rank -- the nth free slot takes the nth mother's child
    # The first two gates discard the junk values in free_rank/mother_rank.
    gets_child = free[:, None] & is_mother[None, :] & (free_rank[:, None] == mother_rank[None, :])

    # Reduce along axis 1: per receiving SLOT, which mother fills it?
    receives = gets_child.any(axis=1)
    from_mother = jnp.where(receives, jnp.argmax(gets_child, axis=1), -1)

    # Gather the mother's data into the receiving slot, then mask.
    # receives[:, None] because pos is (MAX_POPULATION, 2) and the mask is (MAX_POPULATION,)
    new_pos = jnp.where(receives[:, None], creature.pos[from_mother], creature.pos)
    # Children overwrite the post-cost energy in their own slots only.
    new_energy = jnp.where(receives, child_energy[from_mother], new_energy)
    new_alive = creature.alive | receives

    return Creatures(new_pos, new_energy, new_alive)

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





