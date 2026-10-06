import jax
import jax.numpy as jnp

class RandomPolicy:

    action_space: jax.Array
    key: jax.Array

    def __init__(self, actions: jax.Array):
        self.action_space = actions

    def select_action(self, obs: jax.Array, key: jax.Array) -> jax.Array:
        return jax.random.randint(key, (), 0, len(self.action_space))
