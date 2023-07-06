import jax.numpy as jnp

# JAX utils
def get_coords_from_angle(angle):
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])

def get_angle_from_coords(coords):
    cos_theta, sin_theta = coords
    return jnp.arctan2(sin_theta, cos_theta)

def bound_limits(value, lower_bound, upper_bound):
    return jnp.clip(value, lower_bound, upper_bound)

def wrap_angle(angle):
    return ((angle + jnp.pi) % (2 * jnp.pi)) - jnp.pi