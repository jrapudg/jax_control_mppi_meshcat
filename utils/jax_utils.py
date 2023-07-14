import jax.numpy as jnp

# JAX utils
def get_coords_from_angle(angle):
    return [jnp.cos(angle), jnp.sin(angle)]

def get_angle_from_coords(coords):
    cos_theta, sin_theta = coords
    return jnp.arctan2(sin_theta, cos_theta)

def bound_limits(value, lower_bound, upper_bound):
    return jnp.clip(value, lower_bound, upper_bound)

def wrap_angle(angle):
    return ((angle + jnp.pi) % (2 * jnp.pi)) - jnp.pi

def inv22(mat):
    m1, m2 = mat[0]
    m3, m4 = mat[1]
    inv_det = 1.0 / (m1 * m4 - m2 * m3)
    return jnp.array([[m4, -m2], [-m3, m1]]) * inv_det