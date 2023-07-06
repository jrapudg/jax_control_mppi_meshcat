import jax.numpy as jnp
from utils.jax_utils import get_coords_from_angle, get_angle_from_coords, bound_limits, wrap_angle

def pendulum_cost(x, u):
    MAX_TORQUE = 2.0
    MIN_TORQUE = -2.0
    
    cos_theta, sin_theta, thdot = x
    theta = get_angle_from_coords((cos_theta, sin_theta))
    u = jnp.clip(u, MIN_TORQUE, MAX_TORQUE)
    cost = wrap_angle(theta) ** 2 + 0.1 * thdot ** 2 + 0.01 * u ** 2
    return cost

def cartpole_cost(x, u):
    pos, pos_dot, theta, theta_dot = x
    cost = 2 * pos ** 2 + 0.01 * pos_dot ** 2 + 10 * theta ** 2 + 0.01 * theta_dot ** 2 + 0.001 * u ** 2
    return cost

def acrobot_cost(x, u):
    cos_theta1, sin_theta1, cos_theta2, sin_theta2, dtheta1, dtheta2 = x
    theta1 = get_angle_from_coords((cos_theta1, sin_theta1))
    theta2 = get_angle_from_coords((cos_theta2, sin_theta2))
    
    theta1 += jnp.pi
    theta1 = wrap_angle(theta1)
    
    #cost = 12 * theta1 ** 2 + 6 * theta2 ** 2 + 0.002 * dtheta1 ** 2 + 0.001 * dtheta2 ** 2 + 0.001 * (u ** 2)
    cost = 12 * theta1 ** 2 + 6 * theta2 ** 2 + 0.01 * dtheta1 ** 2 + 0.01 * dtheta2 ** 2 + 0.001 * (u ** 2)
    return cost

def hopper_cost(x, u):
    rb = x[0:2]
    rf = x[2:4]
    v = x[4:8]
    
    cost = rb[0] ** 2 + rf[0] ** 2 + 1000 * (rb[1] - 1.0) ** 2 + 1000 * (rf[1] - 0.1) ** 2
    
    #0.001 * jnp.dot(v, v) + 0.0001 * jnp.dot(u, u) + 100 * (rb[1] - 1.0) ** 2 + 100 * (rf[1] - 0.2)** 2
    return cost