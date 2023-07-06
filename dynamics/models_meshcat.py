import jax.numpy as jnp
import jax

GRAVITY = 9.8 # m/s
M1 = 1.0
M2 = 0.5

M = jnp.array([[M1, 0, 0, 0],
               [0, M1, 0, 0], 
               [0, 0, M2, 0], 
               [0, 0, 0, M2]])

M_inv = jnp.array([[1/M1, 0, 0, 0],
                   [0, 1/M1, 0, 0], 
                   [0, 0, 1/M2, 0], 
                   [0, 0, 0, 1/M2]])

def flight_dynamics(x,u):
    rb = x[0:2]
    rf = x[2:4]
    v = x[4:8]
    
    l1 = (rb[0]-rf[0])/jnp.linalg.norm(rb-rf)
    l2 = (rb[1]-rf[1])/jnp.linalg.norm(rb-rf)
    B = jnp.array([[l1, l2],
                   [l2, -l1],
                   [-l1, -l2],
                   [-l2, l1]])
    v_dot = jnp.array([0, -GRAVITY, 0, GRAVITY]) + jnp.dot(jnp.dot(M_inv,B), u)
    x_dot = jnp.concatenate([v, v_dot])
    return x_dot

def stance_dynamics(x,u):
    rb = x[0:2]
    rf = x[2:4]
    v = x[4:8]
    
    l1 = (rb[0]-rf[0])/jnp.linalg.norm(rb-rf)
    l2 = (rb[1]-rf[1])/jnp.linalg.norm(rb-rf)
    B = jnp.array([[l1, l2],
                   [l2, -l1],
                   [0, 0],
                   [0, 0]])
    v_dot = jnp.array([0, -GRAVITY, 0, 0]) + jnp.dot(jnp.dot(M_inv,B), u)
    x_dot = jnp.concatenate([v, v_dot])
    return x_dot

def rk4(dynamics, x, u, h):
    # RK4 integration with zero-order hold on u
    f1 = dynamics(x, u)
    f2 = dynamics(x + 0.5 * h * f1, u)
    f3 = dynamics(x + 0.5 * h * f2, u)
    f4 = dynamics(x + h * f3, u)
    return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

def flight_dynamics_rk4(x, u, h):
    return rk4(flight_dynamics, x, u, h)

def stance_dynamics_rk4(x, u, h):
    return rk4(stance_dynamics, x, u, h)  

def jump_map(x):
    # Assume the foot experiences inelastic collisions
    return jnp.array([*x[0:6], 0, 0])

def guard_function(x):
    rf_y = x[3]
    index_1 = jnp.where(rf_y < 0, 2, 0) # 2 -> collision
    index_2 = jnp.where(rf_y > 0, 1, 0) # 1 -> flight, 0 -> stance
    index = index_1 + index_2 
    return index

def collision_function(x, u, h):
    x = jnp.array([*x[0:3], 0, *x[4:]])
    x = jump_map(x)
    return stance_dynamics_rk4(x, u, h)

def hopper_dynamics(x, u, h):
    return jax.lax.switch(guard_function(x), 
                          [stance_dynamics_rk4, flight_dynamics_rk4, collision_function], 
                          x, u, h)