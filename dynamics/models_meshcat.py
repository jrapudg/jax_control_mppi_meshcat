import jax.numpy as jnp
import jax
from utils.jax_utils import get_coords_from_angle, wrap_angle, inv22

STANCE_STEPS = 25

GRAVITY = 9.81 # m/s
M1 = 5.0
M2 = 1.0

M = jnp.array([[M1, 0, 0, 0],
               [0, M1, 0, 0], 
               [0, 0, M2, 0], 
               [0, 0, 0, M2]])

M_inv = jnp.array([[1/M1, 0, 0, 0],
                   [0, 1/M1, 0, 0], 
                   [0, 0, 1/M2, 0], 
                   [0, 0, 0, 1/M2]])

########################## GENERAL ########################################

def rk4(dynamics, x, u, h):
    # RK4 integration with zero-order hold on u
    f1 = dynamics(x, u)
    f2 = dynamics(x + 0.5 * h * f1, u)
    f3 = dynamics(x + 0.5 * h * f2, u)
    f4 = dynamics(x + h * f3, u)
    return x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

########################## CARTPOLE ########################################

def cartpole_dynamics(x, u):
    mc = 1.0  # mass of the cart in kg (10)
    mp = 0.2   # mass of the pole (point mass at the end) in kg
    l = 0.5   # length of the pole in m
    g = 9.81  # gravity m/s^2

    q = x[:2] 
    qd = x[2:]

    s = jnp.sin(q[1])
    c = jnp.cos(q[1])

    H = jnp.array([[mc+mp, mp*l*c], [mp*l*c, mp*l**2]])
    C = jnp.array([[0, -mp*qd[1]*l*s],[0, 0]])
    G = jnp.array([0, mp*g*l*s])
    B = jnp.array([1, 0])

    qdd = jnp.dot(-inv22(H), (jnp.dot(C, qd) + G - B*u))
    return jnp.array([*qd, *qdd])

def cartpole_dynamics_sim(x, u):
    mc = 1.1  # mass of the cart in kg (10)
    mp = 0.22   # mass of the pole (point mass at the end) in kg
    l = 0.45   # length of the pole in m
    g = 9.81  # gravity m/s^2

    q = x[:2] 
    qd = x[2:]

    s = jnp.sin(q[1])
    c = jnp.cos(q[1])

    H = jnp.array([[mc+mp, mp*l*c], [mp*l*c, mp*l**2]])
    C = jnp.array([[0, -mp*qd[1]*l*s],[0, 0]])
    G = jnp.array([0, mp*g*l*s])
    B = jnp.array([1, 0])

    qdd = jnp.dot(-inv22(H), (jnp.dot(C, qd) + G - B*u))
    return jnp.array([*qd, *qdd])

def cartpole_dynamics_sim_rk4(x, u, h):
    return rk4(cartpole_dynamics_sim, x, u, h)

def cartpole_dynamics_rk4(x, u, h):
    return rk4(cartpole_dynamics, x, u, h)

########################## ACROBOT ########################################

def acrobot_dynamics(x, u):
    g = 9.81
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0
    J1 = 1.0
    J2 = 1.0
    
    theta1, theta2, dtheta1, dtheta2 = x
    
    theta1 = wrap_angle(theta1)
    theta2 = wrap_angle(theta2)
    
    c1, s1 = get_coords_from_angle(theta1)
    c2, s2 = get_coords_from_angle(theta2)
    c12 = jnp.cos(theta1 + theta2)

    # mass matrix
    m11 = m1*l1**2 + J1 + m2*(l1**2 + l2**2 + 2*l1*l2*c2) + J2
    m12 = m2*(l2**2 + l1*l2*c2 + J2)
    m22 = l2**2 * m2 + J2
    M = jnp.array([[m11, m12], [m12, m22]])

    # bias term
    tmp = l1*l2*m2*s2
    b1 = -(2 * dtheta1 * dtheta2 + dtheta2**2)*tmp
    b2 = tmp * dtheta1**2
    B = jnp.array([b1, b2])

    # friction
    c = 1.0
    C = jnp.array([c*dtheta1, c*dtheta2])

    # gravity term
    g1 = ((m1 + m2)*l2*c1 + m2*l2*c12) * g
    g2 = m2*l2*c12*g
    G = jnp.array([g1, g2])
    
    tau = jnp.array([0, u.squeeze()])
    ddtheta = jnp.dot(inv22(M),(tau - B - G - C))
    return jnp.array([dtheta1, dtheta2, *ddtheta])

def acrobot_dynamics_sim(x, u):
    g = 9.81
    l1 = 1.1
    l2 = 1.15
    m1 = 0.9
    m2 = 1.05
    J1 = 1.1
    J2 = 1.2
    
    theta1, theta2, dtheta1, dtheta2 = x
    
    theta1 = wrap_angle(theta1)
    theta2 = wrap_angle(theta2)
    
    c1, s1 = get_coords_from_angle(theta1)
    c2, s2 = get_coords_from_angle(theta2)
    c12 = jnp.cos(theta1 + theta2)

    # mass matrix
    m11 = m1*l1**2 + J1 + m2*(l1**2 + l2**2 + 2*l1*l2*c2) + J2
    m12 = m2*(l2**2 + l1*l2*c2 + J2)
    m22 = l2**2 * m2 + J2
    M = jnp.array([[m11, m12], [m12, m22]])

    # bias term
    tmp = l1*l2*m2*s2
    b1 = -(2 * dtheta1 * dtheta2 + dtheta2**2)*tmp
    b2 = tmp * dtheta1**2
    B = jnp.array([b1, b2])

    # friction
    c = 1.0
    C = jnp.array([c*dtheta1, c*dtheta2])

    # gravity term
    g1 = ((m1 + m2)*l2*c1 + m2*l2*c12) * g
    g2 = m2*l2*c12*g
    G = jnp.array([g1, g2])
    
    tau = jnp.array([0, u.squeeze()])
    ddtheta = jnp.dot(inv22(M),(tau - B - G - C))
    return jnp.array([dtheta1, dtheta2, *ddtheta])

def acrobot_dynamics_sim_rk4(x, u, h):
    return rk4(acrobot_dynamics, x, u, h)

def acrobot_dynamics_rk4(x, u, h):
    return rk4(acrobot_dynamics, x, u, h)

########################## HOPPER ########################################

def flight_dynamics(x, u):
    rb = x[0:2]
    rf = x[2:4]
    v = x[4:8]
    
    l1 = (rb[0]-rf[0])/jnp.linalg.norm(rb-rf)
    l2 = (rb[1]-rf[1])/jnp.linalg.norm(rb-rf)
    B = jnp.array([[l1, l2],
                   [l2, -l1],
                   [-l1, -l2],
                   [-l2, l1]])
    v_dot = jnp.array([0, -GRAVITY, 0, -GRAVITY]) + jnp.dot(jnp.dot(M_inv,B), u)
    x_dot = jnp.concatenate([v, v_dot])
    return x_dot

def stance_dynamics(x, u):
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

def flight_dynamics_rk4(x, u, h):
    return rk4(flight_dynamics, x, u, h)

def stance_dynamics_rk4(x, u, h):
    return rk4(stance_dynamics, x, u, h)  

def jump_map(x):
    # Assume the foot experiences inelastic collisions
    return jnp.array([*x[0:6], 0, 0])

def guard_function(x):
    rf_y = x[3]
    index = jnp.where(rf_y < 0, True, False) # 1 -> collision
    return index

def update_stance(x, u, h, stance_flag, stance_count):
    stance_flag = False
    stance_count = 0
    x = flight_dynamics_rk4(x, u, h)
    return x, stance_flag, stance_count

def keep_stance(x, u, h, stance_flag, stance_count):
    x = stance_dynamics_rk4(x, u, h)
    return x, stance_flag, stance_count

def stance_logic(x, u, h, stance_flag, stance_count):
    stance_count += 1
    x, new_stance_flag, new_stance_count = jax.lax.cond(stance_count == STANCE_STEPS, 
                                                        update_stance, 
                                                        keep_stance, 
                                                        x, u, h, stance_flag, stance_count)
    return x, new_stance_flag, new_stance_count

def collision_function(x, u, h):
    x = jnp.array([*x[0:3], 0, *x[4:]])
    x = jump_map(x)
    stance_flag = True
    return x, stance_flag

def flight_function(x, u, h):
    x = flight_dynamics_rk4(x, u, h)
    stance_flag = False
    return x, stance_flag

def modes_logic(x, u, h, stance_flag, stance_count):
    x, new_stance_flag = jax.lax.cond(guard_function(x), 
                                      collision_function, 
                                      flight_function, 
                                      x, u, h)
    new_stance_count = 0
    return x, new_stance_flag, new_stance_count

def hopper_dynamics(x, u, h, stance_flag, stance_count):
    x, new_stance_flag, new_stance_count = jax.lax.cond(stance_flag,
                                                        stance_logic,
                                                        modes_logic,
                                                        x, u, h, stance_flag, stance_count)
    return x, new_stance_flag, new_stance_count