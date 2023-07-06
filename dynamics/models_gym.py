import jax.numpy as jnp
from utils.jax_utils import get_coords_from_angle, get_angle_from_coords, bound_limits, wrap_angle

def pendulum_dynamics(x, u, h):
    # Inverted Pendulum
    MAX_TORQUE = 2.0
    MIN_TORQUE = -2.0
    MAX_ANGULAR_VELO = 8.0
    MIN_ANGULAR_VELO = -8.0

    MASS = 1.0 # Mass of the pendulum
    LENGTH = 1.0 # Length of the pendulum
    GRAVITY = 10.0 # Gravity

    cos_theta, sin_theta, thdot = x
    theta = get_angle_from_coords((cos_theta, sin_theta))
    u = jnp.clip(u, MIN_TORQUE, MAX_TORQUE)
    thdot += (3 * GRAVITY / (2 * LENGTH) * jnp.sin(theta)) * h \
           + (3.0 / (MASS * LENGTH ** 2) * u) * h
    thdot = jnp.clip(thdot, MIN_ANGULAR_VELO, MAX_ANGULAR_VELO)
    newth = theta + thdot * h
    new_cos_theta, new_sin_theta = get_coords_from_angle(newth)
    new_x = jnp.array([new_cos_theta, new_sin_theta, thdot])
    return new_x

def cartpole_dynamics(x, u, h): 
    FORCE_MAG = 10.0
    MASSPOLE = 0.1
    LENGTH = 0.5
    GRAVITY = 9.8
    MASSCART = 1.0
    TOTAL_MASS = MASSPOLE + MASSCART
    POLE_MASS_LENGTH = MASSPOLE * LENGTH
    
    pos, pos_dot, theta, theta_dot = x
    force = jnp.where(u > 0, FORCE_MAG, -FORCE_MAG)
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    
    temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) \
             / (LENGTH * (4.0 / 3.0 - MASSPOLE * costheta**2 / TOTAL_MASS))
    xacc = temp - POLE_MASS_LENGTH * thetaacc * costheta / TOTAL_MASS
    
    pos_dot = pos_dot + h * xacc
    pos = pos + h * pos_dot
    theta_dot = theta_dot + h * thetaacc
    theta = theta + h * theta_dot
            
    new_x = jnp.array([pos, pos_dot, theta, theta_dot])
    return new_x

def acrobot_dynamics_dsdt(x, u):
    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links
    FORCE_MAG = 1.0
    
    m1 = LINK_MASS_1
    m2 = LINK_MASS_2
    l1 = LINK_LENGTH_1
    lc1 = LINK_COM_POS_1
    lc2 = LINK_COM_POS_2
    I1 = LINK_MOI
    I2 = LINK_MOI
    GRAVITY = 9.8
    
    theta1, theta2, dtheta1, dtheta2 = x
        
    #force1 = jnp.where(u > 0.33, FORCE_MAG, 0)
    #force2 = jnp.where(u < -0.33, -FORCE_MAG, 0)
    #force = force1 + force2
    force = u
    
    d1 = (m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * jnp.cos(theta2)) + I1 + I2)
    d2 = m2 * (lc2**2 + l1 * lc2 * jnp.cos(theta2)) + I2
    phi2 = m2 * lc2 * GRAVITY * jnp.cos(theta1 + theta2 - jnp.pi / 2.0)
    phi1 = (-m2 * l1 * lc2 * dtheta2**2 * jnp.sin(theta2)\
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * jnp.sin(theta2)\
            + (m1 * lc1 + m2 * l1) * GRAVITY * jnp.cos(theta1 - jnp.pi / 2)
            + phi2)
    ddtheta2 = (force + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * jnp.sin(theta2) - phi2)\
               /(m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    
    dsdt = jnp.array([dtheta1, dtheta2, ddtheta1, ddtheta2])
    return dsdt

def acrobot_dynamics_rk4(x, u, h):
    MAX_VEL_1 = 4 * jnp.pi
    MAX_VEL_2 = 9 * jnp.pi
    
    cos_theta1, sin_theta1, cos_theta2, sin_theta2, dtheta1, dtheta2 = x
    theta1 = get_angle_from_coords((cos_theta1, sin_theta1))
    theta2 = get_angle_from_coords((cos_theta2, sin_theta2))
    
    dtheta1 = jnp.clip(dtheta1, -MAX_VEL_1, MAX_VEL_1)
    dtheta2 = jnp.clip(dtheta2, -MAX_VEL_2, MAX_VEL_2)
    
    new_x = jnp.array([theta1, theta2, dtheta1, dtheta2])
    
    # RK4 integration with zero-order hold on u
    f1 = acrobot_dynamics_dsdt(new_x, u)
    f2 = acrobot_dynamics_dsdt(new_x + 0.5 * h * f1, u)
    f3 = acrobot_dynamics_dsdt(new_x + 0.5 * h * f2, u)
    f4 = acrobot_dynamics_dsdt(new_x + h * f3, u)
    
    f = new_x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    
    cos_theta1 = jnp.cos(f[0])
    sin_theta1 = jnp.sin(f[0])
    cos_theta2 = jnp.cos(f[1])
    sin_theta2 = jnp.sin(f[1])
    
    new_x = jnp.array([cos_theta1, sin_theta1, cos_theta2, sin_theta2, f[2], f[3]])
    return new_x