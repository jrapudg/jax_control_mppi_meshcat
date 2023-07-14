import numpy as np
import jax.numpy as jnp
from utils.jax_utils import get_coords_from_angle, get_angle_from_coords, bound_limits, wrap_angle

GRAVITY = 9.81 # m/s
M1 = 5.0

tfinal = 2.2
h = 0.02
Nt = int(np.ceil(tfinal/h))+1
Nx = 8

x_ref = np.zeros((Nt,Nx))
x_ref[:,0] = 1.0
x_ref[:,2] = 1.0

x_ref[:,1] = 1.0 + 0.35*np.sin(2*np.pi/50.0*(np.arange(Nt))-3*np.pi/8)
x_ref[:10,1] = 1.0
x_ref[-53:,1] = 1.0
x_ref[:,3] = -0.35*np.sin(2*np.pi/50.0*(np.arange(Nt)))
x_ref[x_ref[:,3] < 0, 3] = 0
x_ref[-50:,3] = 0

x_ref = jnp.array(x_ref)
u_ref = jnp.array([M1*GRAVITY,0])

Q = jnp.diag(jnp.array([0.5, 1, 0.5, 1, 0.001, 0.001, 0.001, 0.001]))
R = jnp.diag(jnp.array([0.0001, 0.0001]))

def acrobot_cost(x, u):
    theta1, theta2, dtheta1, dtheta2 = x
    theta1 -= jnp.pi/2
    theta1 = wrap_angle(theta1)
    theta2 = wrap_angle(theta2)
    
    cost = 5*theta1 ** 2 + 3*theta2 ** 2 + 0.001 * dtheta1 ** 2 + 0.001 * dtheta2 ** 2 + 0.01 * (u ** 2)
    return cost

def cartpole_cost(x, u):
    xc, theta, dxc, dtheta = x
    theta -= jnp.pi
    theta = wrap_angle(theta)
    
    cost = theta ** 2 + 1.2*xc ** 2 + 0.01 * dxc ** 2 + 0.01 * dtheta ** 2 + 0.001 * (u ** 2)
    return cost

def hopper_cost(x, u, i): 
    x_error = x - x_ref[i]
    #u_error = u - u_ref
    cost = jnp.dot(x_error, jnp.dot(Q,x_error)) + 0.1 * (jnp.linalg.norm(x[0:2] - x[2:4]) - 0.4) **2 + jnp.dot(u, jnp.dot(R,u))
    return cost

"""
def hopper_cost(x, u, i): 
    x_error = x - x_ref[i]
    #u_error = u - u_ref
    cost = jnp.dot(x_error, jnp.dot(Q,x_error)) #+ 0.1 * (jnp.linalg.norm(x[0:2] - x[2:4]) - 1.0) **2 #+ jnp.dot(u_error, jnp.dot(R, u_error)) +  0.01 * (x[0] - x[2])**2
    return cost
"""