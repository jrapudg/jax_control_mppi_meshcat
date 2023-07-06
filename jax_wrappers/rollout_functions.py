import jax

from utils.env_utils import step
from dynamics.models_gym import pendulum_dynamics, cartpole_dynamics, acrobot_dynamics_rk4
from dynamics.models_meshcat import hopper_dynamics
from costs.cost_functions import pendulum_cost, cartpole_cost, acrobot_cost, hopper_cost

def step_wrapper_pendulum(carry, action):
    obs = carry[0]
    h = carry[1]
    next_obs = step(obs, action, pendulum_dynamics, h)
    cost = pendulum_cost(obs, action)
    carry = (next_obs, h)
    output = (next_obs, cost)
    return carry, output

def step_wrapper_cartpole(carry, action):
    obs = carry[0]
    h = carry[1]
    next_obs = step(obs, action, cartpole_dynamics, h)
    cost = cartpole_cost(obs, action)
    carry = (next_obs, h)
    output = (next_obs, cost)
    return carry, output

def step_wrapper_acrobot(carry, action):
    obs = carry[0]
    h = carry[1]
    next_obs = step(obs, action, acrobot_dynamics_rk4, h)
    cost = acrobot_cost(obs, action)
    carry = (next_obs, h)
    output = (next_obs, cost)
    return carry, output

def step_wrapper_hopper(carry, action):
    obs = carry[0]
    h = carry[1]
    next_obs = step(obs, action, hopper_dynamics, h)
    cost = hopper_cost(obs, action)
    carry = (next_obs, h)
    output = (next_obs, cost)
    return carry, output

def load_rollout_jax(step_fn):
    def rollout_aux(obs, actions, h):
        carry = (obs, h)
        _, output = jax.lax.scan(f=step_fn, init=carry, xs=actions)
        return output
    func = jax.jit(jax.vmap(rollout_aux, in_axes=(None, 0, None)))
    return func