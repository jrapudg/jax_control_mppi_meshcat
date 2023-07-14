import jax

from utils.env_utils import step
from dynamics.models_meshcat import hopper_dynamics, acrobot_dynamics_rk4, cartpole_dynamics_rk4
from costs.cost_functions import acrobot_cost, hopper_cost, cartpole_cost

def step_wrapper_cartpole(carry, action):
    obs = carry[0]
    h = carry[1]
    next_obs = step(obs, action, cartpole_dynamics_rk4, h)
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
    i = carry[2]
    stance_flag = carry[3]
    stance_count = carry[4]
    
    next_obs, stance_flag, stance_count = hopper_dynamics(obs, action, h, stance_flag, stance_count)
    cost = hopper_cost(obs, action, i)
    i += 1 
    carry = (next_obs, h, i, stance_flag, stance_count)
    output = (next_obs, cost)
    return carry, output


def load_rollout_jax(step_fn):
    def rollout_aux(obs, actions, h):
        carry = (obs, h)
        _, output = jax.lax.scan(f=step_fn, init=carry, xs=actions)
        return output
    func = jax.jit(jax.vmap(rollout_aux, in_axes=(None, 0, None)))
    return func

def load_rollout_jax_i(step_fn):
    def rollout_aux(obs, actions, h, stance_flag, stance_count):
        carry = (obs, h, 0, stance_flag, stance_count)
        _, output = jax.lax.scan(f=step_fn, init=carry, xs=actions)
        return output
    func = jax.jit(jax.vmap(rollout_aux, in_axes=(None, 0, None, None, None)))
    return func