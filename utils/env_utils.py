import gym
from environments.acrobot import AcrobotEnv

# Simulation utils
def create_env(env_name, seed, dt, render_mode="rgb_array"):
    if env_name == 'Acrobot-v1.1':
        env = AcrobotEnv(render_mode=render_mode)
    else:
        env = gym.make(env_name, render_mode=render_mode) # Change render_mode to 'human' for script
    env.unwrapped.dt = dt
    env.reset(seed=0)
    env.observation_space.seed(seed)
    return env

def step(obs, act, dynamics, h):
    return dynamics(obs, act, h)