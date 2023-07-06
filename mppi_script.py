from utils.env_utils import create_env
from controller.mppi import MPPI

params = {'seed':42,
          #'h':0.05,
          'h':0.1,
          'env_name':'Acrobot-v1.1',
          'render_mode':'human',
          'sample_type':'cubic',
          'n_knots':15,
          'horizon':30,
          'temperature':1.0,
          'n_samples':1000,
          'noise_sigma':0.6}

""" params = {'seed':42,
          #'h':0.05,
          'h':0.1,
          #'env_name':'Pendulum-v1',
          #'env_name':'CartPole-v1',
          #'env_name':'Acrobot-v1',
          'env_name':'Acrobot-v1.1',
          'render_mode':'rgb_array',
          'sample_type':'cubic',
          #'sample_type':'normal',
          'n_knots':15,
          'horizon':30,
          'temperature':1.0,
          'n_samples':1000,
          'noise_sigma':0.6} """

env = create_env(params["env_name"], seed=params['seed'], dt=params['h'], render_mode=params['render_mode'])
controller_jax = MPPI(env, params)
controller_jax.reset_planner()
obs, _ = env.reset()
done = False

while not done:
    action = controller_jax.get_action(obs.reshape((-1,1)))
    obs, rew, done, _, _ = env.step(action)
    env.render()
env.close()