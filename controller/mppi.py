import numpy as np
from scipy.interpolate import CubicSpline
from jax_wrappers.rollout_functions import load_rollout_jax, step_wrapper_pendulum, step_wrapper_acrobot, step_wrapper_cartpole, step_wrapper_hopper

class MPPI:
    def __init__(self, env, params):
        # Hyper-parameters configuration
        self.temperature = params['temperature']
        self.horizon = params['horizon']
        self.n_samples = params['n_samples']
        self.noise_sigma = params['noise_sigma']

        # Get env parameters
        self.env_name = params["env_name"]
        if self.env_name in ['Pendulum-v1']:
            self.act_dim = env.action_space.shape[0]
            self.act_max = env.action_space.high
            self.act_min = env.action_space.low
            
        elif self.env_name in ['Acrobot-v1', 'Acrobot-v1.1']:
            self.act_dim = 1
            self.act_max = 1
            self.act_min = -1
        elif self.env_name in ['Hopper-meshcat']:
            self.act_dim = 2
            self.act_max = 100
            self.act_min = -100
        else:
            self.act_dim = 1
            self.act_max = 1
            self.act_min = -1

        # Rollouts
        self.h = params['h']
        self.sample_type = params['sample_type']
        self.n_knots = params['n_knots']
        self.random_generator = np.random.default_rng(params["seed"])
        
        if self.env_name == 'Pendulum-v1':
            self.rollout_jax = load_rollout_jax(step_wrapper_pendulum)
        elif self.env_name == 'CartPole-v1':
            self.rollout_jax = load_rollout_jax(step_wrapper_cartpole)
        elif self.env_name in ['Acrobot-v1', 'Acrobot-v1.1']:
            self.rollout_jax = load_rollout_jax(step_wrapper_acrobot)
        elif self.env_name in ['Hopper-meshcat']:
            self.rollout_jax = load_rollout_jax(step_wrapper_hopper)
        else:
            self.rollout_jax = None
        
        self.trajectory = None
        self.reset_planner()

    def reset_planner(self):
        self.trajectory = np.zeros((self.horizon, self.act_dim))

    def add_noise(self, size):
        return self.random_generator.normal(size=size) * self.noise_sigma
    
    def sample_delta_u(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            return self.add_noise(size)
        elif self.sample_type == 'cubic':
            indices = np.arange(self.n_knots)*self.horizon//self.n_knots
            size = (self.n_samples, self.n_knots, self.act_dim)
            knot_points = self.add_noise(size)
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            return cubic_spline(np.arange(self.horizon))
        
    def get_u(self, obs): 
        delta_u = self.sample_delta_u()
        actions = self.trajectory + delta_u
        actions = np.clip(actions, self.act_min, self.act_max)
        
        _, costs = self.rollout_jax(obs, actions, self.h) 
        costs = costs.sum(axis=1).squeeze() 

        # MPPI weights calculation
        exp_weights = np.exp(- 1/self.temperature * (costs - np.min(costs)))
        weighted_delta_u = exp_weights.reshape(self.n_samples, 1, 1) * delta_u
        weighted_delta_u = np.sum(weighted_delta_u, axis=0) / ( np.sum(exp_weights) + 1e-10)
        
        
        updated_actions = self.trajectory + weighted_delta_u
        updated_actions = np.clip(updated_actions, self.act_min, self.act_max)
    
        # Pop out first action from the trajectory and repeat last action
        self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
        self.trajectory[-1] = updated_actions[-1]

        # Output first action (MPC)
        action = updated_actions[0] 
        
        if self.env_name in ['CartPole-v1']:
            return 1 if action.squeeze() >= 0.0 else 0
        elif self.env_name in ['Acrobot-v1']:
            return 2 if action.squeeze() >= 0.33 else (0 if action.squeeze() <= -0.33 else 1)
        elif self.env_name in ['Acrobot-v1.1']:
            return action.squeeze()
        else:
            return action