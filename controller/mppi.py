import numpy as np
from scipy.interpolate import CubicSpline
from jax_wrappers.rollout_functions import load_rollout_jax, load_rollout_jax_i, step_wrapper_cartpole, step_wrapper_acrobot, step_wrapper_hopper

GRAVITY = 9.81 # m/s
M1 = 5.0
STANCE_STEPS = 25

class MPPI:
    def __init__(self, env, params):
        # Hyper-parameters configuration
        self.temperature = params['temperature']
        self.horizon = params['horizon']
        self.n_samples = params['n_samples']
        self.noise_sigma = np.array(params['noise_sigma'])
        self.stance_flag = True
        self.stance_count = 0
        self.last_state = None

        # Get env parameters
        self.env_name = params["env_name"]
        if self.env_name in ['Acrobot-meshcat']:
            self.act_dim = 1
            self.act_max = 100
            self.act_min = -100
        elif self.env_name in ['Hopper-meshcat']:
            self.act_dim = 2
            self.act_max = [200,20]
            self.act_min = [-200, -20]
        elif self.env_name in ['Cartpole-meshcat']:
            self.act_dim = 1
            self.act_max = 200
            self.act_min = -200
        else:
            self.act_dim = 1
            self.act_max = 1
            self.act_min = -1

        # Rollouts
        self.h = params['h']
        self.sample_type = params['sample_type']
        self.n_knots = params['n_knots']
        self.random_generator = np.random.default_rng(params["seed"])
        
        if self.env_name in ['Acrobot-meshcat']:
            self.rollout_jax = load_rollout_jax(step_wrapper_acrobot)
        elif self.env_name in ['Cartpole-meshcat']:
            self.rollout_jax = load_rollout_jax(step_wrapper_cartpole)
        elif self.env_name in ['Hopper-meshcat']:
            self.rollout_jax = load_rollout_jax_i(step_wrapper_hopper)
        else:
            self.rollout_jax = None
        
        self.trajectory = None
        self.reset_planner()
    
    def update_stance_variables(self):
        if self.stance_flag:
            self.stance_count += 1
            if self.stance_count == STANCE_STEPS:
                self.stance_count = 0
                self.stance_flag = False
        else:
            self.stance_flag = False if self.last_state[3] > 0 else True
                
    def reset_planner(self, hopper=False):
        self.trajectory = np.zeros((self.horizon, self.act_dim))
        
        if hopper:
            self.trajectory += np.array([M1*GRAVITY,0])
            
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
        self.last_state = obs
        delta_u = self.sample_delta_u()
        actions = self.trajectory + delta_u
        actions = np.clip(actions, self.act_min, self.act_max)
        
        if self.env_name == "Hopper-meshcat":
            self.update_stance_variables()
            _, costs = self.rollout_jax(obs, actions, self.h, self.stance_flag, self. stance_count) 
        else:
            _, costs = self.rollout_jax(obs, actions, self.h) 
        costs_sum = costs.sum(axis=1).squeeze() 

        # MPPI weights calculation
        ## Scale parameters
        min_cost = np.min(costs_sum)
        max_cost = np.max(costs_sum)
        
        exp_weights = np.exp(- 1/self.temperature * ((costs_sum - min_cost)/(max_cost - min_cost)))
        
        weighted_delta_u = exp_weights.reshape(self.n_samples, 1, 1) * delta_u
        weighted_delta_u = np.sum(weighted_delta_u, axis=0) / (np.sum(exp_weights) + 1e-10)
        
        
        updated_actions = self.trajectory + weighted_delta_u
        updated_actions = np.clip(updated_actions, self.act_min, self.act_max)
    
        # Pop out first action from the trajectory and repeat last action
        self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
        self.trajectory[-1] = updated_actions[-1]

        # Output first action (MPC)
        action = updated_actions[0] 
        return action, min_cost