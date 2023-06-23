import gym
from gym import spaces
import pandas as pd
import numpy as np
from scipy.special import softmax

returns = pd.read_pickle('/home/memerling/data/portfolio_opt.pkl')

day_lookback = 42
episode_length = 252
NUM_CHANNELS = returns.shape[1] + 1
val_cutoff = '2022-01-01'

def make_episode(day_lookback, episode_length, flatten):
    data_length = day_lookback + episode_length
    bootstrap = returns.sample(data_length)
    mean, cov = bootstrap.mean(), bootstrap.cov()
    ep = np.random.multivariate_normal(mean, cov, size=data_length)
    ep = np.pad(ep, ((0, 0), (0, 1)), constant_values=0)
    obs = np.lib.stride_tricks.sliding_window_view(ep, (day_lookback, ep.shape[1])).squeeze()
    if flatten:
        obs = obs.reshape(obs.shape[0], -1)
    return obs, ep[day_lookback:]

def safe_tstat(x):
    return np.sum(x) / np.sqrt(np.sum(np.power(x, 2)))

class PortEnv(gym.Env):
    def __init__(self, mode, day_lookback, episode_length, flatten):
        super().__init__()
        
        self.day_lookback = day_lookback
        self.episode_length = episode_length
        self.flatten = flatten
        
        if mode == 'train':
            self.returns = returns[returns.index < val_cutoff]
        elif mode == 'eval':
            self.returns = returns[returns.index >= val_cutoff]
        else:
            raise RuntimeError(f'Unrecognized mode: {mode}')
                
        self.action_space = spaces.Box(-1, 1, shape=(NUM_CHANNELS,), dtype=np.float32)
        
        if flatten:
            image_shape = (day_lookback*NUM_CHANNELS,)
        else:
            image_shape = (day_lookback, NUM_CHANNELS)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(-np.inf, np.inf, shape=image_shape, dtype=np.float32),
            'pos': spaces.Box(0, 1, shape=(NUM_CHANNELS,), dtype=np.float32),
        })
        
    def reset(self):
                
        self.obs, self.next_ret = make_episode(self.day_lookback, self.episode_length, self.flatten)
            
        ## State values ##
        
        self.pos = np.empty(shape=(self.episode_length+1, NUM_CHANNELS))
        self.pos[:] = np.nan
        self.pos[0] = self.pos[-1] = 0
        self.pos[0, -1] = 1 # all cash
        
        self.t = 0
        
        self.dayret = np.empty(shape=self.episode_length+1)
        self.dayret[:] = np.nan
        self.dayret[0] = self.dayret[-1] = 0
        
        ## Current observation
        
        obs = self._get_obs()
        obs['is_terminal'] = False
        obs['is_first'] = True
        
        return obs
        
    def _get_obs(self):
        return {
            'image': self.obs[self.t],
            'pos': self.pos[self.t],
        }
    
    def _get_info(self):
        return {
            'dayret': self.dayret[self.t],
        }
    
    def step(self, action):
        action = softmax(action)
        terminated = self.t >= self.episode_length - 1
        
        if terminated:
            assert not np.isnan(self.dayret).any()
            reward = safe_tstat(self.dayret)
        else:
            reward = 0
            self.pos[self.t+1] = action
            self.dayret[self.t+1] = (action*self.next_ret[self.t]).sum()
        
        self.t += 1

        obs = self._get_obs()
        obs['is_terminal'] = terminated
        obs['is_first'] = False
        
        info = self._get_info()
        
        return obs, reward, terminated, info