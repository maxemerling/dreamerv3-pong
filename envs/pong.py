import gym
from gym import spaces
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.special import softmax
import datetime
import pickle

with open('/home/memerling/data/simple_env_data.pkl', 'rb') as f:
    data = pickle.load(f)

dates = pd.Index(data.keys())
val_cutoff = datetime.date(2022, 1, 1)
NUM_CHANNELS = 16
DAY_LOOKBACK = 5
DAY_CHANNELS = 4

def make_episode(day, context_length, close_delta, flatten):
    base, daily_data, auction_size_z, fill_prices = day
    
    base.index = base.index.tz_convert('America/New_York')
    fill_prices.index = fill_prices.index.tz_convert('America/New_York')

    eod = datetime.time(hour=13) if len(base) <= 210 else datetime.time(hour=16)
    eod = pd.Timestamp.combine(base.index[0].date(), eod).tz_localize('America/New_York') \
        - pd.Timedelta(minutes=close_delta)

    cutoff_eod_mask = base.index <= eod
    num_feat = base.shape[1]
    intraday_obs = np.lib.stride_tricks.sliding_window_view(base[cutoff_eod_mask].values, 
                                                            (context_length, num_feat)).squeeze()
    if flatten:
        intraday_obs = intraday_obs.reshape(-1, context_length*num_feat)

    fill_prices = fill_prices[cutoff_eod_mask].iloc[context_length-1:]

    assert intraday_obs.shape[0] == fill_prices.shape[0]
    
    times = fill_prices.index
    auction_size_z = np.array([auction_size_z])
    mid = (0.5*(fill_prices['ask'] + fill_prices['bid'])).values
    spd = (fill_prices['ask'] - fill_prices['bid']).values
    
    return intraday_obs, daily_data, auction_size_z, mid, spd, eod, times

class PongEnv(gym.Env):
    fee_sec = 8/1000000
    fee_finra = 0.000145
    max_finra = 7.27 ## NOTE: this is not used as of now since we use fees per share and assume the limit is never hit.
    
    def __init__(self, mode, context_length, close_delta, flatten, punish_factor, reward_multiplier):
        super().__init__()
        
        if mode == 'train':
            self.dates = dates[dates < val_cutoff]
        elif mode == 'eval':
            self.dates = dates[dates >= val_cutoff]
        else:
            raise RuntimeError(f'Unrecognized mode: {mode}')
            
        self.episodes = {date: make_episode(data[date], context_length, close_delta, flatten) for date in tqdm(self.dates)}
        
        self.action_space = spaces.Discrete(3)
        
        if flatten:
            image_shape = (context_length*NUM_CHANNELS,)
        else:
            image_shape = (context_length, NUM_CHANNELS)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(-np.inf, np.inf, shape=image_shape, dtype=np.float32),
            'daily_data': spaces.Box(-np.inf, np.inf, shape=(DAY_LOOKBACK*DAY_CHANNELS,), dtype=np.float32),
            'auction_size_z': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            'pos': spaces.Discrete(3), # same as action space
            'time_to_close': spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        })
        
        self.time_to_close_normalizer = pd.Timedelta(minutes=390-context_length-close_delta)
        
        self.punish_factor = punish_factor # multiplier for reward of trades that lose money
        print(f'PUNISH FACTOR: {self.punish_factor}')

        self.reward_multiplier = reward_multiplier # multiplier for reward of all trades
        print(f'REWARD MULTIPLIER: {self.reward_multiplier}')
        
    def reset(self):
        
        ## Episode constants ##
        
        self.date = np.random.choice(self.dates)
        self.intraday_obs, self.daily_data, self.auction_size_z, self.mid, self.spd, self.eod, self.times \
            = self.episodes[self.date]
        self.day_length = len(self.times)
        
        ## State values ##
        
        self.t = 0
        self.pos = 0
        self.open_pos_price = np.nan
        self.close_pos_price = np.nan
        self.fees_per_dollar = 0
        self.returns_per_dollar = 0
        self.action = np.nan
        
        ## Current observation
        
        obs = self._get_obs()
        obs['is_terminal'] = False
        obs['is_first'] = True

        return obs
        
    def _get_obs(self):
        return {
            'image': self.intraday_obs[self.t],
            'daily_data': self.daily_data,
            'auction_size_z': self.auction_size_z,
            'pos': self.pos + 1,
            'time_to_close': (self.eod - self.times[self.t:self.t+1])/self.time_to_close_normalizer
        }
    
    def _get_info(self):
        return {
            'open_pos_price': self.open_pos_price,
            'close_pos_price': self.close_pos_price,
            'fees_per_dollar': self.fees_per_dollar,
            'returns_per_dollar': self.returns_per_dollar,
            # 'action': self.action+1, ### no need, already saved
            'time': self.times[self.t].value
        }
    
    def step(self, action):
        action -= 1
        assert action == -1 or action == 0 or action == 1
        self.action = action

        terminated = self.t+1 >= self.day_length - 1
        
        if not np.isnan(self.close_pos_price): #just closed one bar ago
            self.open_pos_price = np.nan
            self.close_pos_price = np.nan
        self.close_pos_price = np.nan
        self.fees_per_dollar = 0
        self.returns_per_dollar = 0
        
        if self.pos == 0:
            if action != 0:
                assert np.isnan(self.open_pos_price)
                self.open_pos_price = self.mid[self.t] + action*self.spd[self.t]/2
                self.pos = action
        else:
            if action == 0 or terminated:
                assert np.isnan(self.close_pos_price)
                self.close_pos_price = self.mid[self.t] - self.pos*self.spd[self.t]/2
                self.returns_per_dollar = self.pos * ((self.close_pos_price / self.open_pos_price) - 1)
                if self.pos == 1:
                    base_price = self.open_pos_price
                    sale_price = self.close_pos_price
                else: # self.pos == -1
                    sale_price = base_price = self.open_pos_price
                self.fees_per_dollar = (sale_price*self.fee_sec + self.fee_finra)/base_price
                self.pos = 0
                
        reward = self.returns_per_dollar - self.fees_per_dollar
        if reward < 0:
            reward *= self.punish_factor
        reward *= self.reward_multiplier
        
        self.t += 1

        obs = self._get_obs()
        obs['is_terminal'] = terminated
        obs['is_first'] = False
        
        info = self._get_info()

        return obs, reward, terminated, info