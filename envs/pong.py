import gym
from gym import spaces
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.special import softmax
import datetime

train_data = pd.read_pickle('/home/memerling/data/trade_train_data.pkl')
val_data = pd.read_pickle('/home/memerling/data/trade_val_data.pkl')
data_cols = train_data.columns.levels[1].difference(['buy_price', 'sell_price'])

sym_order = [
    'DRIP', 'GUSH', # oil & gas
    'UCO', 'SCO', # crude oil
    'SPXL', 'SPXS', # SP500
    'TQQQ', 'SQQQ', # NASDAQ
    'UVXY', 'SVXY', # VIX
    'TNA', 'TZA', # small cap
    'LABU', 'LABD', # biotech
    'SOXL', 'SOXS', # semiconductor
]
assert set(train_data.columns.levels[0]) == set(sym_order)

def make_episode(day, context_length, close_delta):
    eod = datetime.time(hour=13) if len(day) <= 210 else datetime.time(hour=16)
    eod = pd.Timestamp.combine(day.index[0].date(), eod).tz_localize('America/New_York') \
        - pd.Timedelta(minutes=close_delta)
    
    cutoff_eod = day[day.index <= eod].copy()
    
    # obs = cutoff_eod.loc[:, obs_indexer].values
    obs = cutoff_eod.loc[:, (slice(None), data_cols)]
    ordered = obs[sym_order].values
    obs = np.stack((obs.values, ordered, np.roll(ordered, len(data_cols), axis=1)), axis=-1)
    obs = np.lib.stride_tricks.sliding_window_view(obs, (context_length, *obs.shape[1:]))[1:].squeeze()
    
    cutoff_context = cutoff_eod.iloc[context_length:]
    buy_price = cutoff_context.loc[:, (slice(None), 'buy_price')].values
    sell_price = cutoff_context.loc[:, (slice(None), 'sell_price')].values
    mid = 0.5*(buy_price + sell_price)
    spd = buy_price - sell_price
    
    assert obs.shape[0] == cutoff_context.shape[0]
    
    return obs, eod, mid, spd, cutoff_context.index

def make_nan(*shape):
    arr = np.empty(shape, dtype=float)
    arr[:] = np.nan
    return arr

class IntradayTraderEnv(gym.Env):
    fee_sec = 8/1000000
    fee_finra = 0.000145
    max_finra = 7.27
    
    def __init__(self, mode, context_length, close_delta, liq_thresh, start_cash):
        super().__init__()

        if mode == 'train':
            data = train_data
        elif mode == 'eval':
            data = val_data
        else:
            raise RuntimeError(f"Unrecognized mode: {mode}")
        
        grouped = data.groupby(data.index.date)
        self.dates = list(grouped.groups.keys())
        self.episodes = {date: make_episode(grouped.get_group(date), 
                                            context_length,
                                            close_delta) for date in tqdm(self.dates)}
        self.syms = train_data.columns.levels[0].tolist()
        self.nsyms = len(self.syms)
        self.context_length = context_length
        self.liq_thresh = liq_thresh
        self.start_cash = start_cash
        
        ## The first self.nsyms+1 are logits for the portfolio weights, the last being cash
        ## The last element, a, is a binary decision to trade on these logits or not (trade iff a > 0)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.nsyms+2,), dtype=np.float32)
        
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(-np.inf, np.inf, 
                                    shape=(context_length, self.nsyms*len(data_cols), 3),
                                    dtype=np.float32),
                "pos_weights": spaces.Box(0, 1, shape=(self.nsyms,), dtype=np.float32),
                "time_to_close": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            }
        )
        
        self.time_to_close_normalizer = pd.Timedelta(minutes=390-context_length-close_delta)
        
    def reset(self):
        
        ## Episode Constants ##
        
        self.date = np.random.choice(self.dates)
        
        self.obs, self.eod, self.mid, self.spd, self.times = self.episodes[self.date]
        self.day_length = len(self.times)

        ## State Values ##
        
        self.t = 0
        
        self.pos_weights = make_nan(self.day_length, self.nsyms)
        self.pos_weights[0] = 0

        self.cash = make_nan(self.day_length)
        self.cash[0] = self.start_cash
        
        self.portfolio_value = make_nan(self.day_length)
        self.portfolio_value[0] = self.start_cash
        
        self.shares = self.pos_weights.copy()
        self.target_weights = self.pos_weights.copy()
        
        self.one_step_returns = make_nan(self.day_length)
        self.one_step_returns[0] = 0
        
        self.fees = make_nan(self.day_length)
        self.fees[0] = 0
        
        ## Current Observation ##
        
        observation = self._get_obs()
        
        observation['is_terminal'] = False
        observation['is_first'] = True
        
        return observation
    
    def _get_obs(self):
        return {"image": self.obs[self.t],
                 "pos_weights": self.pos_weights[self.t],
                 "time_to_close": (self.eod - self.times[self.t:self.t+1])/self.time_to_close_normalizer}
    
    def _get_info(self):
        return {"one_step_returns": self.one_step_returns[self.t],
                "cash": self.cash[self.t],
                "portfolio_value": self.portfolio_value[self.t]}
    
    def step(self, action):
             
        current_portfolio_value = np.dot(self.shares[self.t], self.mid[self.t+1]) + self.cash[self.t]
        self.one_step_returns[self.t+1] = current_portfolio_value/self.portfolio_value[self.t] - 1
        if (self.t+1 >= self.day_length - 1) or (self.one_step_returns[self.t+1] < -self.liq_thresh):
            terminated = True
            
            self.target_weights[self.t+1] = 0
            sell_prices = self.mid[self.t+1] - 0.5*self.spd[self.t+1]
            
            shares_sold = self.shares[self.t]
            notional_sold = shares_sold * sell_prices
            self.cash[self.t+1] = np.sum(notional_sold) + self.cash[self.t]
            self.portfolio_value[self.t+1] = self.cash[self.t+1]
            
            self.shares[self.t+1] = 0
            self.pos_weights[self.t+1] = 0                        
        else:
            terminated = False
            
            if action[-1] > 0: # agent decided to trade
                self.target_weights[self.t+1] = softmax(action)[:-2]
                D = (self.target_weights[self.t+1] - self.pos_weights[self.t])*current_portfolio_value

                buysell_prices = self.mid[self.t+1] + 0.5*np.sign(D)*self.spd[self.t+1]
                target_shares_delta = D / buysell_prices
                shares_delta = np.maximum(target_shares_delta, -self.shares[self.t])
                self.shares[self.t+1] = self.shares[self.t] + shares_delta
                assert np.all(self.shares[self.t+1] >= 0)

                notional_ordered = shares_delta*buysell_prices
                self.cash[self.t+1] = self.cash[self.t] - np.sum(notional_ordered)

                notional_sold = -np.minimum(0, notional_ordered)
                shares_sold = -np.minimum(0, shares_delta)
            else: # agent decided not to trade
                self.target_weights[self.t+1] = self.pos_weights[self.t]
                self.shares[self.t+1] = self.shares[self.t]
                self.cash[self.t+1] = self.cash[self.t]
                
                shares_sold = np.zeros_like(self.shares[self.t])
                notional_sold = shares_sold

            pos_notionals = self.shares[self.t+1]*self.mid[self.t+1]
            self.portfolio_value[self.t+1] = np.sum(pos_notionals) + self.cash[self.t+1]
            self.pos_weights[self.t+1] = pos_notionals / self.portfolio_value[self.t+1]
            
        ## Fees ##
        self.fees[self.t+1] = np.sum(self.fee_sec*notional_sold + 
                                     np.minimum(self.max_finra, self.fee_finra*shares_sold))
        
        self.t += 1
        
        observation = self._get_obs()
        info = self._get_info()
        
        observation['is_terminal'] = terminated
        observation['is_first'] = False
        
        if terminated:
            reward = (self.portfolio_value[self.t] - np.nansum(self.fees)) / self.portfolio_value[0]
        else:
#             reward = self.one_step_returns[self.t] - self.fees[self.t]
            reward = 0
        
        return observation, reward, terminated, info
    