# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np # Impor C-API NumPy untuk kecepatan
import pandas as pd
import gym
from gym import spaces
# Pastikan Anda memiliki referensi ke fungsi ini jika akan digunakan
# from utils.metrics import compute_deflated_sharpe_ratio

# Deklarasi kelas sebagai C extension type untuk performa
class BaselineTradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, tickers, int window_size=30, double initial_cash=1e6, double transaction_cost_pct=0.001):
        super().__init__()
        self.df = df.copy()
        self.tickers = tickers
        self.window_size = window_size
        self.initial_cash = np.float64(initial_cash)
        self.transaction_cost_pct = transaction_cost_pct
        self.num_assets = len(tickers)
        self.asset_returns = []

        # Gunakan np.float64 untuk konsistensi dengan agen
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float64)
        feature_dim = self.df.shape[1] - 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, self.num_assets, feature_dim),
            dtype=np.float64
        )
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.asset_holdings = np.zeros(self.num_assets, dtype=np.float64)
        self.done = False
        self.asset_returns = []
        return self._get_observation()

    def _get_observation(self):
        obs = []
        for ticker in self.tickers:
            data = self.df[self.df['ticker'] == ticker].iloc[self.current_step - self.window_size:self.current_step]
            obs.append(data.drop(columns=["date", "ticker"]).values)
        return np.stack(obs, axis=1).astype(np.float64)

    def _get_prices(self, int step):
        prices = []
        for ticker in self.tickers:
            try:
                price = self.df[self.df['ticker'] == ticker].iloc[step]['close']
            except IndexError:
                price = np.nan
            prices.append(price)
        return np.array(prices, dtype=np.float64)

    # Method step adalah target optimasi utama
    def step(self, np.ndarray[np.float64_t, ndim=1] actions):
        # Deklarasikan tipe variabel lokal untuk kecepatan
        cdef double portfolio_value, cost, new_portfolio_value, reward
        cdef np.ndarray[np.float64_t, ndim=1] weights, current_prices, desired_allocation, desired_holdings, delta_holdings, next_prices
        
        if self.done:
            raise RuntimeError("Episode has ended. Please call reset() to start a new episode.")
        
        weights = actions / (np.sum(actions) + 1e-8)
        current_prices = self._get_prices(self.current_step)
        
        if np.isnan(current_prices).any():
            self.done = True
            return self._get_observation(), -1.0, self.done, {'portfolio_value': self.cash}

        portfolio_value = self.cash + np.sum(self.asset_holdings * current_prices)
        desired_allocation = weights * portfolio_value
        desired_holdings = np.divide(desired_allocation, current_prices, out=np.zeros_like(desired_allocation), where=current_prices!=0)

        delta_holdings = np.abs(desired_holdings - self.asset_holdings)
        cost = np.sum(delta_holdings * current_prices * self.transaction_cost_pct)

        self.asset_holdings = desired_holdings
        self.cash = portfolio_value - np.sum(self.asset_holdings * current_prices) - cost

        self.current_step += 1
        if self.current_step >= len(self.df) // self.num_assets:
            self.done = True

        next_prices = self._get_prices(self.current_step)
        if self.done:
            next_prices = current_prices

        new_portfolio_value = self.cash + np.sum(self.asset_holdings * np.nan_to_num(next_prices))
        reward = (new_portfolio_value - portfolio_value) / (portfolio_value + 1e-8)

        if np.isnan(reward) or not np.isfinite(reward):
            reward = -1.0
            self.done = True
        
        self.asset_returns.append(reward)
        obs = self._get_observation()
        info = {'portfolio_value': new_portfolio_value}
        
        if self.done:
            # Jika Anda menggunakan DSR, pastikan fungsi ini juga dioptimalkan jika memungkinkan
            # dsr = compute_deflated_sharpe_ratio(self.asset_returns)
            # info['deflated_sharpe_ratio'] = dsr
            pass

        return obs, reward, self.done, info