import numpy as np
import gym
from gym import spaces
from torch_geometric.data import Data
from utils.metrics import compute_deflated_sharpe_ratio

class GraphTradingEnv(gym.Env):
    def __init__(self, graph_sequence, transaction_cost_pct=0.001, risk_window=30, initial_cash=1e6):
        """
        Args:
            graph_sequence (list of torch_geometric.data.Data): List of daily graph-structured market states
            transaction_cost_pct (float): Biaya transaksi (misalnya 0.001 = 0.1%)
            risk_window (int): Window size untuk menghitung risk-adjusted reward
            initial_cash (float): Modal awal
        """
        super().__init__()
        self.graph_sequence = graph_sequence
        self.transaction_cost_pct = transaction_cost_pct
        self.initial_cash = initial_cash
        self.risk_window = risk_window

        self.num_assets = graph_sequence[0].x.shape[0]
        self.num_features = graph_sequence[0].x.shape[1]

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_assets, self.num_features), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=self.num_assets - 1, shape=(2, None), dtype=np.int64),
            'edge_attr': spaces.Box(low=-np.inf, high=np.inf, shape=(None,), dtype=np.float32)
        })

        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.asset_holdings = np.zeros(self.num_assets)
        self.done = False

        self.history = []

        return self._get_observation()

    def _get_observation(self):
        return self.graph_sequence[self.current_step]

    def step(self, actions):
        actions = np.clip(actions, 0, 1)
        weights = actions / (np.sum(actions) + 1e-8)

        current_graph = self.graph_sequence[self.current_step]
        prices = current_graph.x[:, 0].numpy()  # asumsi kolom 0 adalah harga 'close'

        portfolio_value = self.cash + np.sum(self.asset_holdings * prices)

        new_holdings = (weights * portfolio_value) / (prices + 1e-8)
        trades = np.abs(new_holdings - self.asset_holdings)
        transaction_cost = np.sum(trades * prices) * self.transaction_cost_pct

        self.asset_holdings = new_holdings
        self.cash = 0

        self.current_step += 1
        if self.current_step >= len(self.graph_sequence):
            self.done = True

        next_graph = self.graph_sequence[self.current_step]
        next_prices = next_graph.x[:, 0].numpy()

        new_portfolio_value = np.sum(self.asset_holdings * next_prices)
        profit = new_portfolio_value - portfolio_value - transaction_cost

        self.history.append(profit)

        if len(self.history) >= self.risk_window:
            reward = compute_deflated_sharpe_ratio(self.history[-self.risk_window:])
        else:
            reward = profit  # fallback sementara

        info = {'portfolio_value': new_portfolio_value}

        return self._get_observation(), reward, self.done, info
