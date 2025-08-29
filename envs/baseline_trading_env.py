import numpy as np
import pandas as pd
import gymnasium as gym # <--- Menggunakan Gymnasium standar baru
from gymnasium import spaces
# from utils.metrics import compute_deflated_sharpe_ratio # Opsional

class BaselineTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(
        self,
        df,
        tickers,
        window_size=30,
        initial_cash=1e6,
        transaction_cost_pct=0.001
    ):
        super().__init__()

        self.df = df.copy()
        self.tickers = tickers
        self.window_size = window_size
        self.initial_cash = np.float64(initial_cash)
        self.transaction_cost_pct = transaction_cost_pct
        self.num_assets = len(tickers)
        self.asset_returns = []

        # <--- REVISI 1: Tambahkan 1 fitur baru untuk "bobot portfolio saat ini" ---
        # Jumlah fitur pasar (harga, volume, dll.)
        market_feature_dim = df.shape[1] - 2 
        # Tambahkan 1 fitur untuk bobot portfolio
        self.total_feature_dim = market_feature_dim

        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float64)

        # Definisikan observation space dengan dimensi fitur yang baru
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_assets, self.total_feature_dim),
            dtype=np.float64
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) # <--- API Gymnasium baru
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.asset_holdings = np.zeros(self.num_assets, dtype=np.float64)
        self.done = False
        self.asset_returns = []

        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        # 1. Dapatkan data pasar seperti biasa
        market_obs_list = []
        for ticker in self.tickers:
            data = self.df[self.df['ticker'] == ticker].iloc[self.current_step - self.window_size:self.current_step]
            market_obs_list.append(data.drop(columns=["date", "ticker"]).values)
        market_obs = np.stack(market_obs_list, axis=1).astype(np.float64)

        # <--- REVISI 2: Hitung bobot portfolio saat ini dan tambahkan ke observasi ---
        # Hitung nilai portfolio total saat ini
        current_prices = self._get_prices(self.current_step)
        asset_values = self.asset_holdings * np.nan_to_num(current_prices)
        portfolio_value = self.cash + np.sum(asset_values)
        
        # Hitung bobot (cegah pembagian dengan nol jika portfolio value adalah 0)
        if portfolio_value > 1e-8:
            current_weights = asset_values / portfolio_value
        else:
            current_weights = np.zeros(self.num_assets)
            
        # Ubah shape bobot agar bisa digabungkan dengan observasi pasar
        # Shape bobot: (num_assets,) -> (1, num_assets, 1)
        # Lalu di-broadcast menjadi (window_size, num_assets, 1)
        weights_obs = np.broadcast_to(
            current_weights.reshape(1, self.num_assets, 1),
            (self.window_size, self.num_assets, 1)
        )

        # Gabungkan observasi pasar dengan observasi bobot
        # Hasilnya akan memiliki shape (window_size, num_assets, total_feature_dim)
        #full_obs = np.concatenate((market_obs, weights_obs), axis=2)
        
        return market_obs #full_obs

    def _get_prices(self, step):
        prices = []
        for ticker in self.tickers:
            try:
                price = self.df[self.df['ticker'] == ticker].iloc[step]['close']
            except IndexError:
                price = np.nan
            prices.append(price)
        return np.array(prices, dtype=np.float64)

    def step(self, actions):
        # ==============================================================================
        # [DEBUG] HEADER - Menandai awal dari satu panggilan 'step'
        # print(f"\n{'='*25} [DEBUG] Inside env.step() at step: {self.current_step} {'='*25}")
        # print(f"[DEBUG] KAS AWAL: Rp {self.cash:,.2f} | HOLDINGS AWAL (lembar): {self.asset_holdings}")
        # print(f"[DEBUG] Aksi mentah dari Agen (Target Bobot Baru): {actions}")
        # ==============================================================================

        if self.done:
            raise RuntimeError("Episode has ended. Please call reset() to start a new episode.")

        # --- LANGKAH 1: SETUP ---
        target_weights = actions / (np.sum(actions) + 1e-8)
        current_prices = self._get_prices(self.current_step)
        if np.isnan(current_prices).any():
            self.done = True; return self._get_observation(), -1.0, True, False, {'portfolio_value': self.cash}
        
        # --- LANGKAH 2: HITUNG KEADAAN PORTFOLIO SAAT INI ---
        current_asset_values = self.asset_holdings * current_prices
        portfolio_value_before_trade = self.cash + np.sum(current_asset_values)
        
        current_weights = np.zeros(self.num_assets)
        if portfolio_value_before_trade > 1e-8:
            current_weights = current_asset_values / portfolio_value_before_trade

        # print("\n--- LANGKAH 2: KEADAAN PORTFOLIO SAAT INI ---")
        # print(f"Harga Aset Hari Ini (Rp): {current_prices}")
        # print(f"Nilai Total Portfolio (Rp): {portfolio_value_before_trade:,.2f}")
        # print(f"Bobot Portfolio Saat Ini (%): {current_weights}")

        # --- LANGKAH 3: BUAT RENCANA PERDAGANGAN ---
        weight_diff = target_weights - current_weights
        trade_value = weight_diff * portfolio_value_before_trade
        # Nilai positif berarti "target beli", negatif berarti "target jual"

        # print("\n--- LANGKAH 3: BUAT RENCANA PERDAGANGAN ---")
        # print(f"Target Bobot Baru (%): {target_weights}")
        # print(f"Selisih Bobot (%): {weight_diff}")
        # print(f"Rencana Nilai Perdagangan (Rp): {trade_value}")

        # --- LANGKAH 4: EKSEKUSI PENJUALAN (SELL) ---
        # print("\n--- LANGKAH 4: EKSEKUSI PENJUALAN (SELL) ---")
        has_sold = False
        for i in range(self.num_assets):
            if trade_value[i] < 0:
                has_sold = True
                value_to_sell = -trade_value[i]
                shares_to_sell_float = value_to_sell / (current_prices[i] + 1e-8)
                shares_to_sell_int = np.floor(shares_to_sell_float)
                shares_to_sell_int = min(shares_to_sell_int, self.asset_holdings[i])

                if shares_to_sell_int > 0:
                    actual_sold_value = shares_to_sell_int * current_prices[i]
                    cash_from_sale = actual_sold_value * (1 - self.transaction_cost_pct)
                    # print(f"  Aset-{i}: Jual {shares_to_sell_int} lembar senilai Rp {actual_sold_value:,.2f}")
                    self.asset_holdings[i] -= shares_to_sell_int
                    self.cash += cash_from_sale
                    # print(f"    -> Kas setelah penjualan: Rp {self.cash:,.2f}")
        # if not has_sold:
        #     print("  Tidak ada aset yang dijual.")

        # --- LANGKAH 5: EKSEKUSI PEMBELIAN (BUY) ---
        # print("\n--- LANGKAH 5: EKSEKUSI PEMBELIAN (BUY) ---")
        has_bought = False
        for i in range(self.num_assets):
            if trade_value[i] > 0:
                has_bought = True
                value_to_buy_target = trade_value[i]
                
                # Hitung berapa lembar yang bisa dibeli dengan sisa kas
                spendable_cash = self.cash / (1 + self.transaction_cost_pct)
                shares_buyable = np.floor(spendable_cash / (current_prices[i] + 1e-8))
                shares_target_to_buy = np.floor(value_to_buy_target / (current_prices[i] + 1e-8))
                shares_to_buy = min(shares_target_to_buy, shares_buyable)

                if shares_to_buy > 0:
                    actual_buy_value = shares_to_buy * current_prices[i]
                    total_cost = actual_buy_value * (1 + self.transaction_cost_pct)
                    # print(f"  Aset-{i}: Beli {shares_to_buy} lembar (Total biaya: Rp {total_cost:,.2f})")
                    self.asset_holdings[i] += shares_to_buy
                    self.cash -= total_cost
                    # print(f"    -> Sisa kas: Rp {self.cash:,.2f}")
        # if not has_bought:
        #     print("  Tidak ada aset yang dibeli.")
        
        # --- LANGKAH 6 & 7: TUTUP BUKU & HITUNG HASIL ---
        self.current_step += 1
        if self.current_step >= len(self.df) // self.num_assets: self.done = True
        
        next_prices = self._get_prices(self.current_step)
        if self.done: next_prices = current_prices
        
        portfolio_value_after_trade = self.cash + np.sum(self.asset_holdings * np.nan_to_num(next_prices))
        reward = (portfolio_value_after_trade - portfolio_value_before_trade) / (portfolio_value_before_trade + 1e-8)

        # print("\n--- LANGKAH 6 & 7: HASIL AKHIR ---")
        # print(f"Harga Penutupan Besok: {next_prices}")
        # print(f"KAS AKHIR: Rp {self.cash:,.2f} | HOLDINGS AKHIR (lembar): {self.asset_holdings}")
        # print(f"Nilai Total Portfolio SETELAH trade (berdasarkan harga besok): Rp {portfolio_value_after_trade:,.2f}")
        # print(f"Reward untuk step ini: {reward:.6f}")
        # print(f"{'='*28} [END DEBUG] Step: {self.current_step-1} {'='*28}")

        if np.isnan(reward) or not np.isfinite(reward):
            reward = -1.0; self.done = True
        
        self.asset_returns.append(reward)
        obs = self._get_observation()
        info = {'portfolio_value': portfolio_value_after_trade}

        terminated, truncated = self.done, False
        return obs, reward, terminated, truncated, info