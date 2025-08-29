# models/mlp_policy.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class MLPPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, scaler: StandardScaler, hidden_sizes=[128, 128]):
        super().__init__()
        
        # input_dim untuk MLP adalah total semua fitur yang sudah di-flatten
        input_dim = int(np.prod(obs_shape))
        
        # Simpan statistik normalisasi HANYA untuk fitur pasar
        # scaler.mean_ akan memiliki panjang (misal) 14
        obs_mean = torch.tensor(scaler.mean_, dtype=torch.float64)
        obs_std = torch.tensor(scaler.scale_, dtype=torch.float64)
        
        self.register_buffer('obs_mean', obs_mean)
        self.register_buffer('obs_std', obs_std)
        
        layers = []
        last_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
            
        self.feature_extractor = nn.Sequential(*layers)
        self.action_mean = nn.Linear(last_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float64))
        self.value_head = nn.Linear(last_size, 1)

    def forward(self, obs):
        batch_size = obs.shape[0]
        # --- REVISI UTAMA DI SINI ---
        
        # Flatten observasi untuk memisahkan fitur
        # Shape: (batch * window * assets, total_features)
        obs_flat = obs.reshape(-1, obs.shape[-1])
        # 1. Pisahkan fitur pasar (misal: 14 fitur pertama) dari fitur non-pasar (misal: 1 fitur terakhir)
        market_features = obs_flat[:, :]
        
        # 2. Normalisasi HANYA fitur pasar
        normalized_market_features = (market_features - self.obs_mean) / (self.obs_std + 1e-8)
        # 3. Gabungkan kembali fitur yang sudah dinormalisasi dengan fitur yang tidak dinormalisasi
        #processed_obs_flat = torch.cat((normalized_market_features, portfolio_weight_feature), dim=1)
        
        # Kembalikan ke shape 4D aslinya
        processed_obs = normalized_market_features.reshape(obs.shape)
        
        # --- SELESAI REVISI NORMALISASI ---

        # Flatten semua dimensi untuk masuk ke MLP
        x = processed_obs.reshape(batch_size, -1)
        
        # Sisa dari forward pass tetap sama
        x = self.feature_extractor(x)
        action_mean = self.action_mean(x)
        action_std = self.log_std.exp()
        value = self.value_head(x)

        return action_mean, action_std, value.squeeze(-1)