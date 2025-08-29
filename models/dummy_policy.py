# models/mlp_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_sizes=[128, 128]):
        super().__init__()
        input_dim = int(torch.prod(torch.tensor(obs_shape)))
        layers = []
        last_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        self.feature_extractor = nn.Sequential(*layers)
        self.action_mean = nn.Linear(last_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # untuk aksi kontinu
        self.value_head = nn.Linear(last_size, 1)

    def forward(self, obs):
        batch_size = obs.shape[0]
        #print("obs before\n",obs)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        #mean = obs.mean(dim=(1,2,3), keepdim=True)
        #std = obs.std(dim=(1,2,3), keepdim=True).clamp(min=1e-6)
        #obs = (obs - mean) / (std + 1e-8)
        #print("obs after\n",obs)
        #print()

        x = obs.reshape(batch_size, -1)
        x = self.feature_extractor(x)

        action_mean = self.action_mean(x)
        action_mean = torch.clamp(action_mean, -10.0, 10.0)

        log_std = torch.clamp(self.log_std, min=-20, max=2)
        action_std = log_std.exp()

        value = self.value_head(x)

        if torch.isnan(action_mean).any():
            print("WARNING: NaN detected in action_mean")
        if torch.isnan(value).any():
            print("WARNING: NaN detected in value output")

        return action_mean, action_std, value.squeeze(-1)



