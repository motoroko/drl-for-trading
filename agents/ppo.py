# agents/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
from collections import defaultdict

class PPO:
    def __init__(
        self,
        policy_net,
        env,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        batch_size=64,
        n_steps=512,
        gae_lambda=0.95,
        device='cpu'
    ):
        self.env = env
        self.policy = policy_net.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.device = device
        self.scheduler = None # Akan dibuat di method train

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, state_value = self.policy(state)

        action_std = torch.clamp(action_std, min=1e-3, max=1.0)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_clamped = torch.clamp(action, 0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action_clamped.squeeze(0).cpu().numpy(), log_prob.item(), state_value.item()

    # <--- REVISI: Menggunakan 'terminals' sebagai input untuk kejelasan ---
    def compute_gae_advantages(self, rewards, terminals, values, last_value):
        advantages = []
        last_advantage = 0
        all_values = values + [last_value]
        
        for step in reversed(range(len(rewards))):
            # is_terminal adalah True jika episode berakhir alami (bukan terpotong)
            is_terminal = terminals[step] 
            
            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            # Jika terminal, V(s_{t+1}) = 0
            delta = rewards[step] + self.gamma * all_values[step + 1] * (1 - is_terminal) - all_values[step]
            
            # advantage = delta_t + gamma * lambda * A_{t+1}
            # Jika terminal, A_{t+1} = 0
            advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - is_terminal)
            
            advantages.insert(0, advantage)
            last_advantage = advantage
            
        advantages_tensor = torch.tensor(advantages, dtype=torch.float64).to(self.device)
        values_tensor = torch.tensor(values, dtype=torch.float64).to(self.device)
        returns_tensor = advantages_tensor + values_tensor
        
        return advantages_tensor, returns_tensor

    def update(self, trajectories):
        # Method ini sudah benar karena menerima data dari 'train' yang sudah diproses
        states = torch.tensor(np.array(trajectories['states']), dtype=torch.float64).to(self.device)
        actions = torch.tensor(np.array(trajectories['actions']), dtype=torch.float64).to(self.device)
        old_log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float64).to(self.device)
        
        advantages = trajectories['advantages']
        returns = trajectories['returns']
        values = torch.tensor(trajectories['values'], dtype=torch.float64).to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ... Sisa method update tidak perlu diubah ...
        # (kode loop epoch dan batch sudah benar)
        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_slice = slice(idx, idx + self.batch_size)
                batch_states = states[batch_slice]
                batch_actions = actions[batch_slice]
                batch_old_log_probs = old_log_probs[batch_slice]
                batch_returns = returns[batch_slice]
                batch_advantages = advantages[batch_slice]

                action_mean, action_std, state_values = self.policy(batch_states)
                state_values = state_values.squeeze()

                dist = Normal(action_mean, action_std)
                entropy = dist.entropy().sum(dim=-1).mean()
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - state_values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.005 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        y_true = returns.cpu().numpy()
        y_pred = values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
        clip_fraction = (ratio.gt(1 + self.clip_epsilon) | ratio.lt(1 - self.clip_epsilon)).float().mean().item()

        return {
            'loss': loss.item(), 'policy_gradient_loss': actor_loss.item(),
            'value_loss': critic_loss.item(), 'entropy_loss': -entropy.item(),
            'approx_kl': approx_kl, 'clip_fraction': clip_fraction,
            'explained_variance': explained_var, 'std': action_std.mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'], 'clip_range': self.clip_epsilon
        }

    def train(self, total_timesteps=100000, log_interval=1):
        # <--- REVISI: Menggunakan API Gymnasium reset() ---
        state, info = self.env.reset()
        start_time = time.time()
        
        total_updates = total_timesteps // self.n_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=total_updates
        )
        
        all_episode_rewards, all_episode_lengths = [], []
        current_episode_reward, current_episode_length = 0, 0
        total_steps_so_far, update_count = 0, 0
        
        while total_steps_so_far < total_timesteps:
            trajectories = defaultdict(list)

            for _ in range(self.n_steps):
                total_steps_so_far += 1
                current_episode_length += 1
                
                action, log_prob, value = self.select_action(state)
                # <--- REVISI: Menerima 5 nilai dari env.step() ---
                next_state, reward, terminated, truncated, info = self.env.step(action)

                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['rewards'].append(reward)
                # <--- REVISI: Simpan 'terminated' untuk GAE, bukan 'done' ---
                trajectories['terminals'].append(terminated)
                trajectories['values'].append(value)

                state = next_state
                current_episode_reward += reward
                
                # <--- REVISI: Cek episode selesai dengan 'terminated' atau 'truncated' ---
                if terminated or truncated:
                    all_episode_rewards.append(current_episode_reward)
                    all_episode_lengths.append(current_episode_length)
                    current_episode_reward, current_episode_length = 0, 0
                    # <--- REVISI: Gunakan API Gymnasium reset() ---
                    state, info = self.env.reset()

            with torch.no_grad():
                _, _, last_value = self.policy(torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0))
            
            # <--- REVISI: Kirim 'terminals' ke GAE, bukan 'dones' ---
            advantages, returns = self.compute_gae_advantages(
                trajectories['rewards'], trajectories['terminals'], trajectories['values'], last_value.item()
            )
            
            trajectories['advantages'] = advantages
            trajectories['returns'] = returns
            
            update_info = self.update(trajectories)
            update_count += 1
            self.scheduler.step()
            
            if update_count % log_interval == 0 and len(all_episode_rewards) > 0:
                # Blok logging Anda sudah bagus dan tidak perlu diubah
                log_values = {}
                elapsed = int(time.time() - start_time)
                
                log_values["time"] = {
                    "fps": int(total_steps_so_far / elapsed) if elapsed > 0 else 0,
                    "iterations": update_count, "time_elapsed": elapsed,
                    "total_timesteps": total_steps_so_far,
                }
                log_values["train"] = update_info
                log_values["train"]["n_updates"] = update_count
                
                if len(all_episode_lengths) > 0:
                    log_values["rollout"] = {
                        "ep_len_mean": np.mean(all_episode_lengths[-100:]),
                        "ep_rew_mean": np.mean(all_episode_rewards[-100:]),
                    }

                print("--------------------------------------------------")
                for category, items in log_values.items():
                    if not items: continue
                    print(f"| {category}/".ljust(25) + "  |".rjust(25))
                    for name, value in items.items():
                        if name in ['fps','iterations','time_elapsed','total_timesteps','n_updates']:
                            formatted_value = f"{value:d}"
                        elif abs(value) > 100000 or (abs(value) < 0.001 and value != 0):
                            formatted_value = f"{value:.5e}"
                        else:
                            formatted_value = f"{value:.5f}"
                        print(f"|    {name.ljust(20)} | {formatted_value.rjust(20)} |")
                print("--------------------------------------------------")
                
        return all_episode_rewards, all_episode_lengths