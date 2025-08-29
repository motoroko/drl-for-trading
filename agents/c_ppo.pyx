# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
cimport numpy as np # Impor C-API
import time
from collections import defaultdict

cdef class PPO:
    # Deklarasikan atribut dengan tipe data C untuk akses cepat
    cdef public object env, policy, optimizer, device, scheduler
    cdef public double gamma, clip_epsilon, gae_lambda
    cdef public int epochs, batch_size, n_steps

    def __init__(self, policy_net, env, lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10,
                 batch_size=64, n_steps=512, gae_lambda=0.95, device='cpu'):
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
        # Method ini didominasi PyTorch, jadi tidak ada perubahan Cython yang signifikan
        state = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, state_value = self.policy(state)
        action_std = torch.clamp(action_std, min=1e-3, max=1.0)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_clamped = torch.clamp(action, 0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action_clamped.squeeze(0).cpu().numpy(), log_prob.item(), state_value.item()

    # Fungsi ini mendapat manfaat paling besar dari Cython karena loop-nya
    def compute_gae_advantages(self, list rewards, list dones, list values, double last_value):
        # Deklarasikan tipe variabel lokal untuk menghilangkan overhead Python
        cdef:
            list advantages = []
            double last_advantage = 0.0
            list all_values = values + [last_value]
            int step
            bint is_terminal # bint adalah boolean versi C
            double delta, advantage
        
        for step in reversed(range(len(rewards))):
            is_terminal = dones[step]
            delta = rewards[step] + self.gamma * all_values[step + 1] * (1 - is_terminal) - all_values[step]
            advantage = delta + self.gamma * self.gae_lambda * last_advantage * (1 - is_terminal)
            advantages.insert(0, advantage)
            last_advantage = advantage
            
        advantages_tensor = torch.tensor(advantages, dtype=torch.float64).to(self.device)
        values_tensor = torch.tensor(values, dtype=torch.float64).to(self.device)
        returns_tensor = advantages_tensor + values_tensor
        return advantages_tensor, returns_tensor

    def update(self, trajectories):
        # Method ini didominasi PyTorch, jadi tidak ada perubahan Cython yang signifikan
        states = torch.tensor(np.array(trajectories['states']), dtype=torch.float64).to(self.device)
        actions = torch.tensor(np.array(trajectories['actions']), dtype=torch.float64).to(self.device)
        old_log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float64).to(self.device)
        advantages = trajectories['advantages']
        returns = trajectories['returns']
        values = torch.tensor(trajectories['values'], dtype=torch.float64).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Variabel untuk logging akan diambil dari batch terakhir
        loss, actor_loss, critic_loss, entropy, new_log_probs = 0, 0, 0, 0, 0
        ratio = 0
        
        for _ in range(self.epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_slice = slice(idx, idx + self.batch_size)
                batch_states, batch_actions, batch_old_log_probs = states[batch_slice], actions[batch_slice], old_log_probs[batch_slice]
                batch_returns, batch_advantages = returns[batch_slice], advantages[batch_slice]
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
                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

        y_true, y_pred = returns.cpu().numpy(), values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        approx_kl = (batch_old_log_probs - new_log_probs).mean().item()
        clip_fraction = (ratio.gt(1 + self.clip_epsilon) | ratio.lt(1 - self.clip_epsilon)).float().mean().item()

        return {'loss': loss.item(), 'policy_gradient_loss': actor_loss.item(), 'value_loss': critic_loss.item(),
                'entropy_loss': -entropy.item(), 'approx_kl': approx_kl, 'clip_fraction': clip_fraction,
                'explained_variance': explained_var, 'std': action_std.mean().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr'], 'clip_range': self.clip_epsilon}

    def train(self, total_timesteps=100000, verbose=1, log_interval=1):
        # Method ini adalah orchestrator, didominasi oleh pemanggilan fungsi lain
        # jadi tidak ada perubahan Cython yang signifikan
        state = self.env.reset()
        start_time = time.time()
        total_updates = total_timesteps // self.n_steps
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates)
        all_episode_rewards, all_episode_lengths = [], []
        current_episode_reward, current_episode_length = 0, 0
        total_steps_so_far, update_count = 0, 0

        while total_steps_so_far < total_timesteps:
            trajectories = defaultdict(list)
            for _ in range(self.n_steps):
                total_steps_so_far += 1
                current_episode_length += 1
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['rewards'].append(reward)
                trajectories['dones'].append(done)
                trajectories['values'].append(value)
                state = next_state
                current_episode_reward += reward
                if done:
                    all_episode_rewards.append(current_episode_reward)
                    all_episode_lengths.append(current_episode_length)
                    current_episode_reward, current_episode_length = 0, 0
                    state = self.env.reset()
            
            with torch.no_grad():
                _, _, last_value = self.policy(torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0))
            
            advantages, returns = self.compute_gae_advantages(trajectories['rewards'], trajectories['dones'], trajectories['values'], last_value.item())
            trajectories['advantages'], trajectories['returns'] = advantages, returns
            
            update_info = self.update(trajectories)
            update_count += 1
            self.scheduler.step()
            
            if verbose and update_count % log_interval == 0 and len(all_episode_rewards) > 0:
                log_values = {}
                elapsed = int(time.time() - start_time)
                log_values["time"] = {"fps": int(total_steps_so_far / elapsed) if elapsed > 0 else 0,
                                      "iterations": update_count, "time_elapsed": elapsed,
                                      "total_timesteps": total_steps_so_far}
                log_values["train"] = update_info
                log_values["train"]["n_updates"] = update_count
                log_values["rollout"] = {"ep_len_mean": np.mean(all_episode_lengths[-100:]),
                                         "ep_rew_mean": np.mean(all_episode_rewards[-100:])}
                                         
                print("-----------------------------------------")
                for category, items in log_values.items():
                    if not items: continue
                    print(f"| {category}/".ljust(25) + "|".rjust(24))
                    for name, value in items.items():
                        if abs(value) > 1000 or (abs(value) < 0.001 and value != 0): formatted_value = f"{value:.3e}"
                        else: formatted_value = f"{value:.5f}"
                        print(f"|    {name.ljust(20)} | {formatted_value.rjust(20)} |")
                print("-----------------------------------------")
                
        return all_episode_rewards, all_episode_lengths