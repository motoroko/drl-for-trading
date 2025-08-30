import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
from collections import deque

class PPO_VecEnv:
    """
    Versi PPO yang dioptimalkan untuk bekerja dengan Vectorized Environments (VecEnv).
    """
    def __init__(
        self,
        policy_net,
        env, # env sekarang diasumsikan sebagai VecEnv
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=10,
        n_steps=2048,
        gae_lambda=0.95,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.env = env
        self.n_envs = env.num_envs
        self.policy = policy_net.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.n_steps = n_steps
        # Ukuran mini-batch untuk update, dihitung dari total data rollout
        self.batch_size = (self.n_steps * self.n_envs) // 4
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.scheduler = None

    def compute_gae_advantages(self, rewards, terminals, values, last_values):
        # rewards, terminals, values berbentuk (n_steps, n_envs)
        advantages = np.zeros_like(rewards, dtype=np.float64)
        last_advantage = np.zeros(self.n_envs, dtype=np.float64)
        
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            
            is_terminal = terminals[step]
            delta = rewards[step] + self.gamma * next_values * (1 - is_terminal) - values[step]
            advantages[step] = delta + self.gamma * self.gae_lambda * last_advantage * (1 - is_terminal)
            last_advantage = advantages[step]
        
        returns = advantages + values
        
        advantages_tensor = torch.tensor(advantages, dtype=torch.float64).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float64).to(self.device)
        
        return advantages_tensor, returns_tensor

    def update(self, states, actions, old_log_probs, advantages, returns, values):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            indices = np.random.permutation(states.shape[0])
            for idx in range(0, states.shape[0], self.batch_size):
                batch_indices = indices[idx : idx + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

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

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
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
        try:
            state, info = self.env.reset()
        except:
            state = self.env.reset()
        start_time = time.time()
        
        total_updates = total_timesteps // (self.n_steps * self.n_envs)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates
        )
        
        ep_info_buffer = deque(maxlen=100)
        
        for update_count in range(1, total_updates + 1):
            states_buf = np.zeros((self.n_steps, self.n_envs) + self.env.observation_space.shape, dtype=np.float64)
            actions_buf = np.zeros((self.n_steps, self.n_envs) + self.env.action_space.shape, dtype=np.float64)
            log_probs_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float64)
            rewards_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float64)
            terminals_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float64)
            values_buf = np.zeros((self.n_steps, self.n_envs), dtype=np.float64)

            for step in range(self.n_steps):
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float64).to(self.device)
                    action_mean, action_std, value = self.policy(state_tensor)
                    
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    action_clamped = torch.clamp(action, 0, 1)

                try:
                  next_state, rewards, terminateds, truncateds, infos = self.env.step(action_clamped.cpu().numpy())
                except:
                  next_state, rewards, terminateds, infos = self.env.step(action_clamped.cpu().numpy())
                  truncateds = False

                states_buf[step] = state
                actions_buf[step] = action_clamped.cpu().numpy()
                log_probs_buf[step] = log_prob.cpu().numpy()
                rewards_buf[step] = rewards
                terminals_buf[step] = terminateds
                values_buf[step] = value.cpu().numpy().flatten()
                
                state = next_state

                for info in infos:
                    maybe_ep_info = info.get("episode")
                    if maybe_ep_info:
                        ep_info_buffer.append(maybe_ep_info)
            
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float64).to(self.device)
                _, _, last_values_tensor = self.policy(state_tensor)
                last_values = last_values_tensor.cpu().numpy().flatten()
            
            advantages, returns = self.compute_gae_advantages(rewards_buf, terminals_buf, values_buf, last_values)

            flat_states = torch.tensor(states_buf.reshape(-1, *self.env.observation_space.shape), dtype=torch.float64).to(self.device)
            flat_actions = torch.tensor(actions_buf.reshape(-1, *self.env.action_space.shape), dtype=torch.float64).to(self.device)
            flat_log_probs = torch.tensor(log_probs_buf.flatten(), dtype=torch.float64).to(self.device)
            flat_values = torch.tensor(values_buf.flatten(), dtype=torch.float64).to(self.device)
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            
            update_info = self.update(flat_states, flat_actions, flat_log_probs, flat_advantages, flat_returns, flat_values)
            self.scheduler.step()

            if update_count % log_interval == 0:
                total_steps_so_far = update_count * self.n_steps * self.n_envs
                elapsed = int(time.time() - start_time)
                
                log_values = {}
                log_values["time"] = {
                    "fps": int(total_steps_so_far / elapsed) if elapsed > 0 else 0,
                    "iterations": update_count, "time_elapsed": elapsed,
                    "total_timesteps": total_steps_so_far,
                }
                log_values["train"] = update_info
                log_values["train"]["n_updates"] = update_count
                
                if len(ep_info_buffer) > 0:
                    log_values["rollout"] = {
                        "ep_len_mean": np.mean([ep['l'] for ep in ep_info_buffer]),
                        "ep_rew_mean": np.mean([ep['r'] for ep in ep_info_buffer]),
                    }

                print("-----------------------------------------")
                for category, items in log_values.items():
                    if not items: continue
                    print(f"| {category}/".ljust(25) + "|".rjust(24))
                    for name, value in items.items():
                        if isinstance(value, (int, np.integer)):
                            formatted_value = f"{int(value):d}"
                        elif abs(value) > 1000 or (abs(value) < 0.001 and value != 0):
                            formatted_value = f"{value:.3e}"
                        else:
                            formatted_value = f"{value:.5f}"
                        print(f"|    {name.ljust(20)} | {formatted_value.rjust(20)} |")
                print("-----------------------------------------")