import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
from collections import defaultdict, deque
import csv
import os
from typing import Dict, Any

# Pastikan Anda bisa mengimpor MLPPolicy dan BaselineTradingEnv dari lokasi yang benar
# from models.mlp_policy import MLPPolicy
# from envs.baseline_trading_env import BaselineTradingEnv

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
        n_steps=2048,
        gae_lambda=0.95,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device='cpu'
    ):
        self.env = env
        self.device = torch.device(device)
        self.policy = policy_net.to(self.device)
        
        # Simpan hyperparameter untuk di-save nanti
        self.hp = {
            "lr": lr, "gamma": gamma, "clip_epsilon": clip_epsilon, "epochs": epochs,
            "batch_size": batch_size, "n_steps": n_steps, "gae_lambda": gae_lambda,
            "ent_coef": ent_coef, "vf_coef": vf_coef, "max_grad_norm": max_grad_norm
        }

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.hp["lr"], eps=1e-5)
        self.scheduler = None

    def _get_action_and_value(self, state):
        """ Helper function internal untuk mendapatkan aksi dan value dari policy. """
        state = torch.tensor(state, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, state_value = self.policy(state)

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action_clamped = torch.clamp(action, 0, 1)

        return action_clamped.squeeze(0).cpu().numpy(), log_prob.item(), state_value.item()
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Memprediksi aksi dari observasi, mirip seperti SB3.
        :param obs: Observasi dari lingkungan.
        :param deterministic: True untuk mengambil aksi rata-rata, False untuk sampling acak.
        :return: Aksi dalam bentuk array NumPy.
        """
        state = torch.tensor(obs, dtype=torch.float64).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, _ = self.policy(state)
        
        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            
        action_clamped = torch.clamp(action, 0, 1)
        return action_clamped.squeeze(0).cpu().numpy()

    def compute_gae_advantages(self, rewards, terminals, values, last_value):
        advantages, last_advantage = [], 0
        all_values = values + [last_value]
        for step in reversed(range(self.hp["n_steps"])):
            is_terminal = terminals[step]
            delta = rewards[step] + self.hp["gamma"] * all_values[step + 1] * (1 - is_terminal) - all_values[step]
            advantage = delta + self.hp["gamma"] * self.hp["gae_lambda"] * last_advantage * (1 - is_terminal)
            advantages.insert(0, advantage)
            last_advantage = advantage
        advantages_t = torch.tensor(advantages, dtype=torch.float64).to(self.device)
        returns_t = advantages_t + torch.tensor(values, dtype=torch.float64).to(self.device)
        return advantages_t, returns_t

    def update(self, trajectories):
        states = torch.tensor(np.array(trajectories['states']), dtype=torch.float64).to(self.device)
        actions = torch.tensor(np.array(trajectories['actions']), dtype=torch.float64).to(self.device)
        old_log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float64).to(self.device)
        advantages = trajectories['advantages']
        returns = trajectories['returns']
        values = torch.tensor(trajectories['values'], dtype=torch.float64).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.hp["epochs"]):
            indices = np.random.permutation(self.hp["n_steps"])
            for idx in range(0, self.hp["n_steps"], self.hp["batch_size"]):
                batch_indices = indices[idx : idx + self.hp["batch_size"]]
                new_values, new_log_probs, entropy = self.policy.evaluate_actions(states[batch_indices], actions[batch_indices])
                
                ratio = (new_log_probs - old_log_probs[batch_indices]).exp()
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.hp["clip_epsilon"], 1 + self.hp["clip_epsilon"]) * advantages[batch_indices]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(returns[batch_indices], new_values.squeeze())
                loss = actor_loss + self.hp["vf_coef"] * critic_loss - self.hp["ent_coef"] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp["max_grad_norm"])
                self.optimizer.step()
        
        y_true, y_pred = returns.cpu().numpy(), values.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            'loss': loss.item(), 'policy_gradient_loss': actor_loss.item(),
            'value_loss': critic_loss.item(), 'entropy_loss': -entropy.item(),
            'explained_variance': explained_var, 'std': self.policy.log_std.exp().mean().item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'], 'clip_range': self.hp["clip_epsilon"]
        }

    def train(self, total_timesteps: int, log_interval: int = 1):
        state, info = self.env.reset()
        start_time = time.time()
        
        total_updates = total_timesteps // self.hp["n_steps"]
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_updates)
        
        ep_info_buffer = deque(maxlen=100)
        
        training_start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"PPO_train_{training_start_time}.csv"
        csv_file = open(log_filename, "w", newline="")
        csv_writer = None
        print(f"Logging training metrics to {log_filename}")
        
        total_steps_so_far, update_count = 0, 0
        while total_steps_so_far < total_timesteps:
            trajectories = defaultdict(list)
            for _ in range(self.hp["n_steps"]):
                total_steps_so_far += 1
                
                action, log_prob, value = self._get_action_and_value(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                trajectories['states'].append(state)
                trajectories['actions'].append(action)
                trajectories['log_probs'].append(log_prob)
                trajectories['rewards'].append(reward)
                trajectories['terminals'].append(terminated)
                trajectories['values'].append(value)
                state = next_state
                
                if terminated or truncated:
                    if "episode" in info:
                        ep_info_buffer.append(info["episode"])
                    state, info = self.env.reset()
            
            with torch.no_grad():
                _, _, last_value = self._get_action_and_value(state)
            
            advantages, returns = self.compute_gae_advantages(trajectories['rewards'], trajectories['terminals'], trajectories['values'], last_value)
            trajectories['advantages'], trajectories['returns'] = advantages, returns
            
            update_info = self.update(trajectories)
            self.scheduler.step()
            update_count += 1
            
            if log_interval is not None and update_count % log_interval == 0:
                log_values = self._log_training(start_time, total_steps_so_far, update_count, update_info, ep_info_buffer)
                
                if csv_writer is None:
                    header = list(log_values.keys())
                    csv_writer = csv.DictWriter(csv_file, fieldnames=header)
                    csv_writer.writeheader()
                csv_writer.writerow(log_values)
                csv_file.flush()

        csv_file.close()

    def _log_training(self, start_time, total_steps, update_count, update_info, ep_info_buffer):
        elapsed = int(time.time() - start_time)
        log_values = {
            "time/fps": int(total_steps / elapsed) if elapsed > 0 else 0,
            "time/iterations": update_count,
            "time/time_elapsed": elapsed,
            "time/total_timesteps": total_steps,
        }
        if len(ep_info_buffer) > 0:
            log_values["rollout/ep_len_mean"] = np.mean([ep['l'] for ep in ep_info_buffer])
            log_values["rollout/ep_rew_mean"] = np.mean([ep['r'] for ep in ep_info_buffer])
        
        log_values.update({f"train/{k}": v for k, v in update_info.items()})

        print("-----------------------------------------")
        for name, value in log_values.items():
            if isinstance(value, (int, np.integer)): formatted_value = f"{int(value):d}"
            else: formatted_value = f"{value:.5f}"
            print(f"| {name.ljust(30)} | {formatted_value.rjust(20)} |")
        print("-----------------------------------------")
        return log_values

    def save(self, path: str):
        agent_state = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "hyperparameters": self.hp
        }
        torch.save(agent_state, path)
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path: str, env, policy_class, scaler, device: str = 'cpu'):
        saved_state = torch.load(path, map_location=device)
        
        policy = policy_class(
            obs_shape=env.observation_space.shape,
            action_dim=env.action_space.shape[0],
            scaler=scaler
        ).double().to(device)
        policy.load_state_dict(saved_state["policy_state_dict"])
        
        agent = cls(policy_net=policy, env=env, device=device, **saved_state["hyperparameters"])
        agent.optimizer.load_state_dict(saved_state["optimizer_state_dict"])
        
        # Scheduler perlu dibuat ulang sebelum memuat state-nya
        if saved_state["scheduler_state_dict"]:
            # Kita perlu tahu total_timesteps untuk membuat scheduler dengan benar
            # Untuk sekarang, kita buat placeholder atau lewati
            print("Warning: Scheduler state loaded but may need total_timesteps to be re-initialized correctly in train().")

        print(f"Agent policy and optimizer loaded from {path}")
        return agent
