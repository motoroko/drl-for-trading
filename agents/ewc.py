# agents/ewc_wrapper.py

import torch
from torch import nn
from copy import deepcopy

class EWC:
    def __init__(self, model: nn.Module, dataset, device='cpu', lambda_ewc=1000):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.lambda_ewc = lambda_ewc

        self.model.to(self.device)
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        # Init fisher information matrix
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()}

        self.model.eval()
        for data in self.dataset:
            self.model.zero_grad()
            output = self.model(data)

            # For PPO, output usually action_mean; use sum to get scalar
            loss = output[0].sum()  # sum over action_mean tensor
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.data.clone().pow(2)

        # Average fisher info over dataset size
        for n in fisher:
            fisher[n] /= len(self.dataset)

        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                _loss = self.fisher[n] * (p - self.params[n]).pow(2)
                loss += _loss.sum()
        return (self.lambda_ewc / 2) * loss
