import torch
from copy import deepcopy

class EWCWrapper:
    def __init__(self, model, ewc, lambda_ewc=1000):
        """
        model   : PPO policy/model yang dilatih
        ewc     : instance EWC class yang sudah compute fisher dan simpan params lama
        lambda_ewc : koefisien regularisasi EWC
        """
        self.model = model
        self.ewc = ewc
        self.lambda_ewc = lambda_ewc

    def compute_loss(self, policy_loss):
        """
        Tambahkan penalti EWC ke loss PPO (policy_loss)
        """
        ewc_penalty = self.ewc.penalty(self.model)
        total_loss = policy_loss + self.lambda_ewc / 2 * ewc_penalty
        return total_loss

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def parameters(self):
        return self.model.parameters()

    def train(self, mode=True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
