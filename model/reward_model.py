import torch
import torch.nn as nn
from model import Model
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.body = Model(config)
        self.head = nn.Linear(config.n_embedding)

    def score(self, ids):
        h = self.body(ids)
        last = h[:, -1, :]
        return self.head(last).squeeze(-1)
