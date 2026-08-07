import torch
from dataclasses import dataclass
from torch import nn, functional as F


@dataclass
class Config:
    n_embedding: int = 768
    dropout: float = 0.5

class Model(nn.Module):
    def __init__(self):
        self.transformer = nn.Sequential()
        for _ in range(12):
            self.transformer.append(
                Block()
            )

class LayerNormalization(nn.Module):
    def __init__(self):
        pass

class Attention(nn.Module):
    def __init__(self):
        pass

        
class Block(nn.Module):
    def __init__(self, config):
        self.attention = Attention(config)
        self.mlp = MLP(config)
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embedding, config.n_embedding * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(config.n_embedding * 4, config.n_embedding)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x