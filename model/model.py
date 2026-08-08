import math
import torch
from dataclasses import dataclass
from torch import nn, functional as F


@dataclass
class Config:
    n_embedding: int = 768
    n_head: int = 12
    dropout: float = 0.5
    n_heads: int = 12
    bias: bool = True

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

class CausalAttention(nn.Module):
    def __init__(self, config):
        assert config.n_embedding % config.n_head == 0

        self.c_attention = nn.Linear(config.n_embedding, config.n_embedding * 3, bias= config.bias)

        self.c_proj = nn.Linear(config.n_embedding, config.n_embedding, bias=config.bias)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.n_embedding = config.n_embedding
        self.dropout = config.dropout
        self.num_heads = config.num_heads

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()

        q, v, k = torch.split(self.c_attention, self.n_embedding, dim=-1)

        # sawpping dimension 1 and 2 so the score calculated is per head

        d_k = self.n_embedding // self.num_heads

        q = q.view(batch_size, sequence_length, self.num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, d_k).transpose(1, 2)

        attention = q @ k.transpose(-2, -1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=attention.device), diagonal=1)
        attention = attention.masked_fill(mask, float("-inf"))
        attention = attention / math.sqrt(d_k)
        attention = nn.Softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        y = attention @  v

        y = self.residual_dropout(self.c_proj(y))
        return y


        
class Block(nn.Module):
    def __init__(self, config):
        self.attention = CausalAttention(config)
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