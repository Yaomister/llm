import math
import torch
from dataclasses import dataclass
from torch import nn, functional as F


@dataclass
class Config:
    n_embedding: int = 768
    n_head: int = 12
    dropout: float = 0.5
    bias: bool = True

class Model(nn.Module):
    def __init__(self):
        self.transformer = nn.Sequential()
        for _ in range(12):
            self.transformer.append(
                Block()
            )

class LayerNormalization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embedding))
        self.bias = nn.Parameter(torch.ones(config.n_embedding))

    def forward(self, x):
        return (x - x.mean(-1))/ (torch.var(x) + 1e-5) * self.weight + self.bias

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
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


        d_k = self.n_embedding // self.num_heads

        # sawpping dimension 1 and 2 so the score calculated is per head, and you end with a tensor that is (sequence_length, d_k)
        q = q.view(batch_size, sequence_length, self.num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, d_k).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, d_k).transpose(1, 2)


        # (sequence_length, sequence_length)
        attention = q @ k.transpose(-2, -1)
        # (sequence_length, sequence_length)
        mask = torch.triu(torch.ones(d_k, d_k, dtype=torch.bool, device=attention.device), diagonal=1)
        attention = attention.masked_fill(mask, float("-inf"))
        attention = attention / math.sqrt(d_k)
        attention = nn.Softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        # (sequence_length, d_k)
        y = attention @  v

        y = self.residual_dropout(self.c_proj(y))
        return y


        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = LayerNormalization(config)
        self.ln_2 = LayerNormalization(config)
        
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

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