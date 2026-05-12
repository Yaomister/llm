import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



@dataclass
class Config:
    block_size = 1024
    vocab_size = 50000
    n_layer = 12
    n_head = 12
    d_model = 768


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_model * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(config.d_model * 4, config.d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

        
class Attention(nn.Module):
    def __init__(self, config: Config):
        assert config.d_model % config.n_head == 0
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.config = config
        self.causal_mask = torch.tril(torch.ones(self.config.block_size, self.config.block_size), diagonal=0).view(1, 1, self.config.block_size, self.config.block_size)
        return
    
    def forward(self, x):
        batch, sequence, d_model = x.size()
        d_k = d_model // self.config.n_head
        q = self.q_proj(x)
        q = q.view(batch, sequence, self.config.n_head, d_k).transpose(1, 2)
        v = self.v_proj(x)
        v = v.view(batch, sequence, self.config.n_head, d_k).transpose(1, 2)
        k = self.k_proj(x)
        k = k.view(batch, sequence, self.config.n_head, d_k).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * (1.0/ math.sqrt(d_k))
        attention = attention.masked_fill(self.causal_mask[:, :, sequence, sequence] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(batch, sequence, d_model)
        y = self.c_proj(y)




class Block(nn.Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.attention = Attention(config)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.MLP(config)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        


class GPT(nn.Module):
    def __init__(self, config: Config):
        super(GPT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            heads = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model)
        )
        )








