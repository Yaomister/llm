import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


"""
Implementation of the GPT-2 model
"""

@dataclass
class Config:
    # maximum sequence length
    block_size = 1024
    # the number of tokens (10k BPE merges + 256 byte tokens + 1 <|endoftext|> + some padding)
    vocab_size = 10304
    # the number of layers (blocks)
    n_layer = 12
    # the number of attention heads per layer
    n_head = 12
    # the dimension of the embedding
    d_model = 768
    # how often to evaluate the model
    eval_iter = 100
 
"""
A single causal attention head
"""       
class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # ensure that the embedding dimensions can be divided evenly by the number of layers for multi-head attention
        assert config.d_model % config.n_head == 0
        # the query, value, key projections
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.scale_init = 1
        # regularization
        self.n_head = config.n_head
        self.d_model = config.d_model
        # causal mask (see "Language Models are Unsupervised Multitask Learners" by Radford et. al)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size), diagonal=0).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        batch_size, sequence_length, d_model = x.size()
        # the dimensions of the embedding for a single attention head
        d_k = d_model // self.n_head

        # all are size (batch_size, n_head, sequence_length, d_k), the transpose swapped n_head and sequence_length for the matrix multiplication
        q = self.q_proj(x)
        q = q.view(batch_size, sequence_length, self.n_head, d_k).transpose(1, 2)
        v = self.v_proj(x)
        v = v.view(batch_size, sequence_length, self.n_head, d_k).transpose(1, 2)
        k = self.k_proj(x)
        k = k.view(batch_size, sequence_length, self.n_head, d_k).transpose(1, 2)

        attention = (q @ k.transpose(-2, -1)) * (1.0/ math.sqrt(d_k))
        # apply masking so softmax attributes 0 for future tokens
        attention = attention.masked_fill(self.causal_mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        y = attention @ v
        # (batch_size, sequence_length, d_model), merge the last two dimensions back into d_model
        y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model) 
        y = self.c_proj(y)

        # Flash attention is built into PyTorch
        # y = F.scaled_dot_product_attention(q, v, k, is_causal=True )

        return y

"""
A single decoder block
"""       
class Block(nn.Module):
    def __init__(self, config: Config):
        super(Block, self).__init__()
        self.attention = Attention(config)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

"""
The feed forward layer
"""
class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_model * 4)
        # approximating GELU with tanh runs faster
        self.gelu = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(config.d_model * 4, config.d_model)
        self.fc2.scale_init = 1

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x # (batch_size, sequence_length, d_model)

"""
The GPT architecture
"""
class GPT(nn.Module):
    def __init__(self, config: Config):
        super(GPT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            heads = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model)
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        # the weights are tied to reduce memory, see "Using the Output Embedding to Improve Language Models" by Press et al
        self.transformer.wte.weight = self.lm_head.weight
        # iterate over the submodule and apply weight initialization
        self.apply(self._initialize_weights)

    def from_saved(self, path):
        if path == "gpt-2":
            return
        else:
            self.load_state_dict(torch.load(path))
            print(f"Successfully loaded model weights from ${path}")

    def _initialize_weights(self, module):
        """
        this acts as a way to control the variance. The variance stacks up as you go deeper in the neural network,
        which makes the neurons over confident and make the softmax output collapses a one-hot vector.
        """
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "scale_init"):
                #  scaled by 1/(2√n_layer), 2√n_layer because each block has mlp and attention.
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                # bias explicitly set to 0
                module.bias = nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, x, target = None):
        batch_size, sequence_length = x.size()
        # sequence cannot exceed the maximum context length the causal mask was built for
        assert sequence_length < self.config.block_size
        # the GPT-2 paper deviates from the original transformer here: instead of hardcoding sinusoidal positional encodings, they make it a learnable
        pos = torch.arange(0, sequence_length, dtype=torch.long, device=x.device)

        position_embedding = self.transformer.wpe(pos)
        token_embedding = self.transformer.wte(x)

        # add the token_embedding and position_embedding
        x = token_embedding + position_embedding 

        # forward pass through the attention heads
        for head in self.transformer.heads:
            x = head(x)

        x = self.transformer.ln_f(x)

        logit = self.lm_head(x) # (batch_size, sequence_length, vocab_size)

        loss = None

        if target is not None:
            loss = F.cross_entropy(logit.view(-1, logit.size(-1)), target.view(-1))

        return logit, loss
