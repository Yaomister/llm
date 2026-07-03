"""Implementation of the GPT-2 model."""

import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel


@dataclass
class Config:
    """Hyperparameters for the GPT model architecture."""
    block_size = 1024  # maximum sequence length
    vocab_size = 10304  # 10k BPE merges + 256 byte tokens + 1 <|endoftext|> + some padding
    n_layer = 12  # number of decoder blocks
    n_head = 12  # number of attention heads per layer
    d_model = 768  # embedding dimension
    eval_iter = 100  # how often to evaluate the model


class Attention(nn.Module):
    """A single causal (masked) multi-head self-attention layer."""

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
        """Apply causal self-attention to the input."""
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

        # y = F.scaled_dot_product_attention(q, v, k, is_causal=True )

        return y


class Block(nn.Module):
    """A single decoder block."""

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


class MLP(nn.Module):
    """The feed-forward layer."""

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
        return x  # (batch_size, sequence_length, d_model)


class GPT(nn.Module):
    """The full GPT-2 architecture."""

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

    def configure_optimizer(self, weight_decay, learning_rate):
        """Build an AdamW optimizer that only applies weight decay."""
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        params_with_decay = [p for _, p in params.items() if p.dim() >= 2]
        params_with_no_decay = [p for _, p in params.items() if p.dim() < 2]

        groups = [
            {"params" : params_with_decay, "weight_decay": weight_decay},
            {"params" : params_with_no_decay, "weight_decay": 0.0}
        ]

        num_params_with_decay = sum(p.numel() for p in params_with_decay)
        num_params_with_no_decay = sum(p.numel() for p in params_with_no_decay)

        print(f"the number of decayed parameter tensors: {len(params_with_decay)} with {num_params_with_decay} parameters")
        print(f"the number of non-decayed parameter tensors: {len(params_with_decay)} with {num_params_with_no_decay} parameters")

        optimizer = torch.optim.AdamW(params=groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


    @classmethod
    def from_saved(cls, path):
        """Load a model either from a pretrained HuggingFace GPT-2 checkpoint name (e.g. "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl") or from a local state_dict file."""
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }
        # these stay consistent across all GPT-2 model sizes
        config_args["vocab_size"] = 50257
        config_args['block_size'] = 1024
        if path in config_args:
            model_type = path
            model = GPT(Config(**config_args))
            sd = model.state_dict()
            sd_keys = sd.keys()
            sd_keys = [(k for k in sd_keys if not k.endswith(".attn.bias"))] # this is the causal mask (a buffer not a parameter)

            # load the weights from HuggingFace
            model_hf = GPT2LMHeadModel.from_pretrained(model_type)
            sd_hf = model_hf.state_dict()

            sd_keys_hf = sd_hf.keys()
            sd_keys_hf = [(k for k in sd_keys_hf if not k.endswith(".attn.masked_bias"))]
            sd_keys_hf = [(k for k in sd_keys_hf if not k.endswith(".attn.bias"))]
            # the original GPT-2 implementation used Conv1D layers instead of nn.Linear, so saved it as [out dimension, in dimension] which is opposite of what nn.Linear wants, so a transposition is needed
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

            assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

            # copy over the weights and transpose if needed
            for k in sd_keys_hf:
                if any(k.endswith(transposed)):
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])

        else:
            assert os.path.exists(path), f"failed to load weights from {path}"
            print(f"Successfully loaded model weights from ${path}")
    
            model = GPT(Config())
            model.load_state_dict(torch.load(path))

        return model


    def _initialize_weights(self, module):
        """Initialize weights to control activation variance."""
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "scale_init"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                # bias explicitly set to 0
                module.bias = nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, x, target = None):
        """Run the model on a batch of token ids, returning logits."""
        batch_size, sequence_length = x.size()
        # sequence cannot exceed the maximum context length the causal mask was built for
        assert sequence_length <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # GPT-2 uses learned positional embeddings rather than the original transformer's fixed sinusoidal ones
        pos = torch.arange(0, sequence_length, dtype=torch.long, device=x.device)

        position_embedding = self.transformer.wpe(pos)
        token_embedding = self.transformer.wte(x)

        x = token_embedding + position_embedding

        for head in self.transformer.heads:
            x = head(x)

        x = self.transformer.ln_f(x)

        logit = self.lm_head(x) # (batch_size, sequence_length, vocab_size)

        loss = None

        if target is not None:
            loss = F.cross_entropy(logit.view(-1, logit.size(-1)), target.view(-1))

        return logit, loss
