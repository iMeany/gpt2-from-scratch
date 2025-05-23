import math
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        # self.attn_dropout = 0.1
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        # calculate query, key, value projections
        # nh = number of heads, hs = head size, C = number of channels
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y= att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reshape to (B, T, C)
        # output projection
        y = self.c_proj(y)
        return y
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh') # can also use exact version
        # self.gelu = nn.GELU() # exact version
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # reduce
        x = x + self.mlp(self.ln2(x)) # map
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max context length
    vocab_size: int = 50257 # tokens
    n_layers: int = 12 # number of transformer blocks
    n_heads: int = 12 # number of attention heads
    n_embed: int = 768 # embedding dimension


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.block_size, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        pos = torch.arrange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embed)
        x = tok_emb + pos_emb # broadcast to (B, T, n_embed)
        # forward the transformer blocks
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and linear layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # shape (B, T, vocab_size)
        return logits

def main():
    num_return_sequences = 5
    max_length = 30
    
    model = GPT(GPTConfig())
    model.eval() # set to evaluation mode
    
    model.to("cuda")
    
    enc = tiktoken.get_encoding("gpt2")
    prefix = "Hello, I'm a language model,"
    tokens = enc.encode(prefix)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # shape (1, T)
    x = tokens.to("cuda")
    
    # generate
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
            
    for i in range(num_return_sequences):
        print(f"Generated sequence {i+1}:")
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

if __name__ == "__main__":
    main()
