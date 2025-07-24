import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
lr = 3e-4
eval_iters = 200
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2


class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Compute attention scores / affinities
        # (B, T, hs) @ (B, hs, T).t
        aff = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, T)
        aff = aff.masked_fill(self.tril == 0, float('-inf'))
        aff = F.softmax(aff, dim=-1)

        out = aff @ v  # (B, T, hs)
        return out

class MultiHead(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // block_size) for _ in range(n_head)])
        self.proj = nn.Linear(, n_embd)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )
    def forward(self, x):
        return self.net(x)

class MakeMore(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(256, N_EMBD),
            MultiHead(N_EMBD, N_HEAD),
            nn.LayerNorm(N_EMBD),
            nn.Dropout(DROPOUT),
            *[nn.Sequential(
                MultiHead(N_EMBD, N_HEAD),
                nn.LayerNorm(N_EMBD),
                nn.Dropout(DROPOUT),
                FeedForward(N_EMBD)
            ) for _ in range(N_LAYER)],
            nn.Linear(N_EMBD, 256)
        )


def main():
    torch.manual_seed(42)

    words = open('names.txt').read().splitlines()