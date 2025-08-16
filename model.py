# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=64, n_heads=2, n_layers=2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(block_size, n_embed)

        self.blocks = nn.ModuleList([
            DecoderBlock(n_embed, n_heads)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(n_embed)
        self.output = nn.Linear(n_embed, vocab_size)

        self.block_size = block_size
        self.vocab_size = vocab_size

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        tok_emb = self.token_embed(idx)       # (B, T, n_embed)
        pos_emb = self.position_embed(pos)    # (1, T, n_embed)
        x = tok_emb + pos_emb                 # (B, T, n_embed)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.output(x)
        return logits

class DecoderBlock(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ff = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x, need_weights=False, attn_mask=self._generate_mask(x.size(1), x.device))
        x = x + attn_output
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x

    def _generate_mask(self, size, device):
        # Causal mask (lower triangle)
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
