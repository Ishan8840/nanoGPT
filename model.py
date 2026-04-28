"""Transformer model definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, d_model, head_size, max_seq_len):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.head_size = head_size

    def forward(self, x):
        T = x.shape[1]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei * self.head_size ** -0.5, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, max_seq_len, num_heads):
        super().__init__()
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, max_seq_len) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.model(x)


class Block(nn.Module):
    def __init__(self, d_model, chunk_size, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, max_seq_len=chunk_size)
        self.ffwd = FeedForward(d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, chunk_size, num_heads, num_layers):
        super().__init__()
        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(chunk_size, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, chunk_size, num_heads) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(idx.shape[1], device=idx.device)
        pos_emb = self.pos_encoding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.chunk_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx