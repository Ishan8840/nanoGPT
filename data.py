"""Character-level tokenizer and batch sampling."""
import torch


class CharTokenizer:
    def __init__(self, text):
        chars = ''.join(sorted(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


def load_data(path, train_split=0.9):
    """Returns (tokenizer, train_data, val_data) as CPU long tensors."""
    with open(path, 'r') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(train_split * len(data))
    return tokenizer, data[:n], data[n:]


def get_batch(data, batch_size, chunk_size, device):
    """Sample a random batch of (x, y) pairs from data."""
    ix = torch.randint(0, len(data) - chunk_size, (batch_size,))
    x = torch.stack([data[i:i + chunk_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + chunk_size] for i in ix])
    return x.to(device), y.to(device)