import torch


# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r') as f:
    text = f.read()


chars = ''.join(sorted(set(text)))

# encoding and decoding
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(batch_size, chunk_size):
    x = torch.zeros([batch_size, chunk_size], dtype=torch.long)
    y = torch.zeros([batch_size, chunk_size], dtype=torch.long)

    for i in range(batch_size):
        idx = torch.randint(0, len(encoded) - chunk_size, (1,))
        xb = encoded[idx:idx+chunk_size]
        yb = encoded[idx+1:idx+chunk_size+1]
        x[i] = xb
        y[i] = yb
    
    return x, y


x, y = get_batch(4, 8)

print(x[1])
print(y[1])