"""Train the GPT model and save a checkpoint."""
import torch

from config import Config
from data import load_data, get_batch
from model import GPT
from pathlib import Path


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg):
    """Average loss over a few batches from each split."""
    model.eval()
    out = {}
    for name, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg.batch_size, cfg.chunk_size, cfg.device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = Config()
    print(f"Using device: {cfg.device}")

    Path(cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    tokenizer, train_data, val_data = load_data(cfg.data_path, cfg.train_split)
    print(f"Vocab size: {tokenizer.vocab_size} | "
          f"train: {len(train_data):,} | val: {len(val_data):,}")

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        chunk_size=cfg.chunk_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(cfg.max_steps):
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, cfg)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch(train_data, cfg.batch_size, cfg.chunk_size, cfg.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save checkpoint: weights + config + tokenizer mapping
    torch.save({
        'model_state': model.state_dict(),
        'config': cfg.__dict__,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
    }, cfg.checkpoint_path)
    print(f"Saved checkpoint to {cfg.checkpoint_path}")


if __name__ == '__main__':
    main()