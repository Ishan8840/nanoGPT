"""Load a trained checkpoint and generate text from it."""
import argparse
import torch

from config import Config
from model import GPT


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt['config']
    itos = ckpt['itos']
    stoi = ckpt['stoi']

    model = GPT(
        vocab_size=len(itos),
        d_model=cfg_dict['d_model'],
        chunk_size=cfg_dict['chunk_size'],
        num_heads=cfg_dict['num_heads'],
        num_layers=cfg_dict['num_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, stoi, itos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='',
                        help='Optional starting text. Defaults to a single newline.')
    parser.add_argument('--max-new-tokens', type=int, default=500)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint. Defaults to Config.checkpoint_path.')
    args = parser.parse_args()

    cfg = Config()
    ckpt_path = args.checkpoint or cfg.checkpoint_path

    model, stoi, itos = load_model(ckpt_path, cfg.device)

    if args.prompt:
        ids = [stoi[c] for c in args.prompt if c in stoi]
        if not ids:
            ids = [0]
    else:
        ids = [0]

    context = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    generated = model.generate(context, max_new_tokens=args.max_new_tokens)

    decoded = ''.join(itos[i] for i in generated[0].tolist())
    print(decoded)


if __name__ == '__main__':
    main()