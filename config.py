"""Hyperparameters and shared config."""
from dataclasses import dataclass
import torch


@dataclass
class Config:
    # data
    data_path: str = 'input.txt'
    train_split: float = 0.9

    # model
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 4
    chunk_size: int = 64

    # training
    batch_size: int = 32
    max_steps: int = 5000
    lr: float = 3e-4
    eval_interval: int = 500
    eval_iters: int = 50

    # io
    checkpoint_path: str = 'pretrained/gpt.pt'

    # runtime
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'