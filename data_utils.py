"""
Data Utilities for CharRNN
- Read and preprocess text files
- Build character vocabulary
- Create training/validation chunks
- Batch generation
"""

import os
import string
import random
import urllib.request
import torch


class TextDataset:
    """Handles text loading, encoding, and batching for CharRNN training."""

    def __init__(self, filepath=None, text=None, val_fraction=0.1):
        """
        Args:
            filepath: path to a plain text file
            text: raw text string (alternative to filepath)
            val_fraction: fraction of data to use for validation
        """
        if text is not None:
            self.raw_text = text
        elif filepath is not None:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                self.raw_text = f.read()
        else:
            raise ValueError("Provide either filepath or text")

        # Build vocabulary from the actual text
        self.chars = sorted(list(set(self.raw_text)))
        self.n_characters = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode full text
        self.encoded = torch.tensor([self.char2idx[ch] for ch in self.raw_text], dtype=torch.long)

        # Train/val split
        split_idx = int(len(self.encoded) * (1 - val_fraction))
        self.train_data = self.encoded[:split_idx]
        self.val_data = self.encoded[split_idx:]

        print(f"Dataset: {len(self.raw_text)} chars, {self.n_characters} unique")
        print(f"Train: {len(self.train_data)} chars, Val: {len(self.val_data)} chars")

    def get_random_chunk(self, data, chunk_len, batch_size, device='cpu'):
        """
        Get a random batch of training chunks.
        Returns (input, target) where target is input shifted by 1.
        """
        inputs = []
        targets = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(data) - chunk_len - 1)
            chunk = data[start_idx:start_idx + chunk_len + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
        inp = torch.stack(inputs).to(device)      # (batch_size, chunk_len)
        target = torch.stack(targets).to(device)   # (batch_size, chunk_len)
        return inp, target

    def get_train_batch(self, chunk_len, batch_size, device='cpu'):
        return self.get_random_chunk(self.train_data, chunk_len, batch_size, device)

    def get_val_batch(self, chunk_len, batch_size, device='cpu'):
        return self.get_random_chunk(self.val_data, chunk_len, batch_size, device)


def download_shakespeare(save_dir="data"):
    """Download Tiny Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "shakespeare.txt")
    if not os.path.exists(filepath):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    else:
        print(f"Already exists: {filepath}")
    return filepath


def download_sherlock(save_dir="data"):
    """Download Complete Sherlock Holmes."""
    url = "https://sherlock-holm.es/stories/plain-text/cnus.txt"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "sherlock.txt")
    if not os.path.exists(filepath):
        print("Downloading Sherlock Holmes...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    else:
        print(f"Already exists: {filepath}")
    return filepath
