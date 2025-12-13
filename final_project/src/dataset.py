"""
PyTorch Dataset for sign language landmark sequences.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict


class SignLanguageDataset(Dataset):
    """
    Dataset for sign language recognition from landmark sequences.

    Each sample is a sequence of hand landmarks with a sign label.
    """

    def __init__(
        self,
        landmarks_dir: str,
        annotations_file: str,
        max_seq_length: int = 60,
        subset_classes: List[str] = None
    ):
        """
        Args:
            landmarks_dir: Directory containing .npy landmark files
            annotations_file: Path to WLASL JSON annotations
            max_seq_length: Maximum sequence length (pad/truncate to this)
            subset_classes: List of sign glosses to include (None = all)
        """
        self.landmarks_dir = Path(landmarks_dir)
        self.max_seq_length = max_seq_length

        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Build samples list and label mapping
        self.samples = []  # List of (video_id, label_idx)
        self.gloss_to_idx = {}
        self.idx_to_gloss = {}

        for entry in annotations:
            gloss = entry['gloss']

            # Filter by subset if specified
            if subset_classes and gloss not in subset_classes:
                continue

            # Add to label mapping
            if gloss not in self.gloss_to_idx:
                idx = len(self.gloss_to_idx)
                self.gloss_to_idx[gloss] = idx
                self.idx_to_gloss[idx] = gloss

            # Add each video instance
            for instance in entry.get('instances', []):
                video_id = instance['video_id']
                landmark_file = self.landmarks_dir / f"{video_id}.npy"

                if landmark_file.exists():
                    self.samples.append((video_id, self.gloss_to_idx[gloss]))

        print(f"Loaded {len(self.samples)} samples across {len(self.gloss_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_id, label = self.samples[idx]

        # Load landmarks
        landmark_file = self.landmarks_dir / f"{video_id}.npy"
        landmarks = np.load(landmark_file)  # Shape: (T, 21, 3)

        # Flatten spatial dimensions: (T, 21, 3) -> (T, 63)
        landmarks = landmarks.reshape(landmarks.shape[0], -1)

        # Pad or truncate to max_seq_length
        landmarks = self._pad_or_truncate(landmarks)

        return torch.tensor(landmarks, dtype=torch.float32), label

    def _pad_or_truncate(self, seq: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate sequence to max_seq_length."""
        T = seq.shape[0]

        if T > self.max_seq_length:
            # Truncate (sample evenly or take first N)
            return seq[:self.max_seq_length]
        elif T < self.max_seq_length:
            # Pad with zeros
            padding = np.zeros((self.max_seq_length - T, seq.shape[1]))
            return np.vstack([seq, padding])
        return seq

    @property
    def num_classes(self) -> int:
        return len(self.gloss_to_idx)

    @property
    def input_dim(self) -> int:
        return 21 * 3  # 21 landmarks * 3 coordinates


def get_data_loaders(
    landmarks_dir: str,
    annotations_file: str,
    batch_size: int = 32,
    max_seq_length: int = 60,
    subset_classes: List[str] = None,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset = SignLanguageDataset(
        landmarks_dir=landmarks_dir,
        annotations_file=annotations_file,
        max_seq_length=max_seq_length,
        subset_classes=subset_classes
    )

    # Split dataset
    total = len(dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
