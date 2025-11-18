"""Data loaders for robot training data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class RobotDataset(Dataset):
    """
    Dataset for robot manipulation trajectories.

    Expected data format:
    data_dir/
        episode_0000/
            rgb_000.png
            depth_000.npy
            action_000.npy
            state_000.npy
            command.txt
        episode_0001/
            ...
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize robot dataset.

        Args:
            data_dir: Path to data directory
            split: Dataset split ('train', 'val', 'test')
            transform: Image transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self._default_transform()

        # Find all episodes
        self.episodes = sorted(list(self.data_dir.glob("episode_*")))

        # Split data
        num_episodes = len(self.episodes)
        if split == "train":
            self.episodes = self.episodes[:int(0.8 * num_episodes)]
        elif split == "val":
            self.episodes = self.episodes[int(0.8 * num_episodes):int(0.9 * num_episodes)]
        else:  # test
            self.episodes = self.episodes[int(0.9 * num_episodes):]

        # Index all samples
        self.samples = []
        for episode_dir in self.episodes:
            rgb_files = sorted(episode_dir.glob("rgb_*.png"))
            for rgb_file in rgb_files:
                frame_id = rgb_file.stem.split("_")[1]
                self.samples.append({
                    "episode_dir": episode_dir,
                    "frame_id": frame_id,
                })

        logger.info(f"Loaded {len(self.samples)} samples from {len(self.episodes)} episodes ({split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        episode_dir = sample["episode_dir"]
        frame_id = sample["frame_id"]

        # Load RGB image
        rgb_path = episode_dir / f"rgb_{frame_id}.png"
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = self.transform(rgb)

        # Load depth (if exists)
        depth_path = episode_dir / f"depth_{frame_id}.npy"
        if depth_path.exists():
            depth = np.load(depth_path)
            depth = torch.from_numpy(depth).float().unsqueeze(0)
        else:
            depth = torch.zeros(1, 480, 640)

        # Load action
        action_path = episode_dir / f"action_{frame_id}.npy"
        if action_path.exists():
            action = np.load(action_path)
            action = torch.from_numpy(action).float()
        else:
            action = torch.zeros(8)

        # Load robot state
        state_path = episode_dir / f"state_{frame_id}.npy"
        if state_path.exists():
            state = np.load(state_path)
            state = torch.from_numpy(state).float()
        else:
            state = torch.zeros(8)

        # Load command
        command_path = episode_dir / "command.txt"
        if command_path.exists():
            with open(command_path, "r") as f:
                command = f.read().strip()
        else:
            command = "pick and place"

        return {
            "rgb": rgb,
            "depth": depth,
            "action": action,
            "state": state,
            "command": command,
        }

    def _default_transform(self) -> transforms.Compose:
        """Default image transforms."""
        return transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = RobotDataset(data_dir, split="train")
    val_dataset = RobotDataset(data_dir, split="val")
    test_dataset = RobotDataset(data_dir, split="test")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Created dataloaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """Custom collate function for robot data."""
    rgb = torch.stack([item["rgb"] for item in batch])
    depth = torch.stack([item["depth"] for item in batch])
    action = torch.stack([item["action"] for item in batch])
    state = torch.stack([item["state"] for item in batch])
    commands = [item["command"] for item in batch]

    return {
        "rgb": rgb,
        "depth": depth,
        "action": action,
        "state": state,
        "commands": commands,
    }
