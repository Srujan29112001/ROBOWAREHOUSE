"""
Data processor for robot VLA training data.

Handles loading, preprocessing, and batching of robot manipulation data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RobotManipulationDataset(Dataset):
    """
    PyTorch Dataset for robot manipulation data.

    Loads RGB-D images, text commands, and robot trajectories.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        max_sequence_length: int = 50
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_seq_len = max_sequence_length

        # Load dataset info
        with open(self.data_dir / 'dataset_info.json', 'r') as f:
            self.dataset_info = json.load(f)

        # Split dataset
        total_samples = len(self.dataset_info['samples'])
        if split == 'train':
            self.samples = self.dataset_info['samples'][:int(0.8 * total_samples)]
        elif split == 'val':
            self.samples = self.dataset_info['samples'][int(0.8 * total_samples):int(0.9 * total_samples)]
        else:  # test
            self.samples = self.dataset_info['samples'][int(0.9 * total_samples):]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load RGB image
        rgb_path = self.data_dir / sample['rgb_path']
        rgb_image = cv2.imread(str(rgb_path))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Load depth map
        depth_path = self.data_dir / sample['depth_path']
        depth_map = np.load(str(depth_path))

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=rgb_image, mask=depth_map)
            rgb_image = transformed['image']
            depth_map = transformed['mask']
        else:
            rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
            depth_map = torch.from_numpy(depth_map).unsqueeze(0).float()

        # Process command
        command = sample['command']
        text = command['text'] if command else ""

        # Process trajectory
        if command and command['joint_trajectory']:
            joint_trajectory = torch.tensor(command['joint_trajectory'], dtype=torch.float32)
            gripper_trajectory = torch.tensor(command['gripper_trajectory'], dtype=torch.float32)

            # Pad sequences
            seq_len = joint_trajectory.shape[0]
            if seq_len < self.max_seq_len:
                padding = self.max_seq_len - seq_len
                joint_trajectory = torch.nn.functional.pad(joint_trajectory, (0, 0, 0, padding))
                gripper_trajectory = torch.nn.functional.pad(gripper_trajectory, (0, padding))
        else:
            joint_trajectory = torch.zeros(self.max_seq_len, 7)
            gripper_trajectory = torch.zeros(self.max_seq_len)

        # Process objects
        objects = sample['objects']
        num_objects = len(objects)

        return {
            'rgb': rgb_image,
            'depth': depth_map,
            'text': text,
            'joint_trajectory': joint_trajectory,
            'gripper_trajectory': gripper_trajectory,
            'num_objects': num_objects,
            'success': command['success'] if command else False,
            'sample_id': sample['sample_id']
        }


class RobotDataProcessor:
    """
    Data processor for robot VLA training.

    Handles data loading, preprocessing, and batch creation.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (480, 640),
        use_augmentation: bool = True
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_augmentation = use_augmentation

        # Create transforms
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()

    def _create_train_transform(self) -> A.Compose:
        """Create training data augmentation pipeline."""
        transforms = [
            A.Resize(self.image_size[0], self.image_size[1]),
        ]

        if self.use_augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ])

        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transforms.append(ToTensorV2())

        return A.Compose(transforms)

    def _create_val_transform(self) -> A.Compose:
        """Create validation transform (no augmentation)."""
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def get_train_loader(self) -> DataLoader:
        """Create training data loader."""
        dataset = RobotManipulationDataset(
            self.data_dir,
            split='train',
            transform=self.train_transform
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def get_val_loader(self) -> DataLoader:
        """Create validation data loader."""
        dataset = RobotManipulationDataset(
            self.data_dir,
            split='val',
            transform=self.val_transform
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_loader(self) -> DataLoader:
        """Create test data loader."""
        dataset = RobotManipulationDataset(
            self.data_dir,
            split='test',
            transform=self.val_transform
        )

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def compute_dataset_statistics(self) -> Dict:
        """Compute statistics of the dataset."""
        dataset = RobotManipulationDataset(self.data_dir, split='train', transform=None)

        total_samples = len(dataset)
        successful_samples = 0
        avg_execution_time = 0
        avg_num_objects = 0

        for sample in dataset:
            if sample['success']:
                successful_samples += 1
            avg_num_objects += sample['num_objects']

        return {
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
            'avg_num_objects': avg_num_objects / total_samples if total_samples > 0 else 0
        }


if __name__ == '__main__':
    # Example usage
    processor = RobotDataProcessor(
        data_dir='data/synthetic_robot_dataset',
        batch_size=8,
        num_workers=4
    )

    # Get data loaders
    train_loader = processor.get_train_loader()
    val_loader = processor.get_val_loader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test loading a batch
    for batch in train_loader:
        print("Batch shapes:")
        print(f"  RGB: {batch['rgb'].shape}")
        print(f"  Depth: {batch['depth'].shape}")
        print(f"  Joint trajectory: {batch['joint_trajectory'].shape}")
        print(f"  Gripper trajectory: {batch['gripper_trajectory'].shape}")
        print(f"  Text commands: {len(batch['text'])}")
        break

    # Compute statistics
    stats = processor.compute_dataset_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
