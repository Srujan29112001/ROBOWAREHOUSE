"""
Advanced data augmentation for robot VLA training.

Includes domain-specific augmentations for robotic manipulation.
"""
import numpy as np
import torch
import cv2
from typing import Tuple, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class VLADataAugmentation:
    """
    Data augmentation pipeline for Vision-Language-Action models.

    Includes both standard computer vision augmentations and
    robot-specific augmentations.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (480, 640),
        augmentation_prob: float = 0.5,
        include_robot_aug: bool = True
    ):
        self.image_size = image_size
        self.aug_prob = augmentation_prob
        self.include_robot_aug = include_robot_aug

        self.visual_augmentation = self._create_visual_augmentation()
        self.depth_augmentation = self._create_depth_augmentation()

    def _create_visual_augmentation(self) -> A.Compose:
        """Create visual augmentation pipeline."""
        return A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3
            ),

            # Color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.3
            ),

            # Noise and blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),

            # Pixel-level augmentations
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.2),

            # Weather and lighting
            A.RandomShadow(p=0.1),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),

            # Normalization
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def _create_depth_augmentation(self) -> A.Compose:
        """Create depth-specific augmentation pipeline."""
        return A.Compose([
            # Geometric transformations (same as RGB for consistency)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3
            ),

            # Depth-specific noise
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),

            A.Resize(self.image_size[0], self.image_size[1]),
            ToTensorV2()
        ])

    def augment_text_command(self, text: str) -> str:
        """
        Augment text command with paraphrasing and synonyms.

        Args:
            text: Original text command

        Returns:
            Augmented text command
        """
        if random.random() > self.aug_prob:
            return text

        # Synonym replacement
        synonyms = {
            'pick': ['grasp', 'grab', 'take', 'lift'],
            'place': ['put', 'set', 'position', 'drop'],
            'move': ['transfer', 'relocate', 'shift'],
            'red': ['crimson', 'scarlet'],
            'blue': ['azure', 'navy'],
            'green': ['emerald', 'lime'],
            'box': ['container', 'bin', 'cube'],
            'shelf': ['rack', 'platform'],
        }

        words = text.lower().split()
        augmented_words = []

        for word in words:
            if word in synonyms and random.random() < 0.3:
                augmented_words.append(random.choice(synonyms[word]))
            else:
                augmented_words.append(word)

        return ' '.join(augmented_words)

    def augment_robot_state(
        self,
        joint_angles: np.ndarray,
        noise_std: float = 0.01
    ) -> np.ndarray:
        """
        Add noise to robot joint angles to simulate sensor noise.

        Args:
            joint_angles: Current joint angles (7-DOF)
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Augmented joint angles
        """
        if random.random() > self.aug_prob:
            return joint_angles

        noise = np.random.normal(0, noise_std, joint_angles.shape)
        augmented = joint_angles + noise

        # Clip to valid range [-π, π]
        augmented = np.clip(augmented, -np.pi, np.pi)

        return augmented

    def augment_depth_occlusion(
        self,
        depth_map: np.ndarray,
        occlusion_prob: float = 0.2,
        occlusion_size: Tuple[int, int] = (50, 50)
    ) -> np.ndarray:
        """
        Simulate depth sensor occlusions.

        Args:
            depth_map: Depth map
            occlusion_prob: Probability of adding occlusion
            occlusion_size: Size of occluded region

        Returns:
            Augmented depth map
        """
        if random.random() > occlusion_prob:
            return depth_map

        h, w = depth_map.shape[:2]
        oc_h, oc_w = occlusion_size

        # Random position
        y = random.randint(0, max(1, h - oc_h))
        x = random.randint(0, max(1, w - oc_w))

        # Create occlusion (set to invalid depth)
        depth_map = depth_map.copy()
        depth_map[y:y + oc_h, x:x + oc_w] = 0

        return depth_map

    def apply_domain_randomization(
        self,
        rgb: np.ndarray,
        depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply domain randomization for sim-to-real transfer.

        Args:
            rgb: RGB image
            depth: Depth map

        Returns:
            Randomized RGB and depth
        """
        # Randomize lighting
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.7, 1.3)
            rgb = np.clip(rgb * brightness_factor, 0, 255).astype(np.uint8)

        # Randomize colors
        if random.random() < 0.3:
            hue_shift = random.randint(-10, 10)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Add camera artifacts
        if random.random() < 0.2:
            # Motion blur
            kernel_size = random.choice([3, 5, 7])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = np.ones(kernel_size)
            kernel /= kernel_size
            rgb = cv2.filter2D(rgb, -1, kernel)

        # Depth noise
        if random.random() < 0.3:
            depth_noise = np.random.normal(0, 0.02, depth.shape)
            depth = depth + depth_noise

        return rgb, depth

    def __call__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        text: str,
        robot_state: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all augmentations.

        Args:
            rgb: RGB image
            depth: Depth map
            text: Text command
            robot_state: Robot joint angles (optional)

        Returns:
            Dictionary with augmented data
        """
        # Domain randomization
        if self.include_robot_aug:
            rgb, depth = self.apply_domain_randomization(rgb, depth)

        # Visual augmentation
        augmented_rgb = self.visual_augmentation(image=rgb)['image']

        # Depth augmentation
        if self.include_robot_aug:
            depth = self.augment_depth_occlusion(depth)
        augmented_depth = self.depth_augmentation(image=depth)['image']

        # Text augmentation
        augmented_text = self.augment_text_command(text)

        # Robot state augmentation
        augmented_robot_state = None
        if robot_state is not None and self.include_robot_aug:
            augmented_robot_state = self.augment_robot_state(robot_state)
        elif robot_state is not None:
            augmented_robot_state = robot_state

        result = {
            'rgb': augmented_rgb,
            'depth': augmented_depth,
            'text': augmented_text,
        }

        if augmented_robot_state is not None:
            result['robot_state'] = torch.from_numpy(augmented_robot_state).float()

        return result


class MixupAugmentation:
    """
    Mixup augmentation for robot manipulation data.

    Combines two samples with a random mixing coefficient.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        sample1: Dict[str, torch.Tensor],
        sample2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply mixup to two samples.

        Args:
            sample1: First sample
            sample2: Second sample

        Returns:
            Mixed sample
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix images
        mixed_rgb = lam * sample1['rgb'] + (1 - lam) * sample2['rgb']
        mixed_depth = lam * sample1['depth'] + (1 - lam) * sample2['depth']

        # Mix trajectories
        mixed_joint_traj = lam * sample1['joint_trajectory'] + (1 - lam) * sample2['joint_trajectory']
        mixed_gripper_traj = lam * sample1['gripper_trajectory'] + (1 - lam) * sample2['gripper_trajectory']

        # For text, randomly choose one
        mixed_text = sample1['text'] if random.random() < lam else sample2['text']

        return {
            'rgb': mixed_rgb,
            'depth': mixed_depth,
            'text': mixed_text,
            'joint_trajectory': mixed_joint_traj,
            'gripper_trajectory': mixed_gripper_traj,
            'mixup_lambda': lam
        }


if __name__ == '__main__':
    # Example usage
    augmenter = VLADataAugmentation(
        image_size=(480, 640),
        augmentation_prob=0.5,
        include_robot_aug=True
    )

    # Create dummy data
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.random.rand(480, 640).astype(np.float32)
    text = "Pick the red box"
    robot_state = np.random.randn(7)

    # Apply augmentation
    augmented = augmenter(rgb, depth, text, robot_state)

    print("Augmentation results:")
    print(f"  RGB shape: {augmented['rgb'].shape}")
    print(f"  Depth shape: {augmented['depth'].shape}")
    print(f"  Text: {augmented['text']}")
    if 'robot_state' in augmented:
        print(f"  Robot state shape: {augmented['robot_state'].shape}")
