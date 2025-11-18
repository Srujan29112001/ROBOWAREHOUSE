"""MiDaS depth estimation module."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MiDasDepthEstimator(nn.Module):
    """
    MiDaS depth estimation using DPT-Hybrid model.

    Combines Vision Transformer encoder with CNN decoder for
    accurate depth prediction.
    """

    def __init__(
        self,
        model_type: str = "DPT_Hybrid",
        device: str = "cuda",
        optimize: bool = True,
        fp16: bool = True,
        max_depth: float = 10.0,
        min_depth: float = 0.1,
    ):
        """
        Initialize MiDaS depth estimator.

        Args:
            model_type: Model variant ('DPT_Hybrid', 'DPT_Large', 'MiDaS_small')
            device: Device to run model on
            optimize: Apply optimization (TensorRT, etc.)
            fp16: Use half precision
            max_depth: Maximum depth in meters
            min_depth: Minimum depth in meters
        """
        super().__init__()
        self.device = device
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.fp16 = fp16

        logger.info(f"Loading MiDaS model: {model_type}")

        # Load MiDaS model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)

        if fp16:
            self.model = self.model.half()

        self.model = self.model.to(device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        if optimize:
            self._optimize_model()

        logger.info(f"MiDaS loaded successfully on {device}")

    def _optimize_model(self):
        """Apply model optimizations."""
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True

        # Try to compile model (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict depth maps.

        Args:
            images: Input RGB images [B, 3, H, W] (normalized)

        Returns:
            Depth maps [B, 1, H, W] in meters
        """
        original_size = images.shape[-2:]

        # MiDaS expects specific input size
        if images.shape[-2:] != (384, 384):
            images = F.interpolate(
                images,
                size=(384, 384),
                mode='bilinear',
                align_corners=False
            )

        # Cast to fp16 if needed
        if self.fp16:
            images = images.half()

        # Predict depth
        prediction = self.model(images)

        # Resize back to original size
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=original_size,
            mode='bicubic',
            align_corners=False
        )

        # Normalize to [min_depth, max_depth]
        depth = self._normalize_depth(prediction)

        return depth

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize depth to metric scale."""
        # MiDaS outputs inverse depth
        depth_min = depth.min()
        depth_max = depth.max()

        # Normalize to [0, 1]
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        # Scale to [min_depth, max_depth]
        depth_metric = self.min_depth + depth_normalized * (self.max_depth - self.min_depth)

        return depth_metric

    def estimate_batch(
        self,
        images_np: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate depth for batch of numpy images.

        Args:
            images_np: Batch of RGB images [B, H, W, 3], uint8

        Returns:
            Depth maps [B, H, W] in meters
        """
        # Convert to tensor
        images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
        images = images.to(self.device)

        # Predict
        depth = self.forward(images)

        # Convert back to numpy
        depth_np = depth.squeeze(1).cpu().numpy()

        return depth_np

    def estimate_single(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate depth for single image.

        Args:
            image: RGB image [H, W, 3], uint8

        Returns:
            Depth map [H, W] in meters
        """
        return self.estimate_batch(image[None])[0]

    def visualize_depth(
        self,
        depth: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO,
    ) -> np.ndarray:
        """
        Create visualization of depth map.

        Args:
            depth: Depth map [H, W]
            colormap: OpenCV colormap

        Returns:
            Colored depth visualization [H, W, 3]
        """
        # Normalize to [0, 255]
        depth_normalized = (depth - self.min_depth) / (self.max_depth - self.min_depth)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)

        return depth_colored
