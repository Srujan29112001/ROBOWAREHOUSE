"""DINO v2 object detection and feature extraction."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import numpy as np

logger = logging.getLogger(__name__)


class DinoV2Detector(nn.Module):
    """
    DINO v2 for zero-shot object detection and feature extraction.

    Architecture: Vision Transformer (ViT-Base)
    - 12 layers
    - 768 hidden dimension
    - 12 attention heads
    - 85M parameters
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        feature_dim: int = 768,
        device: str = "cuda",
        quantization: Optional[str] = None,
    ):
        """
        Initialize DINO v2 detector.

        Args:
            model_name: Hugging Face model name
            feature_dim: Feature embedding dimension
            device: Device to run model on
            quantization: Quantization strategy ('int8', 'fp16', None)
        """
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim

        logger.info(f"Loading DINO v2 model: {model_name}")

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Apply quantization
        if quantization == "fp16":
            self.model = self.model.half()
        elif quantization == "int8":
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        self.model = self.model.to(device)
        self.model.eval()

        logger.info(f"DINO v2 loaded successfully on {device}")

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DINO v2.

        Args:
            images: Input images [B, C, H, W]
            return_attention: Whether to return attention maps

        Returns:
            Dictionary containing:
                - features: Global features [B, feature_dim]
                - patch_features: Patch-level features [B, N, feature_dim]
                - attention_maps: Optional attention maps [B, num_heads, N, N]
        """
        # Preprocess images
        if images.shape[-2:] != (518, 518):
            images = F.interpolate(
                images,
                size=(518, 518),
                mode='bilinear',
                align_corners=False
            )

        # Forward pass
        outputs = self.model(
            pixel_values=images,
            output_attentions=return_attention,
        )

        # Extract features
        last_hidden_state = outputs.last_hidden_state  # [B, N+1, D]

        # CLS token (global feature)
        cls_token = last_hidden_state[:, 0]  # [B, D]

        # Patch features (local features)
        patch_features = last_hidden_state[:, 1:]  # [B, N, D]

        result = {
            "features": cls_token,
            "patch_features": patch_features,
        }

        if return_attention:
            result["attention_maps"] = outputs.attentions[-1]

        return result

    def detect_objects(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Detect objects using attention-based approach.

        Args:
            images: Input images [B, C, H, W]
            threshold: Detection threshold

        Returns:
            List of detections per image, each containing:
                - boxes: Bounding boxes [N, 4] (x1, y1, x2, y2)
                - scores: Confidence scores [N]
                - features: Object features [N, feature_dim]
        """
        outputs = self.forward(images, return_attention=True)

        attention_maps = outputs["attention_maps"]  # [B, H, N, N]
        patch_features = outputs["patch_features"]  # [B, N, D]

        batch_size = images.shape[0]
        h = w = int(np.sqrt(patch_features.shape[1]))

        detections = []

        for b in range(batch_size):
            # Average attention across heads
            attn = attention_maps[b].mean(0)  # [N, N]

            # CLS token attention to patches
            cls_attn = attn[0, 1:].reshape(h, w)  # [h, w]

            # Normalize attention
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

            # Find peaks (objects)
            peaks = self._find_peaks(cls_attn, threshold)

            boxes = []
            scores = []
            features_list = []

            for peak in peaks:
                # Convert peak to bounding box
                y, x = peak
                box = self._peak_to_box(y, x, h, w, images.shape[-2:])

                # Get feature for this region
                patch_idx = y * w + x
                feature = patch_features[b, patch_idx]

                boxes.append(box)
                scores.append(cls_attn[y, x].item())
                features_list.append(feature.cpu().numpy())

            detections.append({
                "boxes": np.array(boxes) if boxes else np.zeros((0, 4)),
                "scores": np.array(scores) if scores else np.zeros(0),
                "features": np.array(features_list) if features_list else np.zeros((0, self.feature_dim)),
            })

        return detections

    def _find_peaks(
        self,
        attention_map: torch.Tensor,
        threshold: float,
    ) -> List[Tuple[int, int]]:
        """Find local peaks in attention map."""
        from scipy.ndimage import maximum_filter

        attn_np = attention_map.cpu().numpy()

        # Apply maximum filter
        local_max = maximum_filter(attn_np, size=3)

        # Find peaks
        peaks = (attn_np == local_max) & (attn_np > threshold)

        # Get coordinates
        coords = np.argwhere(peaks)

        return [(int(y), int(x)) for y, x in coords]

    def _peak_to_box(
        self,
        y: int,
        x: int,
        grid_h: int,
        grid_w: int,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Convert peak location to bounding box."""
        img_h, img_w = image_size

        # Grid cell size
        cell_h = img_h / grid_h
        cell_w = img_w / grid_w

        # Box around peak (3x3 cells)
        x1 = max(0, (x - 1) * cell_w)
        y1 = max(0, (y - 1) * cell_h)
        x2 = min(img_w, (x + 2) * cell_w)
        y2 = min(img_h, (y + 2) * cell_h)

        return np.array([x1, y1, x2, y2])

    def extract_features(
        self,
        images: torch.Tensor,
        boxes: Optional[List[np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Extract features for given bounding boxes.

        Args:
            images: Input images [B, C, H, W]
            boxes: Optional list of boxes per image [N, 4]

        Returns:
            Features [B, N, feature_dim] or [B, feature_dim] if no boxes
        """
        outputs = self.forward(images)

        if boxes is None:
            return outputs["features"]

        # Extract features for each box using RoI pooling
        batch_features = []

        for b, image_boxes in enumerate(boxes):
            if len(image_boxes) == 0:
                batch_features.append(torch.zeros((0, self.feature_dim)))
                continue

            # Simple feature extraction (can be improved with RoI Align)
            features = outputs["features"][b:b+1].expand(len(image_boxes), -1)
            batch_features.append(features)

        return batch_features
