"""CLIP for vision-language alignment."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPAlignment(nn.Module):
    """
    CLIP for vision-language feature alignment.

    Projects visual and text features into a shared embedding space
    for matching and grounding.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        device: str = "cuda",
        quantization: Optional[str] = None,
    ):
        """
        Initialize CLIP alignment module.

        Args:
            model_name: CLIP model name
            embed_dim: Embedding dimension
            device: Device to run on
            quantization: Quantization strategy
        """
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim

        logger.info(f"Loading CLIP model: {model_name}")

        # Load CLIP
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Apply quantization
        if quantization == "int8":
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif quantization == "fp16":
            self.model = self.model.half()

        self.model = self.model.to(device)
        self.model.eval()

        # Temperature parameter for similarity scaling
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        logger.info("CLIP alignment module initialized")

    @torch.no_grad()
    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Image embeddings [B, embed_dim]
        """
        # CLIP expects specific preprocessing
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(
                images,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

        # Encode
        image_features = self.model.get_image_features(pixel_values=images)

        # Normalize
        image_features = F.normalize(image_features, dim=-1)

        return image_features

    @torch.no_grad()
    def encode_text(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Text embeddings [B, embed_dim]
        """
        # Tokenize
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Encode
        text_features = self.model.get_text_features(**inputs)

        # Normalize
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between image and text features.

        Args:
            image_features: Image embeddings [N, embed_dim]
            text_features: Text embeddings [M, embed_dim]

        Returns:
            Similarity matrix [N, M]
        """
        # Cosine similarity
        similarity = torch.matmul(image_features, text_features.T)

        # Scale by temperature
        similarity = similarity / self.temperature

        return similarity

    def align_vision_language(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Align vision and language features.

        Args:
            visual_features: Visual features [N, visual_dim]
            text_features: Text features [M, text_dim]

        Returns:
            Dictionary with aligned features and similarity scores
        """
        # Project to CLIP space if needed
        if visual_features.shape[-1] != self.embed_dim:
            visual_proj = self.visual_projection(visual_features)
        else:
            visual_proj = visual_features

        if text_features.shape[-1] != self.embed_dim:
            text_proj = self.text_projection(text_features)
        else:
            text_proj = text_features

        # Normalize
        visual_norm = F.normalize(visual_proj, dim=-1)
        text_norm = F.normalize(text_proj, dim=-1)

        # Compute similarity
        similarity = self.compute_similarity(visual_norm, text_norm)

        # Find best matches
        visual_to_text = similarity.argmax(dim=1)  # [N]
        text_to_visual = similarity.argmax(dim=0)  # [M]

        # Get scores
        visual_scores = similarity.max(dim=1)[0]  # [N]
        text_scores = similarity.max(dim=0)[0]  # [M]

        # Create aligned features by concatenation
        aligned_visual = torch.cat([
            visual_norm,
            text_norm[visual_to_text],
        ], dim=-1)  # [N, 2*embed_dim]

        return {
            "aligned_features": aligned_visual,
            "similarity_matrix": similarity,
            "visual_to_text": visual_to_text,
            "text_to_visual": text_to_visual,
            "visual_scores": visual_scores,
            "text_scores": text_scores,
        }

    def visual_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Project visual features to CLIP space."""
        if not hasattr(self, "_visual_proj"):
            self._visual_proj = nn.Linear(
                features.shape[-1],
                self.embed_dim
            ).to(self.device)

        return self._visual_proj(features)

    def text_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Project text features to CLIP space."""
        if not hasattr(self, "_text_proj"):
            self._text_proj = nn.Linear(
                features.shape[-1],
                self.embed_dim
            ).to(self.device)

        return self._text_proj(features)

    def zero_shot_classification(
        self,
        images: torch.Tensor,
        text_labels: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform zero-shot classification.

        Args:
            images: Input images [B, C, H, W]
            text_labels: List of text labels

        Returns:
            Tuple of (predictions, probabilities)
        """
        # Encode
        image_features = self.encode_images(images)
        text_features = self.encode_text(text_labels)

        # Compute similarity
        similarity = self.compute_similarity(image_features, text_features)

        # Get predictions
        probabilities = F.softmax(similarity, dim=-1)
        predictions = similarity.argmax(dim=-1)

        return predictions, probabilities

    def visual_grounding(
        self,
        image: torch.Tensor,
        text_query: str,
        boxes: torch.Tensor,
    ) -> Tuple[int, float]:
        """
        Ground text query to visual region.

        Args:
            image: Input image [C, H, W]
            text_query: Text description
            boxes: Bounding boxes [N, 4]

        Returns:
            Tuple of (best_box_idx, confidence)
        """
        # Extract regions from boxes (simplified)
        # In practice, would use RoI Align

        num_boxes = len(boxes)

        # Encode text
        text_features = self.encode_text([text_query])  # [1, embed_dim]

        # Encode full image
        image_features = self.encode_images(image.unsqueeze(0))  # [1, embed_dim]

        # Compute similarity (simplified - would extract box features)
        similarity = self.compute_similarity(
            image_features.repeat(num_boxes, 1),
            text_features
        ).squeeze(-1)  # [N]

        # Get best match
        best_idx = similarity.argmax().item()
        confidence = similarity[best_idx].item()

        return best_idx, confidence
