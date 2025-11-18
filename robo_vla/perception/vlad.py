"""NetVLAD for visual place recognition and object instance matching."""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NetVLAD(nn.Module):
    """
    NetVLAD layer for aggregating local features into global descriptor.

    Used for object instance matching and visual place recognition.
    """

    def __init__(
        self,
        num_clusters: int = 64,
        descriptor_dim: int = 512,
        normalize: bool = True,
        alpha: float = 100.0,
    ):
        """
        Initialize NetVLAD layer.

        Args:
            num_clusters: Number of cluster centers (K)
            descriptor_dim: Dimension of local descriptors (D)
            normalize: Whether to L2-normalize output
            alpha: Softmax temperature parameter
        """
        super().__init__()

        self.num_clusters = num_clusters
        self.descriptor_dim = descriptor_dim
        self.normalize = normalize
        self.alpha = alpha

        # Cluster centers (learnable)
        self.centroids = nn.Parameter(
            torch.randn(num_clusters, descriptor_dim)
        )

        # Convolution for soft-assignment
        self.conv = nn.Conv2d(
            descriptor_dim,
            num_clusters,
            kernel_size=1,
            bias=True,
        )

        self._init_params()

        logger.info(
            f"NetVLAD initialized with K={num_clusters}, D={descriptor_dim}"
        )

    def _init_params(self):
        """Initialize parameters."""
        # Initialize convolution weights from centroids
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            -(self.alpha * self.centroids.norm(dim=1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NetVLAD.

        Args:
            x: Local descriptors [B, D, H, W]

        Returns:
            Global descriptor [B, K*D]
        """
        B, D, H, W = x.shape
        N = H * W  # Number of descriptors

        # Soft-assignment (B, K, H, W)
        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign, dim=1)

        # Reshape for computation
        x_flatten = x.view(B, D, -1)  # (B, D, N)
        soft_assign_flatten = soft_assign.view(B, self.num_clusters, -1)  # (B, K, N)

        # VLAD core computation
        # residuals(b,k,d) = sum_n( a_k(x_n) * (x_n - c_k) )
        vlad = torch.zeros(
            [B, self.num_clusters, D],
            dtype=x.dtype,
            device=x.device,
        )

        for k in range(self.num_clusters):
            # Residuals: x - c_k
            residual = x_flatten - self.centroids[k:k+1].T.unsqueeze(0)  # (B, D, N)

            # Weighted sum
            vlad[:, k, :] = (
                residual * soft_assign_flatten[:, k:k+1, :]
            ).sum(dim=-1)

        # Flatten VLAD descriptor
        vlad = vlad.view(B, -1)  # (B, K*D)

        # Intra-normalization
        vlad = F.normalize(vlad, p=2, dim=1)

        # L2 normalization
        if self.normalize:
            vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


class VLADRetrieval:
    """
    VLAD-based object instance retrieval system.

    Maintains a database of object descriptors for matching.
    """

    def __init__(
        self,
        vlad_model: NetVLAD,
        device: str = "cuda",
    ):
        """
        Initialize VLAD retrieval system.

        Args:
            vlad_model: NetVLAD model
            device: Device to run on
        """
        self.vlad = vlad_model.to(device)
        self.device = device

        # Database
        self.descriptors = []
        self.object_ids = []
        self.metadata = []

        logger.info("VLAD retrieval system initialized")

    def add_object(
        self,
        features: torch.Tensor,
        object_id: str,
        metadata: dict = None,
    ) -> None:
        """
        Add object to database.

        Args:
            features: Local features [D, H, W]
            object_id: Unique object identifier
            metadata: Additional metadata
        """
        with torch.no_grad():
            # Compute VLAD descriptor
            features = features.unsqueeze(0).to(self.device)
            descriptor = self.vlad(features).cpu()

            self.descriptors.append(descriptor)
            self.object_ids.append(object_id)
            self.metadata.append(metadata or {})

        logger.debug(f"Added object {object_id} to database")

    def retrieve(
        self,
        query_features: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[list, list, list]:
        """
        Retrieve similar objects.

        Args:
            query_features: Query features [D, H, W]
            top_k: Number of results to return

        Returns:
            Tuple of (object_ids, similarities, metadata)
        """
        if len(self.descriptors) == 0:
            return [], [], []

        with torch.no_grad():
            # Compute query descriptor
            query_features = query_features.unsqueeze(0).to(self.device)
            query_descriptor = self.vlad(query_features).cpu()

            # Stack all descriptors
            db_descriptors = torch.cat(self.descriptors, dim=0)

            # Compute similarities (cosine similarity)
            similarities = F.cosine_similarity(
                query_descriptor,
                db_descriptors,
                dim=1,
            )

            # Get top-k
            top_k = min(top_k, len(similarities))
            scores, indices = torch.topk(similarities, top_k)

            # Gather results
            retrieved_ids = [self.object_ids[i] for i in indices]
            retrieved_scores = scores.tolist()
            retrieved_metadata = [self.metadata[i] for i in indices]

        return retrieved_ids, retrieved_scores, retrieved_metadata

    def save(self, path: str) -> None:
        """Save database to file."""
        data = {
            "descriptors": torch.cat(self.descriptors, dim=0) if self.descriptors else torch.tensor([]),
            "object_ids": self.object_ids,
            "metadata": self.metadata,
        }
        torch.save(data, path)
        logger.info(f"VLAD database saved to {path}")

    def load(self, path: str) -> None:
        """Load database from file."""
        data = torch.load(path)
        descriptors_tensor = data["descriptors"]

        if len(descriptors_tensor) > 0:
            self.descriptors = [
                descriptors_tensor[i:i+1]
                for i in range(len(descriptors_tensor))
            ]
        else:
            self.descriptors = []

        self.object_ids = data["object_ids"]
        self.metadata = data["metadata"]

        logger.info(f"VLAD database loaded from {path} ({len(self.descriptors)} objects)")
