"""Spatial Graph Network for modeling object relationships."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpatialGraphNetwork(nn.Module):
    """
    Spatial Graph Network for modeling object relationships.

    Learns spatial relationships between objects in 3D space:
    - Distance-based edges
    - Containment relationships (inside, on_top_of, next_to)
    - Support relationships
    - Reachability
    """

    def __init__(
        self,
        node_dim: int = 768,
        edge_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize Spatial Graph Network.

        Args:
            node_dim: Dimension of node features (object features)
            edge_dim: Dimension of edge features (spatial relationships)
            hidden_dim: Hidden dimension for graph layers
            num_layers: Number of graph neural network layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Relationship classifier
        self.relationship_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),  # 7 relationship types
        )

        # Relationship types
        self.relationship_types = [
            'none',
            'on_top_of',
            'inside',
            'next_to',
            'behind',
            'in_front_of',
            'supports',
        ]

        logger.info(f"Spatial Graph Network initialized with {num_layers} layers")

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through spatial graph network.

        Args:
            node_features: Object features [N, node_dim]
            positions: 3D positions [N, 3] (x, y, z)
            batch_size: Batch size (if batched)

        Returns:
            Tuple of:
            - Updated node features [N, hidden_dim]
            - Edge features [N, N, hidden_dim]
            - Relationship probabilities [N, N, 7]
        """
        N = node_features.shape[0]

        # Project node features
        nodes = self.node_proj(node_features)  # [N, hidden_dim]

        # Compute edge features from spatial relationships
        edges = self._compute_edge_features(positions)  # [N, N, edge_dim]
        edges = self.edge_encoder(edges)  # [N, N, hidden_dim]

        # Apply graph layers
        for layer in self.graph_layers:
            nodes = layer(nodes, edges)  # [N, hidden_dim]

        # Compute pairwise relationships
        relationships = self._compute_relationships(nodes)  # [N, N, 7]

        return nodes, edges, relationships

    def _compute_edge_features(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge features from 3D positions.

        Args:
            positions: 3D positions [N, 3]

        Returns:
            Edge features [N, N, edge_dim]
        """
        N = positions.shape[0]
        device = positions.device

        # Pairwise distances
        pos_i = positions.unsqueeze(1).expand(N, N, 3)  # [N, N, 3]
        pos_j = positions.unsqueeze(0).expand(N, N, 3)  # [N, N, 3]

        # Distance vector
        delta = pos_j - pos_i  # [N, N, 3]
        distance = torch.norm(delta, dim=-1, keepdim=True)  # [N, N, 1]

        # Normalized direction
        direction = F.normalize(delta, dim=-1)  # [N, N, 3]

        # Relative height (z-axis)
        height_diff = delta[..., 2:3]  # [N, N, 1]

        # Angular features
        # Angle from horizontal plane
        angle_elevation = torch.atan2(
            delta[..., 2],
            torch.norm(delta[..., :2], dim=-1)
        ).unsqueeze(-1)  # [N, N, 1]

        # Angle in xy plane
        angle_azimuth = torch.atan2(
            delta[..., 1],
            delta[..., 0]
        ).unsqueeze(-1)  # [N, N, 1]

        # Distance bins (useful for learning discrete relationships)
        distance_bins = self._discretize_distance(distance)  # [N, N, 10]

        # Direction bins
        direction_bins = self._discretize_direction(direction)  # [N, N, 8]

        # Concatenate all edge features
        edge_features = torch.cat([
            distance,  # 1
            direction,  # 3
            height_diff,  # 1
            angle_elevation,  # 1
            angle_azimuth,  # 1
            distance_bins,  # 10
            direction_bins,  # 8
        ], dim=-1)  # [N, N, 25]

        # Pad or project to edge_dim
        if edge_features.shape[-1] < self.edge_dim:
            padding = torch.zeros(
                N, N, self.edge_dim - edge_features.shape[-1],
                device=device
            )
            edge_features = torch.cat([edge_features, padding], dim=-1)
        elif edge_features.shape[-1] > self.edge_dim:
            edge_features = edge_features[..., :self.edge_dim]

        return edge_features

    def _discretize_distance(
        self,
        distance: torch.Tensor,
        num_bins: int = 10,
        max_distance: float = 5.0,
    ) -> torch.Tensor:
        """One-hot encode distance into bins."""
        bins = torch.linspace(0, max_distance, num_bins + 1, device=distance.device)
        indices = torch.bucketize(distance.squeeze(-1), bins[:-1])
        indices = torch.clamp(indices, 0, num_bins - 1)
        return F.one_hot(indices, num_classes=num_bins).float()

    def _discretize_direction(
        self,
        direction: torch.Tensor,
        num_bins: int = 8,
    ) -> torch.Tensor:
        """Discretize 3D direction into octants."""
        # Project to xy plane and get angle
        angle = torch.atan2(direction[..., 1], direction[..., 0])
        # Bin into 8 directions
        bin_size = 2 * np.pi / num_bins
        indices = ((angle + np.pi) / bin_size).long()
        indices = torch.clamp(indices, 0, num_bins - 1)
        return F.one_hot(indices, num_classes=num_bins).float()

    def _compute_relationships(
        self,
        nodes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise relationships between nodes.

        Args:
            nodes: Node features [N, hidden_dim]

        Returns:
            Relationship probabilities [N, N, 7]
        """
        N = nodes.shape[0]

        # Create pairwise node features
        nodes_i = nodes.unsqueeze(1).expand(N, N, -1)  # [N, N, hidden_dim]
        nodes_j = nodes.unsqueeze(0).expand(N, N, -1)  # [N, N, hidden_dim]

        # Concatenate pairs
        pairs = torch.cat([nodes_i, nodes_j], dim=-1)  # [N, N, 2*hidden_dim]

        # Classify relationships
        relationships = self.relationship_classifier(pairs)  # [N, N, 7]

        # Apply softmax
        relationships = F.softmax(relationships, dim=-1)

        return relationships

    def get_relationships(
        self,
        relationship_probs: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[Tuple[int, int, str]]:
        """
        Extract relationships from probability matrix.

        Args:
            relationship_probs: Relationship probabilities [N, N, 7]
            threshold: Probability threshold

        Returns:
            List of (object_i, object_j, relationship_type) tuples
        """
        N = relationship_probs.shape[0]
        relationships = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                probs = relationship_probs[i, j]
                max_prob, max_idx = torch.max(probs, dim=0)

                if max_prob > threshold and max_idx > 0:  # Skip 'none'
                    rel_type = self.relationship_types[max_idx]
                    relationships.append((i, j, rel_type))

        return relationships


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with edge features."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize Graph Attention Layer.

        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge projection
        self.edge_proj = nn.Linear(hidden_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            nodes: Node features [N, hidden_dim]
            edges: Edge features [N, N, hidden_dim]

        Returns:
            Updated node features [N, hidden_dim]
        """
        N = nodes.shape[0]

        # Project to Q, K, V
        Q = self.q_proj(nodes).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        K = self.k_proj(nodes).view(N, self.num_heads, self.head_dim)  # [N, H, D]
        V = self.v_proj(nodes).view(N, self.num_heads, self.head_dim)  # [N, H, D]

        # Compute attention scores
        scores = torch.einsum('ihd,jhd->hij', Q, K)  # [H, N, N]
        scores = scores / np.sqrt(self.head_dim)

        # Add edge bias
        edge_bias = self.edge_proj(edges)  # [N, N, H]
        edge_bias = edge_bias.permute(2, 0, 1)  # [H, N, N]
        scores = scores + edge_bias

        # Softmax
        attn = F.softmax(scores, dim=-1)  # [H, N, N]
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.einsum('hij,jhd->ihd', attn, V)  # [N, H, D]
        out = out.reshape(N, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual connection and layer norm
        out = self.norm(nodes + out)

        return out
