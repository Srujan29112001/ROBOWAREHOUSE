"""Cross-attention fusion for multimodal features."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for vision-language-action.

    Fuses visual features, language embeddings, and robot state
    using multi-head cross-attention.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 4096,
        state_dim: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention fusion.

        Args:
            visual_dim: Visual feature dimension
            text_dim: Text feature dimension
            state_dim: Robot state dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Projection layers
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Cross-attention layers
        self.visual_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.text_to_visual_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention for final fusion
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        robot_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through cross-attention fusion.

        Args:
            visual_features: Visual features [B, N, visual_dim] or [B, visual_dim]
            text_features: Text features [B, M, text_dim] or [B, text_dim]
            robot_state: Robot state [B, state_dim]

        Returns:
            Tuple of (fused_features, attention_weights)
        """
        batch_size = visual_features.shape[0]

        # Ensure sequence dimension exists
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)  # [B, 1, D]
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # [B, 1, D]

        # Project to hidden dimension
        visual = self.visual_proj(visual_features)  # [B, N, hidden_dim]
        text = self.text_proj(text_features)  # [B, M, hidden_dim]
        state = self.state_proj(robot_state).unsqueeze(1)  # [B, 1, hidden_dim]

        # Cross-attention: Visual attends to Text
        visual_attended, v2t_weights = self.visual_to_text_attn(
            query=visual,
            key=text,
            value=text,
        )
        visual = self.norm1(visual + self.dropout(visual_attended))

        # Cross-attention: Text attends to Visual
        text_attended, t2v_weights = self.text_to_visual_attn(
            query=text,
            key=visual,
            value=visual,
        )
        text = self.norm2(text + self.dropout(text_attended))

        # Concatenate all modalities
        multimodal = torch.cat([visual, text, state], dim=1)  # [B, N+M+1, hidden_dim]

        # Self-attention for final fusion
        fused, self_weights = self.self_attn(
            query=multimodal,
            key=multimodal,
            value=multimodal,
        )
        fused = self.norm3(multimodal + self.dropout(fused))

        # Feed-forward
        fused_ffn = self.ffn(fused)
        fused = self.norm4(fused + self.dropout(fused_ffn))

        # Collect attention weights
        attention_weights = {
            "visual_to_text": v2t_weights,
            "text_to_visual": t2v_weights,
            "self_attention": self_weights,
        }

        return fused, attention_weights

    def get_pooled_features(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Pool fused features to single vector."""
        return fused_features.mean(dim=1)  # [B, hidden_dim]
