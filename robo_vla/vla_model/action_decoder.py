"""Action decoder for generating robot commands."""

import logging
import math
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ActionDecoder(nn.Module):
    """
    Transformer-based action decoder.

    Generates robot action sequences from fused multimodal features.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        num_joints: int = 7,
        gripper_dim: int = 1,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        max_trajectory_steps: int = 10,
    ):
        """
        Initialize action decoder.

        Args:
            hidden_dim: Hidden dimension
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            num_joints: Number of robot joints
            gripper_dim: Gripper control dimension
            feedforward_dim: Feed-forward dimension
            dropout: Dropout rate
            max_trajectory_steps: Maximum trajectory length
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.gripper_dim = gripper_dim
        self.action_dim = num_joints + gripper_dim
        self.max_trajectory_steps = max_trajectory_steps

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Action prediction heads
        self.joint_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_joints),
            nn.Tanh(),  # Output in [-1, 1]
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, gripper_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Auxiliary heads
        self.success_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Positive output
        )

        # Positional encoding for trajectory
        self.pos_encoding = PositionalEncoding(hidden_dim, max_trajectory_steps)

    def forward(
        self,
        memory: torch.Tensor,
        trajectory_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate action trajectory.

        Args:
            memory: Encoded multimodal features [B, L, hidden_dim]
            trajectory_steps: Number of trajectory steps to generate

        Returns:
            Dictionary containing actions and predictions
        """
        batch_size = memory.shape[0]
        device = memory.device

        # Initialize trajectory query
        trajectory_query = torch.zeros(
            batch_size,
            trajectory_steps,
            self.hidden_dim,
            device=device,
        )

        # Add positional encoding
        trajectory_query = self.pos_encoding(trajectory_query)

        # Decode trajectory
        decoded = self.decoder(
            tgt=trajectory_query,
            memory=memory,
        )  # [B, T, hidden_dim]

        # Predict actions for each timestep
        joints = self.joint_head(decoded)  # [B, T, num_joints]
        gripper = self.gripper_head(decoded)  # [B, T, gripper_dim]

        # Scale joint angles to [-π, π]
        joints = joints * math.pi

        # Auxiliary predictions (from final timestep)
        final_features = decoded[:, -1, :]  # [B, hidden_dim]
        success_prob = self.success_head(final_features)  # [B, 1]
        execution_time = self.time_head(final_features)  # [B, 1]

        return {
            "joints": joints,  # [B, T, num_joints]
            "gripper": gripper,  # [B, T, gripper_dim]
            "success_probability": success_prob,  # [B, 1]
            "execution_time": execution_time,  # [B, 1]
        }

    def generate_single_action(
        self,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate single-step action.

        Args:
            memory: Encoded features [B, L, hidden_dim]

        Returns:
            Action [B, action_dim]
        """
        output = self.forward(memory, trajectory_steps=1)

        action = torch.cat([
            output["joints"][:, 0, :],  # [B, num_joints]
            output["gripper"][:, 0, :],  # [B, gripper_dim]
        ], dim=-1)  # [B, action_dim]

        return action


class PositionalEncoding(nn.Module):
    """Positional encoding for trajectory timesteps."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, D]

        Returns:
            Tensor with positional encoding added [B, T, D]
        """
        x = x + self.pe[:, : x.size(1), :]
        return x
