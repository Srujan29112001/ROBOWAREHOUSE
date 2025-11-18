"""Complete VLA (Vision-Language-Action) model."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from robo_vla.perception import PerceptionPipeline
from robo_vla.language import LanguageGroundingModule
from robo_vla.vla_model.cross_attention import CrossAttentionFusion
from robo_vla.vla_model.action_decoder import ActionDecoder

logger = logging.getLogger(__name__)


class VLAModel(nn.Module):
    """
    Complete Vision-Language-Action model for robotic manipulation.

    Integrates perception, language understanding, and action generation
    in an end-to-end trainable architecture.
    """

    def __init__(
        self,
        config: Dict,
        device: str = "cuda",
    ):
        """
        Initialize VLA model.

        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        super().__init__()
        self.config = config
        self.device = device

        logger.info("Initializing VLA Model...")

        # Initialize perception pipeline
        self.perception = PerceptionPipeline(config, device)

        # Initialize language grounding
        self.language = LanguageGroundingModule(config, device)

        # Initialize cross-attention fusion
        self.fusion = CrossAttentionFusion(
            visual_dim=config["vla_model"]["architecture"]["visual_encoder_dim"],
            text_dim=config["vla_model"]["architecture"]["text_encoder_dim"],
            state_dim=config["vla_model"]["action"]["num_joints"] + config["vla_model"]["action"]["gripper_dim"],
            hidden_dim=config["vla_model"]["architecture"]["hidden_dim"],
            num_heads=config["vla_model"]["architecture"]["num_attention_heads"],
            dropout=config["vla_model"]["architecture"]["dropout"],
        ).to(device)

        # Initialize action decoder
        self.action_decoder = ActionDecoder(
            hidden_dim=config["vla_model"]["architecture"]["hidden_dim"],
            num_decoder_layers=config["vla_model"]["architecture"]["num_decoder_layers"],
            num_heads=config["vla_model"]["architecture"]["num_attention_heads"],
            num_joints=config["vla_model"]["action"]["num_joints"],
            gripper_dim=config["vla_model"]["action"]["gripper_dim"],
            feedforward_dim=config["vla_model"]["architecture"]["feedforward_dim"],
            dropout=config["vla_model"]["architecture"]["dropout"],
            max_trajectory_steps=config["vla_model"]["action"]["trajectory_steps"],
        ).to(device)

        logger.info("VLA Model initialized successfully")

    def forward(
        self,
        rgb_images: torch.Tensor,
        text_commands: list,
        robot_state: torch.Tensor,
        trajectory_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VLA model.

        Args:
            rgb_images: RGB images [B, 3, H, W]
            text_commands: List of text commands
            robot_state: Current robot state [B, state_dim]
            trajectory_steps: Number of trajectory steps

        Returns:
            Dictionary with actions and predictions
        """
        # 1. Perception
        perception_output = self.perception.process(rgb_images)
        visual_features = perception_output["intermediate"]["dino_features"]  # [B, 768]

        # 2. Language understanding
        language_output = self.language.process_command(
            text_commands[0],  # Simplified: take first command
            images=rgb_images,
        )
        text_features = language_output["text_embeddings_llama"]  # [B, 4096]

        # 3. Cross-attention fusion
        fused_features, attention_weights = self.fusion(
            visual_features,
            text_features,
            robot_state,
        )  # [B, L, hidden_dim]

        # 4. Action generation
        actions = self.action_decoder(
            memory=fused_features,
            trajectory_steps=trajectory_steps,
        )

        return {
            **actions,
            "attention_weights": attention_weights,
            "visual_features": visual_features,
            "text_features": text_features,
            "fused_features": fused_features,
        }

    @torch.no_grad()
    def predict(
        self,
        rgb_image: torch.Tensor,
        text_command: str,
        robot_state: torch.Tensor,
    ) -> Dict:
        """
        Predict action for given observation and command.

        Args:
            rgb_image: Single RGB image [3, H, W]
            text_command: Text command string
            robot_state: Robot state [state_dim]

        Returns:
            Predicted action and metadata
        """
        self.eval()

        # Add batch dimension
        rgb_batch = rgb_image.unsqueeze(0)
        state_batch = robot_state.unsqueeze(0)

        # Forward pass
        output = self.forward(
            rgb_images=rgb_batch,
            text_commands=[text_command],
            robot_state=state_batch,
        )

        # Remove batch dimension
        result = {
            "joint_commands": output["joints"][0].cpu().numpy(),
            "gripper_command": output["gripper"][0].cpu().numpy(),
            "success_probability": output["success_probability"][0].item(),
            "execution_time": output["execution_time"][0].item(),
        }

        return result
