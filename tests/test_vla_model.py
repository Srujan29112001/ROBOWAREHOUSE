"""Tests for VLA model."""

import pytest
import torch

from robo_vla.vla_model import CrossAttentionFusion, ActionDecoder
from robo_vla.utils import load_config


class TestCrossAttentionFusion:
    """Test cross-attention fusion module."""

    @pytest.fixture
    def fusion(self):
        """Create fusion module."""
        return CrossAttentionFusion(
            visual_dim=768,
            text_dim=4096,
            state_dim=8,
            hidden_dim=256,
            num_heads=8,
        )

    def test_forward(self, fusion):
        """Test forward pass."""
        visual = torch.randn(2, 768)
        text = torch.randn(2, 4096)
        state = torch.randn(2, 8)

        output, attention = fusion(visual, text, state)

        assert output.shape[0] == 2
        assert output.shape[-1] == 256
        assert "visual_to_text" in attention


class TestActionDecoder:
    """Test action decoder."""

    @pytest.fixture
    def decoder(self):
        """Create decoder instance."""
        return ActionDecoder(
            hidden_dim=256,
            num_decoder_layers=4,
            num_heads=8,
            num_joints=7,
            gripper_dim=1,
        )

    def test_forward(self, decoder):
        """Test action generation."""
        memory = torch.randn(2, 10, 256)  # [B, L, D]

        output = decoder(memory, trajectory_steps=5)

        assert "joints" in output
        assert "gripper" in output
        assert "success_probability" in output
        assert output["joints"].shape == (2, 5, 7)
        assert output["gripper"].shape == (2, 5, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
