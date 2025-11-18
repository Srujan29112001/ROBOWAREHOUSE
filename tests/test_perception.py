"""Tests for perception module."""

import pytest
import torch
import numpy as np

from robo_vla.perception import (
    DinoV2Detector,
    MiDasDepthEstimator,
    PointCloudGenerator,
    GaussianSplatting,
)


class TestDinoV2:
    """Test DINO v2 detector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return DinoV2Detector(
            model_name="facebook/dinov2-base",
            device="cpu",
        )

    def test_forward(self, detector):
        """Test forward pass."""
        images = torch.randn(2, 3, 518, 518)
        output = detector(images)

        assert "features" in output
        assert "patch_features" in output
        assert output["features"].shape == (2, 768)

    def test_detect_objects(self, detector):
        """Test object detection."""
        images = torch.randn(1, 3, 640, 480)
        detections = detector.detect_objects(images, threshold=0.5)

        assert len(detections) == 1
        assert "boxes" in detections[0]
        assert "scores" in detections[0]
        assert "features" in detections[0]


class TestMiDas:
    """Test MiDaS depth estimator."""

    @pytest.fixture
    def estimator(self):
        """Create estimator instance."""
        return MiDasDepthEstimator(
            model_type="DPT_Hybrid",
            device="cpu",
        )

    def test_forward(self, estimator):
        """Test depth estimation."""
        images = torch.randn(1, 3, 480, 640)
        depth = estimator(images)

        assert depth.shape == (1, 1, 480, 640)
        assert depth.min() >= estimator.min_depth
        assert depth.max() <= estimator.max_depth


class TestPointCloudGenerator:
    """Test point cloud generator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return PointCloudGenerator(
            fx=525.0,
            fy=525.0,
            cx=320.0,
            cy=240.0,
        )

    def test_generate(self, generator):
        """Test point cloud generation."""
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640).astype(np.float32) * 5.0

        pcd = generator.generate(rgb, depth)

        assert len(pcd.points) > 0
        assert len(pcd.colors) > 0


class TestGaussianSplatting:
    """Test 3D Gaussian Splatting."""

    @pytest.fixture
    def splatting(self):
        """Create splatting instance."""
        return GaussianSplatting(
            num_gaussians=1000,
            device="cpu",
        )

    def test_initialization(self, splatting):
        """Test initialization from point cloud."""
        points = torch.randn(500, 3)
        colors = torch.rand(500, 3)

        splatting.initialize_from_point_cloud(points, colors)

        assert splatting.means is not None
        assert splatting.means.shape == (1000, 3)

    def test_get_covariance(self, splatting):
        """Test covariance matrix computation."""
        points = torch.randn(100, 3)
        colors = torch.rand(100, 3)

        splatting.initialize_from_point_cloud(points, colors)
        covariances = splatting.get_covariance_matrices()

        assert covariances.shape == (1000, 3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
