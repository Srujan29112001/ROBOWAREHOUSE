"""3D Gaussian Splatting for scene representation."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GaussianSplatting(nn.Module):
    """
    3D Gaussian Splatting for differentiable scene representation.

    Represents 3D scenes as collections of 3D Gaussians, enabling
    fast, high-quality novel view synthesis and scene understanding.
    """

    def __init__(
        self,
        num_gaussians: int = 10000,
        optimization_iterations: int = 30,
        learning_rate: float = 0.01,
        device: str = "cuda",
    ):
        """
        Initialize 3D Gaussian Splatting.

        Args:
            num_gaussians: Number of Gaussians to represent the scene
            optimization_iterations: Number of optimization steps
            learning_rate: Learning rate for optimization
            device: Device to run on
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.optimization_iterations = optimization_iterations
        self.learning_rate = learning_rate
        self.device = device

        # Gaussian parameters (will be initialized from point cloud)
        self.means = None  # [N, 3] - 3D positions
        self.log_scales = None  # [N, 3] - log of scales
        self.rotations = None  # [N, 4] - quaternions
        self.opacities = None  # [N, 1] - opacity values
        self.features = None  # [N, 3] - RGB colors

        logger.info(f"3D Gaussian Splatting initialized with {num_gaussians} gaussians")

    def initialize_from_point_cloud(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
    ) -> None:
        """
        Initialize Gaussians from point cloud.

        Args:
            points: Point positions [N, 3]
            colors: Point colors [N, 3]
        """
        num_points = points.shape[0]

        # Sample or pad to num_gaussians
        if num_points > self.num_gaussians:
            indices = torch.randperm(num_points)[:self.num_gaussians]
            points = points[indices]
            colors = colors[indices]
        elif num_points < self.num_gaussians:
            # Duplicate points
            repeats = self.num_gaussians // num_points + 1
            points = points.repeat(repeats, 1)[:self.num_gaussians]
            colors = colors.repeat(repeats, 1)[:self.num_gaussians]

        # Initialize parameters
        self.means = nn.Parameter(points.to(self.device))

        # Initialize scales from nearest neighbor distances
        distances = torch.cdist(points, points)
        distances[distances == 0] = float('inf')
        nn_distances, _ = distances.min(dim=1)
        initial_scale = nn_distances.mean() * 0.5

        self.log_scales = nn.Parameter(
            torch.log(torch.ones(self.num_gaussians, 3, device=self.device) * initial_scale)
        )

        # Initialize rotations as identity quaternions
        self.rotations = nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(self.num_gaussians, 1)
        )

        # Initialize opacities
        self.opacities = nn.Parameter(
            torch.ones(self.num_gaussians, 1, device=self.device) * 0.5
        )

        # Initialize features (RGB colors)
        self.features = nn.Parameter(colors.to(self.device))

        logger.info("Gaussians initialized from point cloud")

    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Compute 3D covariance matrices from scales and rotations.

        Returns:
            Covariance matrices [N, 3, 3]
        """
        # Get scales
        scales = torch.exp(self.log_scales)  # [N, 3]

        # Normalize quaternions
        rotations = F.normalize(self.rotations, dim=1)  # [N, 4]

        # Convert quaternion to rotation matrix
        w, x, y, z = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1),
        ], dim=1)  # [N, 3, 3]

        # Scale matrix
        S = torch.diag_embed(scales)  # [N, 3, 3]

        # Covariance: Σ = R S S^T R^T
        RS = torch.bmm(R, S)  # [N, 3, 3]
        covariance = torch.bmm(RS, RS.transpose(1, 2))  # [N, 3, 3]

        return covariance

    def project_gaussians(
        self,
        viewpoint: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Project 3D Gaussians to 2D for a given viewpoint.

        Args:
            viewpoint: Dictionary with 'intrinsics' and 'extrinsics'

        Returns:
            Dictionary with projected 2D Gaussians
        """
        # Transform to camera space
        world2cam = viewpoint["world_to_camera"]  # [4, 4]

        # Homogeneous coordinates
        means_homo = torch.cat([
            self.means,
            torch.ones(self.num_gaussians, 1, device=self.device)
        ], dim=1)  # [N, 4]

        means_cam = torch.matmul(means_homo, world2cam.T)[:, :3]  # [N, 3]

        # Project to 2D
        intrinsics = viewpoint["intrinsics"]  # [3, 3]
        means_2d = torch.matmul(means_cam, intrinsics.T)  # [N, 3]
        means_2d = means_2d[:, :2] / means_2d[:, 2:3]  # [N, 2]

        # Project covariance to 2D (Jacobian method)
        covariance_3d = self.get_covariance_matrices()  # [N, 3, 3]

        # Simplified 2D covariance (approximation)
        focal_x = intrinsics[0, 0]
        focal_y = intrinsics[1, 1]
        depth = means_cam[:, 2:3]  # [N, 1]

        # Scale covariance by projection
        scale_factor = (focal_x + focal_y) / (2 * depth)  # [N, 1]
        covariance_2d = covariance_3d[:, :2, :2] * scale_factor.unsqueeze(-1) ** 2

        return {
            "means_2d": means_2d,
            "covariance_2d": covariance_2d,
            "depths": means_cam[:, 2],
            "colors": torch.sigmoid(self.features),
            "opacities": torch.sigmoid(self.opacities),
        }

    def rasterize(
        self,
        projected: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Rasterize projected Gaussians to an image.

        Args:
            projected: Projected Gaussians from project_gaussians()
            image_size: Output image size (H, W)

        Returns:
            Rendered image [3, H, W]
        """
        H, W = image_size

        # Create pixel grid
        y_coords = torch.arange(H, device=self.device).float()
        x_coords = torch.arange(W, device=self.device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        pixel_coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        # Initialize output
        output = torch.zeros(3, H, W, device=self.device)
        alpha_accumulated = torch.zeros(1, H, W, device=self.device)

        # Sort Gaussians by depth (back to front)
        depths = projected["depths"]
        sorted_indices = torch.argsort(depths, descending=True)

        # Rasterize each Gaussian
        means_2d = projected["means_2d"][sorted_indices]  # [N, 2]
        colors = projected["colors"][sorted_indices]  # [N, 3]
        opacities = projected["opacities"][sorted_indices]  # [N, 1]

        # Simplified rasterization (full implementation would use tile-based rendering)
        for i in range(min(1000, self.num_gaussians)):  # Limit for speed
            mean = means_2d[i]  # [2]
            color = colors[i]  # [3]
            opacity = opacities[i, 0]  # scalar

            # Gaussian weights
            diff = pixel_coords - mean.view(1, 1, 2)  # [H, W, 2]
            dist_sq = (diff ** 2).sum(dim=-1)  # [H, W]

            # Simplified Gaussian (isotropic)
            sigma = 3.0
            weights = torch.exp(-dist_sq / (2 * sigma ** 2))  # [H, W]

            # Alpha blending
            alpha = weights * opacity  # [H, W]
            alpha = alpha * (1 - alpha_accumulated.squeeze(0))  # [H, W]

            # Accumulate color
            output += color.view(3, 1, 1) * alpha.unsqueeze(0)  # [3, H, W]
            alpha_accumulated += alpha.unsqueeze(0)  # [1, H, W]

        return output

    def optimize(
        self,
        target_image: torch.Tensor,
        viewpoint: Dict[str, torch.Tensor],
    ) -> float:
        """
        Optimize Gaussians to match target image.

        Args:
            target_image: Target RGB image [3, H, W]
            viewpoint: Camera viewpoint

        Returns:
            Final loss value
        """
        optimizer = torch.optim.Adam([
            {"params": [self.means], "lr": self.learning_rate},
            {"params": [self.log_scales], "lr": self.learning_rate * 0.5},
            {"params": [self.rotations], "lr": self.learning_rate * 0.1},
            {"params": [self.opacities], "lr": self.learning_rate * 0.05},
            {"params": [self.features], "lr": self.learning_rate * 0.025},
        ])

        for iteration in range(self.optimization_iterations):
            optimizer.zero_grad()

            # Render
            projected = self.project_gaussians(viewpoint)
            rendered = self.rasterize(projected, target_image.shape[1:])

            # Loss
            loss = F.l1_loss(rendered, target_image)

            # Backward
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}, Loss: {loss.item():.6f}")

        return loss.item()

    def extract_features(self) -> torch.Tensor:
        """
        Extract features from Gaussians for downstream tasks.

        Returns:
            Feature tensor [num_gaussians, feature_dim]
        """
        # Combine all Gaussian attributes into features
        features = torch.cat([
            self.means,  # Position [N, 3]
            torch.exp(self.log_scales),  # Scales [N, 3]
            torch.sigmoid(self.features),  # Colors [N, 3]
            torch.sigmoid(self.opacities),  # Opacity [N, 1]
        ], dim=1)  # [N, 10]

        return features
