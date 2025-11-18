"""Unified perception pipeline integrating all vision components."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from robo_vla.perception.dino import DinoV2Detector
from robo_vla.perception.depth import MiDasDepthEstimator
from robo_vla.perception.point_cloud import PointCloudGenerator
from robo_vla.perception.gaussian_splatting import GaussianSplatting

logger = logging.getLogger(__name__)


class PerceptionPipeline(nn.Module):
    """
    Complete perception pipeline for robotic vision.

    Integrates:
    - DINO v2 for object detection
    - MiDaS for depth estimation
    - Point cloud generation
    - 3D Gaussian Splatting for scene representation
    """

    def __init__(
        self,
        config: Dict,
        device: str = "cuda",
    ):
        """
        Initialize perception pipeline.

        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        super().__init__()
        self.device = device
        self.config = config

        logger.info("Initializing Perception Pipeline...")

        # Initialize components
        self.dino = DinoV2Detector(
            model_name=config["perception"]["dino"]["model_name"],
            feature_dim=config["perception"]["dino"]["feature_dim"],
            device=device,
            quantization=config["perception"]["dino"]["quantization"],
        )

        self.depth_estimator = MiDasDepthEstimator(
            model_type=config["perception"]["midas"]["model_type"],
            device=device,
            optimize=config["perception"]["midas"]["optimize"],
            fp16=config["perception"]["midas"]["fp16"],
            max_depth=config["perception"]["midas"]["max_depth"],
            min_depth=config["perception"]["midas"]["min_depth"],
        )

        self.point_cloud_generator = PointCloudGenerator(
            fx=config["perception"]["point_cloud"]["camera_intrinsics"]["fx"],
            fy=config["perception"]["point_cloud"]["camera_intrinsics"]["fy"],
            cx=config["perception"]["point_cloud"]["camera_intrinsics"]["cx"],
            cy=config["perception"]["point_cloud"]["camera_intrinsics"]["cy"],
            voxel_size=config["perception"]["point_cloud"]["voxel_size"],
            max_points=config["perception"]["point_cloud"]["max_points"],
        )

        self.gaussian_splatting = GaussianSplatting(
            num_gaussians=config["perception"]["gaussian_splatting"]["num_gaussians"],
            optimization_iterations=config["perception"]["gaussian_splatting"]["optimization_iterations"],
            learning_rate=config["perception"]["gaussian_splatting"]["learning_rate"],
            device=device,
        )

        logger.info("Perception Pipeline initialized successfully")

    @torch.no_grad()
    def process(
        self,
        rgb_images: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, any]:
        """
        Process RGB images through full perception pipeline.

        Args:
            rgb_images: Input RGB images [B, 3, H, W], normalized [0, 1]
            return_intermediate: Whether to return intermediate results

        Returns:
            Dictionary containing:
                - detections: List of detections per image
                - depth_maps: Depth maps [B, 1, H, W]
                - point_clouds: List of point clouds
                - scene_features: Scene features from Gaussian Splatting
                - (optional) intermediate results
        """
        batch_size = rgb_images.shape[0]

        logger.debug(f"Processing batch of {batch_size} images")

        # Step 1: Object Detection with DINO v2
        logger.debug("Running DINO v2 object detection...")
        dino_outputs = self.dino.forward(rgb_images, return_attention=True)
        detections = self.dino.detect_objects(rgb_images, threshold=0.5)

        # Step 2: Depth Estimation with MiDaS
        logger.debug("Running MiDaS depth estimation...")
        depth_maps = self.depth_estimator.forward(rgb_images)

        # Step 3: Generate Point Clouds
        logger.debug("Generating point clouds...")
        point_clouds = []

        for i in range(batch_size):
            # Convert to numpy
            rgb_np = (rgb_images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            depth_np = depth_maps[i, 0].cpu().numpy()
            boxes = detections[i]["boxes"]

            # Generate point cloud
            pcd = self.point_cloud_generator.generate(rgb_np, depth_np, boxes)
            point_clouds.append(pcd)

        # Step 4: 3D Gaussian Splatting (for first image as example)
        logger.debug("Building 3D scene representation...")
        pcd_tensors = [self.point_cloud_generator.to_tensor(pcd) for pcd in point_clouds]

        # Initialize Gaussians from first point cloud
        if len(pcd_tensors) > 0:
            self.gaussian_splatting.initialize_from_point_cloud(
                pcd_tensors[0]["points"],
                pcd_tensors[0]["colors"],
            )
            scene_features = self.gaussian_splatting.extract_features()
        else:
            scene_features = None

        # Prepare output
        output = {
            "detections": detections,
            "depth_maps": depth_maps,
            "point_clouds": point_clouds,
            "scene_features": scene_features,
        }

        if return_intermediate:
            output["intermediate"] = {
                "dino_features": dino_outputs["features"],
                "dino_patch_features": dino_outputs["patch_features"],
                "dino_attention": dino_outputs.get("attention_maps"),
                "point_cloud_tensors": pcd_tensors,
            }

        logger.debug("Perception pipeline completed")

        return output

    def process_numpy(
        self,
        rgb_images_np: np.ndarray,
    ) -> Dict:
        """
        Process numpy RGB images.

        Args:
            rgb_images_np: RGB images [B, H, W, 3], uint8

        Returns:
            Processing results
        """
        # Convert to tensor
        rgb_tensor = torch.from_numpy(rgb_images_np).permute(0, 3, 1, 2).float() / 255.0
        rgb_tensor = rgb_tensor.to(self.device)

        # Process
        return self.process(rgb_tensor)

    def get_scene_representation(
        self,
        rgb_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get compact scene representation for VLA model.

        Args:
            rgb_images: Input RGB images [B, 3, H, W]

        Returns:
            Scene features [B, feature_dim]
        """
        outputs = self.process(rgb_images, return_intermediate=True)

        # Combine features from different modalities
        dino_features = outputs["intermediate"]["dino_features"]  # [B, 768]

        # Average features from point clouds
        scene_features_list = []
        for pcd_tensor in outputs["intermediate"]["point_cloud_tensors"]:
            # Get spatial features from point cloud
            points = pcd_tensor["points"]  # [N, 3]
            colors = pcd_tensor["colors"]  # [N, 3]

            # Simple pooling
            spatial_feat = points.mean(dim=0)  # [3]
            color_feat = colors.mean(dim=0)  # [3]

            scene_feat = torch.cat([spatial_feat, color_feat])  # [6]
            scene_features_list.append(scene_feat)

        scene_features_batch = torch.stack(scene_features_list)  # [B, 6]

        # Combine DINO features with scene features
        combined_features = torch.cat([
            dino_features,
            scene_features_batch.to(dino_features.device),
        ], dim=1)  # [B, 774]

        return combined_features

    def visualize_results(
        self,
        rgb_image: np.ndarray,
        output: Dict,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize perception results.

        Args:
            rgb_image: Original RGB image [H, W, 3]
            output: Output from process()
            save_path: Optional path to save visualization

        Returns:
            Visualization image
        """
        import cv2

        # Get first detection
        detection = output["detections"][0]
        depth_map = output["depth_maps"][0, 0].cpu().numpy()

        # Draw bounding boxes
        vis_image = rgb_image.copy()

        for box, score in zip(detection["boxes"], detection["scores"]):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_image,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Visualize depth
        depth_viz = self.depth_estimator.visualize_depth(depth_map)

        # Combine visualizations
        vis_combined = np.hstack([vis_image, depth_viz])

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_combined, cv2.COLOR_RGB2BGR))
            logger.info(f"Visualization saved to {save_path}")

        return vis_combined

    def benchmark(
        self,
        num_iterations: int = 100,
        batch_size: int = 1,
        image_size: Tuple[int, int] = (640, 480),
    ) -> Dict[str, float]:
        """
        Benchmark perception pipeline performance.

        Args:
            num_iterations: Number of iterations to run
            batch_size: Batch size
            image_size: Input image size

        Returns:
            Performance metrics
        """
        import time

        logger.info(f"Benchmarking with {num_iterations} iterations...")

        # Create dummy input
        dummy_input = torch.randn(
            batch_size, 3, image_size[0], image_size[1]
        ).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.process(dummy_input)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            _ = self.process(dummy_input)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        fps = 1.0 / avg_time

        metrics = {
            "total_time_s": total_time,
            "avg_time_ms": avg_time * 1000,
            "fps": fps,
            "throughput": fps * batch_size,
        }

        logger.info(f"Benchmark results: {metrics}")

        return metrics
