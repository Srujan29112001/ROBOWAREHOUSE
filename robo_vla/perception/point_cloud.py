"""Point cloud generation from RGB-D images."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

logger = logging.getLogger(__name__)


class PointCloudGenerator:
    """
    Generate 3D point clouds from RGB-D images.

    Uses camera intrinsics to back-project pixels to 3D space.
    """

    def __init__(
        self,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 320.0,
        cy: float = 240.0,
        voxel_size: float = 0.01,
        max_points: int = 50000,
    ):
        """
        Initialize point cloud generator.

        Args:
            fx: Focal length x (pixels)
            fy: Focal length y (pixels)
            cx: Principal point x (pixels)
            cy: Principal point y (pixels)
            voxel_size: Voxel grid size for downsampling (meters)
            max_points: Maximum number of points after downsampling
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.voxel_size = voxel_size
        self.max_points = max_points

        # Create camera intrinsic matrix
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(
            width=640,
            height=480,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        logger.info("Point cloud generator initialized")

    def generate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> o3d.geometry.PointCloud:
        """
        Generate point cloud from RGB-D image.

        Args:
            rgb: RGB image [H, W, 3], uint8
            depth: Depth map [H, W], float (meters)
            boxes: Optional bounding boxes [N, 4] for object segmentation

        Returns:
            Open3D point cloud
        """
        height, width = depth.shape

        # Create Open3D images
        rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # Convert to mm

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            self.intrinsics,
        )

        # Add object IDs if boxes provided
        if boxes is not None:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            object_ids = self._assign_object_ids(points, rgb, depth, boxes)

            # Store object IDs as extra attribute
            pcd.object_ids = object_ids

        # Downsample
        pcd = self._downsample(pcd)

        return pcd

    def generate_batch(
        self,
        rgb_batch: np.ndarray,
        depth_batch: np.ndarray,
        boxes_batch: Optional[list] = None,
    ) -> list:
        """
        Generate point clouds for batch of RGB-D images.

        Args:
            rgb_batch: Batch of RGB images [B, H, W, 3]
            depth_batch: Batch of depth maps [B, H, W]
            boxes_batch: Optional list of boxes per image

        Returns:
            List of point clouds
        """
        point_clouds = []

        for i in range(len(rgb_batch)):
            boxes = boxes_batch[i] if boxes_batch is not None else None
            pcd = self.generate(rgb_batch[i], depth_batch[i], boxes)
            point_clouds.append(pcd)

        return point_clouds

    def _downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Downsample point cloud using voxel grid."""
        # Voxel grid downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Further downsample if still too many points
        if len(pcd_down.points) > self.max_points:
            # Random downsampling
            indices = np.random.choice(
                len(pcd_down.points),
                self.max_points,
                replace=False,
            )
            pcd_down = pcd_down.select_by_index(indices)

        logger.debug(f"Downsampled point cloud: {len(pcd.points)} -> {len(pcd_down.points)} points")

        return pcd_down

    def _assign_object_ids(
        self,
        points: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        boxes: np.ndarray,
    ) -> np.ndarray:
        """Assign object IDs to points based on bounding boxes."""
        height, width = depth.shape
        num_points = len(points)

        # Initialize object IDs (0 = background)
        object_ids = np.zeros(num_points, dtype=np.int32)

        # Project points back to 2D
        x_2d = (points[:, 0] * self.fx / points[:, 2] + self.cx).astype(int)
        y_2d = (points[:, 1] * self.fy / points[:, 2] + self.cy).astype(int)

        # Filter valid projections
        valid = (x_2d >= 0) & (x_2d < width) & (y_2d >= 0) & (y_2d < height)

        # Assign object IDs based on boxes
        for obj_id, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.astype(int)

            # Points inside this box
            inside = (
                valid &
                (x_2d >= x1) & (x_2d < x2) &
                (y_2d >= y1) & (y_2d < y2)
            )

            object_ids[inside] = obj_id

        return object_ids

    def to_tensor(self, pcd: o3d.geometry.PointCloud) -> Dict[str, torch.Tensor]:
        """Convert point cloud to PyTorch tensors."""
        points = torch.from_numpy(np.asarray(pcd.points)).float()
        colors = torch.from_numpy(np.asarray(pcd.colors)).float()

        result = {
            "points": points,
            "colors": colors,
        }

        if hasattr(pcd, "object_ids"):
            result["object_ids"] = torch.from_numpy(pcd.object_ids).long()

        return result

    def visualize(self, pcd: o3d.geometry.PointCloud) -> None:
        """Visualize point cloud using Open3D viewer."""
        o3d.visualization.draw_geometries([pcd])

    def save(self, pcd: o3d.geometry.PointCloud, filename: str) -> None:
        """Save point cloud to file."""
        o3d.io.write_point_cloud(filename, pcd)
        logger.info(f"Point cloud saved to {filename}")

    def load(self, filename: str) -> o3d.geometry.PointCloud:
        """Load point cloud from file."""
        pcd = o3d.io.read_point_cloud(filename)
        logger.info(f"Point cloud loaded from {filename}")
        return pcd
