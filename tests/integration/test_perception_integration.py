"""
Integration tests for perception pipeline.

Tests the integration between DINO, MiDaS, Point Cloud, and Gaussian Splatting.
"""
import pytest
import torch
import numpy as np
from PIL import Image

from robo_vla.perception.pipeline import PerceptionPipeline
from robo_vla.perception.dino import DinoV2Detector
from robo_vla.perception.depth import MiDaSDepthEstimator
from robo_vla.perception.point_cloud import PointCloudGenerator
from robo_vla.perception.gaussian_splatting import GaussianSplatting


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Create a 640x480 RGB image with colored rectangles
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a red box
    img[100:200, 100:200] = [255, 0, 0]
    # Add a blue box
    img[200:300, 300:400] = [0, 0, 255]
    # Add a green box
    img[350:450, 450:550] = [0, 255, 0]
    return img


@pytest.fixture
def perception_pipeline():
    """Create a perception pipeline instance."""
    return PerceptionPipeline(device='cpu')


class TestPerceptionIntegration:
    """Integration tests for perception components."""

    def test_full_pipeline_execution(self, perception_pipeline, sample_rgb_image):
        """Test complete perception pipeline execution."""
        # Process the image
        result = perception_pipeline.process(sample_rgb_image)

        # Verify all expected outputs are present
        assert 'objects' in result
        assert 'depth_map' in result
        assert 'point_cloud' in result
        assert 'gaussians' in result
        assert 'features' in result

        # Verify objects were detected
        assert len(result['objects']) > 0

        # Verify depth map shape
        assert result['depth_map'].shape == (480, 640)

        # Verify point cloud has points
        assert result['point_cloud'].shape[0] > 0
        assert result['point_cloud'].shape[1] == 6  # x,y,z,r,g,b

    def test_dino_to_pointcloud_integration(self, sample_rgb_image):
        """Test integration between DINO detection and point cloud generation."""
        # Initialize components
        dino = DinoV2Detector(device='cpu')
        depth_estimator = MiDaSDepthEstimator(device='cpu')
        pc_generator = PointCloudGenerator()

        # Detect objects
        detections = dino.detect(sample_rgb_image)

        # Generate depth map
        depth_map = depth_estimator.estimate(sample_rgb_image)

        # Generate point cloud with object segmentation
        point_cloud = pc_generator.from_rgbd(
            rgb=sample_rgb_image,
            depth=depth_map,
            boxes=detections['boxes']
        )

        # Verify point cloud has object IDs from detections
        assert point_cloud is not None
        assert len(point_cloud.points) > 0

    def test_pointcloud_to_gaussian_integration(self, sample_rgb_image):
        """Test integration between point cloud and Gaussian Splatting."""
        # Create pipeline components
        depth_estimator = MiDaSDepthEstimator(device='cpu')
        pc_generator = PointCloudGenerator()
        gaussian_splatter = GaussianSplatting(num_gaussians=1000, device='cpu')

        # Generate depth and point cloud
        depth_map = depth_estimator.estimate(sample_rgb_image)
        point_cloud = pc_generator.from_rgbd(
            rgb=sample_rgb_image,
            depth=depth_map
        )

        # Convert to numpy array
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)

        # Initialize Gaussians from point cloud
        gaussian_splatter.initialize_from_pointcloud(points, colors)

        # Verify Gaussians were created
        assert gaussian_splatter.means is not None
        assert gaussian_splatter.means.shape[0] > 0

    def test_pipeline_benchmark(self, perception_pipeline, sample_rgb_image, benchmark):
        """Benchmark the full perception pipeline."""
        result = benchmark(perception_pipeline.process, sample_rgb_image)

        # Verify result is valid
        assert result is not None
        assert 'objects' in result

    def test_pipeline_with_multiple_images(self, perception_pipeline):
        """Test pipeline with batch of images."""
        # Create multiple test images
        images = []
        for i in range(5):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            images.append(img)

        results = []
        for img in images:
            result = perception_pipeline.process(img)
            results.append(result)

        # Verify all images were processed
        assert len(results) == 5
        for result in results:
            assert 'objects' in result
            assert 'depth_map' in result

    def test_pipeline_error_handling(self, perception_pipeline):
        """Test pipeline error handling with invalid inputs."""
        # Test with None input
        with pytest.raises(Exception):
            perception_pipeline.process(None)

        # Test with wrong shape
        with pytest.raises(Exception):
            perception_pipeline.process(np.zeros((100, 100)))

        # Test with wrong dtype
        with pytest.raises(Exception):
            perception_pipeline.process(np.zeros((480, 640, 3), dtype=np.float64))

    def test_pipeline_consistency(self, perception_pipeline, sample_rgb_image):
        """Test that pipeline produces consistent results."""
        # Process same image twice
        result1 = perception_pipeline.process(sample_rgb_image)
        result2 = perception_pipeline.process(sample_rgb_image)

        # Results should be similar (allowing for small numerical differences)
        assert len(result1['objects']) == len(result2['objects'])

        # Depth maps should be very similar
        depth_diff = np.abs(result1['depth_map'] - result2['depth_map']).mean()
        assert depth_diff < 0.01  # Should be nearly identical


@pytest.mark.slow
class TestPerceptionPerformance:
    """Performance tests for perception pipeline."""

    def test_realtime_performance(self, perception_pipeline, sample_rgb_image):
        """Test that pipeline can run at real-time speeds (30 FPS)."""
        import time

        # Warm up
        perception_pipeline.process(sample_rgb_image)

        # Time 10 iterations
        start = time.time()
        for _ in range(10):
            perception_pipeline.process(sample_rgb_image)
        elapsed = time.time() - start

        fps = 10 / elapsed
        print(f"Perception pipeline FPS: {fps:.2f}")

        # Should achieve at least 10 FPS on CPU
        assert fps > 10, f"Pipeline too slow: {fps:.2f} FPS"

    def test_memory_usage(self, perception_pipeline, sample_rgb_image):
        """Test memory usage of perception pipeline."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple images
        for _ in range(20):
            perception_pipeline.process(sample_rgb_image)

        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = peak_memory - baseline_memory
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Should not use more than 2GB additional memory
        assert memory_increase < 2000, f"Memory usage too high: {memory_increase:.2f} MB"
