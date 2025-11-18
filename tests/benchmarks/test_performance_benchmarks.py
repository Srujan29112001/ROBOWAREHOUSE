"""
Performance benchmarks for all components.

Uses pytest-benchmark to measure and track performance metrics.
"""
import pytest
import torch
import numpy as np
from PIL import Image

from robo_vla.perception.dino import DinoV2Detector
from robo_vla.perception.depth import MiDaSDepthEstimator
from robo_vla.perception.gaussian_splatting import GaussianSplatting
from robo_vla.perception.pipeline import PerceptionPipeline
from robo_vla.language.llama import LlamaQLoRA
from robo_vla.language.grounding import LanguageGroundingModule
from robo_vla.vla_model.vla import VLAModel


@pytest.fixture
def sample_image():
    """Create sample image for benchmarking."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_text():
    """Sample text command."""
    return "Pick the red box from the top shelf"


class TestPerceptionBenchmarks:
    """Benchmarks for perception components."""

    def test_dino_detection_speed(self, benchmark, sample_image):
        """Benchmark DINO v2 object detection."""
        detector = DinoV2Detector(device='cpu')

        def detect():
            return detector.detect(sample_image)

        result = benchmark(detect)
        assert 'boxes' in result

    def test_midas_depth_estimation_speed(self, benchmark, sample_image):
        """Benchmark MiDaS depth estimation."""
        estimator = MiDaSDepthEstimator(device='cpu')

        def estimate():
            return estimator.estimate(sample_image)

        result = benchmark(estimate)
        assert result.shape == (480, 640)

    def test_gaussian_splatting_speed(self, benchmark):
        """Benchmark 3D Gaussian Splatting."""
        splatter = GaussianSplatting(num_gaussians=1000, device='cpu')

        # Initialize with random points
        points = np.random.randn(1000, 3)
        colors = np.random.rand(1000, 3)
        splatter.initialize_from_pointcloud(points, colors)

        def render():
            return splatter.render(viewpoint_camera=None)

        benchmark(render)

    def test_full_perception_pipeline_speed(self, benchmark, sample_image):
        """Benchmark complete perception pipeline."""
        pipeline = PerceptionPipeline(device='cpu')

        def process():
            return pipeline.process(sample_image)

        result = benchmark(process)
        assert 'objects' in result


class TestLanguageBenchmarks:
    """Benchmarks for language components."""

    def test_llama_inference_speed(self, benchmark, sample_text):
        """Benchmark Llama 3.1 inference."""
        model = LlamaQLoRA(device='cpu')

        def process():
            return model.process_command(sample_text)

        result = benchmark(process)
        assert 'intent' in result or 'embeddings' in result

    def test_language_grounding_speed(self, benchmark, sample_text):
        """Benchmark language grounding module."""
        grounding = LanguageGroundingModule(device='cpu')

        def process():
            return grounding.process(sample_text)

        result = benchmark(process)
        assert 'embeddings' in result


class TestVLABenchmarks:
    """Benchmarks for VLA model."""

    def test_vla_inference_speed(self, benchmark, sample_image, sample_text):
        """Benchmark complete VLA model inference."""
        model = VLAModel(device='cpu')
        robot_state = np.zeros(8)

        def predict():
            return model.predict(sample_image, sample_text, robot_state)

        result = benchmark(predict)
        assert 'joint_commands' in result

    def test_vla_batch_processing_speed(self, benchmark):
        """Benchmark VLA batch processing."""
        model = VLAModel(device='cpu')

        # Create batch
        batch_size = 4
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
        commands = ["Pick object" for _ in range(batch_size)]
        states = [np.zeros(8) for _ in range(batch_size)]

        def process_batch():
            results = []
            for img, cmd, state in zip(images, commands, states):
                result = model.predict(img, cmd, state)
                results.append(result)
            return results

        results = benchmark(process_batch)
        assert len(results) == batch_size


class TestMemoryBenchmarks:
    """Benchmarks for memory usage."""

    def test_perception_memory_footprint(self, sample_image):
        """Measure perception pipeline memory footprint."""
        import tracemalloc

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # Initialize pipeline
        pipeline = PerceptionPipeline(device='cpu')

        # Process image
        pipeline.process(sample_image)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_used_mb = (peak - baseline) / 1024 / 1024
        print(f"Perception memory: {memory_used_mb:.2f} MB")

        # Should not exceed 2GB
        assert memory_used_mb < 2000

    def test_vla_model_memory_footprint(self, sample_image, sample_text):
        """Measure VLA model memory footprint."""
        import tracemalloc

        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]

        # Initialize model
        model = VLAModel(device='cpu')

        # Run inference
        model.predict(sample_image, sample_text, np.zeros(8))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_used_mb = (peak - baseline) / 1024 / 1024
        print(f"VLA model memory: {memory_used_mb:.2f} MB")

        # Should not exceed 10GB
        assert memory_used_mb < 10000


class TestThroughputBenchmarks:
    """Benchmarks for system throughput."""

    def test_perception_throughput(self, sample_image):
        """Measure perception pipeline throughput."""
        import time

        pipeline = PerceptionPipeline(device='cpu')

        # Warm up
        for _ in range(5):
            pipeline.process(sample_image)

        # Measure throughput
        num_iterations = 50
        start = time.time()

        for _ in range(num_iterations):
            pipeline.process(sample_image)

        elapsed = time.time() - start
        throughput = num_iterations / elapsed

        print(f"Perception throughput: {throughput:.2f} FPS")

        # Should achieve at least 10 FPS on CPU
        assert throughput > 10

    def test_vla_throughput(self, sample_image, sample_text):
        """Measure VLA model throughput."""
        import time

        model = VLAModel(device='cpu')
        robot_state = np.zeros(8)

        # Warm up
        for _ in range(5):
            model.predict(sample_image, sample_text, robot_state)

        # Measure throughput
        num_iterations = 30
        start = time.time()

        for _ in range(num_iterations):
            model.predict(sample_image, sample_text, robot_state)

        elapsed = time.time() - start
        throughput = num_iterations / elapsed

        print(f"VLA throughput: {throughput:.2f} inferences/sec")

        # Should achieve reasonable throughput
        assert throughput > 1


class TestLatencyBenchmarks:
    """Benchmarks for latency measurements."""

    def test_perception_latency_p50_p95_p99(self, sample_image):
        """Measure perception pipeline latency percentiles."""
        import time

        pipeline = PerceptionPipeline(device='cpu')

        # Warm up
        for _ in range(10):
            pipeline.process(sample_image)

        # Collect latencies
        latencies = []
        for _ in range(100):
            start = time.time()
            pipeline.process(sample_image)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print(f"Perception latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")

        # Verify latency targets
        assert p50 < 100  # 50th percentile < 100ms
        assert p95 < 200  # 95th percentile < 200ms
        assert p99 < 300  # 99th percentile < 300ms

    def test_vla_latency_p50_p95_p99(self, sample_image, sample_text):
        """Measure VLA model latency percentiles."""
        import time

        model = VLAModel(device='cpu')
        robot_state = np.zeros(8)

        # Warm up
        for _ in range(10):
            model.predict(sample_image, sample_text, robot_state)

        # Collect latencies
        latencies = []
        for _ in range(100):
            start = time.time()
            model.predict(sample_image, sample_text, robot_state)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print(f"VLA latency - P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")

        # On CPU, expect higher latencies
        assert p50 < 500
        assert p95 < 1000
        assert p99 < 1500


@pytest.mark.gpu
class TestGPUBenchmarks:
    """Benchmarks specifically for GPU performance."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_perception_gpu_speed(self, benchmark, sample_image):
        """Benchmark perception pipeline on GPU."""
        pipeline = PerceptionPipeline(device='cuda')

        def process():
            return pipeline.process(sample_image)

        result = benchmark(process)
        assert 'objects' in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_vla_gpu_speed(self, benchmark, sample_image, sample_text):
        """Benchmark VLA model on GPU."""
        model = VLAModel(device='cuda')
        robot_state = np.zeros(8)

        def predict():
            return model.predict(sample_image, sample_text, robot_state)

        result = benchmark(predict)
        assert 'joint_commands' in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_usage(self, sample_image, sample_text):
        """Measure GPU memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        # Initialize model
        model = VLAModel(device='cuda')

        # Run inference
        model.predict(sample_image, sample_text, np.zeros(8))

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_gb = (peak_memory - baseline) / 1024**3

        print(f"GPU memory used: {memory_used_gb:.2f} GB")

        # Should fit in RTX 3060 (12GB)
        assert memory_used_gb < 12


class TestScalabilityBenchmarks:
    """Benchmarks for scalability."""

    def test_concurrent_requests_scalability(self, sample_image, sample_text):
        """Test system performance with concurrent requests."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        model = VLAModel(device='cpu')
        robot_state = np.zeros(8)

        def single_request():
            return model.predict(sample_image, sample_text, robot_state)

        # Test with different concurrency levels
        for num_workers in [1, 2, 4, 8]:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                start = time.time()
                futures = [executor.submit(single_request) for _ in range(20)]
                results = [f.result() for f in futures]
                elapsed = time.time() - start

                throughput = len(results) / elapsed
                print(f"Concurrency {num_workers}: {throughput:.2f} req/s")

                assert len(results) == 20

    def test_batch_size_impact(self):
        """Test impact of batch size on performance."""
        import time

        model = VLAModel(device='cpu')

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
            commands = ["Pick object" for _ in range(batch_size)]
            states = [np.zeros(8) for _ in range(batch_size)]

            start = time.time()
            for img, cmd, state in zip(images, commands, states):
                model.predict(img, cmd, state)
            elapsed = time.time() - start

            throughput = batch_size / elapsed
            print(f"Batch size {batch_size}: {throughput:.2f} inferences/sec")
