"""
TensorRT optimization for robot VLA models.

Provides utilities for converting PyTorch models to TensorRT for
optimized inference on NVIDIA GPUs.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import logging

try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available. Install with: pip install torch-tensorrt nvidia-tensorrt")

logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """
    Optimizer for converting PyTorch models to TensorRT.

    Supports various optimization strategies including:
    - FP16/INT8 quantization
    - Dynamic shape optimization
    - Layer fusion
    - Kernel auto-tuning
    """

    def __init__(
        self,
        workspace_size: int = 1 << 30,  # 1GB
        precision: str = 'fp16',  # 'fp32', 'fp16', 'int8'
        max_batch_size: int = 8,
        device: str = 'cuda:0'
    ):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Please install torch-tensorrt.")

        self.workspace_size = workspace_size
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.device = device

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)

    def optimize_model(
        self,
        model: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        output_path: Optional[str] = None,
        calibration_data: Optional[torch.Tensor] = None
    ) -> Union[nn.Module, trt.ICudaEngine]:
        """
        Optimize PyTorch model with TensorRT.

        Args:
            model: PyTorch model to optimize
            input_shapes: Dictionary of input names to shapes
            output_path: Path to save TensorRT engine
            calibration_data: Data for INT8 calibration (if using INT8)

        Returns:
            Optimized model or TensorRT engine
        """
        logger.info(f"Optimizing model with TensorRT ({self.precision} precision)...")

        # Move model to device
        model = model.to(self.device)
        model.eval()

        # Create example inputs
        example_inputs = []
        for name, shape in input_shapes.items():
            # Handle dynamic batch size
            if shape[0] == -1:
                shape = (1,) + shape[1:]
            example_inputs.append(torch.randn(shape).to(self.device))

        # Optimization settings
        if self.precision == 'fp16':
            enabled_precisions = {torch.float16}
        elif self.precision == 'int8':
            if calibration_data is None:
                raise ValueError("INT8 precision requires calibration data")
            enabled_precisions = {torch.int8}
        else:
            enabled_precisions = {torch.float32}

        try:
            # Convert to TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=example_inputs,
                enabled_precisions=enabled_precisions,
                workspace_size=self.workspace_size,
                truncate_long_and_double=True,
                device={
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": int(self.device.split(':')[1]) if ':' in self.device else 0
                }
            )

            logger.info("Model successfully optimized with TensorRT")

            # Save if path provided
            if output_path:
                torch.jit.save(trt_model, output_path)
                logger.info(f"Saved TensorRT model to {output_path}")

            return trt_model

        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            logger.warning("Falling back to PyTorch model")
            return model

    def benchmark_model(
        self,
        model: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model performance.

        Args:
            model: Model to benchmark
            input_shapes: Dictionary of input shapes
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with benchmark results
        """
        import time

        model.eval()
        model = model.to(self.device)

        # Create example inputs
        example_inputs = []
        for name, shape in input_shapes.items():
            if shape[0] == -1:
                shape = (1,) + shape[1:]
            example_inputs.append(torch.randn(shape).to(self.device))

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(*example_inputs)

        # Benchmark
        torch.cuda.synchronize()
        latencies = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = model(*example_inputs)
                torch.cuda.synchronize()
                latency = (time.time() - start) * 1000  # ms
                latencies.append(latency)

        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_fps': 1000.0 / np.mean(latencies)
        }

    def compare_performance(
        self,
        pytorch_model: nn.Module,
        trt_model: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        num_iterations: int = 100
    ) -> None:
        """
        Compare PyTorch and TensorRT model performance.

        Args:
            pytorch_model: Original PyTorch model
            trt_model: TensorRT optimized model
            input_shapes: Dictionary of input shapes
            num_iterations: Number of benchmark iterations
        """
        logger.info("Benchmarking PyTorch model...")
        pytorch_results = self.benchmark_model(pytorch_model, input_shapes, num_iterations)

        logger.info("Benchmarking TensorRT model...")
        trt_results = self.benchmark_model(trt_model, input_shapes, num_iterations)

        logger.info("\nPerformance Comparison:")
        logger.info("=" * 60)
        logger.info(f"{'Metric':<30} {'PyTorch':<15} {'TensorRT':<15} {'Speedup':<10}")
        logger.info("-" * 60)

        speedup_mean = pytorch_results['mean_latency_ms'] / trt_results['mean_latency_ms']
        logger.info(
            f"{'Mean Latency (ms)':<30} "
            f"{pytorch_results['mean_latency_ms']:<15.2f} "
            f"{trt_results['mean_latency_ms']:<15.2f} "
            f"{speedup_mean:<10.2f}x"
        )

        speedup_p95 = pytorch_results['p95_latency_ms'] / trt_results['p95_latency_ms']
        logger.info(
            f"{'P95 Latency (ms)':<30} "
            f"{pytorch_results['p95_latency_ms']:<15.2f} "
            f"{trt_results['p95_latency_ms']:<15.2f} "
            f"{speedup_p95:<10.2f}x"
        )

        logger.info(
            f"{'Throughput (FPS)':<30} "
            f"{pytorch_results['throughput_fps']:<15.2f} "
            f"{trt_results['throughput_fps']:<15.2f} "
            f"{trt_results['throughput_fps'] / pytorch_results['throughput_fps']:<10.2f}x"
        )
        logger.info("=" * 60)


class ONNXExporter:
    """
    Utility for exporting PyTorch models to ONNX format.

    ONNX is an intermediate format that can be further optimized
    or deployed on various platforms.
    """

    def __init__(
        self,
        opset_version: int = 14,
        dynamic_axes: Optional[Dict] = None
    ):
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes or {}

    def export_model(
        self,
        model: nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        output_path: str,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ) -> None:
        """
        Export PyTorch model to ONNX.

        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing
            output_path: Path to save ONNX model
            input_names: Names of input tensors
            output_names: Names of output tensors
        """
        logger.info(f"Exporting model to ONNX: {output_path}")

        model.eval()

        # Default names
        if input_names is None:
            input_names = [f"input_{i}" for i in range(len(example_inputs))]
        if output_names is None:
            output_names = ["output"]

        # Export
        torch.onnx.export(
            model,
            example_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
            verbose=False
        )

        logger.info(f"Model exported to ONNX: {output_path}")

    def verify_onnx(self, onnx_path: str) -> bool:
        """
        Verify ONNX model.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            True if model is valid, False otherwise
        """
        try:
            import onnx
            from onnx import checker

            # Load model
            model = onnx.load(onnx_path)

            # Check model
            checker.check_model(model)

            logger.info(f"ONNX model is valid: {onnx_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False


class ModelQuantizer:
    """
    Quantization utilities for model compression.

    Supports dynamic, static, and QAT (Quantization-Aware Training) quantization.
    """

    def __init__(self, backend: str = 'qnnpack'):
        self.backend = backend
        torch.backends.quantized.engine = backend

    def dynamic_quantize(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.

        Args:
            model: PyTorch model
            dtype: Quantization dtype

        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d, nn.LSTM},
            dtype=dtype
        )

        logger.info("Dynamic quantization complete")
        return quantized_model

    def static_quantize(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader
    ) -> nn.Module:
        """
        Apply static quantization with calibration.

        Args:
            model: PyTorch model
            calibration_loader: DataLoader for calibration

        Returns:
            Quantized model
        """
        logger.info("Applying static quantization...")

        model.eval()

        # Specify quantization config
        model.qconfig = torch.quantization.get_default_qconfig(self.backend)

        # Fuse modules
        model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

        # Prepare for quantization
        model_prepared = torch.quantization.prepare(model)

        # Calibrate
        logger.info("Calibrating model...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                if batch_idx >= 100:  # Use first 100 batches
                    break
                _ = model_prepared(*batch)

        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)

        logger.info("Static quantization complete")
        return model_quantized

    def measure_model_size(self, model: nn.Module) -> float:
        """
        Measure model size in MB.

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        temp_path = Path("/tmp/temp_model.pth")
        torch.save(model.state_dict(), temp_path)
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink()

        return size_mb


if __name__ == '__main__':
    # Example usage
    import sys

    if not TENSORRT_AVAILABLE:
        print("TensorRT not available. Install with:")
        print("  pip install torch-tensorrt nvidia-tensorrt")
        sys.exit(1)

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 120 * 160, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Initialize optimizer
    optimizer = TensorRTOptimizer(precision='fp16', device='cuda:0')

    # Create model
    model = SimpleModel()

    # Define input shapes
    input_shapes = {
        'input': (1, 3, 480, 640)
    }

    # Optimize model
    trt_model = optimizer.optimize_model(
        model,
        input_shapes,
        output_path='model_trt.pth'
    )

    # Compare performance
    optimizer.compare_performance(model, trt_model, input_shapes)
