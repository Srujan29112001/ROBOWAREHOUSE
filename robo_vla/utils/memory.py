"""GPU memory management for RTX 3060 (12GB VRAM)."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Unified memory management for RTX 3060 (12GB VRAM)."""

    def __init__(self, device: str = "cuda:0", reserved_gb: float = 2.0):
        """
        Initialize GPU memory manager.

        Args:
            device: CUDA device string
            reserved_gb: GB to reserve for system operations
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vram_total = self._get_total_memory()
        self.reserved_system = int(reserved_gb * 1024 * 1024 * 1024)
        self.available = self.vram_total - self.reserved_system

        logger.info(f"GPU Memory Manager initialized")
        logger.info(f"Total VRAM: {self.vram_total / 1024**3:.2f} GB")
        logger.info(f"Reserved: {self.reserved_system / 1024**3:.2f} GB")
        logger.info(f"Available: {self.available / 1024**3:.2f} GB")

    def _get_total_memory(self) -> int:
        """Get total GPU memory in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free = self.vram_total - allocated

        return {
            "allocated_gb": allocated / 1024**3,
            "reserved_gb": reserved / 1024**3,
            "free_gb": free / 1024**3,
            "utilization": (allocated / self.vram_total) * 100,
        }

    def optimize_model_loading(
        self,
        models: Dict[str, nn.Module],
        strategies: Optional[Dict[str, str]] = None,
    ) -> Dict[str, nn.Module]:
        """
        Load models with memory optimization.

        Args:
            models: Dictionary of model name to model object
            strategies: Optional dictionary of model name to optimization strategy
                       Options: 'int8', 'int4', 'fp16', 'fp32'

        Returns:
            Dictionary of optimized models
        """
        optimized_models = {}

        for name, model in models.items():
            model_size = self._get_model_size(model)
            strategy = strategies.get(name, "auto") if strategies else "auto"

            logger.info(f"Loading model '{name}' (size: {model_size / 1024**3:.2f} GB)")

            if strategy == "auto":
                strategy = self._auto_select_strategy(model_size)

            logger.info(f"Using optimization strategy: {strategy}")

            if strategy == "int4":
                # 4-bit quantization (requires bitsandbytes)
                optimized_models[name] = self._quantize_int4(model)
            elif strategy == "int8":
                # 8-bit quantization
                optimized_models[name] = self._quantize_int8(model)
            elif strategy == "fp16":
                # Half precision
                optimized_models[name] = model.half().to(self.device)
            else:
                # Full precision
                optimized_models[name] = model.to(self.device)

            # Log memory after loading
            usage = self.get_memory_usage()
            logger.info(f"Memory after loading '{name}': {usage['allocated_gb']:.2f} GB")

        return optimized_models

    def _get_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        return sum(p.numel() * p.element_size() for p in model.parameters())

    def _auto_select_strategy(self, model_size: int) -> str:
        """Automatically select optimization strategy based on model size."""
        size_gb = model_size / 1024**3

        if size_gb > 8:
            return "int4"
        elif size_gb > 4:
            return "int8"
        elif size_gb > 2:
            return "fp16"
        else:
            return "fp32"

    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """Apply INT8 quantization."""
        return quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8
        ).to(self.device)

    def _quantize_int4(self, model: nn.Module) -> nn.Module:
        """Apply INT4 quantization (placeholder - requires bitsandbytes)."""
        logger.warning("INT4 quantization requires bitsandbytes library")
        return model.half().to(self.device)

    def clear_cache(self) -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    def get_optimization_config(self, project_type: str = "robotics") -> Dict:
        """
        Get project-specific optimization configuration.

        Args:
            project_type: Type of project (robotics, healthcare, space)

        Returns:
            Optimization configuration dictionary
        """
        configs = {
            "robotics": {
                "batch_size": 1,
                "precision": "fp16",
                "tensorrt": True,
                "cudnn_benchmark": True,
                "gradient_accumulation_steps": 8,
                "cache_vision_features": True,
                "async_execution": True,
            },
            "healthcare": {
                "batch_size": 4,
                "precision": "mixed",
                "use_amp": True,
                "pin_memory": True,
                "num_workers": 4,
                "prefetch_factor": 2,
                "persistent_workers": True,
            },
            "space": {
                "batch_size": 16,
                "precision": "fp16",
                "gradient_accumulation_steps": 4,
                "compile_model": True,
                "fused_adam": True,
                "channels_last": True,
            },
        }

        return configs.get(project_type, configs["robotics"])

    def profile_model(self, model: nn.Module, input_shape: Tuple) -> Dict:
        """
        Profile model memory and compute requirements.

        Args:
            model: Model to profile
            input_shape: Input tensor shape

        Returns:
            Profiling results
        """
        dummy_input = torch.randn(input_shape).to(self.device)

        # Measure memory before
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)

        # Measure memory after
        mem_after = torch.cuda.memory_allocated()
        mem_peak = torch.cuda.max_memory_allocated()

        return {
            "memory_before_mb": mem_before / 1024**2,
            "memory_after_mb": mem_after / 1024**2,
            "memory_peak_mb": mem_peak / 1024**2,
            "memory_used_mb": (mem_after - mem_before) / 1024**2,
        }
