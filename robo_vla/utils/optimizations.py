"""Project-specific optimization strategies for different use cases."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ProjectOptimizations:
    """
    Project-specific optimization strategies.

    Provides optimized configurations for:
    - Robotics: Real-time inference priority
    - Healthcare: Streaming data priority
    - Space: Batch processing priority
    """

    @staticmethod
    def robotics_optimization() -> Dict[str, Any]:
        """
        Robotics: Real-time inference priority.

        Optimized for:
        - Low latency (< 100ms)
        - Single-image processing
        - Real-time robot control
        - Vision feature caching

        Returns:
            Optimization configuration dictionary
        """
        return {
            'batch_size': 1,  # Real-time single-image processing
            'precision': 'fp16',
            'tensorrt': True,
            'cudnn_benchmark': True,
            'gradient_accumulation_steps': 8,
            'cache_vision_features': True,
            'async_execution': True,

            # Additional robotics-specific settings
            'use_jit': True,  # TorchScript JIT compilation
            'optimize_for_inference': True,
            'enable_fusion': True,  # Operator fusion
            'memory_efficient': True,
            'use_channels_last': True,  # Memory layout optimization

            # Real-time constraints
            'max_latency_ms': 100,
            'priority': 'latency',  # latency over throughput
            'prefetch_data': False,  # Don't prefetch for single samples

            # Device settings
            'pin_memory': True,
            'non_blocking': True,
            'num_workers': 0,  # No multiprocessing for real-time
        }

    @staticmethod
    def healthcare_optimization() -> Dict[str, Any]:
        """
        Healthcare: Streaming data priority.

        Optimized for:
        - Continuous data streams
        - Multiple concurrent patients
        - Data persistence
        - Quality over speed

        Returns:
            Optimization configuration dictionary
        """
        return {
            'batch_size': 4,  # Small batch for streaming
            'precision': 'mixed',
            'use_amp': True,  # Automatic Mixed Precision
            'pin_memory': True,
            'num_workers': 4,
            'prefetch_factor': 2,
            'persistent_workers': True,

            # Healthcare-specific settings
            'quality_priority': True,
            'data_validation': True,
            'enable_checksum': True,
            'redundant_processing': True,  # Process critical data twice
            'audit_logging': True,

            # Streaming settings
            'buffer_size': 64,
            'stream_timeout_ms': 5000,
            'max_queue_size': 128,
            'enable_backpressure': True,

            # Data handling
            'save_intermediate': True,
            'compression': 'lossless',
            'encryption': True,

            # Performance
            'optimize_memory': True,
            'gradient_checkpointing': False,  # Prefer speed
            'compile_mode': 'default',
        }

    @staticmethod
    def space_optimization() -> Dict[str, Any]:
        """
        Space: Batch processing priority.

        Optimized for:
        - Large satellite image batches
        - Maximum throughput
        - Offline processing
        - Memory efficiency

        Returns:
            Optimization configuration dictionary
        """
        return {
            'batch_size': 16,  # Larger batch for throughput
            'precision': 'fp16',
            'gradient_accumulation_steps': 4,
            'compile_model': True,  # torch.compile for speed
            'fused_adam': True,
            'channels_last': True,  # Memory format optimization

            # Space-specific settings
            'priority': 'throughput',
            'large_image_support': True,
            'tile_processing': True,
            'distributed_inference': True,

            # Batch processing
            'prefetch_factor': 4,
            'num_workers': 8,
            'persistent_workers': True,
            'pin_memory': True,

            # Memory optimization
            'gradient_checkpointing': True,
            'cpu_offload': True,  # Offload to CPU when needed
            'activation_checkpointing': True,
            'memory_efficient_attention': True,

            # Compilation
            'torch_compile': True,
            'compile_mode': 'max-autotune',
            'dynamic_shapes': False,  # Static shapes for speed

            # I/O optimization
            'async_io': True,
            'io_threads': 4,
            'read_ahead_mb': 512,
        }

    @staticmethod
    def get_optimization(project_type: str = "robotics") -> Dict[str, Any]:
        """
        Get optimization configuration for specific project type.

        Args:
            project_type: One of 'robotics', 'healthcare', 'space'

        Returns:
            Optimization configuration dictionary

        Raises:
            ValueError: If project_type is invalid
        """
        optimizations = {
            'robotics': ProjectOptimizations.robotics_optimization,
            'healthcare': ProjectOptimizations.healthcare_optimization,
            'space': ProjectOptimizations.space_optimization,
        }

        if project_type not in optimizations:
            raise ValueError(
                f"Invalid project_type: {project_type}. "
                f"Must be one of {list(optimizations.keys())}"
            )

        config = optimizations[project_type]()
        logger.info(f"Loaded {project_type} optimization configuration")
        return config

    @staticmethod
    def compare_optimizations() -> Dict[str, Dict[str, Any]]:
        """
        Compare all optimization strategies.

        Returns:
            Dictionary mapping project type to configuration
        """
        return {
            'robotics': ProjectOptimizations.robotics_optimization(),
            'healthcare': ProjectOptimizations.healthcare_optimization(),
            'space': ProjectOptimizations.space_optimization(),
        }


def training_config_for_rtx3060(project_type: str = "robotics") -> Dict[str, Any]:
    """
    Optimized training configuration for RTX 3060 (12GB VRAM).

    Args:
        project_type: Type of project (robotics, healthcare, space)

    Returns:
        Complete training configuration optimized for RTX 3060
    """
    # Base configuration (common across projects)
    base_config = {
        # Memory optimization
        'gradient_checkpointing': True,
        'gradient_accumulation_steps': 8,
        'mixed_precision': {
            'enabled': True,
            'opt_level': 'O2',  # Apex amp O2 level
            'loss_scale': 'dynamic',
        },

        # Speed optimization
        'compile_model': True,  # PyTorch 2.0 compile
        'cudnn_benchmark': True,
        'pin_memory': True,
        'num_workers': 4,

        # Efficient optimizers
        'optimizer': {
            'type': '8bit_adam',  # bitsandbytes 8-bit Adam
            'lr': 1e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
        },

        # Learning rate schedule
        'scheduler': {
            'type': 'cosine',
            'warmup_steps': 1000,
            'num_cycles': 0.5,
            'min_lr': 1e-6,
        },

        # Data loading
        'dataloader': {
            'batch_size': 4,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'drop_last': True,
        },

        # Checkpointing
        'checkpoint': {
            'save_every_n_steps': 1000,
            'keep_last_n': 3,
            'save_optimizer_state': True,
        },

        # Logging
        'logging': {
            'log_every_n_steps': 100,
            'use_wandb': True,
            'use_tensorboard': True,
        },

        # Validation
        'validation': {
            'val_every_n_steps': 500,
            'val_batch_size': 8,
        },

        # GPU-specific
        'gpu': {
            'device': 'cuda:0',
            'max_memory_gb': 10.0,  # Leave 2GB for system
            'empty_cache_every_n_steps': 100,
        },
    }

    # Project-specific overrides
    project_configs = {
        'robotics': {
            'dataloader': {
                'batch_size': 1,  # Real-time training
                'num_workers': 2,
            },
            'gradient_accumulation_steps': 16,  # Larger accumulation
            'training_mode': 'online',  # Online learning
        },
        'healthcare': {
            'dataloader': {
                'batch_size': 4,
                'num_workers': 4,
            },
            'validation': {
                'val_every_n_steps': 100,  # More frequent validation
                'cross_validation': True,
            },
            'quality_checks': True,
        },
        'space': {
            'dataloader': {
                'batch_size': 8,  # Larger batch
                'num_workers': 8,
            },
            'gradient_accumulation_steps': 4,
            'distributed': {
                'enabled': True,
                'backend': 'nccl',
            },
        },
    }

    # Merge configurations
    config = base_config.copy()
    if project_type in project_configs:
        config.update(project_configs[project_type])

    return config
