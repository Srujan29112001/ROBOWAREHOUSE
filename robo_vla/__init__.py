"""
RoboVLA: Multimodal Robotic Vision-Language-Action System

A production-grade system for warehouse automation combining:
- Vision: DINO v2, MiDaS depth estimation, 3D Gaussian Splatting
- Language: Llama 3.1 8B with QLoRA, CLIP alignment
- Knowledge: Neo4j GraphRAG, ChromaDB vector database
- Action: Transformer-based action generation with SAC reinforcement learning
"""

__version__ = "1.0.0"
__author__ = "RoboVLA Team"

from robo_vla.perception import (
    DinoV2Detector,
    MiDasDepthEstimator,
    PointCloudGenerator,
    GaussianSplatting,
    PerceptionPipeline,
)

from robo_vla.language import (
    LlamaLanguageModel,
    CLIPAlignment,
    LanguageGroundingModule,
)

from robo_vla.knowledge import (
    Neo4jGraphDB,
    ChromaDBVectorStore,
    GraphRAGSystem,
)

from robo_vla.vla_model import (
    CrossAttentionFusion,
    ActionDecoder,
    VLAModel,
)

from robo_vla.rl import (
    SACAgent,
    ReplayBuffer,
    RobotEnvironment,
)

from robo_vla.server import (
    create_app,
    RobotRequest,
    RobotResponse,
)

from robo_vla.utils import (
    load_config,
    setup_logging,
    GPUMemoryManager,
)

__all__ = [
    # Perception
    "DinoV2Detector",
    "MiDasDepthEstimator",
    "PointCloudGenerator",
    "GaussianSplatting",
    "PerceptionPipeline",
    # Language
    "LlamaLanguageModel",
    "CLIPAlignment",
    "LanguageGroundingModule",
    # Knowledge
    "Neo4jGraphDB",
    "ChromaDBVectorStore",
    "GraphRAGSystem",
    # VLA Model
    "CrossAttentionFusion",
    "ActionDecoder",
    "VLAModel",
    # RL
    "SACAgent",
    "ReplayBuffer",
    "RobotEnvironment",
    # Server
    "create_app",
    "RobotRequest",
    "RobotResponse",
    # Utils
    "load_config",
    "setup_logging",
    "GPUMemoryManager",
]
