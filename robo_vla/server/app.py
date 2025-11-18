"""FastAPI application for VLA inference server."""

import asyncio
import base64
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from robo_vla import VLAModel, load_config, setup_logging
from robo_vla.server.models import RobotRequest, RobotResponse

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("vla_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("vla_request_latency_seconds", "Request latency", ["endpoint"])
INFERENCE_LATENCY = Histogram("vla_inference_latency_seconds", "Inference latency")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config_path: Path to config file

    Returns:
        FastAPI application
    """
    # Load config
    config = load_config(config_path)

    # Setup logging
    setup_logging(
        log_level=config["monitoring"]["logging"]["level"],
        log_file=config["monitoring"]["logging"]["file"],
        json_format=config["monitoring"]["logging"]["format"] == "json",
    )

    # Create FastAPI app
    app = FastAPI(
        title="RoboVLA API",
        description="Production API for Vision-Language-Action Robot Control",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config["server"]["cors"]["allow_origins"],
        allow_credentials=config["server"]["cors"]["allow_credentials"],
        allow_methods=config["server"]["cors"]["allow_methods"],
        allow_headers=config["server"]["cors"]["allow_headers"],
    )

    # Thread pool for CPU-bound operations
    executor = ThreadPoolExecutor(max_workers=config["server"]["workers"])

    # Model will be loaded on startup
    app.state.vla_model = None
    app.state.config = config
    app.state.executor = executor

    @app.on_event("startup")
    async def startup_event():
        """Load models on startup."""
        logger.info("Loading VLA model...")
        device = config["system"]["device"]
        app.state.vla_model = VLAModel(config, device=device)
        logger.info("VLA model loaded successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        executor.shutdown(wait=True)
        logger.info("Server shutdown complete")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model_loaded": app.state.vla_model is not None}

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type="text/plain")

    @app.post("/execute", response_model=RobotResponse)
    async def execute_command(request: RobotRequest):
        """
        Execute robot command.

        Args:
            request: Robot request with images and command

        Returns:
            Robot response with actions
        """
        start_time = time.time()

        try:
            # Decode images
            rgb_image = decode_base64_image(request.rgb_image)


            # Convert to tensor
            rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
            rgb_tensor = rgb_tensor.to(config["system"]["device"])

            # Robot state
            robot_state = torch.tensor(request.robot_state, dtype=torch.float32)
            robot_state = robot_state.to(config["system"]["device"])

            # Run inference
            with INFERENCE_LATENCY.time():
                result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    app.state.vla_model.predict,
                    rgb_tensor,
                    request.text_command,
                    robot_state,
                )

            # Create response
            response = RobotResponse(
                joint_commands=result["joint_commands"].flatten().tolist(),
                gripper_command=float(result["gripper_command"].flatten()[0]),
                success_probability=result["success_probability"],
                execution_time=result["execution_time"],
                reasoning="Action generated from VLA model",
            )

            REQUEST_COUNT.labels(endpoint="/execute", status="success").inc()

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            REQUEST_COUNT.labels(endpoint="/execute", status="error").inc()
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint="/execute").observe(latency)

        return response

    @app.websocket("/stream")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time control."""
        await websocket.accept()

        try:
            while True:
                # Receive data
                data = await websocket.receive_json()

                # Process request
                request = RobotRequest(**data)
                response = await execute_command(request)

                # Send response
                await websocket.send_json(response.dict())

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()

    return app


def decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)
