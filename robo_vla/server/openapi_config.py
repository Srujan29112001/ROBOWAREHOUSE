"""
OpenAPI configuration and documentation for RoboVLA API.

Provides comprehensive API documentation with examples, schemas, and tags.
"""
from typing import Dict, Any


# API metadata
API_METADATA = {
    "title": "RoboVLA API",
    "description": """
# Vision-Language-Action Robot Control API

Production-ready API for controlling robots using multimodal AI.

## Features

- 🤖 **Vision-Language-Action** - Control robots with natural language
- 🔄 **Real-time Streaming** - WebSocket support for continuous control
- 📊 **Monitoring** - Built-in Prometheus metrics
- 🚀 **High Performance** - Optimized for low-latency inference
- 🔒 **Production Ready** - Health checks, graceful shutdown, error handling

## Quick Start

### 1. Execute a Command

```python
import requests
import base64

# Load your image
with open("scene.png", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post("http://api.robovla.ai/execute", json={
    "rgb_image": img_base64,
    "depth_image": img_base64,
    "text_command": "Pick the red box",
    "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "safety_constraints": {
        "max_velocity": 1.0,
        "max_acceleration": 0.5
    }
})

result = response.json()
print(f"Joint commands: {result['joint_commands']}")
print(f"Success probability: {result['success_probability']}")
```

### 2. Real-time Control via WebSocket

```python
import asyncio
import websockets
import json

async def control_robot():
    uri = "ws://api.robovla.ai/realtime_control"
    async with websockets.connect(uri) as websocket:
        # Send command
        await websocket.send(json.dumps({
            "rgb_image": img_base64,
            "text_command": "Move to position"
        }))

        # Receive response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(control_robot())
```

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│  FastAPI    │───▶│  VLA Model  │
│ Application │◀───│   Server    │◀───│  Inference  │
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Prometheus  │
                    │  Metrics    │
                    └─────────────┘
```

## Performance

- **Latency**: P50 < 100ms, P95 < 150ms
- **Throughput**: 50+ requests/second (single GPU)
- **Success Rate**: 92% on standard benchmarks

## Support

- 📚 Documentation: https://docs.robovla.ai
- 🐛 Issues: https://github.com/your-org/robovla/issues
- 💬 Discord: https://discord.gg/robovla
    """,
    "version": "1.0.0",
    "contact": {
        "name": "RoboVLA Team",
        "email": "support@robovla.ai",
        "url": "https://robovla.ai"
    },
    "license_info": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    "terms_of_service": "https://robovla.ai/terms"
}

# API tags for grouping endpoints
API_TAGS = [
    {
        "name": "Inference",
        "description": "Robot command execution and action generation"
    },
    {
        "name": "Monitoring",
        "description": "Health checks, metrics, and system status"
    },
    {
        "name": "Real-time",
        "description": "WebSocket streaming for continuous control"
    },
    {
        "name": "Admin",
        "description": "Administrative endpoints (authentication required)"
    }
]

# Example requests and responses
EXAMPLE_REQUEST = {
    "rgb_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "depth_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "text_command": "Pick the red box from the top shelf",
    "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "safety_constraints": {
        "max_velocity": 1.0,
        "max_acceleration": 0.5,
        "max_force": 50.0,
        "workspace_bounds": {
            "x_min": -0.5, "x_max": 0.5,
            "y_min": -0.5, "y_max": 0.5,
            "z_min": 0.0, "z_max": 1.0
        }
    }
}

EXAMPLE_RESPONSE = {
    "joint_commands": [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.1],
    "gripper_command": 0.2,
    "success_probability": 0.95,
    "execution_time": 3.5,
    "detected_objects": [
        {
            "name": "red_box",
            "confidence": 0.98,
            "position": [0.3, 0.1, 0.5],
            "bounding_box": [100, 150, 200, 250]
        }
    ],
    "reasoning": "Detected red box at position (0.3, 0.1, 0.5). Planning grasp from top approach with gripper opening 0.06m. Estimated execution time: 3.5 seconds."
}

# Response schemas with detailed descriptions
RESPONSE_SCHEMAS = {
    "HealthResponse": {
        "description": "Health check response",
        "content": {
            "application/json": {
                "example": {
                    "status": "healthy",
                    "models_loaded": True,
                    "uptime_seconds": 3600,
                    "gpu_available": True,
                    "gpu_utilization": 75.5,
                    "memory_used_gb": 8.2
                }
            }
        }
    },
    "ReadyResponse": {
        "description": "Readiness check response",
        "content": {
            "application/json": {
                "example": {
                    "ready": True,
                    "models_initialized": True,
                    "database_connected": True
                }
            }
        }
    },
    "ErrorResponse": {
        "description": "Error response",
        "content": {
            "application/json": {
                "example": {
                    "error": "InvalidInput",
                    "message": "RGB image decoding failed",
                    "details": {
                        "field": "rgb_image",
                        "reason": "Invalid base64 encoding"
                    },
                    "request_id": "req_123456789",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    }
}

# OpenAPI customization
def custom_openapi() -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced documentation.

    Returns:
        OpenAPI schema dictionary
    """
    return {
        "openapi": "3.0.2",
        "info": API_METADATA,
        "tags": API_TAGS,
        "servers": [
            {
                "url": "https://api.robovla.ai",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.robovla.ai",
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Local development"
            }
        ],
        "externalDocs": {
            "description": "Full documentation",
            "url": "https://docs.robovla.ai"
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        }
    }


# Endpoint documentation
ENDPOINT_DOCS = {
    "execute_command": {
        "summary": "Execute Robot Command",
        "description": """
Execute a robot manipulation command using vision-language-action model.

### Process Flow

1. **Image Processing**: RGB-D images are processed through DINO v2 and MiDaS
2. **Language Understanding**: Text command is parsed by Llama 3.1 LLM
3. **Action Generation**: VLA model generates 7-DOF joint commands
4. **Safety Validation**: Commands are validated against safety constraints

### Input Requirements

- **RGB Image**: 640x480 PNG/JPEG, base64 encoded
- **Depth Image**: Same dimensions as RGB, base64 encoded
- **Text Command**: Natural language instruction (max 200 chars)
- **Robot State**: Current 7-DOF joint angles + gripper state

### Output

- **Joint Commands**: 7 joint angles in radians [-π, π]
- **Gripper Command**: Gripper state [0=closed, 1=open]
- **Success Probability**: Predicted success rate [0, 1]
- **Execution Time**: Estimated time in seconds
        """,
        "response_description": "Successful response with robot commands",
        "responses": {
            200: {
                "description": "Command executed successfully",
                "content": {
                    "application/json": {
                        "example": EXAMPLE_RESPONSE
                    }
                }
            },
            400: RESPONSE_SCHEMAS["ErrorResponse"],
            422: {
                "description": "Validation error",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": [
                                {
                                    "loc": ["body", "robot_state"],
                                    "msg": "ensure this value has at least 8 items",
                                    "type": "value_error.list.min_items"
                                }
                            ]
                        }
                    }
                }
            },
            500: {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "InferenceError",
                            "message": "Model inference failed",
                            "request_id": "req_123456789"
                        }
                    }
                }
            }
        }
    },

    "health_check": {
        "summary": "Health Check",
        "description": """
Check if the service is healthy and operational.

This endpoint should be used for:
- Kubernetes liveness probes
- Load balancer health checks
- Monitoring systems

Returns immediately without model inference.
        """,
        "response_description": "Service health status"
    },

    "ready_check": {
        "summary": "Readiness Check",
        "description": """
Check if the service is ready to handle requests.

Unlike /health, this endpoint verifies:
- Models are loaded and initialized
- Database connections are active
- All dependencies are ready

Use for Kubernetes readiness probes.
        """,
        "response_description": "Service readiness status"
    },

    "metrics": {
        "summary": "Prometheus Metrics",
        "description": """
Expose Prometheus-formatted metrics for monitoring.

### Available Metrics

- `vla_requests_total` - Total requests by endpoint and status
- `vla_request_latency_seconds` - Request latency histogram
- `vla_inference_latency_seconds` - Model inference latency
- `vla_success_rate` - Task success rate
- `gpu_utilization_percent` - GPU utilization
- `gpu_memory_used_bytes` - GPU memory usage

Configure Prometheus to scrape this endpoint.
        """,
        "response_description": "Prometheus metrics in text format"
    },

    "websocket_control": {
        "summary": "Real-time Control Stream",
        "description": """
WebSocket endpoint for real-time robot control.

### Connection

```python
import websockets
import asyncio

async def control():
    uri = "ws://api.robovla.ai/realtime_control"
    async with websockets.connect(uri) as ws:
        # Send commands
        await ws.send(json.dumps(command))
        # Receive responses
        response = await ws.recv()
```

### Message Format

Same as `/execute` endpoint - send RobotRequest JSON, receive RobotResponse JSON.

### Use Cases

- Continuous teleoperation
- Real-time feedback loops
- Streaming manipulation
        """,
        "response_description": "WebSocket connection for bidirectional streaming"
    }
}


# Code examples for different languages
CODE_EXAMPLES = {
    "python": """
import requests
import base64
from PIL import Image
import io

# Load and encode image
with Image.open("scene.jpg") as img:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

# Execute command
response = requests.post(
    "https://api.robovla.ai/execute",
    json={
        "rgb_image": img_base64,
        "depth_image": img_base64,
        "text_command": "Pick the red box",
        "robot_state": [0.0] * 7 + [1.0],
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

if response.status_code == 200:
    result = response.json()
    print(f"Joint commands: {result['joint_commands']}")
    print(f"Success probability: {result['success_probability']:.2%}")
else:
    print(f"Error: {response.json()}")
    """,

    "javascript": """
const fs = require('fs');
const axios = require('axios');

// Load and encode image
const imageBuffer = fs.readFileSync('scene.jpg');
const imageBase64 = imageBuffer.toString('base64');

// Execute command
axios.post('https://api.robovla.ai/execute', {
    rgb_image: imageBase64,
    depth_image: imageBase64,
    text_command: 'Pick the red box',
    robot_state: [0, 0, 0, 0, 0, 0, 0, 1]
}, {
    headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
    }
})
.then(response => {
    console.log('Joint commands:', response.data.joint_commands);
    console.log('Success probability:', response.data.success_probability);
})
.catch(error => {
    console.error('Error:', error.response.data);
});
    """,

    "curl": """
# Save image as base64
IMAGE_BASE64=$(base64 -w 0 scene.jpg)

# Execute command
curl -X POST https://api.robovla.ai/execute \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -d '{
    "rgb_image": "'$IMAGE_BASE64'",
    "depth_image": "'$IMAGE_BASE64'",
    "text_command": "Pick the red box",
    "robot_state": [0, 0, 0, 0, 0, 0, 0, 1]
  }'
    """
}
