"""
Generate OpenAPI documentation for RoboVLA API.

Creates OpenAPI spec file and HTML documentation.
"""
import json
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robo_vla.server.app import create_app
from robo_vla.server.openapi_config import custom_openapi, CODE_EXAMPLES


def generate_openapi_spec(output_format: str = 'json') -> None:
    """
    Generate OpenAPI specification file.

    Args:
        output_format: 'json' or 'yaml'
    """
    # Create app
    app = create_app()

    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Add custom documentation
    custom_schema = custom_openapi()
    openapi_schema.update(custom_schema)

    # Create output directory
    docs_dir = Path(__file__).parent.parent / 'docs' / 'api'
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Save specification
    if output_format == 'json':
        output_file = docs_dir / 'openapi.json'
        with open(output_file, 'w') as f:
            json.dump(openapi_schema, f, indent=2)
    else:
        output_file = docs_dir / 'openapi.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(openapi_schema, f, default_flow_style=False)

    print(f"OpenAPI specification saved to: {output_file}")

    # Generate code examples
    examples_dir = docs_dir / 'examples'
    examples_dir.mkdir(exist_ok=True)

    for lang, code in CODE_EXAMPLES.items():
        ext = {
            'python': 'py',
            'javascript': 'js',
            'curl': 'sh'
        }.get(lang, 'txt')

        example_file = examples_dir / f'example.{ext}'
        with open(example_file, 'w') as f:
            f.write(code)

        print(f"Code example saved to: {example_file}")


def generate_html_docs() -> None:
    """Generate HTML documentation using Swagger UI."""
    docs_dir = Path(__file__).parent.parent / 'docs' / 'api'
    docs_dir.mkdir(parents=True, exist_ok=True)

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoboVLA API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        body {
            margin: 0;
            padding: 0;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
            window.ui = ui;
        };
    </script>
</body>
</html>
    """

    html_file = docs_dir / 'index.html'
    with open(html_file, 'w') as f:
        f.write(html_content)

    print(f"HTML documentation saved to: {html_file}")
    print(f"\nOpen in browser: file://{html_file.absolute()}")


def generate_markdown_docs() -> None:
    """Generate Markdown documentation."""
    docs_dir = Path(__file__).parent.parent / 'docs' / 'api'
    docs_dir.mkdir(parents=True, exist_ok=True)

    markdown = """# RoboVLA API Documentation

## Overview

RoboVLA provides a REST API for controlling robots using vision-language-action models.

## Authentication

All API requests require authentication using either:

1. **Bearer Token** (recommended for production)
2. **API Key** (for development)

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.robovla.ai/execute
```

## Endpoints

### POST /execute

Execute a robot manipulation command.

**Request:**

```json
{
  "rgb_image": "base64_encoded_image",
  "depth_image": "base64_encoded_depth",
  "text_command": "Pick the red box",
  "robot_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "safety_constraints": {
    "max_velocity": 1.0,
    "max_acceleration": 0.5
  }
}
```

**Response:**

```json
{
  "joint_commands": [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.1],
  "gripper_command": 0.2,
  "success_probability": 0.95,
  "execution_time": 3.5,
  "detected_objects": [...],
  "reasoning": "..."
}
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### GET /ready

Readiness check endpoint.

**Response:**

```json
{
  "ready": true,
  "models_initialized": true
}
```

### GET /metrics

Prometheus metrics endpoint.

Returns metrics in Prometheus text format.

### WebSocket /realtime_control

Real-time control via WebSocket.

```python
import websockets
import asyncio
import json

async def control():
    uri = "ws://api.robovla.ai/realtime_control"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(command))
        response = await ws.recv()
        print(json.loads(response))

asyncio.run(control())
```

## Rate Limits

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 10,000 requests/hour
- **Enterprise**: Unlimited

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 422 | Validation Error |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

## Code Examples

### Python

See `examples/example.py` for complete Python example.

### JavaScript

See `examples/example.js` for complete JavaScript example.

### cURL

See `examples/example.sh` for complete cURL example.

## Support

- Documentation: https://docs.robovla.ai
- Issues: https://github.com/your-org/robovla/issues
- Email: support@robovla.ai
"""

    md_file = docs_dir / 'README.md'
    with open(md_file, 'w') as f:
        f.write(markdown)

    print(f"Markdown documentation saved to: {md_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate API documentation')
    parser.add_argument('--format', choices=['json', 'yaml', 'all'], default='all',
                        help='Output format')
    parser.add_argument('--html', action='store_true',
                        help='Generate HTML documentation')
    parser.add_argument('--markdown', action='store_true',
                        help='Generate Markdown documentation')

    args = parser.parse_args()

    # Generate OpenAPI spec
    if args.format in ['json', 'all']:
        generate_openapi_spec('json')
    if args.format in ['yaml', 'all']:
        generate_openapi_spec('yaml')

    # Generate HTML docs
    if args.html or args.format == 'all':
        generate_html_docs()

    # Generate Markdown docs
    if args.markdown or args.format == 'all':
        generate_markdown_docs()

    print("\n✅ Documentation generation complete!")
