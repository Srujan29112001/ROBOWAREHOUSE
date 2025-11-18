"""Server entry point for RoboVLA."""

import argparse
import sys

import uvicorn

from robo_vla import load_config


def main():
    """Main server entry point."""
    parser = argparse.ArgumentParser(
        description="RoboVLA Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (overrides config)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (overrides config)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["critical", "error", "warning", "info", "debug"],
        default=None,
        help="Log level (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI arguments
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    workers = args.workers or config["server"]["workers"]
    log_level = args.log_level or config["server"]["log_level"]
    reload = args.reload or config["server"]["reload"]

    print("="*60)
    print("ROBOVLA INFERENCE SERVER")
    print("="*60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print(f"Log Level: {log_level}")
    print(f"Auto-reload: {reload}")
    print("="*60)
    print(f"\nServer will be available at: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Metrics: http://{host}:{port}/metrics")
    print("="*60)

    # Run server
    uvicorn.run(
        "robo_vla.server.app:create_app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
        factory=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
