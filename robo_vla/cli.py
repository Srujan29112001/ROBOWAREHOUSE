"""Command-line interface for RoboVLA."""

import argparse
import sys
from pathlib import Path

import torch

from robo_vla import VLAModel, load_config, setup_logging
from robo_vla.utils import GPUMemoryManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RoboVLA: Multimodal Robotic Vision-Language-Action System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference")
    predict_parser.add_argument("--image", type=str, required=True, help="Path to RGB image")
    predict_parser.add_argument("--command", type=str, required=True, help="Text command")
    predict_parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check system requirements")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Load config
    config = load_config(args.config)
    setup_logging(log_level="INFO")

    if args.command == "predict":
        return run_predict(args, config)
    elif args.command == "info":
        return show_info(config)
    elif args.command == "check":
        return check_system()
    else:
        parser.print_help()
        return 1


def run_predict(args, config):
    """Run prediction on an image."""
    from PIL import Image
    import numpy as np

    print(f"Loading model on {args.device}...")
    model = VLAModel(config, device=args.device)

    print(f"Loading image from {args.image}...")
    image = Image.open(args.image).convert("RGB")
    rgb_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    rgb_tensor = rgb_tensor.to(args.device)

    # Mock robot state (7 joints + gripper)
    robot_state = torch.zeros(8).to(args.device)

    print(f"Running inference with command: '{args.command}'...")
    result = model.predict(rgb_tensor, args.command, robot_state)

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Joint Commands: {result['joint_commands'].flatten().tolist()}")
    print(f"Gripper Command: {result['gripper_command'].flatten()[0]:.4f}")
    print(f"Success Probability: {result['success_probability']:.2%}")
    print(f"Execution Time: {result['execution_time']:.3f}s")
    print("="*60)

    return 0


def show_info(config):
    """Show system information."""
    print("\n" + "="*60)
    print("ROBOVLA SYSTEM INFORMATION")
    print("="*60)
    print(f"Version: {config['system']['version']}")
    print(f"Device: {config['system']['device']}")
    print(f"Mixed Precision: {config['system']['mixed_precision']}")
    print(f"Compile Models: {config['system']['compile_models']}")
    print("\nHardware Configuration:")
    print(f"  GPU Model: {config['hardware']['gpu']['model']}")
    print(f"  VRAM: {config['hardware']['gpu']['vram_gb']}GB")
    print(f"  Max Batch Size: {config['hardware']['gpu']['max_batch_size']}")
    print(f"  CPU Workers: {config['hardware']['cpu']['num_workers']}")
    print("\nCUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print("="*60)

    return 0


def check_system():
    """Check system requirements."""
    print("\n" + "="*60)
    print("SYSTEM REQUIREMENTS CHECK")
    print("="*60)

    checks = []

    # Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 9)
    checks.append(("Python 3.9+", py_ok, f"{py_version.major}.{py_version.minor}.{py_version.micro}"))

    # PyTorch
    try:
        torch_ok = torch.__version__ >= "2.1.0"
        checks.append(("PyTorch 2.1+", torch_ok, torch.__version__))
    except Exception:
        checks.append(("PyTorch 2.1+", False, "Not installed"))

    # CUDA
    cuda_ok = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_ok else "Not available"
    checks.append(("CUDA Available", cuda_ok, cuda_version))

    # GPU Memory
    if cuda_ok:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_ok = gpu_mem >= 12.0
        checks.append(("GPU Memory (12GB+)", gpu_ok, f"{gpu_mem:.1f}GB"))

    for name, status, info in checks:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{status_str:8} {name:25} {info}")

    print("="*60)

    all_pass = all(check[1] for check in checks)
    if all_pass:
        print("✓ All checks passed!")
        return 0
    else:
        print("✗ Some checks failed. Please review the requirements.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
