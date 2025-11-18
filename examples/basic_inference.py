"""Basic inference example for RoboVLA."""

import torch
import numpy as np
from PIL import Image

from robo_vla import VLAModel, load_config


def main():
    """Run basic inference example."""
    print("🤖 RoboVLA Basic Inference Example\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config()

    # Initialize model
    print("Initializing VLA model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VLAModel(config, device=device)
    print(f"✓ Model loaded on {device}\n")

    # Create dummy input (in practice, load from camera)
    print("Preparing input data...")
    rgb_image = torch.randn(3, 640, 480).to(device)  # Random image for demo
    text_command = "Pick the red box from the top shelf and place it in bin A"
    robot_state = torch.zeros(8).to(device)  # Current joint angles + gripper

    print(f"Command: {text_command}")
    print(f"Robot state: {robot_state.cpu().numpy()}\n")

    # Run inference
    print("Running inference...")
    result = model.predict(
        rgb_image=rgb_image,
        text_command=text_command,
        robot_state=robot_state,
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nJoint Commands (radians):")
    for i, angle in enumerate(result["joint_commands"][0], 1):
        print(f"  Joint {i}: {angle:+.4f} rad ({np.degrees(angle):+.2f}°)")

    print(f"\nGripper Command: {result['gripper_command'][0]:.3f}")
    print(f"  (0.0 = fully closed, 1.0 = fully open)")

    print(f"\nSuccess Probability: {result['success_probability']:.1%}")
    print(f"Estimated Execution Time: {result['execution_time']:.2f} seconds")

    print("\n" + "="*60)
    print("✓ Inference complete!")


if __name__ == "__main__":
    main()
