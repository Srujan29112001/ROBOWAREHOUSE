"""Example API client for RoboVLA server."""

import base64
import json
from io import BytesIO

import numpy as np
import requests
from PIL import Image


class RoboVLAClient:
    """Client for RoboVLA API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url

    def health_check(self) -> dict:
        """Check server health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def execute_command(
        self,
        rgb_image: np.ndarray,
        text_command: str,
        robot_state: list,
    ) -> dict:
        """
        Execute robot command.

        Args:
            rgb_image: RGB image array [H, W, 3]
            text_command: Natural language command
            robot_state: Current robot state

        Returns:
            API response with actions
        """
        # Encode image to base64
        pil_image = Image.fromarray(rgb_image.astype(np.uint8))
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        rgb_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Prepare request
        payload = {
            "rgb_image": rgb_base64,
            "text_command": text_command,
            "robot_state": robot_state,
        }

        # Send request
        response = requests.post(
            f"{self.base_url}/execute",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        return response.json()


def main():
    """Run API client example."""
    print("🤖 RoboVLA API Client Example\n")

    # Initialize client
    client = RoboVLAClient(base_url="http://localhost:8000")

    # Check health
    print("Checking server health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}\n")

    # Prepare input
    print("Preparing input data...")
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    text_command = "Pick the red box from the shelf"
    robot_state = [0.0] * 8

    # Execute command
    print(f"Executing command: '{text_command}'...")
    result = client.execute_command(
        rgb_image=rgb_image,
        text_command=text_command,
        robot_state=robot_state,
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nJoint Commands: {result['joint_commands']}")
    print(f"Gripper Command: {result['gripper_command']}")
    print(f"Success Probability: {result['success_probability']:.1%}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"Reasoning: {result['reasoning']}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
