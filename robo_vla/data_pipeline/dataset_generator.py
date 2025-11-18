"""
Synthetic dataset generator for robot VLA training.

Generates realistic RGB-D images with objects, depth maps, and corresponding
robot manipulation commands for training.
"""
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import random


@dataclass
class ObjectInstance:
    """Represents an object in the scene."""
    object_id: int
    name: str
    position: Tuple[float, float, float]  # x, y, z in meters
    rotation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    color: Tuple[int, int, int]  # RGB
    shape: str  # 'cube', 'sphere', 'cylinder', 'box'
    dimensions: Tuple[float, float, float]  # width, height, depth in meters


@dataclass
class RobotCommand:
    """Represents a robot manipulation command."""
    command_id: int
    text: str
    target_object: str
    source_location: str
    target_location: str
    joint_trajectory: List[List[float]]  # Sequence of joint angles
    gripper_trajectory: List[float]  # Sequence of gripper states
    success: bool
    execution_time: float


class SyntheticDatasetGenerator:
    """
    Generates synthetic training data for robot VLA.

    Creates realistic scenes with objects, generates depth maps,
    and produces corresponding manipulation commands.
    """

    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        camera_intrinsics: Optional[Dict] = None,
        output_dir: str = "data/synthetic"
    ):
        self.width = image_width
        self.height = image_height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Camera intrinsics
        if camera_intrinsics is None:
            self.intrinsics = {
                'fx': 525.0,
                'fy': 525.0,
                'cx': image_width / 2,
                'cy': image_height / 2
            }
        else:
            self.intrinsics = camera_intrinsics

        # Object templates
        self.object_shapes = ['cube', 'sphere', 'cylinder', 'box']
        self.object_colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
        }

        # Command templates
        self.command_templates = [
            "Pick the {color} {shape}",
            "Place the {color} {shape} in the {location}",
            "Move the {color} {shape} to the {location}",
            "Grasp the {color} {shape} from the {source}",
            "Stack the {color} {shape} on top of the {target}",
        ]

        self.locations = ['bin', 'shelf', 'table', 'box', 'container']

    def generate_object(self, object_id: int) -> ObjectInstance:
        """Generate a random object instance."""
        color_name = random.choice(list(self.object_colors.keys()))
        color_rgb = self.object_colors[color_name]
        shape = random.choice(self.object_shapes)

        # Random position in workspace (meters)
        position = (
            random.uniform(0.2, 0.8),  # x: 20-80cm
            random.uniform(-0.3, 0.3),  # y: -30 to 30cm
            random.uniform(0.0, 0.5)   # z: 0-50cm above table
        )

        # Random rotation
        rotation = (
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi),
            random.uniform(0, 2 * np.pi)
        )

        # Object dimensions based on shape
        if shape == 'cube':
            size = random.uniform(0.03, 0.08)  # 3-8cm cubes
            dimensions = (size, size, size)
        elif shape == 'sphere':
            radius = random.uniform(0.02, 0.05)
            dimensions = (radius * 2, radius * 2, radius * 2)
        elif shape == 'cylinder':
            radius = random.uniform(0.02, 0.04)
            height = random.uniform(0.05, 0.10)
            dimensions = (radius * 2, height, radius * 2)
        else:  # box
            dimensions = (
                random.uniform(0.04, 0.10),
                random.uniform(0.03, 0.08),
                random.uniform(0.05, 0.12)
            )

        return ObjectInstance(
            object_id=object_id,
            name=f"{color_name}_{shape}",
            position=position,
            rotation=rotation,
            color=color_rgb,
            shape=shape,
            dimensions=dimensions
        )

    def render_object_2d(
        self,
        obj: ObjectInstance,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render object to 2D image and depth map."""
        # Project 3D position to 2D
        x_3d, y_3d, z_3d = obj.position

        # Camera projection
        x_2d = int(self.intrinsics['fx'] * x_3d / z_3d + self.intrinsics['cx'])
        y_2d = int(self.intrinsics['fy'] * y_3d / z_3d + self.intrinsics['cy'])

        # Calculate object size in pixels
        w, h, d = obj.dimensions
        size_pixels = int(self.intrinsics['fx'] * w / z_3d)

        # Draw object based on shape
        if obj.shape == 'cube' or obj.shape == 'box':
            # Draw rectangle
            top_left = (max(0, x_2d - size_pixels // 2), max(0, y_2d - size_pixels // 2))
            bottom_right = (
                min(self.width, x_2d + size_pixels // 2),
                min(self.height, y_2d + size_pixels // 2)
            )
            cv2.rectangle(image, top_left, bottom_right, obj.color, -1)

            # Update depth map
            depth_map[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = z_3d

        elif obj.shape == 'sphere':
            # Draw circle
            cv2.circle(image, (x_2d, y_2d), size_pixels // 2, obj.color, -1)

            # Update depth map (circular region)
            y, x = np.ogrid[:self.height, :self.width]
            mask = (x - x_2d)**2 + (y - y_2d)**2 <= (size_pixels // 2)**2
            depth_map[mask] = z_3d

        elif obj.shape == 'cylinder':
            # Draw ellipse
            axes = (size_pixels // 2, int(size_pixels * 0.7))
            cv2.ellipse(image, (x_2d, y_2d), axes, 0, 0, 360, obj.color, -1)

            # Update depth map
            y, x = np.ogrid[:self.height, :self.width]
            mask = ((x - x_2d) / axes[0])**2 + ((y - y_2d) / axes[1])**2 <= 1
            depth_map[mask] = z_3d

        return image, depth_map

    def generate_scene(self, num_objects: int = 5) -> Tuple[np.ndarray, np.ndarray, List[ObjectInstance]]:
        """Generate a complete scene with multiple objects."""
        # Initialize image and depth map
        image = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200  # Light gray background
        depth_map = np.ones((self.height, self.width), dtype=np.float32) * 10.0  # Far background

        # Add table surface
        table_color = (139, 90, 43)  # Brown
        cv2.rectangle(image, (0, self.height // 2), (self.width, self.height), table_color, -1)
        depth_map[self.height // 2:, :] = 0.75  # Table at 75cm

        # Generate objects
        objects = []
        for i in range(num_objects):
            obj = self.generate_object(i)
            objects.append(obj)
            image, depth_map = self.render_object_2d(obj, image, depth_map)

        # Add noise to make realistic
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

        depth_noise = np.random.normal(0, 0.01, depth_map.shape)
        depth_map += depth_noise

        return image, depth_map, objects

    def generate_robot_trajectory(
        self,
        start_state: List[float],
        target_object: ObjectInstance,
        action: str = 'pick'
    ) -> Tuple[List[List[float]], List[float]]:
        """Generate robot joint trajectory for manipulation."""
        # Simplified trajectory generation
        # In real scenario, use motion planning algorithms

        num_steps = 10
        joint_trajectory = []
        gripper_trajectory = []

        # Calculate target joint angles (simplified inverse kinematics)
        x, y, z = target_object.position

        # Simple 7-DOF joint calculation (placeholder)
        target_joints = [
            np.arctan2(y, x),  # Base rotation
            -np.pi / 4,  # Shoulder
            np.pi / 2,   # Elbow
            -np.pi / 4,  # Wrist 1
            np.pi / 2,   # Wrist 2
            0.0,         # Wrist 3
            0.0          # Wrist rotation
        ]

        # Linear interpolation from start to target
        for step in range(num_steps):
            alpha = step / (num_steps - 1)
            joints = [
                start + alpha * (target - start)
                for start, target in zip(start_state, target_joints)
            ]
            joint_trajectory.append(joints)

            # Gripper trajectory
            if action == 'pick':
                gripper = 1.0 if step < num_steps // 2 else 0.0  # Open then close
            else:  # place
                gripper = 0.0 if step < num_steps // 2 else 1.0  # Close then open
            gripper_trajectory.append(gripper)

        return joint_trajectory, gripper_trajectory

    def generate_command(self, objects: List[ObjectInstance]) -> RobotCommand:
        """Generate a robot command for the scene."""
        if not objects:
            return None

        # Select random target object
        target_obj = random.choice(objects)

        # Parse object name
        color, shape = target_obj.name.split('_')

        # Select command template
        template = random.choice(self.command_templates)

        # Fill in template
        if '{location}' in template:
            location = random.choice(self.locations)
            text = template.format(color=color, shape=shape, location=location)
        elif '{source}' in template:
            source = random.choice(self.locations)
            text = template.format(color=color, shape=shape, source=source)
        elif '{target}' in template:
            if len(objects) > 1:
                other_obj = random.choice([o for o in objects if o != target_obj])
                other_color, other_shape = other_obj.name.split('_')
                text = template.format(color=color, shape=shape, target=f"{other_color} {other_shape}")
            else:
                text = template.format(color=color, shape=shape, target='base')
        else:
            text = template.format(color=color, shape=shape)

        # Generate trajectory
        start_state = [0.0] * 7  # Home position
        joint_traj, gripper_traj = self.generate_robot_trajectory(start_state, target_obj, 'pick')

        # Simulate success rate
        success = random.random() > 0.1  # 90% success rate
        execution_time = random.uniform(2.0, 5.0)  # 2-5 seconds

        return RobotCommand(
            command_id=random.randint(0, 1000000),
            text=text,
            target_object=target_obj.name,
            source_location='table',
            target_location=random.choice(self.locations),
            joint_trajectory=joint_traj,
            gripper_trajectory=gripper_traj,
            success=success,
            execution_time=execution_time
        )

    def generate_dataset(
        self,
        num_samples: int = 1000,
        num_objects_range: Tuple[int, int] = (3, 7)
    ) -> None:
        """Generate complete dataset and save to disk."""
        print(f"Generating {num_samples} synthetic samples...")

        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'depth').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)

        dataset_info = {
            'num_samples': num_samples,
            'image_size': (self.width, self.height),
            'camera_intrinsics': self.intrinsics,
            'samples': []
        }

        for i in range(num_samples):
            # Generate scene
            num_objects = random.randint(*num_objects_range)
            rgb_image, depth_map, objects = self.generate_scene(num_objects)

            # Generate command
            command = self.generate_command(objects)

            # Save images
            rgb_path = self.output_dir / 'images' / f'rgb_{i:06d}.png'
            depth_path = self.output_dir / 'depth' / f'depth_{i:06d}.npy'
            metadata_path = self.output_dir / 'metadata' / f'meta_{i:06d}.json'

            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            np.save(str(depth_path), depth_map)

            # Save metadata
            sample_metadata = {
                'sample_id': i,
                'rgb_path': str(rgb_path.relative_to(self.output_dir)),
                'depth_path': str(depth_path.relative_to(self.output_dir)),
                'objects': [asdict(obj) for obj in objects],
                'command': asdict(command) if command else None,
                'num_objects': num_objects
            }

            with open(metadata_path, 'w') as f:
                json.dump(sample_metadata, f, indent=2)

            dataset_info['samples'].append(sample_metadata)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")

        # Save dataset info
        with open(self.output_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Dataset generation complete! Saved to {self.output_dir}")
        print(f"Total samples: {num_samples}")
        print(f"Average objects per scene: {np.mean([s['num_objects'] for s in dataset_info['samples']]):.1f}")


if __name__ == '__main__':
    # Generate dataset
    generator = SyntheticDatasetGenerator(output_dir='data/synthetic_robot_dataset')
    generator.generate_dataset(num_samples=1000)
