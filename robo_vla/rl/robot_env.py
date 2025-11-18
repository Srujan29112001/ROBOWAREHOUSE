"""Robot manipulation environment for RL training."""

import logging
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class RobotEnvironment(gym.Env):
    """
    Gymnasium environment for robot manipulation tasks.

    Supports pick-and-place, grasping, and other manipulation primitives.
    """

    def __init__(
        self,
        task: str = "pick_and_place",
        render_mode: Optional[str] = None,
        max_steps: int = 100,
    ):
        """
        Initialize robot environment.

        Args:
            task: Task type ('pick_and_place', 'grasp', 'push')
            render_mode: Render mode ('human', 'rgb_array', None)
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.task = task
        self.render_mode = render_mode
        self.max_steps = max_steps

        # State space: robot joints (7) + gripper (1) + object pose (7) + target pose (7) = 22
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )

        # Action space: joint velocities (7) + gripper (1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )

        # Episode state
        self.current_step = 0
        self.robot_state = np.zeros(8)
        self.object_pose = np.zeros(7)  # x, y, z, qw, qx, qy, qz
        self.target_pose = np.zeros(7)
        self.grasped = False

        logger.info(f"Robot environment initialized for task: {task}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.grasped = False

        # Reset robot to home position
        self.robot_state = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0, 1.0])

        # Random object position on table
        self.object_pose[:3] = np.array([
            np.random.uniform(0.3, 0.6),  # x
            np.random.uniform(-0.3, 0.3),  # y
            0.02,  # z (on table)
        ])
        self.object_pose[3:] = np.array([1, 0, 0, 0])  # Identity quaternion

        # Random target position
        self.target_pose[:3] = np.array([
            np.random.uniform(0.3, 0.6),  # x
            np.random.uniform(-0.3, 0.3),  # y
            0.02,  # z
        ])
        self.target_pose[3:] = np.array([1, 0, 0, 0])

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in environment.

        Args:
            action: Action array [joint_velocities, gripper]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply action (simplified dynamics)
        joint_velocities = action[:7] * 0.1  # Scale velocities
        gripper_command = action[7]

        # Update robot state
        self.robot_state[:7] += joint_velocities
        self.robot_state[:7] = np.clip(self.robot_state[:7], -np.pi, np.pi)
        self.robot_state[7] = gripper_command

        # Simple grasp logic
        ee_pos = self._forward_kinematics(self.robot_state[:7])
        dist_to_object = np.linalg.norm(ee_pos - self.object_pose[:3])

        if dist_to_object < 0.05 and gripper_command < 0.3:
            self.grasped = True

        # Update object position if grasped
        if self.grasped:
            self.object_pose[:3] = ee_pos

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        dist_to_target = np.linalg.norm(self.object_pose[:3] - self.target_pose[:3])
        terminated = dist_to_target < 0.05  # Success

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()
        info["success"] = terminated

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([
            self.robot_state,  # 8
            self.object_pose,  # 7
            self.target_pose,  # 7
        ])

    def _get_info(self) -> dict:
        """Get environment info."""
        ee_pos = self._forward_kinematics(self.robot_state[:7])
        dist_to_object = np.linalg.norm(ee_pos - self.object_pose[:3])
        dist_to_target = np.linalg.norm(self.object_pose[:3] - self.target_pose[:3])

        return {
            "ee_position": ee_pos,
            "distance_to_object": dist_to_object,
            "distance_to_target": dist_to_target,
            "grasped": self.grasped,
            "step": self.current_step,
        }

    def _compute_reward(self) -> float:
        """Compute reward."""
        ee_pos = self._forward_kinematics(self.robot_state[:7])

        # Distance to object (encourage reaching)
        dist_to_object = np.linalg.norm(ee_pos - self.object_pose[:3])
        reach_reward = -0.1 * dist_to_object

        # Grasp reward
        grasp_reward = 10.0 if self.grasped else 0.0

        # Distance to target (encourage placing)
        dist_to_target = np.linalg.norm(self.object_pose[:3] - self.target_pose[:3])
        place_reward = -0.1 * dist_to_target if self.grasped else 0.0

        # Success reward
        success_reward = 100.0 if dist_to_target < 0.05 else 0.0

        # Time penalty
        time_penalty = -0.01

        total_reward = reach_reward + grasp_reward + place_reward + success_reward + time_penalty

        return total_reward

    def _forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Simplified forward kinematics.

        In practice, use a proper kinematics library or robot model.
        """
        # Simplified: just use first 3 joints for position
        x = 0.3 + 0.3 * np.sin(joint_angles[0])
        y = 0.3 * np.cos(joint_angles[1])
        z = 0.2 + 0.2 * joint_angles[2]

        return np.array([x, y, z])

    def render(self):
        """Render environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Robot: {self.robot_state[:3]}")
            print(f"Object: {self.object_pose[:3]}")
            print(f"Target: {self.target_pose[:3]}")
            print(f"Grasped: {self.grasped}")
            print("-" * 40)

    def close(self):
        """Close environment."""
        pass
