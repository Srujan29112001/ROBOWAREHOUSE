"""Pydantic models for API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RobotRequest(BaseModel):
    """Robot command request."""

    rgb_image: str = Field(..., description="Base64 encoded RGB image")
    depth_image: Optional[str] = Field(None, description="Base64 encoded depth image")
    text_command: str = Field(..., description="Natural language command")
    robot_state: List[float] = Field(..., description="Current robot joint angles")
    safety_constraints: Optional[Dict] = Field(default=None, description="Safety constraints")


class RobotResponse(BaseModel):
    """Robot command response."""

    joint_commands: List[float] = Field(..., description="Joint angle commands")
    gripper_command: float = Field(..., description="Gripper command [0, 1]")
    success_probability: float = Field(..., description="Predicted success probability")
    execution_time: float = Field(..., description="Estimated execution time (seconds)")
    reasoning: str = Field(..., description="Explanation of the action")
