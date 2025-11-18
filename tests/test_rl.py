"""Tests for RL components."""

import pytest
import torch
import numpy as np

from robo_vla.rl import SACAgent, ReplayBuffer, RobotEnvironment


class TestReplayBuffer:
    """Test replay buffer."""

    @pytest.fixture
    def buffer(self):
        """Create buffer instance."""
        return ReplayBuffer(
            state_dim=22,
            action_dim=8,
            buffer_size=1000,
            device="cpu",
        )

    def test_add_and_sample(self, buffer):
        """Test adding and sampling transitions."""
        # Add transitions
        for i in range(100):
            state = np.random.randn(22)
            action = np.random.randn(8)
            reward = np.random.rand()
            next_state = np.random.randn(22)
            done = bool(np.random.randint(2))

            buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 100

        # Sample batch
        batch = buffer.sample(32)

        assert batch["states"].shape == (32, 22)
        assert batch["actions"].shape == (32, 8)


class TestSACAgent:
    """Test SAC agent."""

    @pytest.fixture
    def agent(self):
        """Create SAC agent."""
        return SACAgent(
            state_dim=22,
            action_dim=8,
            hidden_dim=64,
            device="cpu",
        )

    def test_select_action(self, agent):
        """Test action selection."""
        state = np.random.randn(22)
        action = agent.select_action(state, eval_mode=False)

        assert action.shape == (8,)
        assert np.all(action >= -1) and np.all(action <= 1)

    def test_train_step(self, agent):
        """Test training step."""
        # Fill buffer
        for i in range(300):
            state = np.random.randn(22)
            action = np.random.randn(8)
            reward = np.random.rand()
            next_state = np.random.randn(22)
            done = False

            agent.replay_buffer.add(state, action, reward, next_state, done)

        # Train
        metrics = agent.train_step()

        assert "critic_loss" in metrics
        assert "actor_loss" in metrics


class TestRobotEnvironment:
    """Test robot environment."""

    @pytest.fixture
    def env(self):
        """Create environment."""
        return RobotEnvironment(task="pick_and_place")

    def test_reset(self, env):
        """Test environment reset."""
        state, info = env.reset()

        assert state.shape == (22,)
        assert "ee_position" in info

    def test_step(self, env):
        """Test environment step."""
        state, _ = env.reset()
        action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action)

        assert next_state.shape == (22,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
