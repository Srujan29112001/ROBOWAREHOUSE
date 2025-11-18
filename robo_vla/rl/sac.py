"""Soft Actor-Critic (SAC) reinforcement learning implementation."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for SAC.

    Stores transitions and samples random batches for training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int = 100000,
        device: str = "cuda",
    ):
        """
        Initialize replay buffer.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            buffer_size: Maximum buffer size
            device: Device to store tensors
        """
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

        # Preallocate memory
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        logger.info(f"Replay buffer initialized with size {buffer_size}")

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.states[self.ptr] = torch.from_numpy(state).float()
        self.actions[self.ptr] = torch.from_numpy(action).float()
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.from_numpy(next_state).float()
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from buffer."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size


class ActorNetwork(nn.Module):
    """SAC actor network (policy)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get action distribution."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample

        # Tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds (tanh correction)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class CriticNetwork(nn.Module):
    """SAC critic network (Q-function)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q-networks."""
        sa = torch.cat([state, action], 1)

        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2


class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.

    Implements SAC algorithm with automatic entropy tuning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        buffer_size: int = 100000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        learning_rate: float = 3e-4,
        device: str = "cuda",
        auto_entropy_tuning: bool = True,
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            buffer_size: Replay buffer size
            batch_size: Training batch size
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy coefficient
            learning_rate: Learning rate
            device: Device to run on
            auto_entropy_tuning: Whether to automatically tune entropy
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)

        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)

        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = Adam([self.log_alpha], lr=learning_rate)
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)

        logger.info(f"SAC agent initialized with state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Select action from policy.

        Args:
            state: Current state
            eval_mode: If True, use mean action (deterministic)

        Returns:
            Action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if eval_mode:
                # Deterministic action
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                # Stochastic action
                action, _ = self.actor.sample(state_tensor)

        return action.cpu().numpy().flatten()

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (entropy coefficient)
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            "alpha": self.alpha.item() if self.auto_entropy_tuning else self.alpha,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, path: str) -> None:
        """Save agent."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy_tuning else None,
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.auto_entropy_tuning and checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp()
        logger.info(f"Agent loaded from {path}")
