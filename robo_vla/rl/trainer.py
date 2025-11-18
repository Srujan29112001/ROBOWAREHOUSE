"""RL training loop for SAC agent."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from robo_vla.rl.sac import SACAgent
from robo_vla.rl.robot_env import RobotEnvironment

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trainer for SAC agent in robot environment.

    Handles training loop, evaluation, and checkpointing.
    """

    def __init__(
        self,
        agent: SACAgent,
        env: RobotEnvironment,
        eval_env: Optional[RobotEnvironment] = None,
        save_dir: str = "./checkpoints",
    ):
        """
        Initialize RL trainer.

        Args:
            agent: SAC agent
            env: Training environment
            eval_env: Evaluation environment
            save_dir: Directory to save checkpoints
        """
        self.agent = agent
        self.env = env
        self.eval_env = eval_env or RobotEnvironment()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.episode = 0
        self.total_steps = 0
        self.best_eval_return = -np.inf

        logger.info("RL Trainer initialized")

    def train(
        self,
        num_episodes: int = 1000,
        eval_freq: int = 10,
        save_freq: int = 50,
        warmup_steps: int = 1000,
        update_freq: int = 1,
    ) -> Dict[str, list]:
        """
        Train agent.

        Args:
            num_episodes: Number of training episodes
            eval_freq: Evaluate every N episodes
            save_freq: Save checkpoint every N episodes
            warmup_steps: Random exploration steps before training
            update_freq: Update agent every N steps

        Returns:
            Training metrics history
        """
        logger.info(f"Starting training for {num_episodes} episodes")

        metrics_history = {
            "episode_returns": [],
            "episode_lengths": [],
            "eval_returns": [],
            "critic_losses": [],
            "actor_losses": [],
        }

        for episode in tqdm(range(num_episodes), desc="Training"):
            self.episode = episode
            episode_return = 0
            episode_length = 0

            state, _ = self.env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                # Select action
                if self.total_steps < warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state, eval_mode=False)

                # Execute action
                next_state, reward, done, truncated, info = self.env.step(action)

                # Store transition
                self.agent.replay_buffer.add(
                    state, action, reward, next_state, done or truncated
                )

                # Update agent
                if self.total_steps > warmup_steps and self.total_steps % update_freq == 0:
                    train_metrics = self.agent.train_step()
                    if train_metrics:
                        metrics_history["critic_losses"].append(train_metrics["critic_loss"])
                        metrics_history["actor_losses"].append(train_metrics["actor_loss"])

                episode_return += reward
                episode_length += 1
                self.total_steps += 1
                state = next_state

            # Log episode metrics
            metrics_history["episode_returns"].append(episode_return)
            metrics_history["episode_lengths"].append(episode_length)

            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"Return={episode_return:.2f}, "
                    f"Length={episode_length}, "
                    f"Success={info.get('success', False)}"
                )

            # Evaluate
            if episode % eval_freq == 0 and episode > 0:
                eval_return = self.evaluate(num_episodes=5)
                metrics_history["eval_returns"].append(eval_return)

                logger.info(f"Evaluation return: {eval_return:.2f}")

                # Save best model
                if eval_return > self.best_eval_return:
                    self.best_eval_return = eval_return
                    self.save_checkpoint("best_model.pth")
                    logger.info(f"New best model saved! Return: {eval_return:.2f}")

            # Save periodic checkpoint
            if episode % save_freq == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_{episode}.pth")

        logger.info("Training completed!")
        return metrics_history

    def evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate agent.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Mean evaluation return
        """
        returns = []
        successes = []

        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            episode_return = 0
            done = False
            truncated = False

            while not (done or truncated):
                action = self.agent.select_action(state, eval_mode=True)
                next_state, reward, done, truncated, info = self.eval_env.step(action)
                episode_return += reward
                state = next_state

            returns.append(episode_return)
            successes.append(info.get("success", False))

        mean_return = np.mean(returns)
        success_rate = np.mean(successes)

        logger.info(f"Eval: Mean return={mean_return:.2f}, Success rate={success_rate:.1%}")

        return mean_return

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        filepath = self.save_dir / filename
        self.agent.save(str(filepath))
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        filepath = self.save_dir / filename
        self.agent.load(str(filepath))
        logger.info(f"Checkpoint loaded: {filepath}")
