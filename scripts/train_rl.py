#!/usr/bin/env python3
"""Script to train RL agent."""

import argparse
import logging

from robo_vla import load_config, setup_logging
from robo_vla.rl import SACAgent, RobotEnvironment, RLTrainer


def main():
    """Main RL training script."""
    parser = argparse.ArgumentParser(description="Train SAC agent")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--save-dir", type=str, default="./rl_checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("Starting SAC training...")

    # Load config
    config = load_config(args.config)
    rl_config = config["rl"]

    # Create environment
    logger.info("Creating robot environment...")
    env = RobotEnvironment(task="pick_and_place")
    eval_env = RobotEnvironment(task="pick_and_place")

    # Create agent
    logger.info("Initializing SAC agent...")
    agent = SACAgent(
        state_dim=rl_config["state_dim"],
        action_dim=rl_config["action_dim"],
        hidden_dim=rl_config["hidden_dim"],
        buffer_size=rl_config["buffer_size"],
        batch_size=rl_config["batch_size"],
        gamma=rl_config["gamma"],
        tau=rl_config["tau"],
        alpha=rl_config["alpha"],
        learning_rate=rl_config["learning_rate"],
        device=args.device,
    )

    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        save_dir=args.save_dir,
    )

    # Train
    logger.info(f"Training for {args.episodes} episodes...")
    metrics = trainer.train(
        num_episodes=args.episodes,
        eval_freq=10,
        save_freq=50,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
