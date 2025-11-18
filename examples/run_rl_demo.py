"""Quick RL training demo."""

import logging

from robo_vla import setup_logging
from robo_vla.rl import SACAgent, RobotEnvironment, RLTrainer


def main():
    """Run quick RL demo."""
    print("🤖 RoboVLA SAC Training Demo\n")

    # Setup
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    # Create environment
    logger.info("Creating robot environment...")
    env = RobotEnvironment(task="pick_and_place", render_mode="human")

    # Create SAC agent
    logger.info("Initializing SAC agent...")
    agent = SACAgent(
        state_dim=22,
        action_dim=8,
        hidden_dim=64,  # Smaller for demo
        buffer_size=10000,
        device="cpu",  # Use CPU for demo
        auto_entropy_tuning=True,
    )

    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        env=env,
        save_dir="./demo_checkpoints",
    )

    # Train for a few episodes
    logger.info("Training for 50 episodes (demo)...")
    metrics = trainer.train(
        num_episodes=50,
        eval_freq=10,
        save_freq=25,
        warmup_steps=100,
    )

    # Show results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Total episodes: 50")
    print(f"Best eval return: {trainer.best_eval_return:.2f}")
    print("\nCheckpoint saved to: ./demo_checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
