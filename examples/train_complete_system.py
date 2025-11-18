"""Complete training example for RoboVLA system."""

import logging

from robo_vla import VLAModel, load_config, setup_logging
from robo_vla.rl import SACAgent, RobotEnvironment, RLTrainer
from robo_vla.training import create_dataloaders, VLATrainer


def train_supervised():
    """Train VLA model with supervised learning."""
    print("=" * 60)
    print("PHASE 1: Supervised Learning from Demonstrations")
    print("=" * 60)

    # Setup
    config = load_config()
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    # Create dummy data directory structure
    logger.info("Note: Create data directory with structure:")
    logger.info("  data/episodes/episode_0000/rgb_000.png, action_000.npy, etc.")

    # Create dataloaders (will fail if no data, but shows how to use)
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir="./data/episodes",
            batch_size=16,
        )

        # Create model
        model = VLAModel(config, device="cuda")

        # Create trainer
        trainer = VLATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            save_dir="./checkpoints/supervised",
            use_wandb=False,
        )

        # Train
        logger.info("Starting supervised training...")
        trainer.train(num_epochs=100)

        logger.info("✓ Supervised training complete!")

    except FileNotFoundError:
        logger.warning("Demo data not found. Skipping supervised training.")
        logger.info("To train, prepare data in: ./data/episodes/")


def train_reinforcement_learning():
    """Train with reinforcement learning."""
    print("\n" + "=" * 60)
    print("PHASE 2: Reinforcement Learning for Grasp Optimization")
    print("=" * 60)

    # Setup
    config = load_config()
    logger = logging.getLogger(__name__)

    # Create environment
    logger.info("Creating robot simulation environment...")
    env = RobotEnvironment(task="pick_and_place", render_mode=None)
    eval_env = RobotEnvironment(task="pick_and_place")

    # Create SAC agent
    logger.info("Initializing SAC agent...")
    agent = SACAgent(
        state_dim=22,  # Robot state + object + target
        action_dim=8,  # 7 joints + gripper
        hidden_dim=256,
        buffer_size=100000,
        device="cuda",
    )

    # Create trainer
    trainer = RLTrainer(
        agent=agent,
        env=env,
        eval_env=eval_env,
        save_dir="./checkpoints/rl",
    )

    # Train
    logger.info("Starting RL training (this may take hours)...")
    metrics = trainer.train(
        num_episodes=1000,
        eval_freq=10,
        save_freq=50,
        warmup_steps=1000,
    )

    logger.info("✓ RL training complete!")
    logger.info(f"Best eval return: {trainer.best_eval_return:.2f}")


def main():
    """Main training pipeline."""
    print("\n" + "🤖" * 30)
    print("COMPLETE ROBO-VLA TRAINING PIPELINE")
    print("🤖" * 30 + "\n")

    # Phase 1: Supervised learning (if data available)
    train_supervised()

    # Phase 2: Reinforcement learning
    train_reinforcement_learning()

    print("\n" + "=" * 60)
    print("✅ COMPLETE TRAINING PIPELINE FINISHED!")
    print("=" * 60)
    print("\nModels saved to:")
    print("  - Supervised: ./checkpoints/supervised/")
    print("  - RL: ./checkpoints/rl/")
    print("\nNext steps:")
    print("  1. Evaluate models on test data")
    print("  2. Deploy to production with Docker/Kubernetes")
    print("  3. Fine-tune on real robot hardware")


if __name__ == "__main__":
    main()
