#!/usr/bin/env python3
"""Script to train VLA model."""

import argparse
import logging

from robo_vla import VLAModel, load_config, setup_logging
from robo_vla.training import create_dataloaders, VLATrainer


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train VLA model")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("Starting VLA training...")

    # Load config
    config = load_config(args.config)

    # Create dataloaders
    logger.info(f"Loading data from {args.data_dir}")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Create model
    logger.info("Initializing VLA model...")
    model = VLAModel(config, device=args.device)

    # Create trainer
    trainer = VLATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
    )

    # Train
    logger.info(f"Training for {args.epochs} epochs...")
    metrics = trainer.train(num_epochs=args.epochs)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    # TODO: Add test evaluation

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
