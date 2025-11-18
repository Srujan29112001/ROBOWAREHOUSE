"""Training CLI for RoboVLA."""

import argparse
import sys
from pathlib import Path

from robo_vla import VLAModel, load_config, setup_logging
from robo_vla.training import Trainer, RobotDataset


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train RoboVLA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to training data directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Path to output directory for checkpoints",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (overrides config)",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(log_level="INFO")

    # Override config with CLI arguments
    if args.batch_size is not None:
        config["vla_model"]["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["vla_model"]["training"]["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["system"]["device"] = args.device

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ROBOVLA TRAINING")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {config['system']['device']}")
    print(f"Batch size: {config['vla_model']['training']['batch_size']}")
    print(f"Learning rate: {config['vla_model']['training']['learning_rate']}")
    print(f"Epochs: {args.epochs}")
    print(f"Weights & Biases: {args.wandb}")
    print("="*60)

    # Initialize model
    print("\nInitializing model...")
    model = VLAModel(config, device=config["system"]["device"])

    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        model.load_state_dict(torch.load(args.resume))

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )

    if args.eval_only:
        print("\nRunning evaluation...")
        results = trainer.evaluate()
        print("\nEvaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("\nStarting training...")
        trainer.train(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
        )
        print("\nTraining complete!")

    return 0


if __name__ == "__main__":
    import torch
    sys.exit(main())
