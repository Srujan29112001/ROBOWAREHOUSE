"""VLA model training pipeline."""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from robo_vla.vla_model import VLAModel

logger = logging.getLogger(__name__)


class VLATrainer:
    """
    Trainer for VLA model.

    Handles supervised learning from demonstration data.
    """

    def __init__(
        self,
        model: VLAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        save_dir: str = "./checkpoints",
        use_wandb: bool = False,
    ):
        """
        Initialize VLA trainer.

        Args:
            model: VLA model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            save_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["vla_model"]["training"]["learning_rate"],
            weight_decay=config["vla_model"]["training"]["weight_decay"],
        )

        # Setup scheduler
        total_steps = config["vla_model"]["training"]["max_steps"]
        warmup_steps = config["vla_model"]["training"]["warmup_steps"]

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
        )

        # Loss functions
        self.action_loss_fn = nn.MSELoss()
        self.success_loss_fn = nn.BCELoss()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        if use_wandb:
            wandb.init(
                project=config["monitoring"]["wandb"]["project"],
                entity=config["monitoring"]["wandb"]["entity"],
                config=config,
            )

        logger.info("VLA Trainer initialized")

    def train(self, num_epochs: int) -> Dict[str, list]:
        """
        Train model.

        Args:
            num_epochs: Number of training epochs

        Returns:
            Training metrics history
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_action_loss": [],
            "val_action_loss": [],
        }

        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            metrics_history["train_loss"].append(train_metrics["loss"])
            metrics_history["train_action_loss"].append(train_metrics["action_loss"])

            # Validation
            val_metrics = self.validate()
            metrics_history["val_loss"].append(val_metrics["loss"])
            metrics_history["val_action_loss"].append(val_metrics["action_loss"])

            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pth")
                logger.info(f"New best model saved! Val loss: {val_metrics['loss']:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/action_loss": train_metrics["action_loss"],
                    "val/loss": val_metrics["loss"],
                    "val/action_loss": val_metrics["action_loss"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                })

        logger.info("Training completed!")
        return metrics_history

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_action_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            rgb = batch["rgb"].to(self.model.device)
            state = batch["state"].to(self.model.device)
            action_gt = batch["action"].to(self.model.device)
            commands = batch["commands"]

            # Forward pass
            output = self.model(
                rgb_images=rgb,
                text_commands=commands,
                robot_state=state,
                trajectory_steps=1,  # Single-step for supervised learning
            )

            # Compute loss
            action_pred = torch.cat([
                output["joints"][:, 0, :],
                output["gripper"][:, 0, :],
            ], dim=-1)

            action_loss = self.action_loss_fn(action_pred, action_gt)

            # Total loss
            loss = action_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": total_loss / num_batches,
                "action_loss": total_action_loss / num_batches,
            })

            self.global_step += 1

        return {
            "loss": total_loss / num_batches,
            "action_loss": total_action_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        total_loss = 0
        total_action_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            rgb = batch["rgb"].to(self.model.device)
            state = batch["state"].to(self.model.device)
            action_gt = batch["action"].to(self.model.device)
            commands = batch["commands"]

            # Forward pass
            output = self.model(
                rgb_images=rgb,
                text_commands=commands,
                robot_state=state,
                trajectory_steps=1,
            )

            # Compute loss
            action_pred = torch.cat([
                output["joints"][:, 0, :],
                output["gripper"][:, 0, :],
            ], dim=-1)

            action_loss = self.action_loss_fn(action_pred, action_gt)
            loss = action_loss

            # Update metrics
            total_loss += loss.item()
            total_action_loss += action_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "action_loss": total_action_loss / num_batches,
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        filepath = self.save_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filename: str) -> None:
        """Load training checkpoint."""
        filepath = self.save_dir / filename

        checkpoint = torch.load(filepath, map_location=self.model.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Checkpoint loaded: {filepath}")
