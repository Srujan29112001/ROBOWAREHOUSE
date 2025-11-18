"""Language grounding module integrating LLM and CLIP."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from robo_vla.language.llama import LlamaLanguageModel
from robo_vla.language.clip_alignment import CLIPAlignment

logger = logging.getLogger(__name__)


class LanguageGroundingModule(nn.Module):
    """
    Unified language grounding combining Llama and CLIP.

    Processes natural language commands and grounds them to visual observations.
    """

    def __init__(
        self,
        config: Dict,
        device: str = "cuda",
    ):
        """
        Initialize language grounding module.

        Args:
            config: Configuration dictionary
            device: Device to run on
        """
        super().__init__()
        self.device = device
        self.config = config

        logger.info("Initializing Language Grounding Module...")

        # Initialize Llama
        self.llama = LlamaLanguageModel(
            model_name=config["language"]["llama"]["model_name"],
            device=device,
            load_in_4bit=config["language"]["llama"]["quantization"]["load_in_4bit"],
            lora_config=config["language"]["llama"]["lora"],
        )

        # Initialize CLIP
        self.clip = CLIPAlignment(
            model_name=config["language"]["clip"]["model_name"],
            embed_dim=config["language"]["clip"]["embed_dim"],
            device=device,
            quantization=config["language"]["clip"]["quantization"],
        )

        # Feature fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 4096, 1024),  # CLIP + Llama
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
        ).to(device)

        logger.info("Language Grounding Module initialized")

    @torch.no_grad()
    def process_command(
        self,
        command: str,
        visual_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Process natural language command.

        Args:
            command: Natural language command string
            visual_features: Optional pre-computed visual features
            images: Optional raw images

        Returns:
            Dictionary containing:
                - parsed: Parsed command structure
                - text_embeddings: Text embeddings
                - grounded_features: Visually grounded features (if images provided)
                - intent: Extracted intent
                - entities: Extracted entities
        """
        logger.debug(f"Processing command: {command}")

        # Parse command with Llama
        parsed = self.llama.parse_robot_command(command)

        # Get text embeddings from Llama
        text_embeddings_llama = self.llama.get_embeddings([command])  # [1, 4096]

        # Get text embeddings from CLIP
        text_embeddings_clip = self.clip.encode_text([command])  # [1, 512]

        # Combine embeddings
        combined_embeddings = torch.cat([
            text_embeddings_clip,
            text_embeddings_llama,
        ], dim=-1)  # [1, 4608]

        # Fuse
        fused_embeddings = self.fusion_layer(combined_embeddings)  # [1, 512]

        result = {
            "parsed": parsed,
            "text_embeddings": fused_embeddings,
            "text_embeddings_llama": text_embeddings_llama,
            "text_embeddings_clip": text_embeddings_clip,
            "intent": parsed.get("intent"),
            "entities": {
                "object": parsed.get("object"),
                "source": parsed.get("source"),
                "target": parsed.get("target"),
            },
        }

        # Visual grounding if images provided
        if images is not None:
            grounding = self.ground_to_visual(
                text_embeddings=fused_embeddings,
                images=images,
                visual_features=visual_features,
            )
            result["grounding"] = grounding

        return result

    def ground_to_visual(
        self,
        text_embeddings: torch.Tensor,
        images: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Ground text to visual observations.

        Args:
            text_embeddings: Text embeddings [B, text_dim]
            images: Visual images [B, C, H, W]
            visual_features: Optional pre-computed visual features

        Returns:
            Grounding results
        """
        # Encode images with CLIP
        image_embeddings = self.clip.encode_images(images)  # [B, 512]

        # Align
        alignment = self.clip.align_vision_language(
            image_embeddings,
            text_embeddings[:, :512],  # Use CLIP portion
        )

        return {
            "aligned_features": alignment["aligned_features"],
            "similarity_scores": alignment["visual_scores"],
            "best_matches": alignment["visual_to_text"],
        }

    def generate_explanation(
        self,
        command: str,
        visual_observations: Optional[List[str]] = None,
    ) -> str:
        """
        Generate natural language explanation of command understanding.

        Args:
            command: Input command
            visual_observations: Optional list of observed objects

        Returns:
            Explanation string
        """
        parsed = self.llama.parse_robot_command(command)

        prompt = f"""Given the robot command: "{command}"

Parsed understanding:
- Intent: {parsed.get('intent', 'Unknown')}
- Object: {parsed.get('object', 'Not specified')}
- Source: {parsed.get('source', 'Not specified')}
- Target: {parsed.get('target', 'Not specified')}

{f"Observed objects: {', '.join(visual_observations)}" if visual_observations else ""}

Provide a brief explanation of what the robot should do."""

        explanation = self.llama.generate(
            [prompt],
            temperature=0.5,
            max_length=256,
        )[0]

        return explanation

    def suggest_alternatives(
        self,
        command: str,
        num_suggestions: int = 3,
    ) -> List[str]:
        """
        Suggest alternative ways to phrase the command.

        Args:
            command: Original command
            num_suggestions: Number of suggestions

        Returns:
            List of alternative phrasings
        """
        prompt = f"""Given the robot command: "{command}"

Generate {num_suggestions} alternative ways to phrase this command.
Each alternative should have the same intent but different wording.

Alternatives:"""

        response = self.llama.generate(
            [prompt],
            temperature=0.8,
            max_length=256,
        )[0]

        # Parse alternatives (simplified)
        alternatives = [
            line.strip() for line in response.split("\n")
            if line.strip() and len(line.strip()) > 10
        ][:num_suggestions]

        return alternatives

    def validate_command(
        self,
        command: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if command is valid and executable.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        parsed = self.llama.parse_robot_command(command)

        # Check intent
        if parsed.get("intent") == "UNKNOWN":
            return False, "Could not understand the intent of the command"

        # Check object
        if not parsed.get("object"):
            return False, "No object specified in the command"

        # Check for ambiguity
        if "or" in command.lower() or "maybe" in command.lower():
            return False, "Command is ambiguous, please be more specific"

        return True, None

    def batch_process(
        self,
        commands: List[str],
        images: Optional[torch.Tensor] = None,
    ) -> List[Dict]:
        """
        Process batch of commands.

        Args:
            commands: List of commands
            images: Optional batch of images

        Returns:
            List of processing results
        """
        results = []

        for i, command in enumerate(commands):
            image = images[i:i+1] if images is not None else None
            result = self.process_command(command, images=image)
            results.append(result)

        return results
