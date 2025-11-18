"""Llama 3.1 8B language model with QLoRA fine-tuning."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__name__)


class LlamaLanguageModel(nn.Module):
    """
    Llama 3.1 8B with QLoRA for robot language understanding.

    Uses 4-bit quantization and LoRA adapters for efficient fine-tuning
    on robot manipulation tasks.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        lora_config: Optional[Dict] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Llama language model.

        Args:
            model_name: Hugging Face model name
            device: Device to run model on
            load_in_4bit: Whether to use 4-bit quantization
            lora_config: LoRA configuration
            cache_dir: Cache directory for model weights
        """
        super().__init__()
        self.device = device
        self.model_name = model_name

        logger.info(f"Loading Llama model: {model_name}")

        # Quantization configuration
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA
        if lora_config:
            self._apply_lora(lora_config)

        self.model.eval()

        logger.info(f"Llama model loaded successfully")
        logger.info(f"Model size: {self._get_model_size():.2f} GB")

    def _apply_lora(self, lora_config: Dict) -> None:
        """Apply LoRA adapters to model."""
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
        )

        self.model = get_peft_model(self.model, peft_config)
        logger.info("LoRA adapters applied")
        logger.info(f"Trainable parameters: {self._get_trainable_params()}")

    def _get_model_size(self) -> float:
        """Get model size in GB."""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_gb = (param_size + buffer_size) / 1024**3
        return size_gb

    def _get_trainable_params(self) -> str:
        """Get trainable parameter count."""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return (
            f"trainable: {trainable_params:,} || "
            f"all: {all_param:,} || "
            f"trainable %: {100 * trainable_params / all_param:.2f}"
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate text from prompts.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            List of generated texts
        """
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )

        return generated_texts

    @torch.no_grad()
    def get_embeddings(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Get text embeddings.

        Args:
            texts: List of input texts

        Returns:
            Embeddings [B, hidden_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Forward pass
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]  # [B, L, D]

        # Pool (use last token)
        embeddings = hidden_states[:, -1, :]  # [B, D]

        return embeddings

    def parse_robot_command(
        self,
        command: str,
    ) -> Dict[str, any]:
        """
        Parse natural language command into structured format.

        Args:
            command: Natural language command

        Returns:
            Dictionary with intent and entities
        """
        # Create parsing prompt
        prompt = f"""Parse the following robot command into structured format.

Command: {command}

Extract:
1. Intent (PICK, PLACE, MOVE, GRASP, etc.)
2. Object (what to manipulate)
3. Source location (where to get it from)
4. Target location (where to put it)
5. Constraints (any special requirements)

Format your response as JSON."""

        # Generate
        response = self.generate([prompt], temperature=0.3)[0]

        # Extract JSON (simplified - would use proper parsing in production)
        try:
            import json
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(response[start:end])
                return parsed
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")

        # Fallback to simple parsing
        return {
            "intent": self._extract_intent(command),
            "object": self._extract_object(command),
            "source": None,
            "target": None,
            "constraints": {},
        }

    def _extract_intent(self, command: str) -> str:
        """Extract intent from command."""
        command_lower = command.lower()

        intent_keywords = {
            "PICK": ["pick", "grab", "take", "get"],
            "PLACE": ["place", "put", "set", "drop"],
            "MOVE": ["move", "transfer", "bring"],
            "GRASP": ["grasp", "hold", "grip"],
            "RELEASE": ["release", "let go", "drop"],
        }

        for intent, keywords in intent_keywords.items():
            if any(kw in command_lower for kw in keywords):
                return intent

        return "UNKNOWN"

    def _extract_object(self, command: str) -> Optional[str]:
        """Extract object from command (simplified)."""
        # Look for common patterns
        import re

        patterns = [
            r"the (\w+\s+\w+)",  # "the red box"
            r"a (\w+\s+\w+)",    # "a blue cube"
            r"(\w+\s+\w+) from", # "red box from"
        ]

        for pattern in patterns:
            match = re.search(pattern, command.lower())
            if match:
                return match.group(1)

        return None

    def save_lora_adapter(self, output_path: str) -> None:
        """Save LoRA adapter weights."""
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(output_path)
            logger.info(f"LoRA adapter saved to {output_path}")
        else:
            logger.warning("Model does not have LoRA adapters")

    def load_lora_adapter(self, adapter_path: str) -> None:
        """Load LoRA adapter weights."""
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )
        logger.info(f"LoRA adapter loaded from {adapter_path}")
