"""
Parameter-Efficient Fine-Tuning (PEFT/LoRA) Configuration for 1B-3B SLMs.
Configured for Qwen2.5-1.5B/3B, Llama-3.2-1B/3B, and SmolLM2 architectures.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import os


@dataclass
class SLMTrainingConfig:
    # Model parameters
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_family: str = "qwen"  # "qwen", "llama", "smollm"
    max_seq_length: int = 2048
    
    # PEFT / LoRA hyper-parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Training hyper-parameters
    output_dir: str = "checkpoints/cgl-slm-lora"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    fp16: bool = False
    bf16: bool = False  # Set to True on MPS or modern CUDA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_peft_config(self) -> Any:
        """Constructs Hugging Face LoraConfig object."""
        try:
            from peft import LoraConfig, TaskType
            return LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type=TaskType.CAUSAL_LM,
            )
        except ImportError:
            # Fallback representation if running in minimal test environment
            return {
                "r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_modules": self.target_modules,
                "lora_dropout": self.lora_dropout,
                "bias": self.bias,
                "task_type": self.task_type,
            }
