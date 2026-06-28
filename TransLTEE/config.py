"""Experiment configuration for TransLTEE."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Model, optimization, and evaluation settings."""

    dataset: str = "ihdp"
    repetition: int = 1
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 1e-6
    weight_decay: float = 1e-6
    sequence_length: int = 100
    history_length: int = 50
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dff: int = 128
    dropout_rate: float = 0.1
    context_tokens: int = 4
    sinkhorn_iterations: int = 20
    sinkhorn_epsilon: float = 0.1
    data_dir: Path = Path("data")
    output_dir: Path = Path("results")
    max_samples: int | None = None

    def validate(self) -> None:
        if self.dataset not in {"ihdp", "news"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if not 1 <= self.repetition <= 10:
            raise ValueError("repetition must be between 1 and 10")
        if self.sequence_length < 2 or self.sequence_length > 100:
            raise ValueError("sequence_length must be between 2 and 100")
        if not 1 <= self.history_length < self.sequence_length:
            raise ValueError("history_length must be smaller than sequence_length")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_dir"] = str(self.data_dir)
        result["output_dir"] = str(self.output_dir)
        return result
