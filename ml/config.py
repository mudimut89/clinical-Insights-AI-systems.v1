from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    project_root: Path
    dataset_slug: str = "imrankhan77/autistic-children-facial-data-set"
    dataset_dirname: str = "data"
    artifacts_dirname: str = "artifacts"

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42

    @property
    def dataset_dir(self) -> Path:
        return self.project_root / "ml" / self.dataset_dirname

    @property
    def raw_dir(self) -> Path:
        return self.dataset_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.dataset_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "ml" / self.artifacts_dirname
