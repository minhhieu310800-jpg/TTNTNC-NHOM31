"""Training package for Caro 11x11 AI.

Includes data generation, dataset management, deduplication, and training pipelines.
"""

from training.dataset import (
    CaroDataset,
    GameRecord,
    TrainingSample,
    apply_symmetry,
    canonical_form,
    deduplicate_samples,
    get_all_symmetries,
    load_dataset,
    save_dataset,
    split_games,
    transform_coords,
)

__all__ = [
    "transform_coords",
    "apply_symmetry",
    "get_all_symmetries",
    "canonical_form",
    "TrainingSample",
    "GameRecord",
    "deduplicate_samples",
    "split_games",
    "save_dataset",
    "load_dataset",
    "CaroDataset",
]
