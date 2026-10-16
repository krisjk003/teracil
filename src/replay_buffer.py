"""
Class-Balanced Experience Replay Buffer for Continual Learning.

Maintains a fixed total capacity (default 500 samples) and dynamically rebalances
exemplars evenly across all observed classes to mitigate catastrophic forgetting.
Only training set images are ever stored in the buffer.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


class ReplayBuffer:
    """
    Deterministic, class-balanced experience replay buffer.
    
    Attributes:
        capacity (int): Maximum total number of exemplars stored across all classes.
        seed (int): Random seed for deterministic exemplar selection.
        exemplars (Dict[int, List[str]]): Mapping from class_idx to list of image paths.
    """

    def __init__(self, capacity: int = 500, seed: int = 42):
        assert capacity > 0, "Replay buffer capacity must be strictly positive"
        self.capacity = capacity
        self.seed = seed
        self.exemplars: Dict[int, List[str]] = {}

    def update_with_class(self, class_label: int, candidate_paths: List[str]) -> None:
        """
        Add candidate training paths for a newly observed class and rebalance
        all stored classes to guarantee equal representation within capacity.
        """
        assert len(candidate_paths) > 0, f"Cannot update replay buffer with empty candidates for class {class_label}"
        
        # Register new class candidates deterministically
        rng = random.Random(self.seed + class_label * 13)
        shuffled = list(candidate_paths)
        rng.shuffle(shuffled)
        self.exemplars[class_label] = shuffled

        # Dynamic rebalancing across all registered classes
        num_classes = len(self.exemplars)
        per_class_quota = self.capacity // num_classes
        remainder = self.capacity % num_classes

        # Assign quotas
        for idx, (cls_lbl, paths) in enumerate(self.exemplars.items()):
            # Distribute remainder to first classes deterministically
            quota = per_class_quota + (1 if idx < remainder else 0)
            self.exemplars[cls_lbl] = paths[:quota]

        # Verify capacity invariant
        total_stored = sum(len(p) for p in self.exemplars.values())
        assert total_stored <= self.capacity, f"Buffer exceeded capacity! {total_stored} > {self.capacity}"

    def get_replay_records(self) -> List[Tuple[str, int]]:
        """Return all stored exemplars as a list of (image_path, class_label) tuples."""
        records: List[Tuple[str, int]] = []
        for cls_lbl, paths in sorted(self.exemplars.items()):
            for p in paths:
                records.append((p, cls_lbl))
        return records

    def get_class_counts(self) -> Dict[int, int]:
        """Return the count of exemplars stored per class."""
        return {cls_lbl: len(paths) for cls_lbl, paths in self.exemplars.items()}

    def save(self, filepath: str) -> None:
        """Save replay buffer state to disk as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "capacity": self.capacity,
            "seed": self.seed,
            "exemplars": {str(k): v for k, v in self.exemplars.items()},
            "class_counts": self.get_class_counts(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load replay buffer state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Replay buffer checkpoint not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.capacity = data["capacity"]
        self.seed = data["seed"]
        self.exemplars = {int(k): v for k, v in data["exemplars"].items()}
