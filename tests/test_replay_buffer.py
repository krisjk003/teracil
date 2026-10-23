"""
Unit Tests for Experience Replay Buffer.
"""

from pathlib import Path
import pytest

from src.replay_buffer import ReplayBuffer


def test_replay_buffer_capacity_and_balancing(tmp_path):
    """Test deterministic rebalancing and total capacity invariance."""
    capacity = 300
    buf = ReplayBuffer(capacity=capacity, seed=42)

    # Stage 1: Add 2 classes with 1000 candidates each (Forest: 0, River: 1)
    cands_0 = [f"forest_{i}.jpg" for i in range(1000)]
    cands_1 = [f"river_{i}.jpg" for i in range(1000)]

    buf.update_with_class(0, cands_0)
    buf.update_with_class(1, cands_1)

    counts = buf.get_class_counts()
    assert sum(counts.values()) == capacity
    # 300 / 2 = 150 per class
    assert counts[0] == 150
    assert counts[1] == 150

    # Stage 2: Add 3rd class (Highway: 2)
    cands_2 = [f"highway_{i}.jpg" for i in range(1000)]
    buf.update_with_class(2, cands_2)

    counts = buf.get_class_counts()
    assert sum(counts.values()) == capacity
    # 300 / 3 = 100 per class
    assert counts[0] == 100
    assert counts[1] == 100
    assert counts[2] == 100

    # Stage 3: Add 4th class (Residential: 3)
    cands_3 = [f"residential_{i}.jpg" for i in range(1000)]
    buf.update_with_class(3, cands_3)

    counts = buf.get_class_counts()
    assert sum(counts.values()) == capacity
    # 300 / 4 = 75 per class
    assert counts[0] == 75
    assert counts[1] == 75
    assert counts[2] == 75
    assert counts[3] == 75



def test_replay_buffer_serialization(tmp_path):
    """Test save and load reproducibility."""
    buf1 = ReplayBuffer(capacity=100, seed=42)
    buf1.update_with_class(0, [f"img_{i}.jpg" for i in range(200)])
    buf1.update_with_class(1, [f"img_{i}.jpg" for i in range(200)])

    save_file = tmp_path / "replay.json"
    buf1.save(str(save_file))

    buf2 = ReplayBuffer(capacity=100, seed=42)
    buf2.load(str(save_file))

    assert buf1.capacity == buf2.capacity
    assert buf1.get_class_counts() == buf2.get_class_counts()
    assert buf1.get_replay_records() == buf2.get_replay_records()
