
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..sequence import IGNORE_LABEL


@dataclass
class LlmCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 64
    max_sequence_length: int | None = None

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("Empty batch")
        lengths = [int(item["input_ids"].size) for item in items]
        target = max(lengths)
        if self.max_sequence_length is not None and target > self.max_sequence_length:


            raise ValueError(
                f"The longest sequence in the batch {target} exceeds max_sequence_length={self.max_sequence_length}"
            )
        if self.pad_to_multiple_of > 1:
            multiple = self.pad_to_multiple_of
            target = ((target + multiple - 1) // multiple) * multiple

        batch_size = len(items)
        input_ids = np.full((batch_size, target), self.pad_token_id, dtype=np.int64)
        labels = np.full((batch_size, target), IGNORE_LABEL, dtype=np.int64)
        has_constraints = ["constraint_kinds" in item for item in items]
        if any(has_constraints) and not all(has_constraints):
            raise ValueError("same as batch cannot be mixed with/without constraint_kinds ")
        constraint_kinds = (
            np.full((batch_size, target), -1, dtype=np.int16)
            if all(has_constraints)
            else None
        )
        attention_mask = np.zeros((batch_size, target), dtype=np.int64)

        for row, item in enumerate(items):
            length = int(item["input_ids"].size)
            input_ids[row, :length] = item["input_ids"]
            labels[row, :length] = item["labels"]
            if constraint_kinds is not None:
                constraint_kinds[row, :length] = item["constraint_kinds"]
            attention_mask[row, :length] = 1

        batch = {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(attention_mask),
            "sample_ids": [item["sample_id"] for item in items],
            "modes": [item["mode"] for item in items],

            "languages": [item.get("language", "unknown") for item in items],
            "quality_buckets": [item.get("quality_bucket", "unknown") for item in items],
            "num_semantic_tokens": torch.tensor(
                [int(item["num_semantic_tokens"]) for item in items], dtype=torch.long
            ),
            "num_melody_tokens": torch.tensor(
                [int(item["num_melody_tokens"]) for item in items], dtype=torch.long
            ),
            "num_condition_tokens": torch.tensor(
                [int(item.get("num_condition_tokens", 0)) for item in items], dtype=torch.long
            ),
            "condition_truncated": torch.tensor(
                [bool(item.get("condition_truncated", False)) for item in items],
                dtype=torch.bool,
            ),
            "semantic_anchor_sections": torch.tensor(
                [int(item.get("semantic_anchor_sections", 0)) for item in items],
                dtype=torch.long,
            ),
            "semantic_anchor_tokens": torch.tensor(
                [int(item.get("semantic_anchor_tokens", 0)) for item in items],
                dtype=torch.long,
            ),
            "semantic_anchor_text_truncated": torch.tensor(
                [
                    int(item.get("semantic_anchor_text_truncated", 0))
                    for item in items
                ],
                dtype=torch.long,
            ),
            "is_instrumental": torch.tensor(
                [bool(item.get("is_instrumental", False)) for item in items], dtype=torch.bool
            ),
            "padded_tokens": int(target * batch_size),
            "real_tokens": int(sum(lengths)),
        }
        if constraint_kinds is not None:
            batch["constraint_kinds"] = torch.from_numpy(constraint_kinds)
        return batch
