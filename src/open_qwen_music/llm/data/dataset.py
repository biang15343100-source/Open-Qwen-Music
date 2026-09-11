
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from ..common.staging import stage_corpus_dir
from ..condition import ConditionRenderConfig
from ..contracts import SequenceMode
from ..grammar import GrammarConfig
from ..registry import TokenRegistry
from ..sequence import BuiltSequence, SequenceBuilder, SequenceConfig, SequenceTooLongError
from .index import ManifestIndex, build_index
from .shards import CorpusReader, TokenSpan


class LlmSampleDataset(Dataset):

    def __init__(
        self,
        corpus_root: str | Path,
        registry: TokenRegistry,
        text_encoder: Any,
        *,
        sequence_config: SequenceConfig | None = None,
        split: str | None = "train",
        seed: int = 0,
        build_missing_index: bool = True,
        strict_metadata: bool = False,
        local_cache_dir: str | Path | None = None,
        condition_config: ConditionRenderConfig | None = None,
        grammar_config: GrammarConfig | None = None,
    ) -> None:
        self.corpus_root = stage_corpus_dir(
            corpus_root,
            cache_root=local_cache_dir,
        )
        self.registry = registry
        self.text_encoder = text_encoder
        self.sequence_config = sequence_config or SequenceConfig()
        self.condition_config = condition_config or ConditionRenderConfig()
        self.grammar_config = grammar_config
        self.split = split
        self.seed = seed

        self.corpus = CorpusReader(self.corpus_root)
        manifest = self.corpus.manifest_path
        if build_missing_index:


            build_index(
                manifest,
                text_encoder=text_encoder,
                strict_metadata=bool(strict_metadata),
                condition_config=self.condition_config,
            )
        self.index = ManifestIndex(manifest)

        if split is None:
            self.record_indices = np.arange(self.index.records, dtype=np.int64)
        else:
            self.record_indices = self.index.indices_where("split", {split})
            if self.record_indices.size == 0:
                raise ValueError(
                    f"Corpus {self.corpus_root} contains no samples for split={split}; "
                    f"available splits: {self.index.values('split')}"
                )

        self._builder: SequenceBuilder | None = None
        self._manifest_handle = None
        self._skipped_too_long = 0
        self._semantic_anchor_token_cache: dict[int, int] = {}


    @property
    def builder(self) -> SequenceBuilder:
        if self._builder is None:
            self._builder = SequenceBuilder(
                self.registry,
                self.text_encoder,
                self.sequence_config,
                self.condition_config,
                self.grammar_config,
            )
        return self._builder

    def _handle(self):
        if self._manifest_handle is None:
            self._manifest_handle = self.corpus.manifest_path.open("rb")
        return self._manifest_handle

    def reset_handles(self) -> None:
        if self._manifest_handle is not None:
            self._manifest_handle.close()
            self._manifest_handle = None
        self.corpus.close()


    def __len__(self) -> int:
        return int(self.record_indices.size)

    def raw_record(self, record_index: int) -> dict[str, Any]:
        handle = self._handle()
        handle.seek(int(self.index.offsets[record_index]))
        line = handle.readline()
        return json.loads(line)

    def semantic_frames(self, record_index: int) -> int:
        return int(self.index.semantic_frames[record_index])

    def melody_frames(self, record_index: int) -> int:
        return int(self.index.melody_frames[record_index])

    def semantic_anchor_token_bound(
        self, record_index: int, mode: SequenceMode
    ) -> int:

        if (
            not self.sequence_config.semantic_section_reanchor
            or not mode.has_melody
            or mode.is_unique_section
        ):
            return 0
        record_index = int(record_index)
        cached = self._semantic_anchor_token_cache.get(record_index)
        if cached is not None:
            return cached
        pools = self.builder.section_anchor_pools(self.raw_record(record_index))
        total = sum(len(tokens) for tokens in pools.get("__ordered__", []))
        self._semantic_anchor_token_cache[record_index] = total
        return total

    def __getitem__(self, key: Any) -> dict[str, Any]:
        if isinstance(key, (int, np.integer)):
            record_index, mode, epoch = int(self.record_indices[int(key)]), SequenceMode.PLAIN, 0
        else:
            record_index, mode_value, epoch = key
            record_index = int(record_index)
            mode = SequenceMode(mode_value)
        record = self.raw_record(record_index)
        semantic = self.corpus.read_semantic(TokenSpan.from_dict(record["semantic"]))
        melody = None
        if record.get("melody"):
            melody = self.corpus.read_melody(TokenSpan.from_dict(record["melody"]))
        try:
            built = self.builder.build(
                record,
                mode=mode,
                semantic=semantic,
                melody=melody,
                epoch=int(epoch),
                sample_index=record_index,
                seed=self.seed,
            )
        except SequenceTooLongError:
            if self.sequence_config.overflow_policy == "error":
                raise


            self._skipped_too_long += 1
            built = self.builder.build(
                record,
                mode=SequenceMode.PLAIN,
                semantic=semantic,
                melody=None,
                epoch=int(epoch),
                sample_index=record_index,
                seed=self.seed,
            )
        return _to_item(built, record)


def _to_item(built: BuiltSequence, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_ids": built.input_ids,
        "labels": built.labels,
        "constraint_kinds": built.constraint_kinds,
        "sample_id": built.sample_id,
        "mode": built.mode.value,
        "num_semantic_tokens": built.num_semantic_tokens,
        "num_melody_tokens": built.num_melody_tokens,
        "num_condition_tokens": built.num_condition_tokens,
        "condition_truncated": built.condition_truncated,
        "semantic_cropped": built.semantic_cropped,
        "semantic_anchor_sections": int(
            built.diagnostics.get("semantic_anchor_sections", 0)
        ),
        "semantic_anchor_tokens": int(
            built.diagnostics.get("semantic_anchor_tokens", 0)
        ),
        "semantic_anchor_text_truncated": int(
            built.diagnostics.get("semantic_anchor_text_truncated", 0)
        ),
        "language": record.get("language") or "unknown",
        "quality_bucket": (record.get("quality") or {}).get("bucket", "unknown"),
        "is_instrumental": bool(record.get("is_instrumental", False)),
    }
