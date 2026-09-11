
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .common.logging import log_kv
from .common.staging import DEFAULT_CACHE_ROOT, stage_model_dir
from .contracts import MELODY_UNVOICED_ID
from .grammar import IGNORE_CONSTRAINT_KIND, AllowedTokenSets
from .registry import TokenRegistry
from .sequence import IGNORE_LABEL


#: *"Loss is applied to both the Melody-CoT region and the final Music Semantic Token


LOSS_REGIONS: tuple[str, ...] = ("semantic", "melody", "boundary")


LOSS_LEAVES: dict[str, str] = {
    "semantic": "semantic",
    "melody_struct": "melody",
    "melody_pitch": "melody",
    "boundary": "boundary",
}


_CONTROL_GROUPS: dict[str, tuple[str, ...]] = {
    "melody_struct": (
        "melody_bos",
        "melody_eos",
        "seg_end",
    ),
    "semantic": ("music_bos", "music_eos"),
    "boundary": ("eos",),
}


_PREFIX_ONLY_CONTROL: tuple[str, ...] = (
    "pad",
    "bos",
    "task_t2m",
    "mode_plain",
    "mode_section",
    "mode_unique_section",
    "cond_bos",
    "cond_eos",
)


EMBEDDING_INIT_MODES: tuple[str, ...] = ("row_norm_matched", "mean_plus_noise")


@dataclass
class ModelOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    num_supervised_tokens: int


    num_loss_tokens: int | None = None


    semantic_predictions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.num_loss_tokens is None:
            self.num_loss_tokens = int(self.num_supervised_tokens)


def _flatten_output(output: ModelOutput):
    return [output.loss, output.metrics, output.semantic_predictions], (
        output.num_supervised_tokens,
        output.num_loss_tokens,
    )


def _unflatten_output(children, context) -> ModelOutput:
    loss, metrics, semantic_predictions = children
    supervised, normalized = context
    return ModelOutput(
        loss=loss,
        metrics=metrics,
        num_supervised_tokens=supervised,
        num_loss_tokens=normalized,
        semantic_predictions=semantic_predictions,
    )


def _register_output_pytree() -> None:
    from torch.utils._pytree import register_pytree_node

    try:
        register_pytree_node(
            ModelOutput,
            _flatten_output,
            _unflatten_output,
            serialized_type_name="open_qwen_music.llm.model.ModelOutput",
        )
    except ValueError:
        pass


_register_output_pytree()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported precision {name},expected one of {sorted(mapping)}")
    return mapping[name]


def _dtype_bits(dtype: torch.dtype) -> int:
    return torch.finfo(dtype).bits


def _storage_dtype(config: Any) -> torch.dtype | None:
    raw = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if isinstance(raw, torch.dtype):
        return raw
    if isinstance(raw, str):
        try:
            return _resolve_dtype(raw)
        except ValueError:
            return None
    return None


MUSIC_EMBEDDING_NAMESPACES: tuple[str, ...] = (
    "control",
    "semantic",
    "melody",
)


def _as_parameter(weight: torch.Tensor | nn.Parameter) -> nn.Parameter:
    return (
        weight
        if isinstance(weight, nn.Parameter)
        else nn.Parameter(weight.detach().clone())
    )


class PartitionedVocabularyEmbedding(nn.Module):

    def __init__(
        self,
        text_weight: torch.Tensor,
        control_weight: torch.Tensor,
        semantic_weight: torch.Tensor,
        melody_weight: torch.Tensor,
        reserved_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.text_weight = nn.Parameter(text_weight.detach().clone())
        self.control_weight = nn.Parameter(control_weight.detach().clone())
        self.semantic_weight = nn.Parameter(semantic_weight.detach().clone())
        self.melody_weight = nn.Parameter(melody_weight.detach().clone())
        self.reserved_weight = nn.Parameter(reserved_weight.detach().clone())
        self.embedding_dim = int(self.text_weight.shape[1])
        self.text_size = int(self.text_weight.shape[0])
        self.control_size = int(self.control_weight.shape[0])
        self.semantic_size = int(self.semantic_weight.shape[0])
        self.melody_size = int(self.melody_weight.shape[0])
        self.active_size = self.control_size + self.semantic_size + self.melody_size
        self.num_embeddings = (
            self.text_size + self.active_size + int(self.reserved_weight.shape[0])
        )

    @classmethod
    def from_module(
        cls, module: nn.Module, registry: TokenRegistry
    ) -> PartitionedVocabularyEmbedding:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("input embedding must expose 2D weight")
        return cls(
            weight[: registry.text_vocab_size],
            weight[registry.control_base : registry.semantic_base],
            weight[registry.semantic_base : registry.melody_base],
            weight[
                registry.melody_base : registry.melody_base + registry.melody_size
            ],
            weight[
                registry.melody_base
                + registry.melody_size : registry.total_vocab_size
            ],
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        output = self.text_weight.new_empty((flat.numel(), self.embedding_dim))
        offset = 0
        boundaries = self._boundaries()
        for low, high, weight in boundaries:
            selected = (flat >= low) & (flat < high)
            output[selected] = F.embedding(flat[selected] - low, weight)
            offset = high
        if offset != self.num_embeddings:
            raise RuntimeError("partitioned embedding boundary does not cover the complete vocabulary")

        zero = output.new_zeros(())
        for _, _, weight in boundaries:
            if weight.numel():
                zero = zero + weight.reshape(-1)[0] * 0.0
        return (output + zero).reshape(*input_ids.shape, self.embedding_dim)

    def _boundaries(self) -> tuple[tuple[int, int, nn.Parameter], ...]:
        text_end = self.text_size
        control_end = text_end + self.control_size
        semantic_end = control_end + self.semantic_size
        melody_end = semantic_end + self.melody_size
        return (
            (0, text_end, self.text_weight),
            (text_end, control_end, self.control_weight),
            (control_end, semantic_end, self.semantic_weight),
            (semantic_end, melody_end, self.melody_weight),
            (melody_end, self.num_embeddings, self.reserved_weight),
        )

    def namespace_weight(self, name: str) -> nn.Parameter:
        if name not in MUSIC_EMBEDDING_NAMESPACES:
            raise KeyError(name)
        return getattr(self, f"{name}_weight")

    def weight_range(self, start: int, end: int) -> torch.Tensor:
        if not 0 <= start <= end <= self.num_embeddings:
            raise IndexError((start, end, self.num_embeddings))
        pieces = [
            weight[max(start, low) - low : min(end, high) - low]
            for low, high, weight in self._boundaries()
            if start < high and end > low
        ]
        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=0)

    def parameters_by_role(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "text": (self.text_weight,),
            "new": (
                self.control_weight,
                self.semantic_weight,
                self.melody_weight,
                self.reserved_weight,
            ),
        }

class PartitionedVocabularyHead(nn.Module):

    def __init__(
        self,
        text_weight: torch.Tensor,
        control_weight: torch.Tensor | nn.Parameter,
        semantic_weight: torch.Tensor | nn.Parameter,
        melody_weight: torch.Tensor | nn.Parameter,
        reserved_weight: torch.Tensor,
        *,
        tie_namespaces: tuple[str, ...] = (),
        semantic_output_residual_rank: int = 0,
    ) -> None:
        super().__init__()
        self.text_weight = nn.Parameter(text_weight.detach().clone())
        self.control_weight = _as_parameter(control_weight)
        self.semantic_weight = _as_parameter(semantic_weight)
        self.melody_weight = _as_parameter(melody_weight)
        self.reserved_weight = nn.Parameter(reserved_weight.detach().clone())
        self.in_features = int(self.text_weight.shape[1])
        self.text_size = int(self.text_weight.shape[0])
        self.control_size = int(self.control_weight.shape[0])
        self.semantic_size = int(self.semantic_weight.shape[0])
        self.melody_size = int(self.melody_weight.shape[0])
        self.active_size = self.control_size + self.semantic_size + self.melody_size
        self.out_features = (
            self.text_size + self.active_size + int(self.reserved_weight.shape[0])
        )
        self.tie_namespaces = tuple(tie_namespaces)
        rank = int(semantic_output_residual_rank)
        if rank < 0:
            raise ValueError("semantic_output_residual_rank required >= 0")
        self.semantic_output_residual_rank = rank
        if rank:
            self.semantic_residual_up = nn.Parameter(
                torch.empty(
                    self.semantic_size,
                    rank,
                    device=self.semantic_weight.device,
                    dtype=self.semantic_weight.dtype,
                )
            )
            self.semantic_residual_down = nn.Parameter(
                torch.zeros(
                    rank,
                    self.in_features,
                    device=self.semantic_weight.device,
                    dtype=self.semantic_weight.dtype,
                )
            )
            nn.init.normal_(self.semantic_residual_up, mean=0.0, std=rank**-0.5)
        else:
            self.register_parameter("semantic_residual_up", None)
            self.register_parameter("semantic_residual_down", None)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        registry: TokenRegistry,
        *,
        shared_weights: Mapping[str, nn.Parameter] | None = None,
        tie_namespaces: tuple[str, ...] = (),
        semantic_output_residual_rank: int = 0,
    ) -> PartitionedVocabularyHead:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("output head must expose 2D weight")
        bias = getattr(module, "bias", None)
        if bias is not None:
            raise ValueError("Partitioned output heads require bias=False")
        shared_weights = dict(shared_weights or {})

        def resolve(name: str, start: int, end: int):
            return shared_weights.get(name, weight[start:end])

        return cls(
            weight[: registry.text_vocab_size],
            resolve("control", registry.control_base, registry.semantic_base),
            resolve("semantic", registry.semantic_base, registry.melody_base),
            resolve(
                "melody",
                registry.melody_base,
                registry.melody_base + registry.melody_size,
            ),
            weight[
                registry.melody_base
                + registry.melody_size : registry.total_vocab_size
            ],
            tie_namespaces=tie_namespaces,
            semantic_output_residual_rank=semantic_output_residual_rank,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        semantic_logits = F.linear(hidden, self.semantic_weight)
        if self.semantic_output_residual_rank:
            semantic_logits = semantic_logits + F.linear(
                F.linear(hidden, self.semantic_residual_down),
                self.semantic_residual_up,
            )
        return torch.cat(
            [
                F.linear(hidden, self.text_weight),
                F.linear(hidden, self.control_weight),
                semantic_logits,
                F.linear(hidden, self.melody_weight),
                F.linear(hidden, self.reserved_weight),
            ],
            dim=-1,
        )

    def _boundaries(self) -> tuple[tuple[int, int, nn.Parameter], ...]:
        text_end = self.text_size
        control_end = text_end + self.control_size
        semantic_end = control_end + self.semantic_size
        melody_end = semantic_end + self.melody_size
        return (
            (0, text_end, self.text_weight),
            (text_end, control_end, self.control_weight),
            (control_end, semantic_end, self.semantic_weight),
            (semantic_end, melody_end, self.melody_weight),
            (melody_end, self.out_features, self.reserved_weight),
        )

    def weight_range(self, start: int, end: int) -> torch.Tensor:
        if not 0 <= start <= end <= self.out_features:
            raise IndexError((start, end, self.out_features))
        pieces = [
            weight[max(start, low) - low : min(end, high) - low]
            for low, high, weight in self._boundaries()
            if start < high and end > low
        ]
        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=0)

    def parameters_by_role(self) -> dict[str, tuple[nn.Parameter, ...]]:
        new_parameters = [
            self.control_weight,
            self.semantic_weight,
            self.melody_weight,
            self.reserved_weight,
        ]
        if self.semantic_output_residual_rank:
            new_parameters.extend(
                [self.semantic_residual_up, self.semantic_residual_down]
            )
        return {
            "text": (self.text_weight,),
            "new": tuple(new_parameters),
        }

def vocabulary_weight_range(module: nn.Module, start: int, end: int) -> torch.Tensor:
    getter = getattr(module, "weight_range", None)
    if callable(getter):
        return getter(start, end)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{type(module).__name__} has no readable vocabulary weight")
    return weight[start:end]


def _partition_vocabulary_modules(
    backbone: nn.Module,
    registry: TokenRegistry,
    *,
    tie_namespaces: tuple[str, ...],
    semantic_output_residual_rank: int,
) -> None:
    unknown = sorted(set(tie_namespaces) - set(MUSIC_EMBEDDING_NAMESPACES))
    if unknown:
        raise ValueError(
            f"tie_embedding_namespaces has unknown value {unknown};"
            f"expected one of {list(MUSIC_EMBEDDING_NAMESPACES)}"
        )
    if semantic_output_residual_rank and "semantic" not in tie_namespaces:
        raise ValueError("Semantic residual is defined only for tied semantic rows")
    input_module = PartitionedVocabularyEmbedding.from_module(
        backbone.get_input_embeddings(),
        registry,
    )
    output = backbone.get_output_embeddings()
    if output is None:
        raise RuntimeError("Base model has no output embedding")
    output_module = PartitionedVocabularyHead.from_module(
        output,
        registry,
        shared_weights={
            name: input_module.namespace_weight(name)
            for name in tie_namespaces
        },
        tie_namespaces=tie_namespaces,
        semantic_output_residual_rank=semantic_output_residual_rank,
    )
    backbone.set_input_embeddings(input_module)
    backbone.set_output_embeddings(output_module)
    backbone.config.tie_word_embeddings = False


def _resolve_tie_namespaces(
    *,
    tie_music_embeddings: bool,
    tie_embedding_namespaces: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if tie_embedding_namespaces is None:
        return MUSIC_EMBEDDING_NAMESPACES if tie_music_embeddings else ()
    resolved = tuple(str(name) for name in tie_embedding_namespaces)
    if len(set(resolved)) != len(resolved):
        raise ValueError("tie_embedding_namespaces cannot be repeated")
    if tie_music_embeddings and resolved != MUSIC_EMBEDDING_NAMESPACES:
        raise ValueError(
            "tie_music_embeddings=true ties every music namespace. Set it to false "
            "before selecting a subset with tie_embedding_namespaces."
        )
    return resolved


class MusicLLM(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        registry: TokenRegistry,
        *,
        loss_chunk_size: int = 4096,
        accuracy_sample_positions: int = 4096,
        region_loss_weights: Mapping[str, float] | None = None,
        melody_class_weights: Mapping[str, Any] | None = None,
        modality_type_embeddings: bool = False,
        grammar_constrained_loss: bool = False,
        grammar_constrained_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.registry = registry
        self.loss_chunk_size = int(loss_chunk_size)
        self.accuracy_sample_positions = int(accuracy_sample_positions)
        self.region_loss_weights = _normalize_region_weights(region_loss_weights)
        class_weights = dict(melody_class_weights or {})
        unvoiced_id = int(class_weights.get("unvoiced_id", MELODY_UNVOICED_ID))
        unvoiced_weight = float(class_weights.get("unvoiced_weight", 1.0))
        if unvoiced_id != MELODY_UNVOICED_ID:
            raise ValueError(
                f"Melody unvoiced_id must be {MELODY_UNVOICED_ID}; got {unvoiced_id}"
            )
        if not math.isfinite(unvoiced_weight) or not 0.75 <= unvoiced_weight <= 1.0:
            raise ValueError("melody unvoiced_weight must be finite and in [0.75, 1.0]")
        self.melody_unvoiced_id = unvoiced_id
        self.melody_unvoiced_weight = unvoiced_weight
        self.grammar_constrained_loss = bool(grammar_constrained_loss)
        self.grammar_constrained_metrics = bool(grammar_constrained_metrics)

        self.embedding_init_stats: dict[str, float] = {}
        self._control_leaf_codes = _control_leaf_codes(registry)
        self._control_codes_cache: torch.Tensor | None = None


        self._semantic_frequency_buckets: torch.Tensor | None = None
        self._constraint_table_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        embedding = self.backbone.get_input_embeddings()
        if embedding.num_embeddings != registry.total_vocab_size:
            raise ValueError(
                f"Base vocabulary size {embedding.num_embeddings} does not match "
                f"registry.total_vocab_size={registry.total_vocab_size}; resize the model first"
            )
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        reference_parameter = next(embedding.parameters())
        self.modality_type_embeddings = (
            nn.Embedding(
                4,
                hidden_size,
                device=reference_parameter.device,
                dtype=reference_parameter.dtype,
            )
            if modality_type_embeddings
            else None
        )
        if self.modality_type_embeddings is not None:

            nn.init.zeros_(self.modality_type_embeddings.weight)


    @classmethod
    def from_pretrained_base(
        cls,
        base_model_path: str | Path,
        registry: TokenRegistry,
        *,
        dtype: str = "bf16",
        attn_implementation: str = "flash_attention_2",
        loss_chunk_size: int = 4096,
        untie_word_embeddings: bool = False,
        partitioned_embeddings: bool = False,
        tie_music_embeddings: bool = False,
        tie_embedding_namespaces: tuple[str, ...] | list[str] | None = None,
        semantic_output_residual_rank: int = 0,
        modality_type_embeddings: bool = False,
        grammar_constrained_loss: bool = False,
        grammar_constrained_metrics: bool = False,
        gradient_checkpointing: bool = False,
        init_mode: str = "row_norm_matched",
        init_std_scale: float = 0.02,
        init_norm_ratio: float = 1.0,
        seed: int = 0,
        accuracy_sample_positions: int = 4096,
        region_loss_weights: Mapping[str, float] | None = None,
        melody_class_weights: Mapping[str, Any] | None = None,
        device: torch.device | str | None = None,
        load_dtype: str | None = None,
    ) -> MusicLLM:
        from transformers import AutoConfig, AutoModelForCausalLM

        base_model_path = str(base_model_path)
        config = AutoConfig.from_pretrained(base_model_path)
        if int(config.vocab_size) != registry.text_vocab_size:
            raise ValueError(
                f"registry text_vocab_size={registry.text_vocab_size} does not match "
                f"base config.vocab_size={config.vocab_size}. Build the registry from "
                "the base model so text and music token ID ranges do not overlap."
            )
        torch_dtype = _resolve_dtype(dtype)
        storage_dtype = _storage_dtype(config)
        if load_dtype is not None:
            read_dtype = _resolve_dtype(load_dtype)
        elif storage_dtype is not None and _dtype_bits(storage_dtype) < _dtype_bits(torch_dtype):
            read_dtype = storage_dtype
        else:
            read_dtype = torch_dtype
        try:
            backbone = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=read_dtype,
                attn_implementation=attn_implementation,
            )
        except (ImportError, ValueError) as error:
            if attn_implementation == "flash_attention_2":
                print(
                    f"flash_attention_2 is unavailable ({error}); falling back to sdpa",
                    flush=True,
                )
                backbone = AutoModelForCausalLM.from_pretrained(
                    base_model_path, dtype=read_dtype, attn_implementation="sdpa"
                )
            else:
                raise

        if device is not None:
            backbone = backbone.to(device)
        if read_dtype != torch_dtype:
            backbone = backbone.to(torch_dtype)
            log_kv(
                "base_model_load",
                {
                    "read_dtype": str(read_dtype).replace("torch.", ""),
                    "train_dtype": str(torch_dtype).replace("torch.", ""),
                    "upcast_device": str(device) if device is not None else "cpu",
                },
            )

        if untie_word_embeddings:
            _untie_output_embeddings(backbone)
        tie_namespaces = _resolve_tie_namespaces(
            tie_music_embeddings=tie_music_embeddings,
            tie_embedding_namespaces=tie_embedding_namespaces,
        )
        if (
            tie_namespaces or semantic_output_residual_rank
        ) and not partitioned_embeddings:
            raise ValueError(
                "namespace tying/residual requires partitioned_embeddings=true"
            )
        init_stats = _resize_and_init(
            backbone,
            registry,
            init_mode=init_mode,
            init_std_scale=init_std_scale,
            init_norm_ratio=init_norm_ratio,
            seed=seed,
        )
        if partitioned_embeddings:
            _partition_vocabulary_modules(
                backbone,
                registry,
                tie_namespaces=tie_namespaces,
                semantic_output_residual_rank=semantic_output_residual_rank,
            )


        if init_stats:
            log_kv(
                "embedding_init",
                {
                    "init_mode": init_mode,
                    "init_norm_ratio": init_norm_ratio,
                    "seed": seed,
                    **init_stats,
                },
            )
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            backbone.config.use_cache = False
        model = cls(
            backbone,
            registry,
            loss_chunk_size=loss_chunk_size,
            accuracy_sample_positions=accuracy_sample_positions,
            region_loss_weights=region_loss_weights,
            melody_class_weights=melody_class_weights,
            modality_type_embeddings=modality_type_embeddings,
            grammar_constrained_loss=grammar_constrained_loss,
            grammar_constrained_metrics=grammar_constrained_metrics,
        )

        # (``inner_model.embedding_init_stats``).
        model.embedding_init_stats = dict(init_stats)
        return model

    @classmethod
    def from_scratch_config(
        cls,
        registry: TokenRegistry,
        *,
        hidden_size: int = 256,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        intermediate_size: int = 512,
        max_position_embeddings: int = 8192,
        dtype: str = "fp32",
        attn_implementation: str = "eager",
        loss_chunk_size: int = 1024,
        region_loss_weights: Mapping[str, float] | None = None,
        melody_class_weights: Mapping[str, Any] | None = None,
        partitioned_embeddings: bool = False,
        tie_music_embeddings: bool = False,
        tie_embedding_namespaces: tuple[str, ...] | list[str] | None = None,
        semantic_output_residual_rank: int = 0,
        modality_type_embeddings: bool = False,
        grammar_constrained_loss: bool = False,
        grammar_constrained_metrics: bool = False,
    ) -> MusicLLM:
        from transformers import AutoModelForCausalLM, Qwen3Config

        config = Qwen3Config(
            vocab_size=registry.total_vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            head_dim=max(8, hidden_size // num_attention_heads),
            tie_word_embeddings=True,
        )
        backbone = AutoModelForCausalLM.from_config(
            config, attn_implementation=attn_implementation
        )
        backbone = backbone.to(_resolve_dtype(dtype))
        tie_namespaces = _resolve_tie_namespaces(
            tie_music_embeddings=tie_music_embeddings,
            tie_embedding_namespaces=tie_embedding_namespaces,
        )
        if (
            tie_namespaces or semantic_output_residual_rank
        ) and not partitioned_embeddings:
            raise ValueError(
                "namespace tying/residual requires partitioned_embeddings=true"
            )
        if partitioned_embeddings:
            _partition_vocabulary_modules(
                backbone,
                registry,
                tie_namespaces=tie_namespaces,
                semantic_output_residual_rank=semantic_output_residual_rank,
            )
        return cls(
            backbone,
            registry,
            loss_chunk_size=loss_chunk_size,
            region_loss_weights=region_loss_weights,
            melody_class_weights=melody_class_weights,
            modality_type_embeddings=modality_type_embeddings,
            grammar_constrained_loss=grammar_constrained_loss,
            grammar_constrained_metrics=grammar_constrained_metrics,
        )


    @property
    def transformer(self) -> nn.Module:
        return getattr(self.backbone, self.backbone.base_model_prefix)

    @property
    def lm_head(self) -> nn.Module:
        head = self.backbone.get_output_embeddings()
        if head is None:
            raise RuntimeError("Base model has no output embedding; cannot compute loss")
        return head

    @property
    def output_dtype(self) -> torch.dtype:
        return next(self.lm_head.parameters()).dtype

    def embedding_parameters_by_role(self) -> dict[str, list[nn.Parameter]]:

        roles: dict[str, list[nn.Parameter]] = {}
        seen: dict[int, str] = {}
        for module in (
            self.backbone.get_input_embeddings(),
            self.backbone.get_output_embeddings(),
        ):
            if module is None:
                continue
            getter = getattr(module, "parameters_by_role", None)
            grouped = (
                getter()
                if callable(getter)
                else {"all": tuple(module.parameters())}
            )
            for role, parameters in grouped.items():
                for parameter in parameters:
                    previous = seen.get(id(parameter))
                    if previous is not None and previous != role:
                        raise RuntimeError(
                            f"same as embedding Parameter also belongs to {previous}/{role}"
                        )
                    if previous is None:
                        seen[id(parameter)] = role
                        roles.setdefault(role, []).append(parameter)
        if self.modality_type_embeddings is not None:
            parameter = self.modality_type_embeddings.weight
            if id(parameter) not in seen:
                roles.setdefault("type", []).append(parameter)
        return roles

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad or not trainable_only
        )


    def _modality_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        registry = self.registry
        kinds = torch.zeros_like(input_ids)
        kinds = torch.where(
            (input_ids >= registry.control_base)
            & (input_ids < registry.control_base + registry.num_control),
            torch.ones_like(kinds),
            kinds,
        )
        kinds = torch.where(
            (input_ids >= registry.semantic_base)
            & (input_ids < registry.semantic_base + registry.semantic_size),
            torch.full_like(kinds, 2),
            kinds,
        )
        return torch.where(
            (input_ids >= registry.melody_base)
            & (input_ids < registry.melody_base + registry.melody_size),
            torch.full_like(kinds, 3),
            kinds,
        )

    def transformer_forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.modality_type_embeddings is None:
            return self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        token_embeddings = self.backbone.get_input_embeddings()(input_ids)
        type_embeddings = self.modality_type_embeddings(
            self._modality_ids(input_ids)
        ).to(token_embeddings.dtype)
        return self.transformer(
            inputs_embeds=token_embeddings + type_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

    def hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.transformer_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        constraint_kinds: torch.Tensor | None = None,
        *,
        compute_accuracy: bool = False,
        compute_diagnostics: bool = False,
        return_semantic_predictions: bool = False,
    ) -> ModelOutput:
        hidden = self.hidden_states(input_ids, attention_mask)
        return self.loss_from_hidden(
            hidden,
            labels,
            constraint_kinds=constraint_kinds,
            compute_accuracy=compute_accuracy,
            compute_diagnostics=compute_diagnostics,
            return_semantic_predictions=return_semantic_predictions,
        )

    def loss_from_hidden(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        constraint_kinds: torch.Tensor | None = None,
        compute_accuracy: bool = False,
        compute_diagnostics: bool = False,
        return_semantic_predictions: bool = False,
    ) -> ModelOutput:

        flat_hidden = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])
        flat_labels = labels[:, 1:].reshape(-1)
        supervised = (flat_labels != IGNORE_LABEL).nonzero(as_tuple=True)[0]
        flat_constraint_kinds: torch.Tensor | None = None
        compute_constraints = (
            self.grammar_constrained_loss or self.grammar_constrained_metrics
        )
        if compute_constraints:
            if constraint_kinds is None:
                raise ValueError(
                    "grammar constrained loss/metrics must be provided when enabling constraint_kinds"
                )
            if constraint_kinds.shape != labels.shape:
                raise ValueError(
                    "constraint_kinds must have the same shape as labels: "
                    f"{tuple(constraint_kinds.shape)} vs {tuple(labels.shape)}"
                )
            flat_constraint_kinds = constraint_kinds[:, 1:].reshape(-1).to(torch.long)
            selected_kinds = flat_constraint_kinds.index_select(0, supervised)
            if bool((selected_kinds == IGNORE_CONSTRAINT_KIND).any()):
                raise RuntimeError("A supervised position is missing its grammar constraint kind")

        if supervised.numel() == 0:


            zero = hidden.sum() * 0.0
            for parameter in self.lm_head.parameters():
                zero = zero + parameter.sum() * 0.0
            return ModelOutput(
                loss=zero,
                metrics={
                    "loss": zero.detach(),
                    "loss_weighted": zero.detach(),
                    "supervised_tokens": torch.tensor(0.0),
                },
                num_supervised_tokens=0,
                num_loss_tokens=0,
            )

        leaves = self._leaf_masks(flat_labels[supervised])
        leaf_counts = {name: int(mask.sum().item()) for name, mask in leaves.items()}


        covered = sum(leaf_counts.values())
        if covered != supervised.numel():
            raise RuntimeError(
                f"{supervised.numel() - covered} labels are outside every loss region. "
                "They may be unmasked text or reserved tokens in the condition region, "
                f"or prefix-only tokens {list(_PREFIX_ONLY_CONTROL)}. Check the label "
                "mask in sequence.py and the padding mask in the collator."
            )


        weights = self.region_loss_weights
        weighted_total = flat_hidden.new_zeros((), dtype=torch.float32)
        plain_total = flat_hidden.new_zeros((), dtype=torch.float32)
        constrained_plain_total = flat_hidden.new_zeros((), dtype=torch.float32)
        metrics: dict[str, torch.Tensor] = {}
        objective_region_sums: dict[str, torch.Tensor] = {}
        constrained_region_sums: dict[str, torch.Tensor] = {}
        full_region_sums: dict[str, torch.Tensor] = {}
        region_counts: dict[str, int] = {}
        total_count = 0
        weighted_count = 0
        for name, mask in leaves.items():
            count = leaf_counts[name]
            if count == 0:
                continue
            region = LOSS_LEAVES[name]
            indices = supervised[mask]
            weighted_constrained_loss: torch.Tensor
            weighted_full_loss: torch.Tensor
            if name == "melody_pitch":
                melody_labels = flat_labels.index_select(0, indices)
                unvoiced = (
                    melody_labels
                    == self.registry.melody_base + self.melody_unvoiced_id
                )
                unvoiced_indices = indices[unvoiced]
                voiced_indices = indices[~unvoiced]
                zero = flat_hidden.new_zeros((), dtype=torch.float32)

                def split_loss(
                    selected: torch.Tensor,
                    empty: torch.Tensor = zero,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    if selected.numel() == 0:
                        return empty, empty
                    return self._chunked_sum_loss(
                        flat_hidden,
                        flat_labels,
                        selected,
                        constraint_kinds=(
                            flat_constraint_kinds.index_select(0, selected)
                            if flat_constraint_kinds is not None
                            else None
                        ),
                    )

                unvoiced_constrained, unvoiced_full = split_loss(unvoiced_indices)
                voiced_constrained, voiced_full = split_loss(voiced_indices)
                constrained_loss = unvoiced_constrained + voiced_constrained
                full_loss = unvoiced_full + voiced_full
                raw_weight_sum = (
                    self.melody_unvoiced_weight * int(unvoiced_indices.numel())
                    + int(voiced_indices.numel())
                )
                normalization = count / max(raw_weight_sum, 1e-12)
                weighted_constrained_loss = normalization * (
                    self.melody_unvoiced_weight * unvoiced_constrained
                    + voiced_constrained
                )
                weighted_full_loss = normalization * (
                    self.melody_unvoiced_weight * unvoiced_full + voiced_full
                )
                metrics["loss_melody_pitch_unvoiced"] = (
                    unvoiced_full / max(int(unvoiced_indices.numel()), 1)
                ).detach()
                metrics["loss_melody_pitch_voiced"] = (
                    voiced_full / max(int(voiced_indices.numel()), 1)
                ).detach()
                metrics["tokens_melody_pitch_unvoiced"] = torch.tensor(
                    float(unvoiced_indices.numel())
                )
                metrics["tokens_melody_pitch_voiced"] = torch.tensor(
                    float(voiced_indices.numel())
                )
            else:
                constrained_loss, full_loss = self._chunked_sum_loss(
                    flat_hidden,
                    flat_labels,
                    indices,
                    constraint_kinds=(
                        flat_constraint_kinds.index_select(0, indices)
                        if flat_constraint_kinds is not None
                        else None
                    ),
                )
                weighted_constrained_loss = constrained_loss
                weighted_full_loss = full_loss
            objective_loss = (
                weighted_constrained_loss
                if self.grammar_constrained_loss
                else weighted_full_loss
            )
            objective_region_sums[region] = (
                objective_loss
                if region not in objective_region_sums
                else objective_region_sums[region] + objective_loss
            )
            constrained_region_sums[region] = (
                constrained_loss
                if region not in constrained_region_sums
                else constrained_region_sums[region] + constrained_loss
            )
            full_region_sums[region] = (
                full_loss
                if region not in full_region_sums
                else full_region_sums[region] + full_loss
            )
            region_counts[region] = region_counts.get(region, 0) + count
            total_count += count
            if weights[region] > 0.0:
                weighted_count += count
            if name != region:


                metrics[f"loss_{name}"] = (full_loss / count).detach()
                if name == "melody_pitch":
                    metrics["loss_weighted_melody_pitch"] = (
                        weighted_full_loss / count
                    ).detach()
                if compute_constraints:
                    metrics[f"loss_constrained_{name}"] = (
                        constrained_loss / count
                    ).detach()
                    metrics[f"tokens_constrained_{name}"] = torch.tensor(
                        float(count)
                    )
                metrics[f"tokens_{name}"] = torch.tensor(float(count))

        for region, objective_region_loss in objective_region_sums.items():
            constrained_region_loss = constrained_region_sums[region]
            full_region_loss = full_region_sums[region]
            count = region_counts[region]
            weighted_total = (
                weighted_total + weights[region] * objective_region_loss
            )

            plain_total = plain_total + full_region_loss.detach()
            constrained_plain_total = (
                constrained_plain_total + constrained_region_loss.detach()
            )
            metrics[f"loss_{region}"] = (full_region_loss / count).detach()
            if compute_constraints:
                metrics[f"loss_constrained_{region}"] = (
                    constrained_region_loss / count
                ).detach()
                metrics[f"tokens_constrained_{region}"] = torch.tensor(
                    float(count)
                )
            metrics[f"tokens_{region}"] = torch.tensor(float(count))

        loss = weighted_total / max(weighted_count, 1)


        metrics["loss"] = (plain_total / max(total_count, 1)).detach()
        if compute_constraints:
            metrics["loss_constrained"] = (
                constrained_plain_total / max(total_count, 1)
            ).detach()
            metrics["tokens_constrained"] = torch.tensor(float(total_count))
        metrics["loss_weighted"] = loss.detach()
        metrics["supervised_tokens"] = torch.tensor(float(total_count))
        if compute_accuracy:
            metrics.update(
                self._accuracy_metrics(flat_hidden, flat_labels, supervised, leaves)
            )
        semantic_predictions = None
        if compute_diagnostics or return_semantic_predictions:
            diagnostics, semantic_predictions = self._semantic_diagnostics(
                flat_hidden,
                labels,
                compute_metrics=compute_diagnostics,
                return_predictions=return_semantic_predictions,
            )
            metrics.update(diagnostics)
        return ModelOutput(
            loss=loss,
            metrics=metrics,
            num_supervised_tokens=total_count,
            num_loss_tokens=weighted_count,
            semantic_predictions=semantic_predictions,
        )

    def set_semantic_frequency_buckets(self, buckets: torch.Tensor | None) -> None:

        if buckets is None:
            self._semantic_frequency_buckets = None
            return
        resolved = torch.as_tensor(buckets, dtype=torch.int8).reshape(-1)
        if resolved.numel() != self.registry.semantic_size:
            raise ValueError(
                "Semantic frequency-bucket length must equal semantic_size="
                f"{self.registry.semantic_size}; got {resolved.numel()}"
            )
        if bool(((resolved < 0) | (resolved > 3)).any()):
            raise ValueError("Semantic frequency bucket can only take 0/1/2/3")
        self._semantic_frequency_buckets = resolved

    def _leaf_masks(self, selected_labels: torch.Tensor) -> dict[str, torch.Tensor]:
        registry = self.registry
        codes = self._control_codes_on(selected_labels.device)
        in_control = (selected_labels >= registry.control_base) & (
            selected_labels < registry.control_base + registry.num_control
        )

        control_index = (selected_labels - registry.control_base).clamp_(
            0, registry.num_control - 1
        )
        control_code = codes.index_select(0, control_index) * in_control
        semantic_ns = (selected_labels >= registry.semantic_base) & (
            selected_labels < registry.semantic_base + registry.semantic_size
        )
        melody_ns = (selected_labels >= registry.melody_base) & (
            selected_labels < registry.melody_base + registry.melody_size
        )
        return {
            "semantic": semantic_ns | (control_code == _LEAF_CODES["semantic"]),
            "melody_pitch": melody_ns,
            "melody_struct": control_code == _LEAF_CODES["melody_struct"],
            "boundary": control_code == _LEAF_CODES["boundary"],
        }

    def _control_codes_on(self, device: torch.device) -> torch.Tensor:
        cache = self._control_codes_cache
        if cache is None or cache.device != device:
            cache = self._control_leaf_codes.to(device)
            self._control_codes_cache = cache
        return cache

    def _constraint_table_on(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._constraint_table_cache
        if cache is None or cache[0].device != device:
            cache = AllowedTokenSets(self.registry, device).padded_table()
            self._constraint_table_cache = cache
        return cache


    def _chunked_sum_loss(
        self,
        flat_hidden: torch.Tensor,
        flat_labels: torch.Tensor,
        indices: torch.Tensor,
        *,
        constraint_kinds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head = self.lm_head
        chunk = self.loss_chunk_size if self.loss_chunk_size > 0 else indices.numel()
        objective_total = flat_hidden.new_zeros((), dtype=torch.float32)
        full_total = flat_hidden.new_zeros((), dtype=torch.float32)
        use_checkpoint = self.training and torch.is_grad_enabled() and chunk < indices.numel()
        constraint_table: torch.Tensor | None = None
        constraint_lengths: torch.Tensor | None = None
        if constraint_kinds is not None:
            constraint_table, constraint_lengths = self._constraint_table_on(
                flat_hidden.device
            )
        for start in range(0, indices.numel(), chunk):
            piece = indices[start : start + chunk]
            hidden_piece = flat_hidden.index_select(0, piece)
            label_piece = flat_labels.index_select(0, piece)
            kind_piece = (
                constraint_kinds[start : start + chunk]
                if constraint_kinds is not None
                else None
            )
            if kind_piece is None:
                if use_checkpoint:
                    full = checkpoint(
                        _head_cross_entropy,
                        head,
                        hidden_piece,
                        label_piece,
                        use_reentrant=False,
                    )
                else:
                    full = _head_cross_entropy(head, hidden_piece, label_piece)
                objective, full = full, full
            else:
                assert constraint_table is not None
                assert constraint_lengths is not None
                if use_checkpoint:
                    pair = checkpoint(
                        _head_cross_entropies,
                        head,
                        hidden_piece,
                        label_piece,
                        kind_piece,
                        constraint_table,
                        constraint_lengths,
                        use_reentrant=False,
                    )
                else:
                    pair = _head_cross_entropies(
                        head,
                        hidden_piece,
                        label_piece,
                        kind_piece,
                        constraint_table,
                        constraint_lengths,
                    )
                objective, full = pair[0], pair[1]
            objective_total = objective_total + objective
            full_total = full_total + full
        return objective_total, full_total

    @torch.no_grad()
    def _accuracy_metrics(
        self,
        flat_hidden: torch.Tensor,
        flat_labels: torch.Tensor,
        supervised: torch.Tensor,
        leaves: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        limit = max(1, self.accuracy_sample_positions)
        groups: dict[str, torch.Tensor] = {}
        for name, mask in leaves.items():
            indices = supervised[mask]
            if indices.numel() == 0:
                continue
            groups[name] = indices
            region = LOSS_LEAVES[name]
            if region != name:
                previous = groups.get(region)
                groups[region] = (
                    indices if previous is None else torch.cat([previous, indices])
                )
        for name, indices in groups.items():
            indices = _subsample_positions(indices, limit)
            hidden_piece = flat_hidden.index_select(0, indices)
            logits = self.lm_head(hidden_piece)
            predictions = logits.argmax(dim=-1)
            targets = flat_labels.index_select(0, indices)
            metrics[f"acc_{name}"] = (predictions == targets).float().mean().detach()
            if name == "melody_pitch":
                unvoiced_global = (
                    self.registry.melody_base + self.melody_unvoiced_id
                )
                target_unvoiced = targets == unvoiced_global
                predicted_unvoiced = predictions == unvoiced_global
                metrics["target_rate_melody_unvoiced"] = (
                    target_unvoiced.float().mean().detach()
                )
                metrics["predicted_rate_melody_unvoiced"] = (
                    predicted_unvoiced.float().mean().detach()
                )
                if target_unvoiced.any():
                    metrics["recall_melody_unvoiced"] = (
                        predicted_unvoiced[target_unvoiced].float().mean().detach()
                    )
                    metrics["acc_melody_pitch_unvoiced"] = (
                        (predictions[target_unvoiced] == targets[target_unvoiced])
                        .float()
                        .mean()
                        .detach()
                    )
                target_voiced = ~target_unvoiced
                if target_voiced.any():
                    metrics["acc_melody_pitch_voiced"] = (
                        (predictions[target_voiced] == targets[target_voiced])
                        .float()
                        .mean()
                        .detach()
                    )
        return metrics

    @torch.no_grad()
    def _semantic_diagnostics(
        self,
        flat_hidden: torch.Tensor,
        labels: torch.Tensor,
        *,
        compute_metrics: bool,
        return_predictions: bool,
        ece_bins: int = 15,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:

        if ece_bins <= 1:
            raise ValueError(f"ece_bins must be greater than 1; got {ece_bins}")
        target_matrix = labels[:, 1:]
        width = int(target_matrix.shape[1])
        flat_targets = target_matrix.reshape(-1)
        registry = self.registry
        semantic_mask = (flat_targets >= registry.semantic_base) & (
            flat_targets < registry.semantic_base + registry.semantic_size
        )
        all_indices = semantic_mask.nonzero(as_tuple=True)[0]
        if all_indices.numel() == 0:
            empty_predictions = (
                torch.full_like(labels, -1) if return_predictions else None
            )
            return {}, empty_predictions

        if return_predictions:
            evaluated = all_indices
        else:
            evaluated = _subsample_positions(
                all_indices, max(1, self.accuracy_sample_positions)
            )

        count = int(evaluated.numel())
        if return_predictions and not compute_metrics:


            predicted_targets = torch.full_like(target_matrix, -1)
            predicted_flat = predicted_targets.reshape(-1)
            chunk = max(1, min(int(self.loss_chunk_size or count), 256))
            for start in range(0, count, chunk):
                indices = evaluated[start : start + chunk]
                logits = self.lm_head(flat_hidden.index_select(0, indices))
                constrained = (
                    logits[
                        :,
                        registry.semantic_base : registry.semantic_base
                        + registry.semantic_size,
                    ].argmax(dim=-1)
                    + registry.semantic_base
                )
                predicted_flat.index_copy_(0, indices, constrained)
            return {}, F.pad(predicted_targets, (1, 0), value=-1)

        nll = torch.empty(count, dtype=torch.float32, device=flat_hidden.device)
        top1 = torch.empty(count, dtype=torch.long, device=flat_hidden.device)
        constrained_top1 = torch.empty(
            count, dtype=torch.long, device=flat_hidden.device
        )
        top5_correct = torch.empty(count, dtype=torch.bool, device=flat_hidden.device)
        top10_correct = torch.empty(count, dtype=torch.bool, device=flat_hidden.device)
        reciprocal_rank = torch.empty(
            count, dtype=torch.float32, device=flat_hidden.device
        )
        confidence = torch.empty(count, dtype=torch.float32, device=flat_hidden.device)
        entropy = torch.empty(count, dtype=torch.float32, device=flat_hidden.device)
        brier = torch.empty(count, dtype=torch.float32, device=flat_hidden.device)


        chunk = max(1, min(int(self.loss_chunk_size or count), 256))
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            indices = evaluated[start:stop]
            hidden_piece = flat_hidden.index_select(0, indices)
            targets = flat_targets.index_select(0, indices)
            logits = self.lm_head(hidden_piece).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            nll[start:stop] = -target_log_probs

            k = min(10, int(logits.shape[-1]))
            values, indices_topk = torch.topk(logits, k=k, dim=-1)
            del values
            top1[start:stop] = indices_topk[:, 0]
            constrained_top1[start:stop] = (
                logits[
                    :,
                    registry.semantic_base : registry.semantic_base
                    + registry.semantic_size,
                ].argmax(dim=-1)
                + registry.semantic_base
            )
            top5_correct[start:stop] = (
                indices_topk[:, : min(5, k)] == targets.unsqueeze(1)
            ).any(dim=1)
            top10_correct[start:stop] = (
                indices_topk == targets.unsqueeze(1)
            ).any(dim=1)

            target_logits = logits.gather(1, targets.unsqueeze(1))
            ranks = 1 + (logits > target_logits).sum(dim=-1)
            reciprocal_rank[start:stop] = ranks.to(torch.float32).reciprocal()
            max_probability, _ = probabilities.max(dim=-1)
            confidence[start:stop] = max_probability
            entropy[start:stop] = -(probabilities * log_probs).sum(dim=-1)
            target_probability = probabilities.gather(
                1, targets.unsqueeze(1)
            ).squeeze(1)
            brier[start:stop] = (
                probabilities.square().sum(dim=-1)
                - 2.0 * target_probability
                + 1.0
            )

        metrics: dict[str, torch.Tensor] = {}
        if compute_metrics:
            correct = top1 == flat_targets.index_select(0, evaluated)

            def add_mean(name: str, values: torch.Tensor) -> None:
                if values.numel() == 0:
                    return
                metrics[name] = values.float().mean()
                prefixes = (
                    "normalized_entropy_",
                    "confidence_",
                    "entropy_",
                    "brier_",
                    "mrr_",
                    "loss_",
                    "acc_",
                )
                suffix = next(
                    (name.removeprefix(prefix) for prefix in prefixes if name.startswith(prefix)),
                    name,
                )
                metrics[f"tokens_{suffix}"] = values.new_tensor(
                    float(values.numel()), dtype=torch.float32
                )

            add_mean("loss_semantic_tokens", nll)
            add_mean("acc_semantic_top1", correct)
            add_mean(
                "acc_semantic_constrained_top1",
                constrained_top1 == flat_targets.index_select(0, evaluated),
            )
            add_mean("acc_semantic_top5", top5_correct)
            add_mean("acc_semantic_top10", top10_correct)
            add_mean("mrr_semantic_tokens", reciprocal_rank)
            add_mean("confidence_semantic_tokens", confidence)
            add_mean("entropy_semantic_tokens", entropy)
            add_mean(
                "normalized_entropy_semantic_tokens",
                entropy / math.log(float(registry.total_vocab_size)),
            )
            add_mean("brier_semantic_tokens", brier)

            rows = torch.div(evaluated, width, rounding_mode="floor")
            columns = evaluated.remainder(width)
            semantic_by_row = semantic_mask.reshape(labels.shape[0], width)
            sequence_lengths = semantic_by_row.sum(dim=1)

            ordinal_matrix = semantic_by_row.long().cumsum(dim=1) - 1
            ordinals = ordinal_matrix[rows, columns]
            lengths = sequence_lengths.index_select(0, rows).clamp_min(1)
            relative = (ordinals.to(torch.float32) + 0.5) / lengths.to(torch.float32)
            for bucket, low, high in (
                ("q1", 0.0, 0.25),
                ("q2", 0.25, 0.50),
                ("q3", 0.50, 0.75),
                ("q4", 0.75, 1.01),
            ):
                selected = (relative >= low) & (relative < high)
                add_mean(f"loss_semantic_position_{bucket}", nll[selected])
                add_mean(f"acc_semantic_position_{bucket}", correct[selected])
            tail = ordinals >= (lengths - 500).clamp_min(0)
            add_mean("loss_semantic_position_tail500", nll[tail])
            add_mean("acc_semantic_position_tail500", correct[tail])

            durations = lengths.to(torch.float32) / 25.0
            for bucket, low, high in (
                ("0_60s", 0.0, 60.0),
                ("60_120s", 60.0, 120.0),
                ("120_180s", 120.0, 180.0),
                ("180_300s", 180.0, 300.0),
                ("300s_plus", 300.0, float("inf")),
            ):
                selected = (durations >= low) & (durations < high)
                add_mean(f"loss_semantic_duration_{bucket}", nll[selected])
                add_mean(f"acc_semantic_duration_{bucket}", correct[selected])

            frequency_buckets = self._semantic_frequency_buckets
            if frequency_buckets is not None:
                local_ids = flat_targets.index_select(0, evaluated) - registry.semantic_base
                frequency_buckets = frequency_buckets.to(
                    device=local_ids.device, non_blocking=True
                )
                assignments = frequency_buckets.index_select(0, local_ids)
                for bucket_id, bucket in enumerate(("head", "middle", "tail", "unseen")):
                    selected = assignments == bucket_id
                    add_mean(f"loss_semantic_frequency_{bucket}", nll[selected])
                    add_mean(f"acc_semantic_frequency_{bucket}", correct[selected])

            calibration_bin = (
                confidence.mul(float(ece_bins)).floor().long().clamp_(0, ece_bins - 1)
            )
            confidence_sum = torch.bincount(
                calibration_bin, weights=confidence, minlength=ece_bins
            )
            correct_sum = torch.bincount(
                calibration_bin,
                weights=correct.to(torch.float32),
                minlength=ece_bins,
            )
            bin_count = torch.bincount(calibration_bin, minlength=ece_bins)


            for index in range(ece_bins):
                prefix = f"_calibration_semantic_bin_{index:02d}"
                metrics[f"{prefix}_confidence_sum"] = confidence_sum[index]
                metrics[f"{prefix}_correct_sum"] = correct_sum[index]
                metrics[f"{prefix}_count"] = bin_count[index].to(torch.float32)

        semantic_predictions = None
        if return_predictions:
            predicted_targets = torch.full_like(target_matrix, -1)
            predicted_targets.reshape(-1).index_copy_(
                0, evaluated, constrained_top1
            )
            semantic_predictions = F.pad(predicted_targets, (1, 0), value=-1)
        return metrics, semantic_predictions


def _subsample_positions(indices: torch.Tensor, limit: int) -> torch.Tensor:
    count = indices.numel()
    if count <= limit:
        return indices
    positions = torch.linspace(0, count - 1, limit, device=indices.device).round_()
    return indices.index_select(0, positions.to(torch.long))


def _head_cross_entropy(
    head: nn.Module, hidden: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    logits = head(hidden)
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def _head_cross_entropies(
    head: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    constraint_kinds: torch.Tensor,
    constraint_table: torch.Tensor,
    constraint_lengths: torch.Tensor,
) -> torch.Tensor:

    logits = head(hidden).float()
    full = F.cross_entropy(logits, labels, reduction="sum")
    constrained = logits.new_zeros(())
    for raw_kind in torch.unique(constraint_kinds).tolist():
        kind = int(raw_kind)
        rows = constraint_kinds == kind
        count = int(constraint_lengths[kind].item())
        allowed = constraint_table[kind, :count]
        row_logits = logits[rows]
        row_labels = labels[rows]
        target_is_allowed = (row_labels[:, None] == allowed[None, :]).any(dim=1)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(
                target_is_allowed.all(),
                f"target is outside grammar constraint kind={kind}",
            )
        elif not bool(target_is_allowed.all()):  # pragma: no cover -  torch
            raise RuntimeError(f"target does not belong to grammar constraint kind={kind}")
        target_logits = row_logits.gather(1, row_labels[:, None]).squeeze(1)
        constrained = constrained + (
            torch.logsumexp(row_logits.index_select(1, allowed), dim=1)
            - target_logits
        ).sum()
    return torch.stack([constrained, full])


def _untie_output_embeddings(backbone: nn.Module) -> None:
    input_embeddings = backbone.get_input_embeddings()
    output_embeddings = backbone.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("Base model has no output embedding; cannot untie weights")
    if output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr():
        return
    output_embeddings.weight = nn.Parameter(input_embeddings.weight.detach().clone())
    backbone.config.tie_word_embeddings = False


@torch.no_grad()
def embedding_statistics(
    weight: torch.Tensor, *, row_chunk: int = 16384
) -> dict[str, float]:
    rows, dim = int(weight.shape[0]), int(weight.shape[1])
    device = weight.device
    total = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_sq = 0.0
    sum_all = 0.0
    norms = torch.empty(rows, dtype=torch.float32, device=device)
    for start in range(0, rows, row_chunk):
        block = weight[start : start + row_chunk].detach().to(torch.float32)
        total += block.sum(dim=0).to(torch.float64)
        sum_all += float(block.sum().item())
        sum_sq += float(block.pow(2).sum().item())
        norms[start : start + block.shape[0]] = block.norm(dim=1)
    count = rows * dim
    mean_vector = (total / rows).to(torch.float32)
    elementwise_mean = sum_all / count
    variance = max(sum_sq / count - elementwise_mean**2, 0.0)
    quantiles = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=device)
    p01, p05, median, p95, p99 = torch.quantile(norms, quantiles).tolist()
    return {
        "rows": float(rows),
        "dim": float(dim),
        "elementwise_mean": elementwise_mean,
        "elementwise_std": math.sqrt(variance),
        "mean_vector_norm": float(mean_vector.norm().item()),
        "mean_vector_std": float(mean_vector.std().item()),
        "row_norm_mean": float(norms.mean().item()),
        "row_norm_p01": p01,
        "row_norm_p05": p05,
        "row_norm_median": median,
        "row_norm_p95": p95,
        "row_norm_p99": p99,
    }


@torch.no_grad()
def new_row_init_plan(
    reference: torch.Tensor,
    *,
    init_mode: str = "row_norm_matched",
    init_std_scale: float = 0.02,
    init_norm_ratio: float = 1.0,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    if init_mode not in EMBEDDING_INIT_MODES:
        raise ValueError(
            f"Unknown init_mode={init_mode!r}; expected one of {list(EMBEDDING_INIT_MODES)}"
        )
    if not math.isfinite(init_norm_ratio) or init_norm_ratio <= 0.0:
        raise ValueError(f"init_norm_ratio={init_norm_ratio} must be a finite positive number")
    stats = embedding_statistics(reference)
    dim = int(reference.shape[1])
    center = reference.detach().to(torch.float32).mean(dim=0)
    center_sq = float(center.pow(2).sum())
    median = stats["row_norm_median"]


    stats["min_init_norm_ratio"] = math.sqrt(center_sq) / max(median, 1e-12)
    if init_mode == "mean_plus_noise":
        if init_norm_ratio != 1.0:


            raise ValueError(
                "init_norm_ratio is only valid with init_mode='row_norm_matched'; "
                "use init_std_scale with init_mode='mean_plus_noise' "
                f"(current ratio={init_norm_ratio})"
            )


        std = stats["elementwise_std"] * init_std_scale
        target = math.sqrt(center_sq + dim * std**2)
    else:

        target = median * init_norm_ratio
        residual = target**2 - center_sq
        if residual <= 0.0:
            raise ValueError(
                f"init_norm_ratio={init_norm_ratio} is too small: target row norm "
                f"{target:.5f} does not exceed mean-vector norm "
                f"{math.sqrt(center_sq):.5f}. The minimum ratio for this base model is "
                f"{stats['min_init_norm_ratio']:.4f} (= ‖mean‖ / median(‖old row‖)). "
                "Use another initialization mode for a smaller new-row norm."
            )
        std = math.sqrt(residual / dim)
    stats["target_row_norm"] = target
    stats["expected_pairwise_cos"] = center_sq / max(target**2, 1e-24)
    if stats["expected_pairwise_cos"] > 0.75:


        warnings.warn(
            f"init_norm_ratio={init_norm_ratio} gives an expected pairwise cosine of "
            f"{stats['expected_pairwise_cos']:.3f} because the mean vector accounts for "
            "most of the target norm; new token rows may separate slowly",
            RuntimeWarning,
            stacklevel=2,
        )
    return center, std, stats


_LEAF_CODES: dict[str, int] = {"melody_struct": 1, "semantic": 2, "boundary": 3}


def _control_leaf_codes(registry: TokenRegistry) -> torch.Tensor:
    codes = torch.zeros(registry.num_control, dtype=torch.long)
    name_to_leaf: dict[str, str] = {}
    for leaf, names in _CONTROL_GROUPS.items():
        for name in names:
            name_to_leaf[name] = leaf
    for index, name in enumerate(registry.control_names):

        leaf = name_to_leaf.get(name)
        if leaf is None and name.startswith("section_"):
            leaf = "melody_struct"
        if leaf is not None:
            codes[index] = _LEAF_CODES[leaf]
    return codes


def _normalize_region_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    resolved = dict.fromkeys(LOSS_REGIONS, 1.0)
    if not weights:
        return resolved
    unknown = sorted(set(weights) - set(LOSS_REGIONS))
    if unknown:
        hint = ""
        if "control" in unknown:


            hint = (
                "; use 'melody' for structural tokens, 'semantic' for music boundaries, "
                "and 'boundary' for sequence EOS"
            )
        raise KeyError(
            f"region_loss_weights contains unknown regions {unknown}; expected "
            f"{list(LOSS_REGIONS)}{hint}"
        )
    for name, value in weights.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"region_loss_weights[{name}]={value} must be a finite non-negative number")
        resolved[name] = value
    if all(value == 0.0 for value in resolved.values()):
        raise ValueError("All region_loss_weights are zero; no region can produce gradients")
    return resolved


@torch.no_grad()
def _resize_and_init(
    backbone: nn.Module,
    registry: TokenRegistry,
    *,
    init_mode: str = "row_norm_matched",
    init_std_scale: float = 0.02,
    init_norm_ratio: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    if init_mode not in EMBEDDING_INIT_MODES:
        raise ValueError(
            f"Unknown init_mode={init_mode!r}; expected one of {list(EMBEDDING_INIT_MODES)}"
        )
    old_size = backbone.get_input_embeddings().num_embeddings
    target = registry.total_vocab_size
    if target < old_size:
        raise ValueError(
            f"registry.total_vocab_size={target} is smaller than the base vocabulary "
            f"{old_size}; shrinking the table is unsupported"
        )
    if target == old_size:
        return {}


    backbone.resize_token_embeddings(target, mean_resizing=False)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    stats: dict[str, float] = {}

    def initialize(module: nn.Module, tag: str) -> None:
        weight = module.weight
        dim = int(weight.shape[1])
        center, std, reference = new_row_init_plan(
            weight[:old_size],
            init_mode=init_mode,
            init_std_scale=init_std_scale,
            init_norm_ratio=init_norm_ratio,
        )


        noise = torch.randn(
            (target - old_size, dim), generator=generator, dtype=torch.float32
        ).to(center.device)
        new_rows = center.unsqueeze(0) + noise * std
        weight[old_size:] = new_rows.to(weight.dtype)
        new_norms = new_rows.norm(dim=1)
        stats.update(
            {
                f"{tag}/old_row_norm_median": reference["row_norm_median"],
                f"{tag}/old_row_norm_p05": reference["row_norm_p05"],
                f"{tag}/old_row_norm_p95": reference["row_norm_p95"],
                f"{tag}/old_elementwise_std": reference["elementwise_std"],
                f"{tag}/mean_vector_norm": reference["mean_vector_norm"],
                f"{tag}/new_noise_std": std,
                f"{tag}/new_row_norm_median": float(new_norms.median()),
                f"{tag}/new_over_old_norm": float(new_norms.median())
                / max(reference["row_norm_median"], 1e-12),


                f"{tag}/target_row_norm": reference["target_row_norm"],
                f"{tag}/expected_pairwise_cos": reference["expected_pairwise_cos"],
                f"{tag}/min_init_norm_ratio": reference["min_init_norm_ratio"],
            }
        )

    input_embeddings = backbone.get_input_embeddings()
    initialize(input_embeddings, "input_embeddings")
    output_embeddings = backbone.get_output_embeddings()
    if (
        output_embeddings is not None
        and output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr()
    ):
        initialize(output_embeddings, "output_embeddings")
    stats["num_new_rows"] = float(target - old_size)
    return stats


def build_model(
    config: dict[str, Any],
    registry: TokenRegistry,
    *,
    device: torch.device | str | None = None,
) -> MusicLLM:
    section = dict(config.get("model", {}) or {})
    train_section = dict(config.get("train", {}) or {})
    if section.get("from_scratch"):
        return MusicLLM.from_scratch_config(
            registry,
            hidden_size=int(section.get("hidden_size", 256)),
            num_hidden_layers=int(section.get("num_hidden_layers", 2)),
            num_attention_heads=int(section.get("num_attention_heads", 4)),
            num_key_value_heads=int(section.get("num_key_value_heads", 2)),
            intermediate_size=int(section.get("intermediate_size", 512)),
            max_position_embeddings=int(section.get("max_position_embeddings", 8192)),
            dtype=str(section.get("dtype", "fp32")),
            attn_implementation=str(section.get("attn_implementation", "eager")),
            loss_chunk_size=int(section.get("loss_chunk_size", 1024)),
            region_loss_weights=section.get("region_loss_weights"),
            melody_class_weights=section.get("melody_class_weights"),
            partitioned_embeddings=bool(
                section.get("partitioned_embeddings", False)
            ),
            tie_music_embeddings=bool(
                section.get("tie_music_embeddings", False)
            ),
            tie_embedding_namespaces=section.get("tie_embedding_namespaces"),
            semantic_output_residual_rank=int(
                section.get("semantic_output_residual_rank", 0)
            ),
            modality_type_embeddings=bool(
                section.get("modality_type_embeddings", False)
            ),
            grammar_constrained_loss=bool(
                section.get("grammar_constrained_loss", False)
            ),
            grammar_constrained_metrics=bool(
                section.get("grammar_constrained_metrics", False)
            ),
        )
    base_path = section.get("base_model_path")
    if not base_path:
        raise KeyError("model.base_model_path not configured(or use model.from_scratch: true)")


    if "local_cache_dir" in section:
        cache_root = section["local_cache_dir"]
    else:
        cache_root = DEFAULT_CACHE_ROOT
    base_path = stage_model_dir(base_path, cache_root=cache_root)
    return MusicLLM.from_pretrained_base(
        base_path,
        registry,
        dtype=str(section.get("dtype", "bf16")),
        attn_implementation=str(section.get("attn_implementation", "flash_attention_2")),
        loss_chunk_size=int(section.get("loss_chunk_size", 4096)),
        untie_word_embeddings=bool(section.get("untie_word_embeddings", False)),
        partitioned_embeddings=bool(
            section.get("partitioned_embeddings", False)
        ),
        tie_music_embeddings=bool(
            section.get("tie_music_embeddings", False)
        ),
        tie_embedding_namespaces=section.get("tie_embedding_namespaces"),
        semantic_output_residual_rank=int(
            section.get("semantic_output_residual_rank", 0)
        ),
        modality_type_embeddings=bool(
            section.get("modality_type_embeddings", False)
        ),
        grammar_constrained_loss=bool(
            section.get("grammar_constrained_loss", False)
        ),
        grammar_constrained_metrics=bool(
            section.get("grammar_constrained_metrics", False)
        ),
        gradient_checkpointing=bool(section.get("gradient_checkpointing", False)),
        init_mode=str(section.get("embedding_init_mode", "row_norm_matched")),
        init_std_scale=float(section.get("init_std_scale", 0.02)),
        init_norm_ratio=float(section.get("init_norm_ratio", 1.0)),
        seed=int(train_section.get("seed", 0)),
        accuracy_sample_positions=int(section.get("accuracy_sample_positions", 4096)),
        region_loss_weights=section.get("region_loss_weights"),
        melody_class_weights=section.get("melody_class_weights"),
        device=device,
        load_dtype=section.get("load_dtype"),
    )
