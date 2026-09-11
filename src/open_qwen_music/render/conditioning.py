
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import (
    DynamicsPredictorConfig,
    DynamicsTargetConfig,
    SemanticDynamicsPredictor,
)

SEMANTIC_VOCAB_SIZE = 32_768
SEMANTIC_EMBEDDING_ASSET_SCHEMA = "oqm.render-semantic-embedding.v1"
SEMANTIC_EMBEDDING_READY_SCHEMA = "oqm.render-semantic-embedding-ready.v1"
SEMANTIC_EMBEDDING_READY_STATUS = "RENDER_SEMANTIC_EMBEDDING_READY"
SEMANTIC_EMBEDDING_SOURCES = (
    "tokenizer_effective_codebook",
    "tokenizer_and_latent_fit",
)
DESCRIPTION_MAX_TOKENS = 256
LYRICS_MAX_TOKENS = 1_536


SUPPORTED_LYRICS_MAX_TOKENS = (1_536, 1_792)
LYRICS_ENCODER_LAYERS = 6
CONDITIONING_INITIALIZATION_MODES = ("dit_xavier",)
GELU_APPROXIMATIONS = ("none", "tanh")
LYRICS_NORM_TYPES = ("layernorm", "rmsnorm")
TEXT_CONTEXT_COMPOSITIONS = ("concatenate",)
TEXT_CONTEXT_LAYOUTS = ("description_then_lyrics",)
TEXT_CONTEXT_SEPARATOR_MODES = ("none",)
TEXT_CONTEXT_SEGMENT_EMBEDDING_MODES = ("none",)
NULL_CONTEXT_LAYOUTS = ("single_token", "preserve_text_mask")
LYRICS_ATTENTION_DIRECTIONS = ("bidirectional",)
LYRICS_NORM_STYLES = ("pre_norm_with_final_norm",)
LYRICS_PROJECTION_ORDERS = ("before_encoder",)
ROPE_STYLES = ("half_split",)
REWRITER_SCHEMA_VERSION = "oqm.rewriter-output.v1"
_FLOATING_REVISIONS = {
    "",
    "main",
    "master",
    "latest",
    "head",
    "unknown",
    "unresolved",
    "none",
    "null",
    "required",
    "required_pinned_revision",
}


def _require_pinned_revision(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    revision = value.strip()
    lowered = revision.lower()
    if lowered in _FLOATING_REVISIONS or lowered.startswith("pin-"):
        raise ValueError(
            f"{name} must be a pinned, traceable revision; {value!r} is not allowed"
        )
    return revision


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    digest = value.lower()
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a hexadecimal SHA-256") from exc
    return digest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _load_semantic_embedding_asset(
    config: Mapping[str, Any],
    *,
    expected_tokenizer_revision: str,
    expected_source_dim: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any], bool]:

    allowed = {
        "ready_path",
        "ready_sha256",
        "artifact_sha256",
        "asset_revision",
        "embedding_key",
        "tensor_sha256",
        "trainable",
        "expected_source",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            f"semantic_embedding_asset contains  unknown fields:{sorted(unknown)}"
        )
    ready_value = config.get("ready_path")
    if not isinstance(ready_value, (str, Path)) or not str(ready_value):
        raise ValueError("semantic_embedding_asset.ready_path must be a non-empty path")
    ready_path = Path(ready_value).expanduser().resolve(strict=True)
    expected_ready_sha = _require_sha256(
        "semantic embedding ready_sha256",
        config.get("ready_sha256"),
    )
    actual_ready_sha = _file_sha256(ready_path)
    if actual_ready_sha != expected_ready_sha:
        raise RuntimeError(
            "semantic embedding READY SHA inconsistent:"
            f"expected={expected_ready_sha} actual={actual_ready_sha}"
        )
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("semantic embedding READY cannot be parsed") from exc
    if not isinstance(ready, Mapping):
        raise TypeError("semantic embedding READY must be a JSON object")
    if (
        ready.get("schema_version") != SEMANTIC_EMBEDDING_READY_SCHEMA
        or ready.get("status") != SEMANTIC_EMBEDDING_READY_STATUS
    ):
        raise RuntimeError("semantic embedding READY schema/status is incompatible")
    asset_revision = _require_sha256(
        "semantic embedding asset_revision",
        config.get("asset_revision"),
    )
    if ready.get("asset_revision") != asset_revision:
        raise RuntimeError("semantic embedding asset revision does not match READY")
    tokenizer_revision = _require_sha256(
        "semantic tokenizer_revision",
        expected_tokenizer_revision,
    )
    if ready.get("tokenizer_revision") != tokenizer_revision:
        raise RuntimeError("semantic embedding asset is bound to a different Tokenizer revision")
    contract = ready.get("semantic_contract")
    if (
        not isinstance(contract, Mapping)
        or float(contract.get("frame_hz", -1.0)) != 25.0
        or int(contract.get("codebook_size", -1)) != SEMANTIC_VOCAB_SIZE
        or int(contract.get("codebooks", -1)) != 1
    ):
        raise RuntimeError("semantic embedding asset violates the 25 Hz, single-codebook, 32,768-entry contract")
    artifact = ready.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("semantic embedding READY is missing an artifact entry")
    artifact_path = Path(str(artifact.get("path") or ""))
    if not artifact_path.is_absolute():
        artifact_path = ready_path.parent / artifact_path
    artifact_path = artifact_path.resolve(strict=True)
    expected_artifact_sha = _require_sha256(
        "semantic embedding artifact.sha256",
        config.get("artifact_sha256"),
    )
    if artifact.get("sha256") != expected_artifact_sha:
        raise RuntimeError("semantic embedding artifact SHA does not match READY")
    if _file_sha256(artifact_path) != expected_artifact_sha:
        raise RuntimeError("semantic embedding artifact SHA-256 mismatch")
    ready_source = str(ready.get("source") or "")
    if ready_source not in SEMANTIC_EMBEDDING_SOURCES:
        raise RuntimeError(f"semantic embedding READY source is invalid: {ready_source!r}")
    if artifact_path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "Loading public semantic embedding requires safetensors"
            ) from exc
        try:
            with safe_open(
                artifact_path,
                framework="pt",
                device="cpu",
            ) as handle:
                metadata = handle.metadata() or {}
                embeddings = {
                    name: handle.get_tensor(name)
                    for name in handle.keys()
                }
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("semantic embedding safetensors could not be loaded") from exc
        expected_metadata = {
            "asset_revision": asset_revision,
            "tokenizer_revision": tokenizer_revision,
            "source": ready_source,
        }
        mismatches = {
            key: {"expected": value, "actual": metadata.get(key)}
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "semantic embedding safetensors metadata is incompatible: "
                f"{mismatches}"
            )
    else:
        try:
            payload = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("semantic embedding artifact could not be loaded safely") from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != SEMANTIC_EMBEDDING_ASSET_SCHEMA
            or payload.get("asset_revision") != asset_revision
            or payload.get("tokenizer_revision") != tokenizer_revision
        ):
            raise RuntimeError("semantic embedding artifact identity is incompatible")
        payload_source = str(payload.get("source") or "")
        if payload_source != ready_source:
            raise RuntimeError("semantic embedding READY and artifact sources differ")
        embeddings = payload.get("embeddings")
    expected_source = config.get("expected_source")
    if expected_source is not None:
        if expected_source not in SEMANTIC_EMBEDDING_SOURCES:
            raise ValueError(
                f"semantic_embedding_asset.expected_source is invalid:{expected_source!r}"
            )
        if ready_source != expected_source:
            raise RuntimeError(
                "semantic embedding: the  source is inconsistent:"
                f"expected={expected_source} actual={ready_source}"
            )
    embedding_key = str(config.get("embedding_key") or "")
    declared = ready.get("embeddings")
    if (
        not embedding_key
        or not isinstance(declared, Mapping)
        or not isinstance(embeddings, Mapping)
        or embedding_key not in declared
        or embedding_key not in embeddings
    ):
        raise RuntimeError("semantic embedding key was replaced by the joint READY and artifact declaration")
    weight = embeddings[embedding_key]
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.shape[0] != SEMANTIC_VOCAB_SIZE
        or not weight.is_floating_point()
        or not torch.isfinite(weight).all()
    ):
        raise RuntimeError("semantic embedding tensor must be finite with shape [32768, D]")
    if expected_source_dim is not None and int(weight.shape[1]) != int(
        expected_source_dim
    ):
        raise RuntimeError(
            "semantic embedding source dimis inconsistent with the configuration:"
            f"expected={expected_source_dim} actual={int(weight.shape[1])}"
        )
    expected_tensor_sha = _require_sha256(
        "semantic_embedding_asset.tensor_sha256",
        config.get("tensor_sha256"),
    )
    declared_tensor = declared[embedding_key]
    if (
        not isinstance(declared_tensor, Mapping)
        or declared_tensor.get("sha256") != expected_tensor_sha
        or declared_tensor.get("shape") != list(weight.shape)
        or declared_tensor.get("dtype")
        != str(weight.dtype).removeprefix("torch.")
        or _tensor_sha256(weight) != expected_tensor_sha
    ):
        raise RuntimeError("semantic embedding tensor identity, shape, or dtype does not match the declaration")
    trainable = config.get("trainable", False)
    if not isinstance(trainable, bool):
        raise TypeError("semantic_embedding_asset.trainable must be bool")
    provenance = {
        "ready_path": str(ready_path),
        "ready_sha256": actual_ready_sha,
        "artifact_path": str(artifact_path),
        "artifact_sha256": expected_artifact_sha,
        "asset_revision": asset_revision,
        "embedding_key": embedding_key,
        "tensor_sha256": expected_tensor_sha,
        "tokenizer_revision": tokenizer_revision,
        "source_dim": int(weight.shape[1]),
        "trainable": trainable,
        "source": ready_source,
        "vae_independent": ready_source == "tokenizer_effective_codebook",
    }
    return weight.detach().float().contiguous(), provenance, trainable


@dataclass(frozen=True)
class TextEncoderProvenance:

    model_id: str
    model_revision: str
    tokenizer_revision: str
    cache_revision: str

    def validate(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("text encoder model_id cannot be empty")
        for name in ("model_revision", "tokenizer_revision", "cache_revision"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"text encoder {name} must be a string")
        _require_pinned_revision("text encoder model_revision", self.model_revision)
        _require_pinned_revision("text tokenizer_revision", self.tokenizer_revision)
        _require_pinned_revision("text cache_revision", self.cache_revision)

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TextEncoderProvenance":
        if not isinstance(value, Mapping):
            raise TypeError("text provenance must be a mapping")
        aliases = {
            "revision": "model_revision",
            "text_encoder_revision": "model_revision",
            "text_tokenizer_revision": "tokenizer_revision",
            "text_cache_revision": "cache_revision",
        }
        canonical_fields = tuple(cls.__dataclass_fields__)
        unknown = set(value) - set(canonical_fields) - set(aliases)
        if unknown:
            raise ValueError(
                f"text provenance contains unknown fields: {sorted(unknown)}"
            )
        normalized = {
            name: value[name] for name in canonical_fields if name in value
        }
        for destination in canonical_fields:
            candidates = [
                (name, value[name])
                for name in (destination, *(
                    source
                    for source, target in aliases.items()
                    if target == destination
                ))
                if name in value
            ]
            if not candidates:
                continue
            first_name, first_value = candidates[0]
            conflicts = [
                name for name, candidate in candidates[1:] if candidate != first_value
            ]
            if conflicts:
                names = [first_name, *conflicts]
                raise ValueError(
                    f"text provenance aliases for {destination} conflict: {names}"
                )
            normalized[destination] = first_value
        result = cls(
            model_id=normalized.get("model_id", ""),
            model_revision=normalized.get("model_revision", ""),
            tokenizer_revision=normalized.get("tokenizer_revision", ""),
            cache_revision=normalized.get("cache_revision", ""),
        )
        result.validate()
        return result


def validate_cache_provenance(
    actual: Mapping[str, Any] | TextEncoderProvenance,
    expected: TextEncoderProvenance,
) -> None:

    observed = (
        actual
        if isinstance(actual, TextEncoderProvenance)
        else TextEncoderProvenance.from_mapping(actual)
    )
    expected.validate()
    if observed != expected:
        mismatches = {
            key: (getattr(expected, key), getattr(observed, key))
            for key in asdict(expected)
            if getattr(expected, key) != getattr(observed, key)
        }
        raise RuntimeError(f"Text condition cache revision mismatch: {mismatches}")


def _infer_encoder_dim(encoder: nn.Module) -> int | None:
    for owner in (encoder, getattr(encoder, "config", None)):
        if owner is None:
            continue
        for name in ("hidden_size", "embedding_dim", "dim", "d_model"):
            value = getattr(owner, name, None)
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise TypeError(
                        f"text encoder {name} must be a positive integer,received{value!r}"
                    )
                return value
    return None


TEXT_HIDDEN_STATE_SELECTIONS = ("last_hidden_state",)
TEXT_POSITION_ID_POLICIES = ("attention_mask_cumsum",)
TEXT_EMPTY_POLICIES = ("zero_valid_tokens",)
TEXT_COMPACTION_POLICIES = ("stable_valid_tokens_right_padded",)


def _extract_hidden_state(
    output: Any,
    *,
    selection: str = "last_hidden_state",
) -> torch.Tensor:
    if selection not in TEXT_HIDDEN_STATE_SELECTIONS:
        raise ValueError(
            "text hidden state selection must be"
            f"{TEXT_HIDDEN_STATE_SELECTIONS}"
        )
    if isinstance(output, torch.Tensor):
        hidden = output
    elif isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    else:
        hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError(
            "text encoder must return Tensor[B,L,D] or containinglast_hidden_state"
        )
    return hidden


class FrozenQwenEmbeddingAdapter(nn.Module):

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tokenizer_revision: str,
        cache_revision: str,
        encoder: nn.Module | None = None,
        tokenizer: Any | None = None,
        hidden_size: int | None = None,
        local_path: str | Path | None = None,
        local_files_only: bool = True,
        use_fast_tokenizer: bool = True,
        padding_side: str = "left",
        truncation_side: str = "right",
        truncation_policy: str = "allow",
        add_special_tokens: bool = True,
        trust_remote_code: bool = False,
        hidden_state_selection: str = "last_hidden_state",
        position_id_policy: str = "attention_mask_cumsum",
        encoder_use_cache: bool = False,
        empty_text_policy: str = "zero_valid_tokens",
        frozen_eval_mode: bool = True,
    ) -> None:
        super().__init__()
        self.provenance = TextEncoderProvenance(
            model_id=model_id,
            model_revision=revision,
            tokenizer_revision=tokenizer_revision,
            cache_revision=cache_revision,
        )
        self.provenance.validate()
        self.encoder = encoder
        self._tokenizer = tokenizer
        self.local_path = Path(local_path).resolve() if local_path else None
        inferred = _infer_encoder_dim(encoder) if encoder is not None else None
        if hidden_size is not None and (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError("text encoder hidden_size must be a positive integer")
        if inferred is not None and inferred <= 0:
            raise ValueError("cannot accept non-positive text encoder hidden_size")
        if hidden_size is not None and inferred is not None and hidden_size != inferred:
            raise ValueError(
                "injectiontext encoder hidden_sizeis inconsistent with the configuration:"
                f"configured={hidden_size} inferred={inferred}"
            )
        self.hidden_size = hidden_size or inferred or 0
        if self.hidden_size <= 0:
            raise ValueError("must explicitly provide verifiable text encoder hidden_size")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be bool")
        if not local_files_only:
            raise ValueError(
                "Renderer text conditioning disables implicit downloads; local_files_only must be true"
            )
        self.local_files_only = True
        if not isinstance(use_fast_tokenizer, bool):
            raise TypeError("use_fast_tokenizer must be bool")
        self.use_fast_tokenizer = use_fast_tokenizer
        if padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be left or right")
        if truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side must be left or right")
        if truncation_policy not in {"allow", "reject"}:
            raise ValueError("truncation_policy must be allow or reject")
        if not isinstance(add_special_tokens, bool):
            raise TypeError("add_special_tokens must be bool")
        if trust_remote_code is not False:
            raise ValueError("Render text encoder requires trust_remote_code=false")
        if hidden_state_selection not in TEXT_HIDDEN_STATE_SELECTIONS:
            raise ValueError(
                "hidden_state_selection must be"
                f"{TEXT_HIDDEN_STATE_SELECTIONS}"
            )
        if position_id_policy not in TEXT_POSITION_ID_POLICIES:
            raise ValueError(
                "position_id_policy must be"
                f"{TEXT_POSITION_ID_POLICIES}"
            )
        if encoder_use_cache is not False:
            raise ValueError("Render text encoder use_cache must be false")
        if empty_text_policy not in TEXT_EMPTY_POLICIES:
            raise ValueError(
                f"empty_text_policy must be one of {TEXT_EMPTY_POLICIES}"
            )
        if frozen_eval_mode is not True:
            raise ValueError("The frozen text encoder must remain in evaluation mode")
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.truncation_policy = truncation_policy
        self.add_special_tokens = add_special_tokens
        self.trust_remote_code = trust_remote_code
        self.hidden_state_selection = hidden_state_selection
        self.position_id_policy = position_id_policy
        self.encoder_use_cache = encoder_use_cache
        self.empty_text_policy = empty_text_policy
        self.frozen_eval_mode = frozen_eval_mode
        if self._tokenizer is not None:
            self._configure_tokenizer(self._tokenizer)
        if encoder is not None:
            self._freeze_encoder()

    def _configure_tokenizer(self, tokenizer: Any) -> None:
        if self.use_fast_tokenizer and getattr(tokenizer, "is_fast", True) is not True:
            raise RuntimeError("Render text conditioning requires a fast tokenizer")
        try:
            tokenizer.padding_side = self.padding_side
            tokenizer.truncation_side = self.truncation_side
        except (AttributeError, TypeError) as exc:
            raise TypeError("tokenizer must support fixed padding/truncation side") from exc

    def _freeze_encoder(self) -> None:
        if self.encoder is None:
            return
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def _lazy_load_encoder(self) -> nn.Module:
        if self.encoder is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise RuntimeError(
                    "Text encoding requires transformers; CPU tests may inject a test encoder"
                ) from exc
            source = str(self.local_path or self.provenance.model_id)
            if self.local_path is not None and not self.local_path.is_dir():
                raise FileNotFoundError(f"Local text encoder directory does not exist: {self.local_path}")
            self.encoder = AutoModel.from_pretrained(
                source,
                revision=(
                    None
                    if self.local_path is not None
                    else self.provenance.model_revision
                ),
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            inferred = _infer_encoder_dim(self.encoder)
            if inferred is None:
                raise RuntimeError("cannot start from the Qwen encoder config to infer hidden_size")
            if self.hidden_size and self.hidden_size != inferred:
                raise RuntimeError(
                    "text encoder hidden size is inconsistent with the configuration:"
                    f"expected={self.hidden_size} actual={inferred}"
                )
            self.hidden_size = inferred
            self._freeze_encoder()
        return self.encoder

    def _lazy_load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "String text input requires a transformers tokenizer; tests may pass input_ids directly"
                ) from exc
            source = str(self.local_path or self.provenance.model_id)
            if self.local_path is not None and not self.local_path.is_dir():
                raise FileNotFoundError(
                    f"Local Qwen tokenizer directory does not exist: {self.local_path}"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                revision=(
                    None
                    if self.local_path is not None
                    else self.provenance.tokenizer_revision
                ),
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.use_fast_tokenizer,
            )
            self._configure_tokenizer(self._tokenizer)
        return self._tokenizer

    def tokenize(
        self,
        texts: str | Sequence[str],
        *,
        max_length: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise TypeError("text max_length must be an integer")
        if max_length <= 0:
            raise ValueError("text max_length must be positive")
        tokenizer = self._lazy_load_tokenizer()
        values = [texts] if isinstance(texts, str) else list(texts)
        if not values or not all(isinstance(value, str) for value in values):
            raise TypeError("texts must be a non-empty string or a sequence of non-empty strings")
        encoded = tokenizer(
            values,
            padding=True,
            truncation=self.truncation_policy == "allow",
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("tokenizer output must be a mapping")
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("tokenizer output is missing input_ids and attention_mask")
        input_ids = torch.as_tensor(encoded["input_ids"]).to(
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.as_tensor(encoded["attention_mask"]).to(
            device=device,
            dtype=torch.bool,
        )
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("tokenizer must return input_ids and attention_mask with the same [B, L] shape")
        if input_ids.shape[0] != len(values) or input_ids.shape[1] > max_length:
            if (
                self.truncation_policy == "reject"
                and input_ids.shape[1] > max_length
            ):
                raise ValueError(
                    f"text token count exceeds max_length={max_length}; silent truncation is disabled"
                )
            raise RuntimeError("tokenizer batch size or max_length violates the text encoder contract")
        if input_ids.numel() and int(input_ids.min()) < 0:
            raise ValueError("tokenizer input_ids must not contain negative numbers")
        empty_rows = torch.tensor(
            [not value.strip() for value in values],
            device=attention_mask.device,
            dtype=torch.bool,
        )
        if bool(empty_rows.any()):


            attention_mask = attention_mask.masked_fill(
                empty_rows.unsqueeze(1),
                False,
            )
        return input_ids, attention_mask

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise TypeError("text input_ids must be int64 with shape [B, L]")
        if (
            attention_mask.dtype != torch.bool
            or attention_mask.shape != input_ids.shape
        ):
            raise TypeError("text attention_mask must be a bool tensor matching input_ids")
        if input_ids.device != attention_mask.device:
            raise ValueError("text input_ids and attention_mask must be on the same device")
        active_rows = attention_mask.any(dim=1)
        if input_ids.shape[1] == 0 or not bool(active_rows.any()):


            return torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                self.hidden_size,
                device=input_ids.device,
                dtype=torch.float32,
            )
        encoder = self._lazy_load_encoder()
        first_tensor = next(
            iter(encoder.parameters()),
            next(iter(encoder.buffers()), None),
        )
        if first_tensor is not None and first_tensor.device != input_ids.device:
            encoder.to(device=input_ids.device)
        self._freeze_encoder()
        active_input_ids = input_ids[active_rows]
        active_attention_mask = attention_mask[active_rows]


        position_ids = (
            active_attention_mask.long().cumsum(dim=-1) - 1
        ).masked_fill(~active_attention_mask, 0)
        call_kwargs = {
            "input_ids": active_input_ids,
            "attention_mask": active_attention_mask.to(dtype=torch.long),
            "position_ids": position_ids,
            "return_dict": True,


            "use_cache": self.encoder_use_cache,
        }
        try:
            signature = inspect.signature(encoder.forward)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            call_kwargs = {
                name: value
                for name, value in call_kwargs.items()
                if name in signature.parameters
            }
        with torch.no_grad():
            output = encoder(**call_kwargs)
            active_hidden = _extract_hidden_state(
                output,
                selection=self.hidden_state_selection,
            )
        if active_hidden.shape[:2] != input_ids[active_rows].shape:
            raise RuntimeError(
                "text encoder not maintained token axis:"
                f"input={tuple(input_ids[active_rows].shape)} "
                f"output={tuple(active_hidden.shape)}"
            )
        if self.hidden_size and active_hidden.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"text encoder hidden={active_hidden.shape[-1]},configured="
                f"{self.hidden_size}"
            )
        self.hidden_size = int(active_hidden.shape[-1])
        if (
            not active_hidden.is_floating_point()
            or active_hidden.device != input_ids.device
        ):
            raise RuntimeError("text encoder must return floating-point hidden states on the input device")
        hidden = active_hidden.new_zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.hidden_size,
        )
        hidden[active_rows] = active_hidden
        masked = hidden.detach().masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(masked).all():
            raise RuntimeError("text encoder valid token output contains NaN/Inf")
        return masked

    def train(self, mode: bool = True) -> "FrozenQwenEmbeddingAdapter":

        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
        *,
        style: str = "half_split",
    ) -> None:
        super().__init__()
        if not isinstance(head_dim, int) or isinstance(head_dim, bool):
            raise TypeError("RoPE head_dim must be an integer")
        if head_dim <= 0 or head_dim % 2:
            raise ValueError(f"RoPE head_dim must be a positive even number, received {head_dim}")
        if (
            isinstance(base, bool)
            or not isinstance(base, (int, float))
            or not math.isfinite(base)
            or base <= 1.0
        ):
            raise ValueError("RoPE base must be finite and greater than 1")
        if style not in ROPE_STYLES:
            raise ValueError(f"RoPE style must be one of {ROPE_STYLES}")


        self.head_dim = head_dim
        self.base = float(base)
        self.style = style

    def _inverse_frequency(
        self,
        device: torch.device,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:


        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        return torch.pow(
            torch.tensor(self.base, device=device, dtype=compute_dtype),
            -torch.arange(
                0,
                self.head_dim,
                2,
                device=device,
                dtype=compute_dtype,
            )
            / float(self.head_dim),
        )

    def forward(
        self, length: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(length, int) or isinstance(length, bool):
            raise TypeError("RoPE length must be an integer")
        if length < 0:
            raise ValueError("RoPE length must not be negative")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("RoPE output dtype must be a floating point type")
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        positions = torch.arange(length, device=device, dtype=compute_dtype)
        angles = torch.outer(
            positions,
            self._inverse_frequency(device, dtype=dtype),
        )
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:

    if value.ndim != 4 or not value.is_floating_point():
        raise TypeError("RoPE value must be floating point with shape [B, H, L, D]")
    if cosine.shape != sine.shape or cosine.ndim != 2:
        raise ValueError("RoPE cosine and sine must have shape [L, D]")
    if cosine.shape != value.shape[-2:]:
        raise ValueError(
            "RoPE cosine and sine must match the value of [L,D] consistent:"
            f"value={tuple(value.shape)} cosine={tuple(cosine.shape)}"
        )
    if (
        cosine.device != value.device
        or sine.device != value.device
        or cosine.dtype != value.dtype
        or sine.dtype != value.dtype
    ):
        raise TypeError("RoPE value, cosine, and sine must share a device and dtype")
    return value * cosine.view(1, 1, cosine.shape[0], cosine.shape[1]) + _rotate_half(
        value
    ) * sine.view(1, 1, sine.shape[0], sine.shape[1])


class LyricsSwiGLU(nn.Module):

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = hidden.chunk(2, dim=-1)
        return F.silu(gate) * value


class LyricsSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        head_dim: int | None = None,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        output_bias: bool = True,
        rope_base: float = 10_000.0,
        rope_style: str = "half_split",
    ) -> None:
        super().__init__()
        for name, value in {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"lyrics attention {name} must be an integer")
            if value <= 0:
                raise ValueError(f"lyrics attention {name} must be positive")
        if head_dim is None:
            if hidden_size % num_heads:
                raise ValueError("lyrics hidden_size must be divisible by num_heads")
            head_dim = hidden_size // num_heads
        elif not isinstance(head_dim, int) or isinstance(head_dim, bool):
            raise TypeError("lyrics attention head_dim must be an integer")
        if head_dim <= 0:
            raise ValueError("lyrics attention head_dim must be positive")
        if head_dim % 2:
            raise ValueError("lyrics attention head_dim must be an even number")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError(
                "lyrics attention dropout must be a finite value in [0, 1)"
            )
        if not isinstance(qkv_bias, bool):
            raise TypeError("lyrics attention qkv_bias must be bool")
        if not isinstance(output_bias, bool):
            raise TypeError("lyrics attention output_bias must be bool")
        if (
            isinstance(rope_base, bool)
            or not isinstance(rope_base, (int, float))
            or not math.isfinite(rope_base)
            or rope_base <= 1.0
        ):
            raise ValueError("lyrics attention rope_base must be finite and greater than 1")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * self.inner_dim, bias=qkv_bias)
        self.output = nn.Linear(
            self.inner_dim,
            hidden_size,
            bias=output_bias,
        )
        self.rope = RotaryEmbedding(
            self.head_dim,
            base=rope_base,
            style=rope_style,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_is_full: bool | None = None,
        mask_validated: bool = False,
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or not hidden.is_floating_point()
            or hidden.shape[-1] != self.hidden_size
        ):
            raise TypeError(
                f"lyrics attention hidden states must be floating point with shape [B, L, {self.hidden_size}]"
            )
        batch, length, _ = hidden.shape
        if attention_mask.dtype != torch.bool or attention_mask.shape != (
            batch,
            length,
        ):
            raise TypeError("lyrics attention_mask must be bool with shape [B, L]")
        if attention_mask.device != hidden.device:
            raise ValueError("lyrics hidden states and attention_mask must be on the same device")
        if not mask_validated:
            mask_is_full = bool(attention_mask.all())
        elif not isinstance(mask_is_full, bool):
            raise TypeError("mask_is_full must be a bool for verified lyrics attention")
        safe_mask = attention_mask
        empty_rows = ~attention_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_mask = attention_mask.clone()
            safe_mask[empty_rows, 0] = True
            mask_is_full = bool(safe_mask.all())
        clean_hidden = hidden.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        query, key, value = self.qkv(clean_hidden).chunk(3, dim=-1)
        target_shape = (batch, length, self.num_heads, self.head_dim)
        query = query.view(target_shape).transpose(1, 2)
        key = key.view(target_shape).transpose(1, 2)
        value = value.view(target_shape).transpose(1, 2)
        cosine, sine = self.rope(length, device=hidden.device, dtype=query.dtype)
        query = apply_rope(query, cosine, sine)
        key = apply_rope(key, cosine, sine)
        dropout = self.dropout if self.training else 0.0
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None if mask_is_full else safe_mask[:, None, None, :],
            dropout_p=dropout,
        )
        projected = self.output(
            attended.transpose(1, 2).reshape(batch, length, self.inner_dim)
        )
        return projected.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


class LyricsEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        head_dim: int | None,
        ffn_expansion: float,
        ffn_activation: str,
        gelu_approximation: str,
        dropout: float,
        qkv_bias: bool,
        non_qkv_linear_bias: bool,
        rope_base: float,
        rope_style: str,
        norm_eps: float,
        norm_type: str,
    ) -> None:
        super().__init__()
        intermediate = max(1, round(hidden_size * float(ffn_expansion)))
        if ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("lyrics ffn_activation must be gelu or swiglu")
        if gelu_approximation not in GELU_APPROXIMATIONS:
            raise ValueError(
                "lyrics gelu_approximation must be"
                f"{GELU_APPROXIMATIONS}"
            )
        if not isinstance(non_qkv_linear_bias, bool):
            raise TypeError("lyrics non_qkv_linear_bias must be bool")
        if norm_type not in LYRICS_NORM_TYPES:
            raise ValueError(f"lyrics norm_type must be one of {LYRICS_NORM_TYPES}")
        norm_class = nn.LayerNorm if norm_type == "layernorm" else nn.RMSNorm
        self.attention_norm = norm_class(hidden_size, eps=norm_eps)
        self.attention = LyricsSelfAttention(
            hidden_size,
            num_heads,
            head_dim=head_dim,
            dropout=dropout,
            qkv_bias=qkv_bias,
            output_bias=non_qkv_linear_bias,
            rope_base=rope_base,
            rope_style=rope_style,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = norm_class(hidden_size, eps=norm_eps)


        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_size,
                (2 if ffn_activation == "swiglu" else 1) * intermediate,
                bias=non_qkv_linear_bias,
            ),
            (
                LyricsSwiGLU()
                if ffn_activation == "swiglu"
                else nn.GELU(approximate=gelu_approximation)
            ),
            nn.Dropout(dropout),
            nn.Linear(
                intermediate,
                hidden_size,
                bias=non_qkv_linear_bias,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_is_full: bool = False,
    ) -> torch.Tensor:
        valid = attention_mask.unsqueeze(-1)
        hidden = hidden.masked_fill(~valid, 0.0)
        hidden = hidden + self.attention_dropout(
            self.attention(
                self.attention_norm(hidden),
                attention_mask,
                mask_is_full,
                True,
            )
        )
        hidden = hidden.masked_fill(~valid, 0.0)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden.masked_fill(~valid, 0.0)


class LyricsRoPEEncoder(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        num_layers: int = LYRICS_ENCODER_LAYERS,
        num_heads: int = 8,
        head_dim: int | None = None,
        ffn_expansion: float = 4.0,
        ffn_activation: str = "gelu",
        gelu_approximation: str = "tanh",
        dropout: float = 0.0,
        qkv_bias: bool = True,
        non_qkv_linear_bias: bool = True,
        rope_base: float = 10_000.0,
        rope_style: str = "half_split",
        norm_eps: float = 1.0e-5,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        if not isinstance(num_layers, int) or isinstance(num_layers, bool):
            raise TypeError("lyrics encoder num_layers must be an integer")
        if num_layers != LYRICS_ENCODER_LAYERS:
            raise ValueError(
                f"lyrics encoder must be {LYRICS_ENCODER_LAYERS} layer, received {num_layers}"
            )
        if (
            isinstance(ffn_expansion, bool)
            or not isinstance(ffn_expansion, (int, float))
            or not math.isfinite(ffn_expansion)
            or ffn_expansion <= 0
        ):
            raise ValueError("lyrics encoder ffn_expansion must be a finite positive number")
        if ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("lyrics encoder ffn_activation must be gelu or swiglu")
        if gelu_approximation not in GELU_APPROXIMATIONS:
            raise ValueError(
                "lyrics encoder gelu_approximation must be"
                f"{GELU_APPROXIMATIONS}"
            )
        if not isinstance(qkv_bias, bool):
            raise TypeError("lyrics encoder qkv_bias must be bool")
        if not isinstance(non_qkv_linear_bias, bool):
            raise TypeError("lyrics encoder non_qkv_linear_bias must be bool")
        if norm_type not in LYRICS_NORM_TYPES:
            raise ValueError(
                f"lyrics encoder norm_type must be one of {LYRICS_NORM_TYPES}"
            )
        for name, value in {"rope_base": rope_base, "norm_eps": norm_eps}.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"lyrics encoder {name} must be a finite positive number")
        self.layers = nn.ModuleList(
            [
                LyricsEncoderBlock(
                    hidden_size,
                    num_heads,
                    head_dim=head_dim,
                    ffn_expansion=ffn_expansion,
                    ffn_activation=ffn_activation,
                    gelu_approximation=gelu_approximation,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                    non_qkv_linear_bias=non_qkv_linear_bias,
                    rope_base=rope_base,
                    rope_style=rope_style,
                    norm_eps=norm_eps,
                    norm_type=norm_type,
                )
                for _ in range(num_layers)
            ]
        )
        norm_class = nn.LayerNorm if norm_type == "layernorm" else nn.RMSNorm
        self.final_norm = norm_class(hidden_size, eps=norm_eps)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or not hidden.is_floating_point()
            or hidden.shape[-1] != self.final_norm.normalized_shape[0]
        ):
            raise TypeError(
                "lyrics encoder hidden must be the same as hidden_size matching floating point with shape [B, L, D]"
            )
        if attention_mask.dtype != torch.bool or attention_mask.shape != hidden.shape[:2]:
            raise TypeError("lyrics encoder attention_mask must be bool with shape [B, L]")
        if attention_mask.device != hidden.device:
            raise ValueError("lyrics hidden states and attention_mask must be on the same device")
        if hidden.shape[1] == 0:
            return hidden
        mask_is_full = bool(attention_mask.all())
        for layer in self.layers:
            hidden = layer(hidden, attention_mask, mask_is_full)
        hidden = self.final_norm(hidden)
        return hidden.masked_fill(~attention_mask.unsqueeze(-1), 0.0)


@dataclass
class ConditioningOutput:
    semantic_embeddings: torch.Tensor
    semantic_mask: torch.Tensor
    text_context: torch.Tensor
    text_mask: torch.Tensor
    text_drop_mask: torch.Tensor
    provenance: dict[str, str]
    relative_dynamics_embeddings: torch.Tensor | None = None

    @property
    def context(self) -> torch.Tensor:
        return self.text_context

    @property
    def context_mask(self) -> torch.Tensor:
        return self.text_mask


class GlobalLoudnessEmbedding(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        mean_lufs: float,
        std_lu: float,
        clamp_std: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError("global loudness hidden_size must be a positive integer")
        for name, value in {
            "mean_lufs": mean_lufs,
            "std_lu": std_lu,
            "clamp_std": clamp_std,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"global loudness {name} must be a finite value")
        if float(std_lu) <= 0.0 or float(clamp_std) <= 0.0:
            raise ValueError("global loudness std_lu and clamp_std must be positive")
        if (
            not isinstance(initialization_seed, int)
            or isinstance(initialization_seed, bool)
            or initialization_seed < 0
        ):
            raise ValueError("global loudness initialization_seed must be a non-negative integer")
        self.hidden_size = hidden_size
        self.mean_lufs = float(mean_lufs)
        self.std_lu = float(std_lu)
        self.clamp_std = float(clamp_std)
        rng_state = torch.get_rng_state()
        torch.manual_seed(initialization_seed)
        try:
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        finally:
            torch.set_rng_state(rng_state)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, integrated_lufs: torch.Tensor) -> torch.Tensor:
        if not isinstance(integrated_lufs, torch.Tensor):
            raise TypeError("integrated_lufs must be a tensor")
        if integrated_lufs.ndim == 2 and integrated_lufs.shape[1] == 1:
            integrated_lufs = integrated_lufs[:, 0]
        if integrated_lufs.ndim != 1 or not integrated_lufs.is_floating_point():
            raise TypeError("integrated_lufs must be floating point with shape [B] or [B, 1]")
        if not torch.isfinite(integrated_lufs).all():
            raise ValueError("integrated_lufs contains NaN/Inf")
        normalized = (integrated_lufs.float() - self.mean_lufs) / self.std_lu
        normalized = normalized.clamp(-self.clamp_std, self.clamp_std)
        parameter = self.mlp[0].weight
        return self.mlp(
            normalized.to(device=parameter.device, dtype=parameter.dtype).unsqueeze(-1)
        )


class RelativeDynamicsConditioning(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        checkpoint_step: int = 8_250,
        target_kind: str = "relative_parent_lufs",
        relative_scale_lu: float = 10.0,
        include_activity: bool = True,
        freeze_predictor: bool = True,
        initialization_seed: int = 2026090304,
    ) -> None:
        super().__init__()
        if (
            not isinstance(hidden_size, int)
            or isinstance(hidden_size, bool)
            or hidden_size <= 0
        ):
            raise ValueError("relative dynamics hidden_size must be a positive integer")
        if not isinstance(include_activity, bool):
            raise TypeError("dynamics include_activity must be bool")
        if freeze_predictor is not True:
            raise ValueError("The current DiT requires a frozen V2 dynamics predictor")
        if (
            not isinstance(checkpoint_step, int)
            or isinstance(checkpoint_step, bool)
            or checkpoint_step < 0
        ):
            raise ValueError("dynamics checkpoint_step must be a non-negative integer")
        if target_kind not in {
            "relative_parent_lufs",
            "relative_short_window_lufs",
        }:
            raise ValueError("dynamics target_kind is unsupported")
        if (
            isinstance(relative_scale_lu, bool)
            or not isinstance(relative_scale_lu, (int, float))
            or not math.isfinite(float(relative_scale_lu))
            or float(relative_scale_lu) <= 0.0
        ):
            raise ValueError("dynamics relative_scale_lu must be a finite positive number")
        if (
            not isinstance(initialization_seed, int)
            or isinstance(initialization_seed, bool)
            or initialization_seed < 0
        ):
            raise ValueError("dynamics initialization_seed must be a non-negative integer")
        source = Path(checkpoint_path).expanduser().resolve(strict=True)
        expected_sha = _require_sha256(
            "dynamics checkpoint_sha256",
            checkpoint_sha256,
        )
        actual_sha = _file_sha256(source)
        if actual_sha != expected_sha:
            raise RuntimeError(
                "dynamics checkpoint SHA inconsistent:"
                f"expected={expected_sha} actual={actual_sha}"
            )
        try:
            payload = torch.load(
                source,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("dynamics checkpoint could not be loaded safely") from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("format_version")
            != "oqm.render.dynamics-predictor.ckpt.v1"
            or not isinstance(payload.get("model_config"), Mapping)
            or not isinstance(payload.get("target_config"), Mapping)
            or not isinstance(payload.get("model"), Mapping)
        ):
            raise RuntimeError("dynamics checkpoint format is incompatible")
        predictor_config = DynamicsPredictorConfig.from_mapping(
            payload["model_config"]
        )
        target_config = DynamicsTargetConfig.from_mapping(
            payload["target_config"]
        )
        observed_step = payload.get("global_step")
        if observed_step != checkpoint_step:
            raise RuntimeError(
                "dynamics checkpoint step inconsistent:"
                f"expected={checkpoint_step} actual={observed_step}"
            )
        expected_target = DynamicsTargetConfig(target_kind=target_kind)
        if target_config != expected_target:
            raise RuntimeError(
                "dynamics targetContract inconsistent:"
                f"expected={expected_target} actual={target_config}"
            )
        data_contract = payload.get("data_contract")
        if target_kind == "relative_short_window_lufs":
            expected_data_contract = {
                "sample_unit": "short_window",
                "target_basis": (
                    "local_400ms_lufs_minus_dataset_short_window_integrated_lufs"
                ),
                "absolute_reference": "dataset_short_window_integrated_lufs",
            }
            if (
                not isinstance(data_contract, Mapping)
                or dict(data_contract) != expected_data_contract
            ):
                raise RuntimeError(
                    "short-window dynamics checkpointData contract is inconsistent:"
                    f"expected={expected_data_contract} actual={data_contract}"
                )
        self.hidden_size = int(hidden_size)
        self.relative_scale_lu = float(relative_scale_lu)
        self.include_activity = include_activity
        self.checkpoint_provenance = {
            "path": str(source),
            "sha256": actual_sha,
            "global_step": int(observed_step),
            "format_version": str(payload["format_version"]),
            "model_config": predictor_config.__dict__,
            "target_config": target_config.__dict__,
            "data_contract": (
                dict(data_contract)
                if isinstance(data_contract, Mapping)
                else None
            ),
        }
        self.relative_min_lu = float(target_config.relative_min_lu)
        self.relative_max_lu = float(target_config.relative_max_lu)

        feature_size = 2 if include_activity else 1
        rng_state = torch.get_rng_state()
        try:
            self.predictor = SemanticDynamicsPredictor(predictor_config)
            self.predictor.load_state_dict(payload["model"], strict=True)
            self.predictor.requires_grad_(False)
            self.predictor.eval()
            torch.manual_seed(initialization_seed)
            self.adapter = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        finally:
            torch.set_rng_state(rng_state)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def train(self, mode: bool = True) -> "RelativeDynamicsConditioning":
        super().train(mode)

        self.predictor.eval()
        return self

    def forward(
        self,
        semantic_ids: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        if semantic_ids.dtype != torch.long or semantic_ids.ndim != 2:
            raise TypeError("dynamics semantic_ids must be int64 with shape [B, T]")
        if frame_mask.dtype != torch.bool or frame_mask.shape != semantic_ids.shape:
            raise TypeError("dynamics frame_mask must be bool with shape [B, T]")
        lengths = frame_mask.sum(dim=1)
        canonical = (
            torch.arange(
                semantic_ids.shape[1],
                device=frame_mask.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if bool((lengths <= 0).any()) or not torch.equal(frame_mask, canonical):
            raise ValueError("Each dynamics input must be a non-empty continuous prefix")


        with torch.no_grad():
            prediction = self.predictor(semantic_ids, frame_mask)
            relative = prediction.relative_loudness_lu.float().clamp(
                self.relative_min_lu,
                self.relative_max_lu,
            )
            activity = prediction.activity_probability.float().mul(2.0).sub(1.0)
            features = [
                relative.clamp(
                    self.relative_min_lu,
                    self.relative_max_lu,
                )
                / self.relative_scale_lu
            ]
            if self.include_activity:
                features.append(activity)
            values = torch.stack(features, dim=-1)
        parameter = self.adapter[0].weight
        embedded = self.adapter(
            values.to(device=parameter.device, dtype=parameter.dtype)
        )
        return embedded.masked_fill(~frame_mask.unsqueeze(-1), 0.0)


def build_null_text_context(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    null_context: torch.Tensor,
    *,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    if (
        context.ndim != 3
        or not context.is_floating_point()
        or context_mask.dtype != torch.bool
        or context_mask.shape != context.shape[:2]
    ):
        raise TypeError("null context requires floating-point context [B, L, D] and bool mask [B, L]")
    if (
        null_context.ndim != 1
        or not null_context.is_floating_point()
        or null_context.shape[0] != context.shape[-1]
    ):
        raise TypeError("null_context must be a floating-point vector matching the context hidden size")
    if (
        context.device != context_mask.device
        or context.device != null_context.device
    ):
        raise ValueError("null context inputs must be on the same device")
    if layout not in NULL_CONTEXT_LAYOUTS:
        raise ValueError(f"null_context_layout must be one of {NULL_CONTEXT_LAYOUTS}")

    batch, length, width = context.shape
    if length == 0:
        context = context.new_zeros(batch, 1, width)
        context_mask = torch.zeros(
            batch,
            1,
            dtype=torch.bool,
            device=context.device,
        )
        length = 1
    null_value = null_context.to(dtype=context.dtype).view(1, 1, width)
    if layout == "single_token":
        null_values = torch.zeros_like(context)
        null_values[:, :1] = null_value
        null_mask = torch.zeros_like(context_mask)
        null_mask[:, :1] = True
        return null_values, null_mask


    null_mask = context_mask.clone()
    empty = ~null_mask.any(dim=1)
    if bool(empty.any()):
        null_mask[empty, 0] = True
    null_values = null_value.expand(batch, length, width).clone()
    return null_values.masked_fill(~null_mask.unsqueeze(-1), 0.0), null_mask


class RenderConditioner(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        text_encoder: FrozenQwenEmbeddingAdapter,
        text_encoder_dim: int | None = None,
        semantic_vocab_size: int = SEMANTIC_VOCAB_SIZE,
        description_max_tokens: int = DESCRIPTION_MAX_TOKENS,
        description_projection_bias: bool = False,
        lyrics_max_tokens: int = LYRICS_MAX_TOKENS,
        lyrics_num_layers: int = LYRICS_ENCODER_LAYERS,
        lyrics_num_heads: int = 8,
        lyrics_head_dim: int | None = None,
        lyrics_ffn_expansion: float = 4.0,
        lyrics_ffn_activation: str = "gelu",
        lyrics_gelu_approximation: str = "tanh",
        lyrics_qkv_bias: bool = True,
        lyrics_non_qkv_linear_bias: bool = True,
        lyrics_rope_base: float = 10_000.0,
        lyrics_rope_style: str = "half_split",
        lyrics_norm_eps: float = 1.0e-5,
        lyrics_norm_type: str = "layernorm",
        lyrics_attention_direction: str = "bidirectional",
        lyrics_norm_style: str = "pre_norm_with_final_norm",
        lyrics_projection_order: str = "before_encoder",
        lyrics_projection_bias: bool = False,
        null_context_init_std: float = 0.02,
        text_drop_granularity: str = "sample",
        text_drop_scope: str = "joint_description_lyrics",
        null_context_tokens: int = 1,
        null_context_layout: str = "single_token",
        text_context_composition: str = "concatenate",
        text_context_layout: str = "description_then_lyrics",
        text_context_separator: str = "none",
        text_context_segment_embedding: str = "none",
        text_compaction_policy: str = "stable_valid_tokens_right_padded",
        initialization: str = "dit_xavier",
        semantic_embedding_init_std: float | None = None,
        semantic_embedding_asset: Mapping[str, Any] | None = None,
        semantic_tokenizer_revision: str | None = None,
        semantic_projection_init: str = "default",
        semantic_source_dim: int | None = None,
        semantic_projection_bias: bool = False,
        global_loudness_conditioning: bool = False,
        global_loudness_mean_lufs: float = -14.0,
        global_loudness_std_lu: float = 5.0,
        global_loudness_clamp_std: float = 4.0,
        global_loudness_initialization_seed: int = 20260903,
        dynamics_conditioning: bool = False,
        dynamics_checkpoint: str | Path | None = None,
        dynamics_checkpoint_sha256: str | None = None,
        dynamics_checkpoint_step: int = 8_250,
        dynamics_target_kind: str = "relative_parent_lufs",
        dynamics_freeze_predictor: bool = True,
        dynamics_include_activity: bool = True,
        dynamics_relative_scale_lu: float = 10.0,
        dynamics_initialization_seed: int = 2026090304,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        integer_values = {
            "hidden_size": hidden_size,
            "semantic_vocab_size": semantic_vocab_size,
            "description_max_tokens": description_max_tokens,
            "lyrics_max_tokens": lyrics_max_tokens,
            "lyrics_num_layers": lyrics_num_layers,
            "lyrics_num_heads": lyrics_num_heads,
        }
        if text_encoder_dim is not None:
            integer_values["text_encoder_dim"] = text_encoder_dim
        if lyrics_head_dim is not None:
            integer_values["lyrics_head_dim"] = lyrics_head_dim
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"conditioner {name} must be an integer")
            if value <= 0:
                raise ValueError(f"conditioner {name} must be positive")
        if semantic_vocab_size != SEMANTIC_VOCAB_SIZE:
            raise ValueError(
                f"Render semantic vocabulary must be 32768;  received {semantic_vocab_size}"
            )
        if description_max_tokens != DESCRIPTION_MAX_TOKENS:
            raise ValueError("description_max_tokens must be 256")
        if lyrics_max_tokens not in SUPPORTED_LYRICS_MAX_TOKENS:
            raise ValueError(
                "lyrics max tokenonly supportsRendererData short=1536or"
                "Losslessfull parent=1792"
            )
        if (
            isinstance(lyrics_ffn_expansion, bool)
            or not isinstance(lyrics_ffn_expansion, (int, float))
            or not math.isfinite(lyrics_ffn_expansion)
            or lyrics_ffn_expansion <= 0
        ):
            raise ValueError("lyrics_ffn_expansion must be a finite positive number")
        if lyrics_ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("lyrics_ffn_activation must be gelu or swiglu")
        if lyrics_gelu_approximation not in GELU_APPROXIMATIONS:
            raise ValueError(
                "lyrics_gelu_approximation must be"
                f"{GELU_APPROXIMATIONS}"
            )
        if (
            lyrics_ffn_activation != "gelu"
            and lyrics_gelu_approximation != "none"
        ):
            raise ValueError(
                "Non-GELU lyrics FFN requires lyrics_gelu_approximation=none"
            )
        for name, value in {
            "description_projection_bias": description_projection_bias,
            "lyrics_qkv_bias": lyrics_qkv_bias,
            "lyrics_non_qkv_linear_bias": lyrics_non_qkv_linear_bias,
            "lyrics_projection_bias": lyrics_projection_bias,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        for name, value in {
            "lyrics_rope_base": lyrics_rope_base,
            "lyrics_norm_eps": lyrics_norm_eps,
            "null_context_init_std": null_context_init_std,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if lyrics_rope_style not in ROPE_STYLES:
            raise ValueError(f"lyrics_rope_style must be one of {ROPE_STYLES}")
        if lyrics_attention_direction not in LYRICS_ATTENTION_DIRECTIONS:
            raise ValueError(
                "lyrics_attention_direction must be"
                f"{LYRICS_ATTENTION_DIRECTIONS}"
            )
        if lyrics_norm_style not in LYRICS_NORM_STYLES:
            raise ValueError(
                f"lyrics_norm_style must be one of {LYRICS_NORM_STYLES}"
            )
        if lyrics_projection_order not in LYRICS_PROJECTION_ORDERS:
            raise ValueError(
                "lyrics_projection_order must be"
                f"{LYRICS_PROJECTION_ORDERS}"
            )
        if text_drop_granularity != "sample":
            raise ValueError("text_drop_granularity only supports sample")
        if text_drop_scope != "joint_description_lyrics":
            raise ValueError(
                "text_drop_scope only supports joint_description_lyrics"
            )
        if null_context_tokens != 1:
            raise ValueError("null_context_tokens is fixed at 1")
        if null_context_layout not in NULL_CONTEXT_LAYOUTS:
            raise ValueError(
                f"null_context_layout must be one of {NULL_CONTEXT_LAYOUTS}"
            )
        if text_context_composition not in TEXT_CONTEXT_COMPOSITIONS:
            raise ValueError(
                "text_context_composition must be"
                f"{TEXT_CONTEXT_COMPOSITIONS}"
            )
        if text_context_layout not in TEXT_CONTEXT_LAYOUTS:
            raise ValueError(
                f"text_context_layout must be one of {TEXT_CONTEXT_LAYOUTS}"
            )
        if text_context_separator not in TEXT_CONTEXT_SEPARATOR_MODES:
            raise ValueError(
                "text_context_separator must be"
                f"{TEXT_CONTEXT_SEPARATOR_MODES}"
            )
        if (
            text_context_segment_embedding
            not in TEXT_CONTEXT_SEGMENT_EMBEDDING_MODES
        ):
            raise ValueError(
                "text_context_segment_embedding must be"
                f"{TEXT_CONTEXT_SEGMENT_EMBEDDING_MODES}"
            )
        if text_compaction_policy not in TEXT_COMPACTION_POLICIES:
            raise ValueError(
                "text_compaction_policy must be"
                f"{TEXT_COMPACTION_POLICIES}"
            )
        if initialization not in CONDITIONING_INITIALIZATION_MODES:
            raise ValueError(
                "conditioning initialization must be"
                f"{CONDITIONING_INITIALIZATION_MODES}"
            )
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(dropout)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("conditioner dropout must be a finite value in [0, 1)")
        if semantic_embedding_init_std is not None and (
            isinstance(semantic_embedding_init_std, bool)
            or not isinstance(semantic_embedding_init_std, (int, float))
            or not math.isfinite(semantic_embedding_init_std)
            or semantic_embedding_init_std <= 0
        ):
            raise ValueError(
                "semantic_embedding_init_std must be null or a finite positive number"
            )
        if (
            semantic_embedding_asset is not None
            and semantic_embedding_init_std is not None
        ):
            raise ValueError(
                "structured semantic embedding assets cannot use a random embedding initialization standard deviation"
            )
        if semantic_embedding_asset is not None and not isinstance(
            semantic_embedding_asset, Mapping
        ):
            raise TypeError("semantic_embedding_asset must be a mapping or null")
        if semantic_embedding_asset is not None and semantic_tokenizer_revision is None:
            raise ValueError(
                "structured semantic embedding assets must declare semantic_tokenizer_revision"
            )
        if semantic_projection_init not in {"default", "latent_projection"}:
            raise ValueError(
                "semantic_projection_init must be default or latent_projection"
            )
        if semantic_embedding_asset is None and semantic_projection_init != "default":
            raise ValueError(
                "semantic_projection_init=latent_projection requires a structured embedding asset"
            )
        if semantic_source_dim is not None and (
            not isinstance(semantic_source_dim, int)
            or isinstance(semantic_source_dim, bool)
            or semantic_source_dim <= 0
        ):
            raise ValueError("semantic_source_dim must be null or a positive integer")
        if (
            semantic_embedding_asset is None
            and semantic_source_dim is not None
            and semantic_embedding_init_std is None
        ):
            raise ValueError(
                "Random low-dimensionalsemantic embedding must be explicitly declared"
                "semantic_embedding_init_std"
            )
        if not isinstance(semantic_projection_bias, bool):
            raise TypeError("semantic_projection_bias must be bool")
        if not isinstance(global_loudness_conditioning, bool):
            raise TypeError("global_loudness_conditioning must be bool")
        if not isinstance(dynamics_conditioning, bool):
            raise TypeError("dynamics_conditioning must be bool")
        if dynamics_conditioning and (
            dynamics_checkpoint is None or dynamics_checkpoint_sha256 is None
        ):
            raise ValueError("Enabled dynamics conditioning requires a checkpoint path and SHA-256")
        if not dynamics_conditioning and (
            dynamics_checkpoint is not None or dynamics_checkpoint_sha256 is not None
        ):
            raise ValueError("Disabled dynamics conditioning must not retain checkpoint configuration")
        encoder_dim = text_encoder_dim or text_encoder.hidden_size
        if encoder_dim <= 0:
            raise ValueError("conditioner requires the known text encoder hidden_size")
        if text_encoder.hidden_size and text_encoder.hidden_size != encoder_dim:
            raise ValueError(
                "conditioner text_encoder_dim and adapter hidden_size differ"
            )
        self.hidden_size = hidden_size
        self.description_max_tokens = description_max_tokens
        self.lyrics_max_tokens = lyrics_max_tokens
        self.text_drop_granularity = text_drop_granularity
        self.text_drop_scope = text_drop_scope
        self.null_context_tokens = int(null_context_tokens)
        self.null_context_layout = null_context_layout
        self.text_context_composition = text_context_composition
        self.text_context_layout = text_context_layout
        self.text_context_separator = text_context_separator
        self.text_context_segment_embedding = text_context_segment_embedding
        self.text_compaction_policy = text_compaction_policy
        self.lyrics_attention_direction = lyrics_attention_direction
        self.lyrics_norm_style = lyrics_norm_style
        self.lyrics_projection_order = lyrics_projection_order
        self.initialization = initialization
        self.semantic_embedding_init_std = (
            float(semantic_embedding_init_std)
            if semantic_embedding_init_std is not None
            else None
        )
        self.semantic_projection_init = semantic_projection_init
        self.text_encoder = text_encoder
        self.semantic_embedding_asset_config: dict[str, Any] | None = None
        self.semantic_embedding_asset_provenance: dict[str, Any] | None = None
        if semantic_embedding_asset is None:
            semantic_rng_state = (
                torch.get_rng_state()
                if semantic_embedding_init_std is not None
                else None
            )
            if semantic_source_dim is None:
                self.semantic_embedding = nn.Embedding(
                    semantic_vocab_size,
                    self.hidden_size,
                )
                self.semantic_projection: nn.Module = nn.Identity()
            else:


                assert semantic_rng_state is not None
                _ = nn.Embedding(semantic_vocab_size, self.hidden_size)
                shared_post_embedding_rng_state = torch.get_rng_state()
                torch.set_rng_state(semantic_rng_state)
                self.semantic_embedding = nn.Embedding(
                    semantic_vocab_size,
                    int(semantic_source_dim),
                )
                self.semantic_projection = nn.Linear(
                    int(semantic_source_dim),
                    self.hidden_size,
                    bias=semantic_projection_bias,
                )
                torch.set_rng_state(shared_post_embedding_rng_state)
        else:


            semantic_rng_state = torch.get_rng_state()
            _ = nn.Embedding(semantic_vocab_size, self.hidden_size)
            shared_post_embedding_rng_state = torch.get_rng_state()
            torch.set_rng_state(semantic_rng_state)
            weight, provenance, trainable = _load_semantic_embedding_asset(
                semantic_embedding_asset,
                expected_tokenizer_revision=str(semantic_tokenizer_revision),
                expected_source_dim=semantic_source_dim,
            )
            self.semantic_embedding = nn.Embedding.from_pretrained(
                weight,
                freeze=not trainable,
            )
            self.semantic_projection = (
                nn.Identity()
                if weight.shape[1] == self.hidden_size
                else nn.Linear(
                    int(weight.shape[1]),
                    self.hidden_size,
                    bias=semantic_projection_bias,
                )
            )

            torch.set_rng_state(shared_post_embedding_rng_state)
            self.semantic_embedding_asset_config = json.loads(
                json.dumps(dict(semantic_embedding_asset), sort_keys=True)
            )
            self.semantic_embedding_asset_provenance = provenance
        if self.semantic_embedding_init_std is not None:
            assert semantic_rng_state is not None
            post_embedding_rng_state = torch.get_rng_state()


            torch.set_rng_state(semantic_rng_state)
            nn.init.normal_(
                self.semantic_embedding.weight,
                mean=0.0,
                std=self.semantic_embedding_init_std,
            )
            torch.set_rng_state(post_embedding_rng_state)


        self.description_projection = nn.Linear(
            encoder_dim,
            self.hidden_size,
            bias=description_projection_bias,
        )
        self.lyrics_projection = nn.Linear(
            encoder_dim,
            self.hidden_size,
            bias=lyrics_projection_bias,
        )
        self.lyrics_encoder = LyricsRoPEEncoder(
            self.hidden_size,
            num_layers=lyrics_num_layers,
            num_heads=lyrics_num_heads,
            head_dim=lyrics_head_dim,
            ffn_expansion=lyrics_ffn_expansion,
            ffn_activation=lyrics_ffn_activation,
            gelu_approximation=lyrics_gelu_approximation,
            dropout=dropout,
            qkv_bias=lyrics_qkv_bias,
            non_qkv_linear_bias=lyrics_non_qkv_linear_bias,
            rope_base=lyrics_rope_base,
            rope_style=lyrics_rope_style,
            norm_eps=lyrics_norm_eps,
            norm_type=lyrics_norm_type,
        )
        self.null_context = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(
            self.null_context,
            mean=0.0,
            std=float(null_context_init_std),
        )
        if self.initialization == "dit_xavier":


            post_construction_rng_state = torch.get_rng_state()
            self._initialize_dit_xavier()
            torch.set_rng_state(post_construction_rng_state)


        self.global_loudness_mlp = (
            GlobalLoudnessEmbedding(
                self.hidden_size,
                mean_lufs=global_loudness_mean_lufs,
                std_lu=global_loudness_std_lu,
                clamp_std=global_loudness_clamp_std,
                initialization_seed=global_loudness_initialization_seed,
            )
            if global_loudness_conditioning
            else None
        )


        self.relative_dynamics_conditioning = (
            RelativeDynamicsConditioning(
                self.hidden_size,
                checkpoint_path=dynamics_checkpoint,
                checkpoint_sha256=str(dynamics_checkpoint_sha256),
                checkpoint_step=dynamics_checkpoint_step,
                target_kind=dynamics_target_kind,
                relative_scale_lu=dynamics_relative_scale_lu,
                include_activity=dynamics_include_activity,
                freeze_predictor=dynamics_freeze_predictor,
                initialization_seed=dynamics_initialization_seed,
            )
            if dynamics_conditioning
            else None
        )

    def embed_global_loudness(
        self, integrated_lufs: torch.Tensor | None
    ) -> torch.Tensor | None:

        if self.global_loudness_mlp is None:
            if integrated_lufs is not None:
                raise ValueError("The current conditioner does not enable global loudness conditioning")
            return None
        if integrated_lufs is None:
            raise ValueError("Enabling global loudness conditioning requires integrated_lufs")
        return self.global_loudness_mlp(integrated_lufs)

    def embed_relative_dynamics(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
    ) -> torch.Tensor | None:

        if self.relative_dynamics_conditioning is None:
            return None
        return self.relative_dynamics_conditioning(semantic_ids, semantic_mask)

    def _initialize_dit_xavier(self) -> None:

        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

        modules = [
            self.semantic_projection,
            self.description_projection,
            self.lyrics_projection,
            self.lyrics_encoder,
        ]
        for module in modules:
            module.apply(initialize)

    def initialize_semantic_projection_from_latent(
        self,
        latent_projection: nn.Linear,
    ) -> None:

        if self.semantic_projection_init == "default":
            return
        if not isinstance(self.semantic_projection, nn.Linear):
            raise RuntimeError(
                "semantic_projection_init=latent_projection requires LinearProjection"
            )
        if (
            self.semantic_projection.in_features != latent_projection.in_features
            or self.semantic_projection.out_features
            != latent_projection.out_features
        ):
            raise RuntimeError(
                "semantic projection and latent projection shapes is incompatible:"
                f"semantic={tuple(self.semantic_projection.weight.shape)} "
                f"latent={tuple(latent_projection.weight.shape)}"
            )
        with torch.no_grad():
            self.semantic_projection.weight.copy_(latent_projection.weight)

    def lookup_semantic(self, semantic_ids: torch.Tensor) -> torch.Tensor:

        if semantic_ids.dtype != torch.long or semantic_ids.ndim != 2:
            raise TypeError("semantic_ids must be int64 with shape [B, T]")
        return self.semantic_embedding(semantic_ids)

    def embed_semantic(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
    ) -> torch.Tensor:

        source = self.lookup_semantic(semantic_ids)
        semantic = self.semantic_projection(source)
        return semantic.masked_fill(~semantic_mask.unsqueeze(-1), 0.0)

    @property
    def provenance(self) -> TextEncoderProvenance:
        return self.text_encoder.provenance

    @staticmethod
    def _compact_valid_tokens(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if (
            hidden.ndim != 3
            or attention_mask.dtype != torch.bool
            or attention_mask.shape != hidden.shape[:2]
        ):
            raise ValueError("Text compression requires hidden states [B, L, D] and a bool mask [B, L]")
        batch, length, width = hidden.shape
        if length == 0:
            return (
                hidden.new_zeros(batch, 1, width),
                torch.zeros(batch, 1, dtype=torch.bool, device=hidden.device),
            )
        positions = torch.arange(length, device=hidden.device).expand(batch, -1)

        order = torch.where(attention_mask, positions, positions + length).argsort(
            dim=1
        )
        compacted = hidden.gather(1, order.unsqueeze(-1).expand(-1, -1, width))
        lengths = attention_mask.sum(dim=1)
        compacted_mask = positions < lengths.unsqueeze(1)
        return (
            compacted.masked_fill(~compacted_mask.unsqueeze(-1), 0.0),
            compacted_mask,
        )

    @staticmethod
    def sample_text_drop_mask(
        batch_size: int,
        probability: float,
        *,
        generator: torch.Generator,
        device: torch.device | None = None,
    ) -> torch.Tensor:

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("text drop batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("text drop batch_size must be positive")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError("text_drop_probability must be in [0, 1]")

        sampled = (
            torch.rand(batch_size, generator=generator, device="cpu")
            < float(probability)
        )
        return sampled.to(device=device)

    @staticmethod
    def _validate_ids_and_mask(
        name: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_tokens: int,
    ) -> None:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise TypeError(f"{name}_input_ids must be int64 with shape [B, L]")
        if attention_mask.dtype != torch.bool:
            raise TypeError(f"{name}_mask must be bool")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(f"{name}_mask must have the same shape as input_ids")
        if input_ids.shape[1] > max_tokens:
            raise ValueError(
                f"{name} length {input_ids.shape[1]} exceeds the limit {max_tokens}"
            )

    def _encode_text(
        self,
        *,
        name: str,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        cached_embeddings: torch.Tensor | None,
        max_tokens: int,
        cache_provenance: Mapping[str, Any] | TextEncoderProvenance | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cached_embeddings is not None and input_ids is not None:
            raise ValueError(
                f"provide exactly one of {name}_input_ids or {name}_embeddings"
            )
        if cached_embeddings is not None:
            if cached_embeddings.ndim != 3 or not cached_embeddings.is_floating_point():
                raise TypeError(f"{name}_embeddings must be floating point with shape [B, L, D]")
            if attention_mask is None:
                raise ValueError(f"cache {name} embedding must provide mask")
            if attention_mask.dtype != torch.bool:
                raise TypeError(f"{name}_mask must be bool")
            if cached_embeddings.shape[:2] != attention_mask.shape:
                raise ValueError(f"{name} embedding and mask shapes differ")
            if cached_embeddings.device != attention_mask.device:
                raise ValueError(f"{name} embedding and mask must be on the same device")
            if cached_embeddings.shape[1] > max_tokens:
                raise ValueError(f"{name} cache length exceeds {max_tokens}")
            if cache_provenance is None:
                raise RuntimeError(
                    f"cached {name} embeddings are missing revision metadata"
                )
            validate_cache_provenance(cache_provenance, self.provenance)
            if not torch.isfinite(cached_embeddings).all():
                raise ValueError(f"{name}_embeddings contains NaN/Inf")
            if (~attention_mask).any() and torch.count_nonzero(
                cached_embeddings[~attention_mask]
            ):
                raise ValueError(f"{name}_embeddings at padding positions must be zero")
            hidden = cached_embeddings
        else:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    f"{name} must provide input_ids+mask or cached_embeddings+mask"
                )
            self._validate_ids_and_mask(
                name,
                input_ids,
                attention_mask,
                max_tokens=max_tokens,
            )
            hidden = self.text_encoder(input_ids, attention_mask)
        if not torch.isfinite(hidden).all():
            raise ValueError(f"{name} encoder output contains NaN/Inf")
        return self._compact_valid_tokens(hidden, attention_mask)

    def _replace_with_null(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = context.shape[0]
        if context.shape[1] == 0:
            context = context.new_zeros(batch, 1, self.hidden_size)
            context_mask = torch.zeros(
                batch, 1, dtype=torch.bool, device=context.device
            )

        effective_drop = drop_mask | ~context_mask.any(dim=1)
        null_values, null_mask = build_null_text_context(
            context,
            context_mask,
            self.null_context.to(device=context.device),
            layout=self.null_context_layout,
        )
        context = torch.where(effective_drop.view(batch, 1, 1), null_values, context)
        context_mask = torch.where(
            effective_drop.view(batch, 1), null_mask, context_mask
        )
        return (
            context.masked_fill(~context_mask.unsqueeze(-1), 0.0),
            context_mask,
            effective_drop,
        )

    def forward(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
        *,
        description_input_ids: torch.Tensor | None = None,
        description_mask: torch.Tensor | None = None,
        lyrics_input_ids: torch.Tensor | None = None,
        lyrics_mask: torch.Tensor | None = None,
        description_embeddings: torch.Tensor | None = None,
        lyrics_embeddings: torch.Tensor | None = None,
        cache_provenance: Mapping[str, Any] | TextEncoderProvenance | None = None,
        text_drop_mask: torch.Tensor | None = None,
        force_null_text: bool = False,
    ) -> ConditioningOutput:
        if semantic_ids.dtype != torch.long or semantic_ids.ndim != 2:
            raise TypeError("semantic_ids must be int64 with shape [B, T]")
        if (
            semantic_mask.dtype != torch.bool
            or semantic_mask.shape != semantic_ids.shape
        ):
            raise TypeError("semantic_mask must be a bool tensor matching semantic_ids")
        if semantic_mask.device != semantic_ids.device:
            raise ValueError("semantic_ids and semantic_mask must be on the same device")
        if semantic_ids.device != self.semantic_embedding.weight.device:
            raise ValueError("semantic inputs and conditioner parameters must be on the same device")
        if semantic_ids.shape[0] <= 0 or semantic_ids.shape[1] <= 0:
            raise ValueError("semantic batch and frame dimensions must be non-empty")
        semantic_lengths = semantic_mask.sum(dim=1)
        if not bool((semantic_lengths > 0).all()):
            raise ValueError("Each sample must contain at least one valid semantic frame")
        canonical_semantic_mask = (
            torch.arange(
                semantic_ids.shape[1],
                device=semantic_mask.device,
            ).unsqueeze(0)
            < semantic_lengths.unsqueeze(1)
        )
        if not torch.equal(semantic_mask, canonical_semantic_mask):
            raise ValueError("semantic_mask must be a contiguous valid prefix followed by right padding")
        if semantic_ids.numel() and (
            int(semantic_ids.min()) < 0
            or int(semantic_ids.max()) >= SEMANTIC_VOCAB_SIZE
        ):
            raise ValueError("semantic IDs, including padding positions, must be in [0, 32767]")
        batch = semantic_ids.shape[0]
        semantic = self.embed_semantic(semantic_ids, semantic_mask)
        relative_dynamics = self.embed_relative_dynamics(
            semantic_ids,
            semantic_mask,
        )

        description_hidden, description_mask = self._encode_text(
            name="description",
            input_ids=description_input_ids,
            attention_mask=description_mask,
            cached_embeddings=description_embeddings,
            max_tokens=self.description_max_tokens,
            cache_provenance=cache_provenance,
        )
        lyrics_hidden, lyrics_mask = self._encode_text(
            name="lyrics",
            input_ids=lyrics_input_ids,
            attention_mask=lyrics_mask,
            cached_embeddings=lyrics_embeddings,
            max_tokens=self.lyrics_max_tokens,
            cache_provenance=cache_provenance,
        )
        if description_hidden.shape[0] != batch or lyrics_hidden.shape[0] != batch:
            raise ValueError("semantic, description, and lyrics batch sizes must match")
        if (
            description_hidden.shape[-1] != self.description_projection.in_features
            or lyrics_hidden.shape[-1] != self.lyrics_projection.in_features
        ):
            raise ValueError(
                "text embedding hidden dimension does not match the frozen encoder or projection contract"
            )
        description_hidden = description_hidden.to(
            device=self.description_projection.weight.device,
            dtype=self.description_projection.weight.dtype,
        )
        description_mask = description_mask.to(
            device=self.description_projection.weight.device
        )
        lyrics_hidden = lyrics_hidden.to(
            device=self.lyrics_projection.weight.device,
            dtype=self.lyrics_projection.weight.dtype,
        )
        lyrics_mask = lyrics_mask.to(device=self.lyrics_projection.weight.device)
        description = self.description_projection(description_hidden)
        description = description.masked_fill(~description_mask.unsqueeze(-1), 0.0)
        lyrics = self.lyrics_projection(lyrics_hidden)
        lyrics = self.lyrics_encoder(lyrics, lyrics_mask)
        context = torch.cat((description, lyrics), dim=1)
        context_mask = torch.cat((description_mask, lyrics_mask), dim=1)

        if text_drop_mask is None:
            text_drop_mask = torch.zeros(
                batch, dtype=torch.bool, device=semantic_ids.device
            )
        if text_drop_mask.dtype != torch.bool or text_drop_mask.shape != (batch,):
            raise TypeError("text_drop_mask must be bool with shape [B]")
        if text_drop_mask.device != semantic_ids.device:
            raise ValueError("text_drop_mask and semantic inputs must be on the same device")
        if not isinstance(force_null_text, bool):
            raise TypeError("force_null_text must be bool")
        if force_null_text:
            text_drop_mask = torch.ones_like(text_drop_mask)
        context, context_mask, text_drop_mask = self._replace_with_null(
            context, context_mask, text_drop_mask
        )
        return ConditioningOutput(
            semantic_embeddings=semantic,
            semantic_mask=semantic_mask,
            text_context=context,
            text_mask=context_mask,
            text_drop_mask=text_drop_mask,
            provenance=self.provenance.to_dict(),
            relative_dynamics_embeddings=relative_dynamics,
        )

    def null_from_semantic(
        self,
        semantic_embeddings: torch.Tensor,
        semantic_mask: torch.Tensor,
        *,
        text_mask: torch.Tensor | None = None,
        relative_dynamics_embeddings: torch.Tensor | None = None,
    ) -> ConditioningOutput:

        if (
            semantic_embeddings.ndim != 3
            or semantic_embeddings.shape[-1] != self.hidden_size
            or not semantic_embeddings.is_floating_point()
        ):
            raise ValueError(f"semantic_embeddings must have shape [B, T, {self.hidden_size}]")
        if (
            semantic_mask.dtype != torch.bool
            or semantic_mask.shape != semantic_embeddings.shape[:2]
        ):
            raise TypeError("semantic_mask must be bool with shape [B, T]")
        if semantic_mask.device != semantic_embeddings.device:
            raise ValueError("semantic_embeddings and semantic_mask must be on the same device")
        if semantic_embeddings.shape[0] <= 0 or semantic_embeddings.shape[1] <= 0:
            raise ValueError("semantic_embeddings batch and frame dimensions must be non-empty")
        if not bool(semantic_mask.any(dim=1).all()):
            raise ValueError("Each sample must contain at least one valid semantic frame")
        lengths = semantic_mask.sum(dim=1)
        canonical = (
            torch.arange(
                semantic_embeddings.shape[1],
                device=semantic_mask.device,
            ).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(semantic_mask, canonical):
            raise ValueError("semantic_mask must be a contiguous valid prefix followed by right padding")
        valid = semantic_mask.unsqueeze(-1)
        if not torch.isfinite(
            torch.where(
                valid,
                semantic_embeddings,
                torch.zeros_like(semantic_embeddings),
            )
        ).all():
            raise ValueError("Valid semantic embedding positions contain NaN/Inf")
        semantic_embeddings = semantic_embeddings.masked_fill(~valid, 0.0)
        if relative_dynamics_embeddings is not None:
            if (
                relative_dynamics_embeddings.shape != semantic_embeddings.shape
                or not relative_dynamics_embeddings.is_floating_point()
                or relative_dynamics_embeddings.device
                != semantic_embeddings.device
            ):
                raise ValueError(
                    "relative_dynamics_embeddings must be a floating-point tensor matching the semantic input shape and device"
                )
            if not torch.isfinite(
                torch.where(
                    valid,
                    relative_dynamics_embeddings,
                    torch.zeros_like(relative_dynamics_embeddings),
                )
            ).all():
                raise ValueError("Valid relative dynamics positions contain NaN/Inf")
            relative_dynamics_embeddings = (
                relative_dynamics_embeddings.masked_fill(~valid, 0.0)
            )
        batch = semantic_embeddings.shape[0]


        if text_mask is None:
            if self.null_context_layout != "single_token":
                raise ValueError(
                    "preserve_text_mask null layout requires conditional text_mask"
                )
            text_mask = torch.ones(
                batch,
                1,
                dtype=torch.bool,
                device=semantic_embeddings.device,
            )
        if (
            text_mask.dtype != torch.bool
            or text_mask.ndim != 2
            or text_mask.shape[0] != batch
            or text_mask.device != semantic_embeddings.device
        ):
            raise TypeError("text_mask must be bool with shape [B, L] on the semantic input device")
        context, context_mask = build_null_text_context(
            torch.zeros(
                batch,
                text_mask.shape[1],
                self.hidden_size,
                device=semantic_embeddings.device,
                dtype=self.null_context.dtype,
            ),
            text_mask,
            self.null_context.to(device=semantic_embeddings.device),
            layout=self.null_context_layout,
        )
        return ConditioningOutput(
            semantic_embeddings=semantic_embeddings,
            semantic_mask=semantic_mask,
            text_context=context,
            text_mask=context_mask,
            text_drop_mask=torch.ones(
                batch, dtype=torch.bool, device=semantic_embeddings.device
            ),
            provenance=self.provenance.to_dict(),
            relative_dynamics_embeddings=relative_dynamics_embeddings,
        )


    def encode_strings(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
        descriptions: str | Sequence[str],
        lyrics: str | Sequence[str],
        *,
        force_null_text: bool = False,
    ) -> ConditioningOutput:
        description_ids, description_mask = self.text_encoder.tokenize(
            descriptions,
            max_length=self.description_max_tokens,
            device=semantic_ids.device,
        )
        lyrics_ids, lyrics_mask = self.text_encoder.tokenize(
            lyrics,
            max_length=self.lyrics_max_tokens,
            device=semantic_ids.device,
        )
        return self(
            semantic_ids,
            semantic_mask,
            description_input_ids=description_ids,
            description_mask=description_mask,
            lyrics_input_ids=lyrics_ids,
            lyrics_mask=lyrics_mask,
            force_null_text=force_null_text,
        )


def build_render_conditioner_from_config(
    *,
    hidden_size: int,
    condition_config: Mapping[str, Any],
    text_encoder_revision: str,
    text_tokenizer_revision: str,
    text_cache_revision: str,
    semantic_tokenizer_revision: str,
    encoder: nn.Module | None = None,
) -> RenderConditioner:

    if not isinstance(condition_config, Mapping):
        raise TypeError("conditioning configuration must be a mapping")
    text_config = condition_config.get("text_encoder")
    if not isinstance(text_config, Mapping):
        raise TypeError("conditioning.text_encoder must be a mapping")
    adapter = FrozenQwenEmbeddingAdapter(
        model_id=str(text_config["model_id"]),
        revision=text_encoder_revision,
        tokenizer_revision=text_tokenizer_revision,
        cache_revision=text_cache_revision,
        encoder=encoder,
        hidden_size=int(text_config["hidden_size"]),
        local_path=text_config.get("local_path"),
        local_files_only=True,
        use_fast_tokenizer=bool(text_config.get("use_fast_tokenizer", True)),
        padding_side=str(text_config.get("padding_side", "left")),
        truncation_side=str(text_config.get("truncation_side", "right")),
        truncation_policy=str(text_config.get("truncation_policy", "allow")),
        add_special_tokens=bool(text_config.get("add_special_tokens", True)),
        trust_remote_code=bool(text_config.get("trust_remote_code", False)),
        hidden_state_selection=str(
            text_config.get("hidden_state_selection", "last_hidden_state")
        ),
        position_id_policy=str(
            text_config.get("position_id_policy", "attention_mask_cumsum")
        ),
        encoder_use_cache=bool(text_config.get("encoder_use_cache", False)),
        empty_text_policy=str(
            text_config.get("empty_text_policy", "zero_valid_tokens")
        ),
        frozen_eval_mode=bool(text_config.get("frozen_eval_mode", True)),
    )
    return RenderConditioner(
        hidden_size=hidden_size,
        text_encoder=adapter,
        text_encoder_dim=int(text_config["hidden_size"]),
        semantic_vocab_size=int(
            condition_config.get("semantic_vocab_size", SEMANTIC_VOCAB_SIZE)
        ),
        description_max_tokens=int(
            condition_config.get("description_max_tokens", DESCRIPTION_MAX_TOKENS)
        ),
        description_projection_bias=bool(
            condition_config.get("description_projection_bias", False)
        ),
        lyrics_max_tokens=int(
            condition_config.get("lyrics_max_tokens", LYRICS_MAX_TOKENS)
        ),
        lyrics_num_layers=int(
            condition_config.get("lyrics_encoder_layers", LYRICS_ENCODER_LAYERS)
        ),
        lyrics_num_heads=int(condition_config.get("lyrics_encoder_heads", 8)),
        lyrics_head_dim=condition_config.get("lyrics_head_dim"),
        lyrics_ffn_expansion=float(condition_config.get("lyrics_ffn_expansion", 4.0)),
        lyrics_ffn_activation=str(
            condition_config.get("lyrics_ffn_activation", "gelu")
        ),
        lyrics_gelu_approximation=str(
            condition_config.get("lyrics_gelu_approximation", "tanh")
        ),
        lyrics_qkv_bias=bool(condition_config.get("lyrics_qkv_bias", True)),
        lyrics_non_qkv_linear_bias=bool(
            condition_config.get("lyrics_non_qkv_linear_bias", True)
        ),
        lyrics_rope_base=float(condition_config.get("lyrics_rope_base", 10_000.0)),
        lyrics_rope_style=str(
            condition_config.get("lyrics_rope_style", "half_split")
        ),
        lyrics_norm_eps=float(condition_config.get("lyrics_norm_eps", 1.0e-5)),
        lyrics_norm_type=str(
            condition_config.get("lyrics_norm_type", "layernorm")
        ),
        lyrics_attention_direction=str(
            condition_config.get("lyrics_attention_direction", "bidirectional")
        ),
        lyrics_norm_style=str(
            condition_config.get("lyrics_norm_style", "pre_norm_with_final_norm")
        ),
        lyrics_projection_order=str(
            condition_config.get("lyrics_projection_order", "before_encoder")
        ),
        lyrics_projection_bias=bool(
            condition_config.get("lyrics_projection_bias", False)
        ),
        null_context_init_std=float(
            condition_config.get("null_context_init_std", 0.02)
        ),
        text_drop_granularity=str(
            condition_config.get("text_drop_granularity", "sample")
        ),
        text_drop_scope=str(
            condition_config.get("text_drop_scope", "joint_description_lyrics")
        ),
        null_context_tokens=int(condition_config.get("null_context_tokens", 1)),
        null_context_layout=str(
            condition_config.get("null_context_layout", "single_token")
        ),
        text_context_composition=str(
            condition_config.get("text_context_composition", "concatenate")
        ),
        text_context_layout=str(
            condition_config.get(
                "text_context_layout",
                "description_then_lyrics",
            )
        ),
        text_context_separator=str(
            condition_config.get("text_context_separator", "none")
        ),
        text_context_segment_embedding=str(
            condition_config.get("text_context_segment_embedding", "none")
        ),
        text_compaction_policy=str(
            condition_config.get(
                "text_compaction_policy",
                "stable_valid_tokens_right_padded",
            )
        ),
        initialization=str(
            condition_config.get("initialization", "dit_xavier")
        ),
        semantic_embedding_init_std=condition_config.get(
            "semantic_embedding_init_std"
        ),
        semantic_embedding_asset=condition_config.get("semantic_embedding_asset"),
        semantic_tokenizer_revision=semantic_tokenizer_revision,
        semantic_projection_init=str(
            condition_config.get("semantic_projection_init", "default")
        ),
        semantic_source_dim=condition_config.get("semantic_source_dim"),
        semantic_projection_bias=bool(
            condition_config.get("semantic_projection_bias", False)
        ),
        global_loudness_conditioning=bool(
            condition_config.get("global_loudness_conditioning", False)
        ),
        global_loudness_mean_lufs=float(
            condition_config.get("global_loudness_mean_lufs", -14.0)
        ),
        global_loudness_std_lu=float(
            condition_config.get("global_loudness_std_lu", 5.0)
        ),
        global_loudness_clamp_std=float(
            condition_config.get("global_loudness_clamp_std", 4.0)
        ),
        global_loudness_initialization_seed=int(
            condition_config.get("global_loudness_initialization_seed", 20260903)
        ),
        dynamics_conditioning=bool(
            condition_config.get("dynamics_conditioning", False)
        ),
        dynamics_checkpoint=condition_config.get("dynamics_checkpoint"),
        dynamics_checkpoint_sha256=condition_config.get(
            "dynamics_checkpoint_sha256"
        ),
        dynamics_checkpoint_step=int(
            condition_config.get("dynamics_checkpoint_step", 8_250)
        ),
        dynamics_target_kind=str(
            condition_config.get(
                "dynamics_target_kind",
                "relative_parent_lufs",
            )
        ),
        dynamics_freeze_predictor=bool(
            condition_config.get("dynamics_freeze_predictor", True)
        ),
        dynamics_include_activity=bool(
            condition_config.get("dynamics_include_activity", True)
        ),
        dynamics_relative_scale_lu=float(
            condition_config.get("dynamics_relative_scale_lu", 10.0)
        ),
        dynamics_initialization_seed=int(
            condition_config.get("dynamics_initialization_seed", 2026090304)
        ),
        dropout=float(condition_config.get("dropout", 0.0)),
    )


def load_cached_embeddings(
    path: str | Path,
    *,
    expected_provenance: TextEncoderProvenance,
) -> tuple[torch.Tensor, torch.Tensor]:

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("text embedding cache must be a mapping")
    validate_cache_provenance(payload.get("provenance") or {}, expected_provenance)
    embedding = payload.get("embeddings")
    mask = payload.get("mask")
    if not isinstance(embedding, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("cache is missing embedding and mask tensors")
    if (
        embedding.ndim != 3
        or mask.dtype != torch.bool
        or embedding.shape[:2] != mask.shape
    ):
        raise ValueError("cache embedding and mask shape or dtype is invalid")
    return embedding, mask
