
from __future__ import annotations

import hashlib
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..contracts import SEMANTIC_CODEBOOK_SIZE

STAGE4_PROBE_ARTIFACT_VERSION = "oqm.llm.stage4-semantic-probe.v1"
STAGE4_PROBE_PROTOCOL = "oqm.eval.stage4-head-probe.teacher-forced-top1.v1"
STAGE4_REFERENCE_TARGET_PROTOCOL = "stage4-head-output-from-reference-semantic"


def _sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision_is_missing(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "unknown",
        "none",
        "null",
    }


def _bootstrap_repository(repository: str | Path) -> Path:
    source = Path(repository).resolve() / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Stage4 repository is missing its src directory: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def export_stage4_probe_artifact(
    checkpoint: str | Path,
    output: str | Path,
    *,
    semantic_tokenizer_revision: str | None = None,
    source_checkpoint_path: str | Path | None = None,
    source_checkpoint_sha256: str | None = None,
) -> Path:

    if _revision_is_missing(semantic_tokenizer_revision):
        raise ValueError(
            "Exporting a Stage4 probe requires semantic_tokenizer_revision so the "
            "artifact can be checked against the LLM corpus"
        )
    checkpoint = Path(checkpoint).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Stage4 probe artifact already exists: {output}")
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if int(state.get("stage", 0)) != 4:
        raise RuntimeError(
            f"Stage4 probe export requires a stage=4 checkpoint; got stage={state.get('stage')}"
        )
    config = state.get("config")
    model_state = state.get("model")
    if not isinstance(config, dict) or not isinstance(model_state, dict):
        raise RuntimeError("checkpoint is missing config/model")
    model_config = dict(config.get("model") or {})
    insertion = int(model_config.get("quantizer_insertion_layer", 0))
    total_layers = int(model_config.get("num_layers", 0))
    if not 0 < insertion < total_layers:
        raise RuntimeError(
            "Invalid quantizer_insertion_layer/num_layers: "
            f"{insertion}/{total_layers}"
        )

    selected: dict[str, torch.Tensor] = {}
    for name, tensor in model_state.items():
        if name.startswith(("quantizer.", "heads.")):
            selected[name] = tensor
            continue
        if not name.startswith("encoder.layers."):
            continue
        parts = name.split(".")
        layer = int(parts[2])
        if layer < insertion:
            continue
        parts[2] = str(layer - insertion)
        selected[".".join(parts)] = tensor
    if not any(name.startswith("quantizer.") for name in selected):
        raise RuntimeError("Checkpoint does not contain quantizer weights")
    if not any(name.startswith("heads.") for name in selected):
        raise RuntimeError("Checkpoint does not contain Stage4 head weights")
    expected_upper = total_layers - insertion
    found_upper = {
        int(name.split(".")[2])
        for name in selected
        if name.startswith("encoder.layers.")
    }
    if found_upper != set(range(expected_upper)):
        raise RuntimeError(
            f"Upper encoder layers are incomplete: expected={list(range(expected_upper))}, "
            f"found={sorted(found_upper)}"
        )

    vocab_path = Path(str((config.get("data") or {}).get("vocab") or "")).resolve()
    vocab_metadata: dict[str, Any] | None = None
    if vocab_path.is_file():
        vocab_metadata = {
            "path": str(vocab_path),
            "bytes": vocab_path.stat().st_size,
            "sha256": _sha256(vocab_path),
        }
    checkpoint_stat = checkpoint.stat()
    identity_path = (
        Path(source_checkpoint_path).resolve()
        if source_checkpoint_path is not None
        else checkpoint
    )
    identity_stat = identity_path.stat() if identity_path.is_file() else checkpoint_stat
    artifact = {
        "format_version": STAGE4_PROBE_ARTIFACT_VERSION,
        "protocol": STAGE4_PROBE_PROTOCOL,
        "stage": 4,
        "config": config,
        "model": selected,
        "semantic_tokenizer_revision": semantic_tokenizer_revision,
        "source_checkpoint": {
            "path": str(identity_path),
            "bytes": identity_stat.st_size,
            "mtime_ns": identity_stat.st_mtime_ns,

            "sha256": source_checkpoint_sha256,
        },
        "ctc_vocab": vocab_metadata,
        "upper_encoder": {
            "source_start": insertion,
            "source_end": total_layers,
            "num_layers": expected_upper,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(artifact, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


@dataclass(frozen=True)
class Stage4ProbeInfo:
    protocol: str
    semantic_tokenizer_revision: str | None
    source_checkpoint: dict[str, Any]
    vocab_path: str | None
    vocab_kind: str | None
    upper_layers: int


class Stage4AudioTargetStore:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as payload:
            required = {
                "sample_ids",
                "mel_values",
                "mel_offsets",
                "chroma_values",
                "chroma_offsets",
            }
            missing = required - set(payload.files)
            if missing:
                raise KeyError(f"Stage4 target store is missing fields: {sorted(missing)}")
            self.sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            self.mel_values = np.asarray(payload["mel_values"], dtype=np.float32)
            self.mel_offsets = np.asarray(payload["mel_offsets"], dtype=np.int64)
            self.chroma_values = np.asarray(payload["chroma_values"], dtype=np.float32)
            self.chroma_offsets = np.asarray(payload["chroma_offsets"], dtype=np.int64)
            raw_metadata = (
                str(payload["metadata"].item()) if "metadata" in payload.files else "{}"
            )
        import json

        self.metadata = json.loads(raw_metadata)
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Stage4 target store contains duplicate sample_id values")
        if self.mel_offsets.shape != (len(self.sample_ids) + 1,):
            raise ValueError("mel_offsets has an invalid length")
        if self.chroma_offsets.shape != (len(self.sample_ids) + 1,):
            raise ValueError("chroma_offsets has an invalid length")
        if self.mel_offsets[0] != 0 or self.mel_offsets[-1] != len(self.mel_values):
            raise ValueError("mel_offsets and mel_values are inconsistent")
        if self.chroma_offsets[0] != 0 or self.chroma_offsets[-1] != len(
            self.chroma_values
        ):
            raise ValueError("chroma_offsets and chroma_values are inconsistent")
        self._indices = {
            sample_id: index for index, sample_id in enumerate(self.sample_ids)
        }

    def get(self, sample_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        index = self._indices.get(str(sample_id))
        if index is None:
            return None
        mel_start, mel_end = self.mel_offsets[index : index + 2]
        chroma_start, chroma_end = self.chroma_offsets[index : index + 2]
        return (
            self.mel_values[int(mel_start) : int(mel_end)],
            self.chroma_values[int(chroma_start) : int(chroma_end)],
        )


class Stage4SemanticProbe(nn.Module):

    def __init__(
        self,
        *,
        quantizer: nn.Module,
        encoder: nn.Module,
        heads: nn.Module,
        attention_causal: bool,
        convolution_causal: bool,
        causal_heads: bool,
        autocast_dtype: torch.dtype | None,
        ctc_tokenizer: Any | None,
        info: Stage4ProbeInfo,
    ) -> None:
        super().__init__()
        self.quantizer = quantizer
        self.encoder = encoder
        self.heads = heads
        self.attention_causal = bool(attention_causal)
        self.convolution_causal = bool(convolution_causal)
        self.causal_heads = bool(causal_heads)
        self.autocast_dtype = autocast_dtype
        self.ctc_tokenizer = ctc_tokenizer
        self.info = info

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        repository: str | Path,
        device: torch.device | str,
        precision: str = "bf16",
        ctc_vocab_path: str | Path | None = None,
        expected_tokenizer_revision: str | None = None,
    ) -> Stage4SemanticProbe:
        _bootstrap_repository(repository)
        from open_qwen_music.tokenizer.conformer import ConformerEncoder
        from open_qwen_music.tokenizer.frontend import resolve_causality
        from open_qwen_music.tokenizer.heads import MultiTaskHeads
        from open_qwen_music.tokenizer.quantizer import build_quantizer
        from open_qwen_music.tokenizer.text import CharacterTokenizer

        artifact_path = Path(artifact_path).resolve()
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if payload.get("format_version") != STAGE4_PROBE_ARTIFACT_VERSION:
            raise RuntimeError(
                "Unsupported Stage4 probe artifact version: "
                f"{payload.get('format_version')!r}"
            )
        revision = payload.get("semantic_tokenizer_revision")
        if expected_tokenizer_revision:
            if _revision_is_missing(revision):
                raise RuntimeError(
                    "Stage4 probe does not record semantic_tokenizer_revision, so it "
                    "cannot be checked against LLM corpus revision "
                    f"{expected_tokenizer_revision}"
                )
            if str(revision) != str(expected_tokenizer_revision):
                raise RuntimeError(
                    "Stage4 probe tokenizer revision does not match the LLM corpus: "
                    f"probe={revision}, corpus={expected_tokenizer_revision}"
                )
        config = dict(payload["config"])
        model_config = dict(config["model"])
        feature_config = dict(config["features"])
        head_config = dict(config["heads"])
        quantizer_config = {
            **dict(config["quantizer"]),
            "codebook_size": int(config["semantic_contract"]["codebook_size"]),
        }
        if str(quantizer_config.get("type")) != "cosine_vq":
            raise RuntimeError(
                "The Semantic-ID lookup probe requires cosine_vq; got "
                f"{quantizer_config.get('type')!r}"
            )
        dim = int(model_config["dim"])
        upper_layers = int((payload.get("upper_encoder") or {})["num_layers"])
        position_encoding = str(model_config.get("position_encoding", "rope"))
        quantizer = build_quantizer(quantizer_config, input_dim=dim)
        encoder = ConformerEncoder(
            dim=dim,
            num_layers=upper_layers,
            num_heads=int(model_config["num_heads"]),
            ffn_dim=int(model_config["ffn_dim"]),
            conv_kernel=int(model_config["conv_kernel"]),
            dropout=float(model_config["dropout"]),
            position_encoding=position_encoding,
        )
        heads = MultiTaskHeads(
            model_dim=dim,
            ctc_vocab_size=int(head_config["ctc_vocab_size"]),
            mel_bins=int(feature_config["n_mels"]),
            chroma_bins=12,
            convnext_blocks=int(head_config.get("convnext_blocks", 2)),
            ctc_blank_id=int(head_config.get("ctc_blank_id", 0)),
            ctc_normalize_by_target_length=bool(
                head_config.get("ctc_normalize_by_target_length", True)
            ),
            mel_loss_mode=str(head_config.get("mel_loss_mode", "signed_l1")),
            chroma_loss_mode=str(head_config.get("chroma_loss_mode", "signed_l1")),
        )
        wrapper = nn.Module()
        wrapper.quantizer = quantizer
        wrapper.encoder = encoder
        wrapper.heads = heads
        missing, unexpected = wrapper.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Stage4 probe weights are incompatible: missing={missing}, "
                f"unexpected={unexpected}"
            )

        vocab = Path(
            str(
                ctc_vocab_path
                or ((payload.get("ctc_vocab") or {}).get("path"))
                or ((config.get("data") or {}).get("vocab"))
                or ""
            )
        )
        tokenizer = CharacterTokenizer.from_file(vocab) if vocab.is_file() else None
        vocab_metadata = payload.get("ctc_vocab") or {}
        if tokenizer is not None and vocab_metadata.get("sha256"):
            actual = _sha256(vocab.resolve())
            if actual != str(vocab_metadata["sha256"]):
                raise RuntimeError(
                    "CTC vocabulary SHA does not match: "
                    f"artifact={vocab_metadata['sha256']}, actual={actual}"
                )

        causality = resolve_causality(model_config)
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": None,
            "float32": None,
        }
        if precision not in dtype_map:
            raise ValueError(f"Unsupported Stage4 probe precision: {precision!r}")
        model = cls(
            quantizer=quantizer,
            encoder=encoder,
            heads=heads,
            attention_causal=bool(causality["attention_causal"]),
            convolution_causal=bool(causality["conformer_conv_causal"]),
            causal_heads=bool(model_config.get("causal_heads", True)),
            autocast_dtype=dtype_map[precision],
            ctc_tokenizer=tokenizer,
            info=Stage4ProbeInfo(
                protocol=str(payload.get("protocol") or STAGE4_PROBE_PROTOCOL),
                semantic_tokenizer_revision=(
                    str(revision) if revision is not None else None
                ),
                source_checkpoint=dict(payload.get("source_checkpoint") or {}),
                vocab_path=str(vocab.resolve()) if vocab.is_file() else None,
                vocab_kind=getattr(tokenizer, "kind", None),
                upper_layers=upper_layers,
            ),
        )
        model.requires_grad_(False)
        model.eval()
        return model.to(device)

    @torch.no_grad()
    def forward(
        self,
        token_ids: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2 or frame_mask.shape != token_ids.shape:
            raise ValueError(
                f"token_ids and frame_mask must both have shape [B, T]; got "
                f"{tuple(token_ids.shape)}/{tuple(frame_mask.shape)}"
            )
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        frame_mask = frame_mask.bool()
        if bool((token_ids[frame_mask] < 0).any()) or bool(
            (token_ids[frame_mask] >= SEMANTIC_CODEBOOK_SIZE).any()
        ):
            raise ValueError("Stage4 probe received an out-of-range semantic local ID")
        quantizer = self.quantizer
        if not hasattr(quantizer, "codebook") or not hasattr(quantizer, "output_proj"):
            raise RuntimeError("Stage4 probe artifact does not contain a cosine VQ")


        with torch.autocast(device_type=token_ids.device.type, enabled=False):
            codebook = F.normalize(quantizer.codebook.float(), dim=-1)
            quantized = F.embedding(token_ids, codebook)
            hidden = quantizer.output_proj(quantized.float())
            hidden = hidden.masked_fill(~frame_mask.unsqueeze(-1), 0.0)

        autocast_enabled = self.autocast_dtype is not None and token_ids.device.type != "cpu"
        with torch.autocast(
            device_type=token_ids.device.type,
            dtype=self.autocast_dtype or torch.float32,
            enabled=autocast_enabled,
        ):
            hidden = self.encoder.forward_range(
                hidden,
                frame_mask,
                start=0,
                end=None,
                attention_causal=self.attention_causal,
                convolution_causal=self.convolution_causal,
            )
            target_length = int(token_ids.shape[1]) * 4
            predictions = self.heads.predict(
                hidden,
                mel_target_length=target_length,
                chroma_target_length=target_length,
                mask=frame_mask,
                causal=self.attention_causal and self.causal_heads,
            )
        return predictions

    def encode_lyrics(self, text: str) -> list[int]:
        if self.ctc_tokenizer is None:
            return []
        return [int(value) for value in self.ctc_tokenizer.encode(text)]


def _collapse_ctc(ids: Sequence[int], blank_id: int = 0) -> list[int]:
    output: list[int] = []
    previous: int | None = None
    for value in ids:
        value = int(value)
        if value != blank_id and value != previous:
            output.append(value)
        previous = value
    return output


def _edit_operations(
    reference: Sequence[int], hypothesis: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return Levenshtein edit, substitution, deletion, and insertion counts."""

    previous = [(column, 0, 0, column) for column in range(len(hypothesis) + 1)]
    for row, ref_value in enumerate(reference, start=1):
        current = [(row, 0, row, 0)]
        for column, hyp_value in enumerate(hypothesis, start=1):
            diagonal = previous[column - 1]
            mismatch = int(ref_value != hyp_value)
            candidates = (
                (
                    diagonal[0] + mismatch,
                    diagonal[1] + mismatch,
                    diagonal[2],
                    diagonal[3],
                ),
                (
                    previous[column][0] + 1,
                    previous[column][1],
                    previous[column][2] + 1,
                    previous[column][3],
                ),
                (
                    current[-1][0] + 1,
                    current[-1][1],
                    current[-1][2],
                    current[-1][3] + 1,
                ),
            )
            best_cost = min(item[0] for item in candidates)
            current.append(next(item for item in candidates if item[0] == best_cost))
        previous = current
    return previous[-1]


def _ctc_required_frames(target: Sequence[int]) -> int:
    repeats = sum(left == right for left, right in zip(target, target[1:]))
    return len(target) + repeats


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-8:
        return None
    return float((left * right).sum().item() / denominator.item())


def _cosine_mean(
    left: torch.Tensor, right: torch.Tensor, *, epsilon: float = 1e-8
) -> tuple[float, int]:
    norms = torch.linalg.vector_norm(right.float(), dim=-1)
    active = norms > epsilon
    if not bool(active.any()):
        return 0.0, 0
    values = F.cosine_similarity(left[active].float(), right[active].float(), dim=-1)
    return float(values.sum().item()), int(values.numel())


def _add(
    numerators: dict[str, float],
    denominators: dict[str, float],
    name: str,
    value: float,
    weight: float = 1.0,
) -> None:
    if not math.isfinite(value) or weight <= 0:
        return
    numerators[name] = numerators.get(name, 0.0) + value * weight
    denominators[name] = denominators.get(name, 0.0) + weight


@torch.no_grad()
def score_stage4_pair_batch(
    reference: dict[str, torch.Tensor],
    hypothesis: dict[str, torch.Tensor],
    *,
    frame_lengths: torch.Tensor,
    lyric_targets: Sequence[Sequence[int]],
    blank_id: int = 0,
    mel_targets: Sequence[torch.Tensor | None] | None = None,
    chroma_targets: Sequence[torch.Tensor | None] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:

    numerators: dict[str, float] = {}
    denominators: dict[str, float] = {}
    batch = int(frame_lengths.numel())
    if len(lyric_targets) != batch:
        raise ValueError(
            f"lyric_targets length does not match batch size: "
            f"{len(lyric_targets)} != {batch}"
        )
    if mel_targets is not None and len(mel_targets) != batch:
        raise ValueError("mel_targets length does not match batch size")
    if chroma_targets is not None and len(chroma_targets) != batch:
        raise ValueError("chroma_targets length does not match batch size")
    for row in range(batch):
        frames = int(frame_lengths[row].item())
        if frames <= 0:
            continue
        target = [int(value) for value in lyric_targets[row]]
        _add(
            numerators,
            denominators,
            "ctc/target_available_rate",
            float(bool(target)),
        )
        _add(
            numerators,
            denominators,
            "ctc/target_feasible_rate",
            float(bool(target) and _ctc_required_frames(target) <= frames),
        )
        for source_name, predictions in (
            ("reference", reference),
            ("hypothesis", hypothesis),
        ):
            logits = predictions["ctc_logits"][row, :frames].float()
            probabilities = logits.softmax(dim=-1)
            argmax = logits.argmax(dim=-1)
            decoded = _collapse_ctc(argmax.tolist(), blank_id=blank_id)
            _add(
                numerators,
                denominators,
                f"ctc/{source_name}/blank_ratio",
                float((argmax == blank_id).float().mean().item()),
            )
            posterior_entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1)
            _add(
                numerators,
                denominators,
                f"ctc/{source_name}/posterior_entropy",
                float(posterior_entropy.mean().item()),
            )
            _add(
                numerators,
                denominators,
                f"ctc/{source_name}/mean_confidence",
                float(probabilities.max(dim=-1).values.mean().item()),
            )
            if target:
                edits, substitutions, deletions, insertions = _edit_operations(
                    target, decoded
                )
                target_count = float(len(target))
                _add(
                    numerators,
                    denominators,
                    f"ctc/{source_name}/token_error_rate",
                    edits / target_count,
                    target_count,
                )
                _add(
                    numerators,
                    denominators,
                    f"ctc/{source_name}/substitution_rate",
                    substitutions / target_count,
                    target_count,
                )
                _add(
                    numerators,
                    denominators,
                    f"ctc/{source_name}/deletion_rate",
                    deletions / target_count,
                    target_count,
                )
                _add(
                    numerators,
                    denominators,
                    f"ctc/{source_name}/insertion_rate",
                    insertions / target_count,
                    target_count,
                )
                _add(
                    numerators,
                    denominators,
                    f"ctc/{source_name}/exact_match",
                    float(edits == 0),
                )
                if _ctc_required_frames(target) <= frames:
                    flat_target = torch.tensor(
                        target, dtype=torch.long, device=logits.device
                    )
                    ctc = F.ctc_loss(
                        logits.log_softmax(dim=-1).unsqueeze(1),
                        flat_target,
                        torch.tensor([frames], device=logits.device),
                        torch.tensor([len(target)], device=logits.device),
                        blank=blank_id,
                        reduction="sum",
                        zero_infinity=True,
                    )
                    _add(
                        numerators,
                        denominators,
                        f"ctc/{source_name}/nll_per_target_token",
                        float(ctc.item()) / target_count,
                        target_count,
                    )

        target_frames = frames * 4
        reference_mel = reference["mel"][row, :target_frames].float()
        hypothesis_mel = hypothesis["mel"][row, :target_frames].float()
        mel_difference = hypothesis_mel - reference_mel
        mel_elements = float(mel_difference.numel())
        _add(
            numerators,
            denominators,
            "mel/hypothesis_to_reference/l1",
            float(mel_difference.abs().mean().item()),
            mel_elements,
        )
        reference_norm = float(torch.linalg.vector_norm(reference_mel).item())
        _add(
            numerators,
            denominators,
            "mel/hypothesis_to_reference/spectral_convergence",
            float(torch.linalg.vector_norm(mel_difference).item())
            / max(reference_norm, 1e-8),
        )
        cosine_sum, cosine_count = _cosine_mean(hypothesis_mel, reference_mel)
        if cosine_count:
            _add(
                numerators,
                denominators,
                "mel/hypothesis_to_reference/frame_cosine",
                cosine_sum / cosine_count,
                float(cosine_count),
            )
        mel_bins = int(reference_mel.shape[-1])
        edges = (0, mel_bins // 3, 2 * mel_bins // 3, mel_bins)
        for band, low, high in zip(("low", "middle", "high"), edges, edges[1:]):
            difference = mel_difference[:, low:high]
            _add(
                numerators,
                denominators,
                f"mel/hypothesis_to_reference/l1_{band}",
                float(difference.abs().mean().item()),
                float(difference.numel()),
            )
        correlation = _pearson(
            hypothesis_mel.mean(dim=-1), reference_mel.mean(dim=-1)
        )
        if correlation is not None:
            _add(
                numerators,
                denominators,
                "mel/hypothesis_to_reference/energy_envelope_correlation",
                correlation,
            )

        reference_chroma = reference["chroma"][row, :target_frames].float()
        hypothesis_chroma = hypothesis["chroma"][row, :target_frames].float()
        chroma_difference = hypothesis_chroma - reference_chroma
        chroma_elements = float(chroma_difference.numel())
        _add(
            numerators,
            denominators,
            "chroma/hypothesis_to_reference/l1",
            float(chroma_difference.abs().mean().item()),
            chroma_elements,
        )
        chroma_reference_norm = float(
            torch.linalg.vector_norm(reference_chroma).item()
        )
        _add(
            numerators,
            denominators,
            "chroma/hypothesis_to_reference/spectral_convergence",
            float(torch.linalg.vector_norm(chroma_difference).item())
            / max(chroma_reference_norm, 1e-8),
        )
        cosine_sum, cosine_count = _cosine_mean(
            hypothesis_chroma, reference_chroma
        )
        if cosine_count:
            _add(
                numerators,
                denominators,
                "chroma/hypothesis_to_reference/frame_cosine",
                cosine_sum / cosine_count,
                float(cosine_count),
            )
        dominant = (
            hypothesis_chroma.argmax(dim=-1) == reference_chroma.argmax(dim=-1)
        ).float()
        _add(
            numerators,
            denominators,
            "chroma/hypothesis_to_reference/dominant_pitch_class_accuracy",
            float(dominant.mean().item()),
            float(dominant.numel()),
        )
        profile_left = hypothesis_chroma.mean(dim=0, keepdim=True)
        profile_right = reference_chroma.mean(dim=0, keepdim=True)
        _add(
            numerators,
            denominators,
            "chroma/hypothesis_to_reference/global_profile_cosine",
            float(F.cosine_similarity(profile_left, profile_right, dim=-1).item()),
        )
        if target_frames > 1:
            reference_change = (
                reference_chroma[1:].argmax(dim=-1)
                != reference_chroma[:-1].argmax(dim=-1)
            )
            hypothesis_change = (
                hypothesis_chroma[1:].argmax(dim=-1)
                != hypothesis_chroma[:-1].argmax(dim=-1)
            )
            true_positive = int((reference_change & hypothesis_change).sum().item())
            precision = true_positive / max(int(hypothesis_change.sum().item()), 1)
            recall = true_positive / max(int(reference_change.sum().item()), 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            _add(
                numerators,
                denominators,
                "chroma/hypothesis_to_reference/change_f1",
                f1,
            )

        audio_mel = None if mel_targets is None else mel_targets[row]
        audio_chroma = None if chroma_targets is None else chroma_targets[row]
        if mel_targets is not None or chroma_targets is not None:
            _add(
                numerators,
                denominators,
                "audio_target/available_rate",
                float(audio_mel is not None and audio_chroma is not None),
            )
        if audio_mel is not None:
            target_mel = torch.as_tensor(
                audio_mel, dtype=torch.float32, device=reference_mel.device
            )
            _add(
                numerators,
                denominators,
                "audio_target/mel_aligned_fraction",
                min(int(reference_mel.shape[0]), int(target_mel.shape[0]))
                / max(int(target_mel.shape[0]), 1),
            )
            _add(
                numerators,
                denominators,
                "audio_target/mel_length_error_frames",
                float(int(reference_mel.shape[0]) - int(target_mel.shape[0])),
            )
            for source_name, source_mel in (
                ("reference", reference_mel),
                ("hypothesis", hypothesis_mel),
            ):
                aligned = min(int(source_mel.shape[0]), int(target_mel.shape[0]))
                if aligned <= 0:
                    continue
                prediction = source_mel[:aligned]
                target_audio = target_mel[:aligned]
                difference = prediction - target_audio
                _add(
                    numerators,
                    denominators,
                    f"mel/{source_name}_to_audio/l1",
                    float(difference.abs().mean().item()),
                    float(difference.numel()),
                )
                _add(
                    numerators,
                    denominators,
                    f"mel/{source_name}_to_audio/spectral_convergence",
                    float(torch.linalg.vector_norm(difference).item())
                    / max(float(torch.linalg.vector_norm(target_audio).item()), 1e-8),
                )
                cosine_sum, cosine_count = _cosine_mean(prediction, target_audio)
                if cosine_count:
                    _add(
                        numerators,
                        denominators,
                        f"mel/{source_name}_to_audio/frame_cosine",
                        cosine_sum / cosine_count,
                        float(cosine_count),
                    )
        if audio_chroma is not None:
            target_chroma = torch.as_tensor(
                audio_chroma,
                dtype=torch.float32,
                device=reference_chroma.device,
            )
            _add(
                numerators,
                denominators,
                "audio_target/chroma_aligned_fraction",
                min(int(reference_chroma.shape[0]), int(target_chroma.shape[0]))
                / max(int(target_chroma.shape[0]), 1),
            )
            _add(
                numerators,
                denominators,
                "audio_target/chroma_length_error_frames",
                float(
                    int(reference_chroma.shape[0]) - int(target_chroma.shape[0])
                ),
            )
            for source_name, source_chroma in (
                ("reference", reference_chroma),
                ("hypothesis", hypothesis_chroma),
            ):
                aligned = min(
                    int(source_chroma.shape[0]), int(target_chroma.shape[0])
                )
                if aligned <= 0:
                    continue
                prediction = source_chroma[:aligned]
                target_audio = target_chroma[:aligned]
                difference = prediction - target_audio
                _add(
                    numerators,
                    denominators,
                    f"chroma/{source_name}_to_audio/l1",
                    float(difference.abs().mean().item()),
                    float(difference.numel()),
                )
                _add(
                    numerators,
                    denominators,
                    f"chroma/{source_name}_to_audio/spectral_convergence",
                    float(torch.linalg.vector_norm(difference).item())
                    / max(float(torch.linalg.vector_norm(target_audio).item()), 1e-8),
                )
                cosine_sum, cosine_count = _cosine_mean(prediction, target_audio)
                if cosine_count:
                    _add(
                        numerators,
                        denominators,
                        f"chroma/{source_name}_to_audio/frame_cosine",
                        cosine_sum / cosine_count,
                        float(cosine_count),
                    )
                dominant = (
                    prediction.argmax(dim=-1) == target_audio.argmax(dim=-1)
                ).float()
                _add(
                    numerators,
                    denominators,
                    f"chroma/{source_name}_to_audio/"
                    "dominant_pitch_class_accuracy",
                    float(dominant.mean().item()),
                    float(dominant.numel()),
                )
    return numerators, denominators


__all__ = [
    "STAGE4_PROBE_ARTIFACT_VERSION",
    "STAGE4_PROBE_PROTOCOL",
    "STAGE4_REFERENCE_TARGET_PROTOCOL",
    "Stage4AudioTargetStore",
    "Stage4ProbeInfo",
    "Stage4SemanticProbe",
    "export_stage4_probe_artifact",
    "score_stage4_pair_batch",
]
