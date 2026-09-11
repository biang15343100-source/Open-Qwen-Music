
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .contracts import SEMANTIC_CODEBOOK_SIZE


SEMANTIC_CORRUPTION_FORMAT_VERSION = "oqm.render.semantic-corruption.v1"
SEMANTIC_ERROR_CALIBRATION_SCHEMA = "oqm.render-semantic-error-calibration.v1"
SEMANTIC_ERROR_CALIBRATION_STATUS = "RENDER_SEMANTIC_ERROR_CALIBRATION_READY"
SEMANTIC_ERROR_METRIC = "semantic_teacher_forced_accuracy_at_1"
SEMANTIC_TOP_K_POLICY = "tokenizer_codebook_cosine_knn"
SEMANTIC_DISTRACTOR_ASSET_SCHEMA = "oqm.render-semantic-distractors.v1"
SEMANTIC_DISTRACTOR_READY_SCHEMA = "oqm.render-semantic-distractors-ready.v1"
SEMANTIC_DISTRACTOR_READY_STATUS = "RENDER_SEMANTIC_DISTRACTORS_READY"
SEMANTIC_DISTRACTOR_SOURCE = "tokenizer_codebook_cosine_knn"
SEMANTIC_DISTRACTOR_METRIC = "cosine"
SEMANTIC_CORRUPTION_MODES = ("none", "emdc_knn")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a 64-character SHA-256")
    digest = value.lower()
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a hexadecimal SHA-256") from exc
    return digest


def _resolve_local_file(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"{field} must be a non-empty local path")
    text = str(value)
    if text.startswith("file://"):
        text = text.removeprefix("file://")
        if not text.startswith("/"):
            raise ValueError(f"{field} file URI must contain an absolute path")
    elif "://" in text:
        raise ValueError(f"{field}only supports local files")
    try:
        path = Path(text).expanduser().resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{field} does not exist: {text}") from exc
    if not path.is_file():
        raise ValueError(f"{field} must be a file: {path}")
    return path


def _load_json_object(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{field} could not be parsed: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} must be a JSON object")
    return value


def _stable_seed(
    base_seed: int,
    *,
    global_step: int,
    sample_id: str,
) -> int:
    payload = (
        f"{SEMANTIC_CORRUPTION_FORMAT_VERSION}:{base_seed}:"
        f"{global_step}:{sample_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


@dataclass(frozen=True)
class SemanticCorruptionConfig:
    format_version: str = SEMANTIC_CORRUPTION_FORMAT_VERSION
    mode: str = "none"
    clean_steps: int = 0
    ramp_steps: int = 0
    top_k: int = 0
    top_k_policy: str = "none"
    temperature: float = 1.0
    seed: int = 0
    calibration: Mapping[str, Any] | None = None
    asset: Mapping[str, Any] | None = None
    robust_sample_probability: float = 1.0

    def validate(self) -> None:
        if self.format_version != SEMANTIC_CORRUPTION_FORMAT_VERSION:
            raise ValueError(
                "semantic_corruption.format_version must be "
                f"{SEMANTIC_CORRUPTION_FORMAT_VERSION}"
            )
        if self.mode not in SEMANTIC_CORRUPTION_MODES:
            raise ValueError(
                f"semantic_corruption.mode must be one of {SEMANTIC_CORRUPTION_MODES}"
            )
        for name, value in {
            "clean_steps": self.clean_steps,
            "ramp_steps": self.ramp_steps,
            "top_k": self.top_k,
            "seed": self.seed,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"semantic_corruption.{name} must be a non-negative integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) <= 0.0
        ):
            raise ValueError("semantic_corruption.temperature must be a finite positive number")
        if (
            isinstance(self.robust_sample_probability, bool)
            or not isinstance(self.robust_sample_probability, (int, float))
            or not math.isfinite(float(self.robust_sample_probability))
            or not 0.0 <= float(self.robust_sample_probability) <= 1.0
        ):
            raise ValueError(
                "semantic_corruption.robust_sample_probability must be finite and in [0, 1]"
            )
        if self.mode == "none":
            if (
                self.clean_steps != 0
                or self.ramp_steps != 0
                or self.top_k != 0
                or self.top_k_policy != "none"
                or float(self.temperature) != 1.0
                or self.seed != 0
                or self.calibration is not None
                or self.asset is not None
                or float(self.robust_sample_probability) != 1.0
            ):
                raise ValueError(
                    "semantic_corruption.mode=none requires clean, ramp, top_k, and seed "
                    "to be zero; top_k_policy=none; temperature and "
                    "robust_sample_probability=1; and calibration and asset=null"
                )
        else:
            calibration = SemanticErrorCalibration.from_mapping(self.calibration)
            if calibration.replacement_probability <= 0.0:
                raise ValueError("emdc_knn requires an LLM accuracy@1 below 1")
            if not isinstance(self.asset, Mapping):
                raise TypeError("emdc_knn requires a versioned distractor asset")
            if self.top_k <= 0:
                raise ValueError("emdc_knn top_k must be a positive integer")
            if self.top_k_policy != SEMANTIC_TOP_K_POLICY:
                raise ValueError(
                    "emdc_knn top_k_policy must be "
                    f"{SEMANTIC_TOP_K_POLICY}"
                )
            if float(self.robust_sample_probability) <= 0.0:
                raise ValueError("emdc_knn robust_sample_probability must be greater than 0")

    @property
    def replacement_probability(self) -> float:
        self.validate()
        if self.mode == "none":
            return 0.0
        return SemanticErrorCalibration.from_mapping(
            self.calibration
        ).replacement_probability

    def probability_for_step(self, global_step: int) -> float:

        self.validate()
        if not isinstance(global_step, int) or isinstance(global_step, bool):
            raise TypeError("semantic corruption global_step must be an integer")
        if global_step < 0:
            raise ValueError("semantic corruption global_step must not be negative")
        if self.mode == "none":
            return 0.0
        target_probability = self.replacement_probability
        update_index = global_step + 1
        if update_index <= self.clean_steps:
            return 0.0
        if self.ramp_steps > 0 and update_index <= self.clean_steps + self.ramp_steps:
            progress = (update_index - self.clean_steps) / self.ramp_steps
            return target_probability * progress
        return target_probability

    def effective_probability_for_step(self, global_step: int) -> float:

        return self.probability_for_step(global_step) * float(
            self.robust_sample_probability
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)


        if self.robust_sample_probability == 1.0:
            result.pop("robust_sample_probability")
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "SemanticCorruptionConfig":
        normalized = dict(value or {})
        unknown = set(normalized) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"semantic_corruption contains unknown fields: {sorted(unknown)}"
            )
        config = cls(
            **{
                key: normalized[key]
                for key in cls.__dataclass_fields__
                if key in normalized
            }
        )
        config.validate()
        return config


@dataclass(frozen=True)
class SemanticErrorCalibration:
    format_version: str
    metric: str
    accuracy_at_1: float
    negative_log_likelihood_sum_nats: float
    cross_entropy_nats: float
    correct_tokens: int
    evaluated_tokens: int
    tokenizer_revision: str
    token_registry_sha256: str
    llm_checkpoint_sha256: str
    evaluation_dataset_revision: str
    evaluation_manifest_sha256: str
    evaluation_split: str
    evaluation_report_path: str
    evaluation_report_sha256: str

    def validate(self) -> None:
        if self.format_version != SEMANTIC_ERROR_CALIBRATION_SCHEMA:
            raise ValueError(
                "semantic corruption calibration format_version must be "
                f"{SEMANTIC_ERROR_CALIBRATION_SCHEMA}"
            )
        if self.metric != SEMANTIC_ERROR_METRIC:
            raise ValueError(
                f"semantic corruption calibration metric must be {SEMANTIC_ERROR_METRIC}"
            )
        if (
            isinstance(self.accuracy_at_1, bool)
            or not isinstance(self.accuracy_at_1, (int, float))
            or not math.isfinite(float(self.accuracy_at_1))
            or not 0.0 <= float(self.accuracy_at_1) <= 1.0
        ):
            raise ValueError("semantic corruption accuracy_at_1 must be in [0, 1]")
        if (
            isinstance(self.negative_log_likelihood_sum_nats, bool)
            or not isinstance(
                self.negative_log_likelihood_sum_nats,
                (int, float),
            )
            or not math.isfinite(
                float(self.negative_log_likelihood_sum_nats)
            )
            or float(self.negative_log_likelihood_sum_nats) < 0.0
        ):
            raise ValueError(
                "semantic corruption negative_log_likelihood_sum_nats "
                "must be a finite non-negative number"
            )
        if (
            isinstance(self.cross_entropy_nats, bool)
            or not isinstance(self.cross_entropy_nats, (int, float))
            or not math.isfinite(float(self.cross_entropy_nats))
            or float(self.cross_entropy_nats) < 0.0
        ):
            raise ValueError(
                "semantic corruption cross_entropy_nats must be a finite non-negative number"
            )
        if (
            not isinstance(self.evaluated_tokens, int)
            or isinstance(self.evaluated_tokens, bool)
            or self.evaluated_tokens <= 0
        ):
            raise ValueError("semantic corruption evaluated_tokens must be a positive integer")
        if (
            not isinstance(self.correct_tokens, int)
            or isinstance(self.correct_tokens, bool)
            or not 0 <= self.correct_tokens <= self.evaluated_tokens
        ):
            raise ValueError(
                "semantic corruption correct_tokens must be in [0, evaluated_tokens]"
            )
        empirical_accuracy = self.correct_tokens / self.evaluated_tokens
        if not math.isclose(
            float(self.accuracy_at_1),
            empirical_accuracy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "semantic corruption accuracy_at_1 must equal "
                "correct_tokens/evaluated_tokens"
            )
        empirical_cross_entropy = (
            float(self.negative_log_likelihood_sum_nats)
            / self.evaluated_tokens
        )
        if not math.isclose(
            float(self.cross_entropy_nats),
            empirical_cross_entropy,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "semantic corruption cross_entropy_nats must equal "
                "negative_log_likelihood_sum_nats/evaluated_tokens"
            )
        if (
            not isinstance(self.tokenizer_revision, str)
            or not self.tokenizer_revision.strip()
        ):
            raise ValueError("semantic corruption calibration is missing the tokenizer revision")
        _require_sha256(
            self.tokenizer_revision,
            field="semantic corruption tokenizer_revision",
        )
        _require_sha256(
            self.token_registry_sha256,
            field="semantic corruption token_registry_sha256",
        )
        if (
            not isinstance(self.evaluation_dataset_revision, str)
            or not self.evaluation_dataset_revision.strip()
            or self.evaluation_dataset_revision.strip().lower().startswith(
                ("pin-", "new-", "unresolved:")
            )
            or self.evaluation_dataset_revision.strip().lower()
            in {"main", "master", "latest", "head", "unknown"}
        ):
            raise ValueError(
                "semantic corruption evaluation_dataset_revision must be specified"
            )
        _require_sha256(
            self.evaluation_manifest_sha256,
            field="semantic corruption evaluation_manifest_sha256",
        )
        if self.evaluation_split not in {"valid", "test"}:
            raise ValueError(
                "semantic corruption evaluation_split must be valid or test; train is not allowed"
            )
        if (
            not isinstance(self.evaluation_report_path, str)
            or not self.evaluation_report_path.strip()
        ):
            raise ValueError(
                "semantic corruption calibration is missing evaluation_report_path"
            )
        _require_sha256(
            self.llm_checkpoint_sha256,
            field="semantic corruption llm_checkpoint_sha256",
        )
        _require_sha256(
            self.evaluation_report_sha256,
            field="semantic corruption evaluation_report_sha256",
        )

    def verify_report(self) -> dict[str, Any]:

        self.validate()
        report_path = _resolve_local_file(
            self.evaluation_report_path,
            field="semantic corruption evaluation_report_path",
        )
        expected_sha = _require_sha256(
            self.evaluation_report_sha256,
            field="semantic corruption evaluation_report_sha256",
        )
        if _file_sha256(report_path) != expected_sha:
            raise RuntimeError("semantic corruption evaluation report SHA does not match")
        report = _load_json_object(
            report_path,
            field="semantic corruption evaluation report",
        )
        if (
            report.get("schema_version") != SEMANTIC_ERROR_CALIBRATION_SCHEMA
            or report.get("status") != SEMANTIC_ERROR_CALIBRATION_STATUS
        ):
            raise RuntimeError(
                "semantic corruption evaluation report schema or status is incompatible with"
            )
        if report.get("teacher_forced") is not True:
            raise RuntimeError(
                "semantic corruption evaluation report must use teacher-forced evaluation"
            )
        semantic_contract = report.get("semantic_contract")
        if (
            not isinstance(semantic_contract, Mapping)
            or semantic_contract.get("frame_hz") != 25.0
            or semantic_contract.get("codebook_size") != SEMANTIC_CODEBOOK_SIZE
            or semantic_contract.get("codebooks") != 1
        ):
            raise RuntimeError(
                "semantic corruption evaluation report violates the 25 Hz, single-codebook, 32768-token contract"
            )
        expected = {
            "metric": self.metric,
            "accuracy_at_1": float(self.accuracy_at_1),
            "negative_log_likelihood_sum_nats": float(
                self.negative_log_likelihood_sum_nats
            ),
            "cross_entropy_nats": float(self.cross_entropy_nats),
            "correct_tokens": self.correct_tokens,
            "evaluated_tokens": self.evaluated_tokens,
            "tokenizer_revision": self.tokenizer_revision,
            "token_registry_sha256": self.token_registry_sha256,
            "llm_checkpoint_sha256": self.llm_checkpoint_sha256,
            "evaluation_dataset_revision": self.evaluation_dataset_revision,
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "evaluation_split": self.evaluation_split,
        }
        mismatches = {
            name: {"expected": value, "actual": report.get(name)}
            for name, value in expected.items()
            if report.get(name) != value
        }
        if mismatches:
            raise RuntimeError(
                "Semantic corruption calibration does not match the published evaluation report: "
                f"{mismatches}"
            )
        return {
            "schema_version": SEMANTIC_ERROR_CALIBRATION_SCHEMA,
            "status": SEMANTIC_ERROR_CALIBRATION_STATUS,
            "teacher_forced": True,
            "semantic_contract": dict(semantic_contract),
            "path": str(report_path),
            "sha256": expected_sha,
            "top_k_reference": self.top_k_reference,
            **expected,
        }

    @property
    def replacement_probability(self) -> float:
        self.validate()
        return 1.0 - float(self.accuracy_at_1)

    @property
    def top_k_reference(self) -> float:

        self.validate()
        try:
            value = 2.0 * math.exp(float(self.cross_entropy_nats))
        except OverflowError as exc:
            raise ValueError(
                "LLM cross_entropy_nats produces an out-of-range top-k reference"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                "LLM cross_entropy_nats produces a non-finite top-k reference"
            )
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "SemanticErrorCalibration":
        if not isinstance(value, Mapping):
            raise TypeError("semantic corruption calibration must be a mapping")
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            raise ValueError(
                "semantic corruption calibration fields must be exactly"
                f"{sorted(required)}"
            )
        calibration = cls(**{name: value[name] for name in required})
        calibration.validate()
        return calibration


@dataclass(frozen=True)
class SemanticCorruptionResult:
    semantic_ids: torch.Tensor
    replacement_mask: torch.Tensor
    probability: float

    @property
    def replacements(self) -> int:
        return int(self.replacement_mask.sum().item())


class SemanticDistractorTable:

    def __init__(
        self,
        *,
        neighbor_ids: torch.Tensor,
        neighbor_scores: torch.Tensor,
        tokenizer_revision: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(neighbor_ids, torch.Tensor)
            or neighbor_ids.ndim != 2
            or neighbor_ids.shape[0] != SEMANTIC_CODEBOOK_SIZE
            or neighbor_ids.dtype
            not in {torch.int32, torch.int64}
        ):
            raise TypeError("neighbor_ids must be an int32 or int64 tensor with shape [32768, K]")
        if (
            not isinstance(neighbor_scores, torch.Tensor)
            or neighbor_scores.shape != neighbor_ids.shape
            or neighbor_scores.dtype != torch.float32
        ):
            raise TypeError("neighbor_scores must be a float32 tensor with the same shape")
        if neighbor_ids.shape[1] <= 0:
            raise ValueError("Semantic distractor top_k must be greater than 0")
        if not torch.isfinite(neighbor_scores).all():
            raise ValueError("neighbor_scores contains NaN/Inf")
        if neighbor_scores.numel() and (
            float(neighbor_scores.min()) < -1.000001
            or float(neighbor_scores.max()) > 1.000001
        ):
            raise ValueError("neighbor_scores must be in the cosine-similarity range [-1, 1]")
        if neighbor_ids.numel() and (
            int(neighbor_ids.min()) < 0
            or int(neighbor_ids.max()) >= SEMANTIC_CODEBOOK_SIZE
        ):
            raise ValueError("neighbor_ids exceeds the semantic vocabulary")
        rows = torch.arange(SEMANTIC_CODEBOOK_SIZE).view(-1, 1)
        if bool((neighbor_ids.cpu() == rows).any()):
            raise ValueError("Semantic distractor neighbor lists must not contain the source token")
        if neighbor_ids.shape[1] > 1 and bool(
            (neighbor_scores[:, 1:] > neighbor_scores[:, :-1]).any()
        ):
            raise ValueError("neighbor_scores must be sorted by non-increasing cosine similarity")
        sorted_ids = neighbor_ids.detach().cpu().sort(dim=1).values
        if sorted_ids.shape[1] > 1 and bool(
            (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()
        ):
            raise ValueError("Semantic distractor neighbor IDs must be unique within each row")
        tokenizer_revision = _require_sha256(
            tokenizer_revision,
            field="Semantic distractor tokenizer_revision",
        )


        self.neighbor_ids = neighbor_ids.detach().cpu().contiguous()
        self.neighbor_scores = neighbor_scores.detach().cpu().contiguous()
        self.tokenizer_revision = tokenizer_revision
        self.provenance = dict(provenance or {})

    @property
    def top_k(self) -> int:
        return int(self.neighbor_ids.shape[1])

    def corrupt(
        self,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
        *,
        sample_ids: Sequence[str],
        global_step: int,
        config: SemanticCorruptionConfig,
    ) -> SemanticCorruptionResult:
        config.validate()
        if config.mode != "emdc_knn":
            raise ValueError("SemanticDistractorTable can only be used with emdc_knn")
        if config.top_k > self.top_k:
            raise ValueError(
                "semantic corruption top_k exceeds distractor asset capacity:"
                f"requested={config.top_k} available={self.top_k}"
            )
        if semantic_ids.dtype != torch.long or semantic_ids.ndim != 2:
            raise TypeError("semantic_ids must be an int64 tensor with shape [B, T]")
        if (
            semantic_mask.dtype != torch.bool
            or semantic_mask.shape != semantic_ids.shape
        ):
            raise TypeError("semantic_mask must be a bool tensor with shape [B, T]")
        if semantic_mask.device != semantic_ids.device:
            raise ValueError("semantic_ids and semantic_mask must be on the same device")
        if len(sample_ids) != semantic_ids.shape[0] or not all(
            isinstance(value, str) and value for value in sample_ids
        ):
            raise ValueError("EMDC requires every sample to have a unique, non-empty sample_id")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("EMDC sample_id values must be unique within a batch")
        if semantic_ids.numel() and (
            int(semantic_ids.min()) < 0
            or int(semantic_ids.max()) >= SEMANTIC_CODEBOOK_SIZE
        ):
            raise ValueError("semantic_ids must be in [0, 32767]")
        conditional_probability = config.probability_for_step(global_step)
        effective_probability = config.effective_probability_for_step(global_step)


        result = semantic_ids.detach().cpu().clone()
        replaced = torch.zeros_like(result, dtype=torch.bool)
        if effective_probability == 0.0:
            return SemanticCorruptionResult(
                semantic_ids=result.to(device=semantic_ids.device),
                replacement_mask=replaced.to(device=semantic_ids.device),
                probability=effective_probability,
            )
        cpu_mask = semantic_mask.detach().cpu()
        for row, sample_id in enumerate(sample_ids):
            valid_positions = torch.nonzero(
                cpu_mask[row],
                as_tuple=False,
            ).flatten()
            generator = torch.Generator(device="cpu").manual_seed(
                _stable_seed(
                    config.seed,
                    global_step=global_step,
                    sample_id=sample_id,
                )
            )
            robust_sample_probability = float(config.robust_sample_probability)
            if robust_sample_probability < 1.0 and float(
                torch.rand((), generator=generator)
            ) >= robust_sample_probability:
                continue
            decisions = (
                torch.rand(
                    valid_positions.numel(),
                    generator=generator,
                )
                < conditional_probability
            )
            selected_positions = valid_positions[decisions]
            if selected_positions.numel() == 0:
                continue
            original = result[row, selected_positions]
            logits = (
                self.neighbor_scores[original, : config.top_k]
                / float(config.temperature)
            )
            probabilities = torch.softmax(logits, dim=-1)
            neighbor_columns = torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(1)
            replacements = self.neighbor_ids[
                original,
                neighbor_columns,
            ]
            if bool((replacements == original).any()):
                raise RuntimeError("Semantic distractor asset violates the exclude-self contract")
            result[row, selected_positions] = replacements.to(dtype=result.dtype)
            replaced[row, selected_positions] = True
        return SemanticCorruptionResult(
            semantic_ids=result.to(device=semantic_ids.device),
            replacement_mask=replaced.to(device=semantic_ids.device),
            probability=effective_probability,
        )


def verify_semantic_error_calibration(
    calibration_config: Mapping[str, Any] | None,
    *,
    expected_tokenizer_revision: str,
) -> tuple[SemanticErrorCalibration, dict[str, Any]]:

    calibration = SemanticErrorCalibration.from_mapping(calibration_config)
    tokenizer_revision = _require_sha256(
        expected_tokenizer_revision,
        field="semantic corruption expected_tokenizer_revision",
    )
    if calibration.tokenizer_revision != tokenizer_revision:
        raise RuntimeError(
            "semantic corruption calibration is bound to a different tokenizer revision"
        )
    return calibration, calibration.verify_report()


def load_semantic_distractor_table(
    asset_config: Mapping[str, Any],
    *,
    expected_tokenizer_revision: str,
) -> SemanticDistractorTable:
    if not isinstance(asset_config, Mapping):
        raise TypeError("semantic distractor asset must be a mapping")
    required = {
        "ready_path",
        "ready_sha256",
        "artifact_sha256",
        "asset_revision",
        "neighbor_ids_sha256",
        "neighbor_scores_sha256",
        "source_embedding_tensor_sha256",
    }
    allowed = required | {"expected_source"}
    missing = required - set(asset_config)
    unknown = set(asset_config) - allowed
    if missing or unknown:
        raise ValueError(
            "semantic distractor asset fields are incomplete or unknown:"
            f"missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    tokenizer_revision = _require_sha256(
        expected_tokenizer_revision,
        field="semantic distractor expected_tokenizer_revision",
    )
    ready_path = _resolve_local_file(
        asset_config["ready_path"],
        field="semantic distractor READY",
    )
    expected_ready_sha = _require_sha256(
        asset_config["ready_sha256"],
        field="semantic distractor ready_sha256",
    )
    if _file_sha256(ready_path) != expected_ready_sha:
        raise RuntimeError("semantic distractor READY SHA does not match")
    ready = _load_json_object(
        ready_path,
        field="semantic distractor READY",
    )
    if (
        ready.get("schema_version") != SEMANTIC_DISTRACTOR_READY_SCHEMA
        or ready.get("status") != SEMANTIC_DISTRACTOR_READY_STATUS
    ):
        raise RuntimeError("semantic distractor READY schema or status is incompatible with")
    expected_source = str(
        asset_config.get("expected_source") or SEMANTIC_DISTRACTOR_SOURCE
    )
    if expected_source != SEMANTIC_DISTRACTOR_SOURCE:
        raise ValueError("semantic distractor expected_source is invalid")
    if ready.get("source") != expected_source:
        raise RuntimeError("semantic distractor source does not match")
    if ready.get("tokenizer_revision") != tokenizer_revision:
        raise RuntimeError("semantic distractor is bound to a different tokenizer revision")
    asset_revision = _require_sha256(
        asset_config.get("asset_revision"),
        field="semantic distractor asset_revision",
    )
    if ready.get("asset_revision") != asset_revision:
        raise RuntimeError("semantic distractor asset revision does not match")
    source_embedding_sha = _require_sha256(
        asset_config.get("source_embedding_tensor_sha256"),
        field="semantic distractor source_embedding_tensor_sha256",
    )
    if ready.get("source_embedding_tensor_sha256") != source_embedding_sha:
        raise RuntimeError("semantic distractor is not bound to the expected tokenizer embedding")
    if (
        ready.get("metric") != SEMANTIC_DISTRACTOR_METRIC
        or ready.get("exclude_self") is not True
    ):
        raise RuntimeError("semantic distractor metric or exclude_self contract is incompatible")
    ready_top_k = ready.get("top_k")
    if (
        not isinstance(ready_top_k, int)
        or isinstance(ready_top_k, bool)
        or ready_top_k <= 0
        or ready_top_k >= SEMANTIC_CODEBOOK_SIZE
    ):
        raise RuntimeError("semantic distractor READY top_k is invalid")
    artifact = ready.get("artifact")
    if not isinstance(artifact, Mapping):
        raise RuntimeError("semantic distractor READY is missing the artifact identity")
    artifact_path = Path(str(artifact.get("path") or ""))
    if not artifact_path.is_absolute():
        artifact_path = ready_path.parent / artifact_path
    artifact_path = _resolve_local_file(
        artifact_path,
        field="semantic distractor artifact",
    )
    artifact_sha = _require_sha256(
        asset_config["artifact_sha256"],
        field="semantic distractor artifact_sha256",
    )
    if (
        artifact.get("sha256") != artifact_sha
        or _file_sha256(artifact_path) != artifact_sha
    ):
        raise RuntimeError("semantic distractor artifact SHA does not match")
    artifact_size = artifact.get("size_bytes")
    if (
        not isinstance(artifact_size, int)
        or isinstance(artifact_size, bool)
        or artifact_size <= 0
        or artifact_path.stat().st_size != artifact_size
    ):
        raise RuntimeError("semantic distractor artifact size does not match")
    report = ready.get("report")
    if not isinstance(report, Mapping):
        raise RuntimeError("semantic distractor READY is missing the report identity")
    report_path = Path(str(report.get("path") or ""))
    if not report_path.is_absolute():
        report_path = ready_path.parent / report_path
    report_path = _resolve_local_file(
        report_path,
        field="semantic distractor REPORT",
    )
    report_sha = _require_sha256(
        report.get("sha256"),
        field="semantic distractor report.sha256",
    )
    if _file_sha256(report_path) != report_sha:
        raise RuntimeError("semantic distractor REPORT SHA does not match")
    report_payload = _load_json_object(
        report_path,
        field="semantic distractor REPORT",
    )
    ready_report_payload = {
        name: value for name, value in ready.items() if name != "report"
    }
    if dict(report_payload) != ready_report_payload:
        raise RuntimeError("semantic distractor READY and REPORT contents do not match")
    try:
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("semantic distractor artifact could not be loaded safely") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != SEMANTIC_DISTRACTOR_ASSET_SCHEMA
        or payload.get("asset_revision") != asset_revision
        or payload.get("source") != expected_source
        or payload.get("tokenizer_revision") != tokenizer_revision
        or payload.get("source_embedding_tensor_sha256") != source_embedding_sha
        or payload.get("metric") != SEMANTIC_DISTRACTOR_METRIC
        or payload.get("exclude_self") is not True
        or payload.get("top_k") != ready_top_k
    ):
        raise RuntimeError("semantic distractor artifact identity is incompatible")
    neighbor_ids = payload.get("neighbor_ids")
    neighbor_scores = payload.get("neighbor_scores")
    if (
        not isinstance(neighbor_ids, torch.Tensor)
        or neighbor_ids.dtype != torch.int32
        or neighbor_ids.shape != (SEMANTIC_CODEBOOK_SIZE, ready_top_k)
        or not isinstance(neighbor_scores, torch.Tensor)
        or neighbor_scores.dtype != torch.float32
        or neighbor_scores.shape != (SEMANTIC_CODEBOOK_SIZE, ready_top_k)
    ):
        raise RuntimeError(
            "semantic distractor artifact neighbor tensor shape or dtype is incompatible with"
        )
    expected_ids_sha = _require_sha256(
        asset_config["neighbor_ids_sha256"],
        field="semantic distractor neighbor_ids_sha256",
    )
    expected_scores_sha = _require_sha256(
        asset_config["neighbor_scores_sha256"],
        field="semantic distractor neighbor_scores_sha256",
    )
    if (
        _tensor_sha256(neighbor_ids) != expected_ids_sha
        or _tensor_sha256(neighbor_scores) != expected_scores_sha
    ):
        raise RuntimeError("semantic distractor tensor SHA does not match")
    expected_tensor_metadata = {
        "neighbor_ids": {
            "shape": [SEMANTIC_CODEBOOK_SIZE, ready_top_k],
            "dtype": "int32",
            "sha256": expected_ids_sha,
        },
        "neighbor_scores": {
            "shape": [SEMANTIC_CODEBOOK_SIZE, ready_top_k],
            "dtype": "float32",
            "sha256": expected_scores_sha,
        },
    }
    if (
        ready.get("neighbors") != expected_tensor_metadata
        or payload.get("tensors") != expected_tensor_metadata
    ):
        raise RuntimeError("semantic distractor tensor metadata does not match")
    return SemanticDistractorTable(
        neighbor_ids=neighbor_ids,
        neighbor_scores=neighbor_scores,
        tokenizer_revision=tokenizer_revision,
        provenance={
            "ready_path": str(ready_path),
            "ready_sha256": expected_ready_sha,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha,
            "asset_revision": asset_revision,
            "neighbor_ids_sha256": expected_ids_sha,
            "neighbor_scores_sha256": expected_scores_sha,
            "source_embedding_tensor_sha256": source_embedding_sha,
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "top_k": ready_top_k,
            "source": expected_source,
        },
    )
