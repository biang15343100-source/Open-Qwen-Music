"""Load the community Open-Qwen-Music weight bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from open_qwen_music.common.config import load_config
from open_qwen_music.llm.model import MusicLLM
from open_qwen_music.llm.registry import TokenRegistry
from open_qwen_music.render.cache import FrozenLatentStats
from open_qwen_music.render.conditioning import REWRITER_SCHEMA_VERSION
from open_qwen_music.render.contracts import STFT_CONTRACT_VERSION
from open_qwen_music.render.dit_factory import build_render_dit_components_from_config
from open_qwen_music.render.inference import InferenceRevisions, RenderInferencePipeline
from open_qwen_music.render.refiner import build_refiner_from_mapping
from open_qwen_music.render.spec_vae import SpecVAE, SpecVAEConfig
from open_qwen_music.render.stft import STFTConfig, StereoSTFT
from open_qwen_music.tokenizer.deployment import DeploymentMusicTokenizer


BUNDLE_FORMAT = "open-qwen-music.weight-bundle.v1"
DEFAULT_REPO_ID = "oqmtest1451/open-qwen-music-weights"
DEFAULT_REVISION = "64b37b84fe6a62c5a08d3366470ca9d4f28b5e67"


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_safetensors(module: nn.Module, directory: Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("Install open-qwen-music[render] to load model weights") from exc

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = _mapping(index.get("weight_map"), name=str(index_path))
        shard_names = sorted(set(str(value) for value in weight_map.values()))
        expected_keys = set(weight_map)
    else:
        shard_names = ["model.safetensors"]
        expected_keys = set(module.state_dict())

    loaded_keys: set[str] = set()
    unexpected_keys: set[str] = set()
    for shard_name in shard_names:
        shard_path = (directory / shard_name).resolve()
        if directory.resolve() not in shard_path.parents:
            raise ValueError(f"Weight shard escapes its component directory: {shard_name}")
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing weight shard: {shard_path}")
        state = load_file(str(shard_path), device="cpu")
        loaded_keys.update(state)
        incompatible = module.load_state_dict(state, strict=False)
        unexpected_keys.update(incompatible.unexpected_keys)

    model_keys = set(module.state_dict())
    missing = sorted(model_keys - loaded_keys)
    extra = sorted((loaded_keys - model_keys) | unexpected_keys)
    undeclared = sorted(loaded_keys - expected_keys)
    if missing or extra or undeclared:
        raise RuntimeError(
            "Weight keys do not match the model architecture: "
            f"missing={missing[:8]} extra={extra[:8]} undeclared={undeclared[:8]}"
        )


@dataclass(frozen=True)
class OpenQwenMusicWeightBundle:
    """A resolved local snapshot of the published model bundle."""

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path = DEFAULT_REPO_ID,
        *,
        revision: str = DEFAULT_REVISION,
        cache_dir: str | Path | None = None,
    ) -> "OpenQwenMusicWeightBundle":
        candidate = Path(source).expanduser()
        if candidate.is_dir():
            root = candidate.resolve()
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - dependency error
                raise RuntimeError(
                    "Install open-qwen-music[render] to download model weights"
                ) from exc
            root = Path(
                snapshot_download(
                    repo_id=str(source),
                    revision=revision,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            ).resolve()
        manifest_path = root / "open_qwen_music.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Model manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != BUNDLE_FORMAT:
            raise ValueError(f"Unsupported model bundle: {manifest.get('format_version')!r}")
        return cls(root=root, manifest=manifest)

    def component_dir(self, name: str) -> Path:
        components = _mapping(self.manifest.get("components"), name="components")
        component = _mapping(components.get(name), name=f"components.{name}")
        path = (self.root / str(component["path"])).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Component path escapes the model snapshot: {path}")
        if not path.is_dir():
            raise FileNotFoundError(f"Component directory not found: {path}")
        return path

    def verify_component_artifact(self, name: str) -> Path:
        """Verify the published identity file for one component before loading it."""
        directory = self.component_dir(name)
        index_path = directory / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = _mapping(index.get("weight_map"), name=str(index_path))
            files = [index_path]
            for shard_name in sorted(set(str(value) for value in weight_map.values())):
                shard_path = (directory / shard_name).resolve()
                if directory.resolve() not in shard_path.parents:
                    raise ValueError(
                        f"Weight shard escapes its component directory: {shard_name}"
                    )
                if not shard_path.is_file():
                    raise FileNotFoundError(f"Missing weight shard: {shard_path}")
                files.append(shard_path)
            components = _mapping(self.manifest.get("components"), name="components")
            component = _mapping(components.get(name), name=f"components.{name}")
            declared_checksums = component.get("files_sha256")
            checksums = (
                _mapping(declared_checksums, name=f"components.{name}.files_sha256")
                if declared_checksums is not None
                else {}
            )
            checksums_path = self.root / "SHA256SUMS"
            if not checksums and checksums_path.is_file():
                for line in checksums_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    checksum, relative = line.split(maxsplit=1)
                    relative = relative.lstrip("* ")
                    if relative.startswith("./"):
                        relative = relative[2:]
                    checksums[relative] = checksum.lower()
            if checksums:
                for path in files:
                    root_relative = path.relative_to(self.root).as_posix()
                    component_relative = path.relative_to(directory).as_posix()
                    expected = checksums.get(component_relative) or checksums.get(
                        root_relative
                    )
                    if expected is None:
                        raise RuntimeError(
                            f"Published file checksums do not cover {root_relative}"
                        )
                    actual = _sha256(path)
                    if actual != expected:
                        raise RuntimeError(
                            f"Artifact SHA-256 mismatch for {root_relative}: "
                            f"expected={expected}, actual={actual}"
                        )
            return index_path

        artifact_path = directory / "model.safetensors"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Component artifact not found: {artifact_path}")
        identities = _mapping(self.manifest.get("artifact_sha256"), name="artifact_sha256")
        expected = str(identities.get(name, "")).lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise ValueError(f"Invalid artifact SHA-256 for component {name!r}")
        actual = _sha256(artifact_path)
        if actual != expected:
            raise RuntimeError(
                f"Artifact SHA-256 mismatch for {name}: expected={expected}, actual={actual}"
            )
        return artifact_path

    @property
    def semantic_tokenizer_revision(self) -> str:
        contracts = _mapping(self.manifest.get("contracts"), name="contracts")
        semantic = _mapping(contracts.get("semantic_token"), name="semantic_token")
        return str(semantic["revision"])

    def validate_language_model_training_recipe(self) -> None:
        components = _mapping(self.manifest.get("components"), name="components")
        language_model = _mapping(
            components.get("language_model"), name="components.language_model"
        )
        training = _mapping(
            language_model.get("training"), name="components.language_model.training"
        )
        expected = {
            "sequence_mode": "plain",
            "stage1_updates": 5_000,
            "stage2_updates": 5_000,
            "total_updates": 10_000,
        }
        mismatches = {
            name: {"expected": value, "actual": training.get(name)}
            for name, value in expected.items()
            if training.get(name) != value
        }
        identities = _mapping(
            self.manifest.get("artifact_sha256"), name="artifact_sha256"
        )
        if training.get("artifact_sha256") != identities.get("language_model"):
            mismatches["artifact_sha256"] = {
                "expected": identities.get("language_model"),
                "actual": training.get("artifact_sha256"),
            }
        if mismatches:
            raise ValueError(f"Language-model training recipe mismatch: {mismatches}")

    def resolve_text_encoder(self, *, cache_dir: str | Path | None = None) -> Path:
        external = _mapping(self.manifest.get("external_models"), name="external_models")
        spec = _mapping(external.get("renderer_text_encoder"), name="renderer_text_encoder")
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "Install open-qwen-music[render] to download the text encoder"
            ) from exc
        return Path(
            snapshot_download(
                repo_id=str(spec["repo_id"]),
                revision=str(spec["revision"]),
                cache_dir=None if cache_dir is None else str(cache_dir),
            )
        ).resolve()

    def load_semantic_tokenizer(
        self, *, device: str | torch.device = "cpu"
    ) -> DeploymentMusicTokenizer:
        directory = self.component_dir("semantic_tokenizer")
        self.verify_component_artifact("semantic_tokenizer")
        artifact = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError("Install open-qwen-music[render] to load model weights") from exc
        artifact["state_dict"] = load_file(str(directory / "model.safetensors"), device="cpu")
        model = DeploymentMusicTokenizer(
            artifact,
            revision=self.semantic_tokenizer_revision,
        )
        return model.requires_grad_(False).to(device).eval()

    def load_language_model(
        self,
        registry: TokenRegistry,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        model_options: Mapping[str, Any] | None = None,
    ) -> MusicLLM:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError("Install open-qwen-music[render] to load the language model") from exc
        options = dict(model_options or {})
        implementation = str(options.get("attn_implementation", "sdpa"))
        directory = self.component_dir("language_model")
        self.validate_language_model_training_recipe()
        self.verify_component_artifact("language_model")
        try:
            backbone = AutoModelForCausalLM.from_pretrained(
                str(directory), dtype=dtype, attn_implementation=implementation
            )
        except (ImportError, ValueError) as error:
            if implementation != "flash_attention_2":
                raise
            print(f"flash_attention_2 is unavailable ({error}); using sdpa", flush=True)
            backbone = AutoModelForCausalLM.from_pretrained(
                str(directory), dtype=dtype, attn_implementation="sdpa"
            )
        model = MusicLLM(
            backbone,
            registry,
            loss_chunk_size=int(options.get("loss_chunk_size", 256)),
            accuracy_sample_positions=int(options.get("accuracy_sample_positions", 4096)),
            melody_class_weights=options.get("melody_class_weights"),
            modality_type_embeddings=bool(options.get("modality_type_embeddings", False)),
            grammar_constrained_loss=bool(options.get("grammar_constrained_loss", False)),
            grammar_constrained_metrics=bool(options.get("grammar_constrained_metrics", True)),
        )
        return model.requires_grad_(False).to(device).eval()

    def load_render_pipeline(
        self,
        *,
        device: str | torch.device,
        text_encoder_dir: str | Path | None = None,
    ) -> tuple[RenderInferencePipeline, dict[str, Any]]:
        target = torch.device(device)
        renderer_dir = self.component_dir("acoustic_renderer")
        self.verify_component_artifact("acoustic_renderer")
        renderer_config = load_config(renderer_dir / "config.yaml")
        conditioning = _mapping(renderer_config.get("conditioning"), name="conditioning")
        embedding = _mapping(
            conditioning.get("semantic_embedding_asset"),
            name="semantic_embedding_asset",
        )
        embedding["ready_path"] = str(
            self.root / "assets/semantic-embedding/READY.json"
        )
        conditioning["semantic_embedding_asset"] = embedding
        text_encoder = _mapping(conditioning.get("text_encoder"), name="text_encoder")
        text_encoder["local_path"] = str(
            Path(text_encoder_dir).resolve()
            if text_encoder_dir is not None
            else self.resolve_text_encoder()
        )
        conditioning["text_encoder"] = text_encoder
        renderer_config["conditioning"] = conditioning

        dit, conditioner = build_render_dit_components_from_config(renderer_config)
        wrapper = nn.ModuleDict({"dit": dit, "conditioner": conditioner})
        _load_safetensors(wrapper, renderer_dir)
        precision = str(renderer_config.get("inference", {}).get("precision", "bf16"))
        renderer_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        wrapper.requires_grad_(False).to(device=target, dtype=renderer_dtype).eval()

        vae_dir = self.component_dir("acoustic_vae")
        self.verify_component_artifact("acoustic_vae")
        vae_config = load_config(vae_dir / "config.yaml")
        vae = SpecVAE(SpecVAEConfig.from_dict(dict(vae_config["model"]["spec_vae"])))
        _load_safetensors(vae, vae_dir)
        vae.requires_grad_(False).to(target).eval()

        refiner_dir = self.component_dir("bandwidth_refiner")
        self.verify_component_artifact("bandwidth_refiner")
        refiner_config = load_config(refiner_dir / "config.yaml")
        refiner = build_refiner_from_mapping(refiner_config["model"]["refiner"])
        _load_safetensors(refiner, refiner_dir)
        refiner.requires_grad_(False).to(target).eval()

        revisions = _mapping(renderer_config.get("revisions"), name="revisions")
        vae.revision = str(revisions["vae_revision"])
        refiner.revision = str(revisions["refiner_revision"])
        stft = StereoSTFT(STFTConfig.from_mapping(vae_config["stft"])).to(target).eval()
        stft.revision = STFT_CONTRACT_VERSION

        stats_path = self.root / "assets/latent-stats.json"
        stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_sha = _sha256(stats_path)
        stats = FrozenLatentStats(
            path=stats_path,
            sha256=stats_sha,
            payload=stats_payload,
            mean=np.asarray(stats_payload["mean"], dtype=np.float64),
            std=np.asarray(stats_payload["std"], dtype=np.float64),
        )
        inference_revisions = InferenceRevisions.from_mapping(
            {
                "checkpoint_revision": revisions["checkpoint_revision"],
                "tokenizer_revision": revisions["tokenizer_revision"],
                "vae_revision": revisions["vae_revision"],
                "refiner_revision": revisions["refiner_revision"],
                "text_encoder_revision": revisions["text_encoder_revision"],
                "text_tokenizer_revision": revisions["text_tokenizer_revision"],
                "text_cache_revision": revisions["text_cache_revision"],
                "rewriter_revision": revisions["rewriter_revision"],
                "latent_cache_revision": revisions["latent_cache_revision"],
                "latent_stats_sha256": stats_sha,
            }
        )
        identities = _mapping(self.manifest.get("artifact_sha256"), name="artifact_sha256")
        inference = _mapping(renderer_config.get("inference"), name="inference")
        pipeline = RenderInferencePipeline(
            model=dit,
            conditioner=conditioner,
            spec_decoder=vae,
            refiner=refiner,
            inverse_stft=stft,
            latent_stats=stats,
            flow_config=renderer_config["flow"],
            revisions=inference_revisions,
            require_input_tokenizer_revision=True,
            artifact_identities={
                "dit_checkpoint_sha256": str(identities["acoustic_renderer"]),
                "vae_checkpoint_sha256": str(identities["acoustic_vae"]),
                "refiner_checkpoint_sha256": str(identities["bandwidth_refiner"]),
                "latent_stats_sha256": stats_sha,
            },
            inverse_stft_revision=STFT_CONTRACT_VERSION,
            precision=precision,
        )
        preset = {
            "solver": str(inference.get("solver", "euler")),
            "num_steps": int(inference.get("num_steps", 8)),
            "cfg_scale": float(inference.get("cfg_scale", 1.0)),
            "use_refiner": True,
            "wav_subtype": "PCM_24",
            "flac_subtype": "PCM_24",
            "peak_policy": "reject",
            "max_output_peak": 1.0,
            "max_output_true_peak": 1.0,
            "rewriter_schema_version": REWRITER_SCHEMA_VERSION,
        }
        return pipeline, preset
