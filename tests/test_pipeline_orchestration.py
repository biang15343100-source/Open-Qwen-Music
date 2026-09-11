from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import open_qwen_music.pipeline as pipeline
from open_qwen_music.release import DEFAULT_REVISION


class _Condition:
    description = "A calm acoustic song"

    @staticmethod
    def render_tags() -> str:
        return "folk; calm"

    @staticmethod
    def render_lyrics() -> str:
        return "Sing softly"


class _Renderer:
    revisions = SimpleNamespace(rewriter_revision="open-qwen-music-rewriter-v1")

    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    def render(self, *args, **kwargs):
        self.arguments = kwargs
        return object()


def test_cpu_fixture_runs_the_complete_inference_orchestration(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "language-model"
    model_dir.mkdir()
    text_encoder = tmp_path / "text-encoder"
    text_encoder.mkdir()
    renderer = _Renderer()
    bundle = SimpleNamespace(
        semantic_tokenizer_revision="semantic-tokenizer-v1",
        component_dir=lambda name: model_dir,
        load_language_model=lambda *args, **kwargs: object(),
        resolve_text_encoder=lambda **kwargs: text_encoder,
        load_render_pipeline=lambda **kwargs: (
            renderer,
            {"solver": "heun", "num_steps": 100, "cfg_scale": 7.0, "use_refiner": True},
        ),
    )
    resolved: dict[str, object] = {}

    def fake_from_pretrained(source, **kwargs):
        resolved.update(source=source, **kwargs)
        return bundle

    generation: dict[str, object] = {}

    def fake_generation(config, **kwargs):
        generation.update(kwargs)
        Path(kwargs["render_requests_path"]).write_text("fixture\n", encoding="utf-8")

    request = SimpleNamespace(
        sample_id="t2m-037",
        semantic_ids=np.array([1, 2, 3], dtype=np.int64),
        semantic_tokenizer_revision="semantic-tokenizer-v1",
        condition=_Condition(),
    )

    def fake_publish(destination: Path, *, output, preset):
        destination.write_bytes(b"RIFF-fixture")
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        Path(str(destination) + ".json").write_text(
            json.dumps({"output": {"sha256": digest}}), encoding="utf-8"
        )
        return {"output": {"sha256": digest}}

    monkeypatch.setattr(
        pipeline.OpenQwenMusicWeightBundle, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(pipeline, "run_generation", fake_generation)
    monkeypatch.setattr(pipeline, "read_render_requests", lambda path: [request])
    monkeypatch.setattr(pipeline, "_publish_render_output", fake_publish)

    root = Path(__file__).resolve().parents[1]
    manifest_path = pipeline.run_pipeline(
        config_path=root / "configs/inference/pipeline.yaml",
        prompts_path=root / "examples/prompts.jsonl",
        output_dir=tmp_path / "outputs",
        device="cpu",
        seed=17,
    )

    assert resolved["revision"] == DEFAULT_REVISION
    assert generation["mode"] == "plain"
    assert generation["temperature"] == 0.7
    assert generation["top_p"] == 0.98
    assert renderer.arguments["use_refiner"] is True
    assert renderer.arguments["seed"] == 17
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest[0]["sample_id"] == "t2m-037"
    assert Path(manifest[0]["output"]).is_file()
    assert Path(manifest[0]["metadata"]).is_file()
