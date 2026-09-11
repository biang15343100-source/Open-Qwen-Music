from pathlib import Path

from oqm_preprocess.registry import Registry


def test_registry_expands_environment_paths(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    monkeypatch.setenv("OQM_TEST_DATASET_ROOT", str(dataset_root))
    (registry_dir / "dataset.yaml").write_text(
        """
dataset_id: 1
slug: test-dataset
name: Test Dataset
sources:
- storage_class: tar
  root: ${OQM_TEST_DATASET_ROOT:-./fallback}/data
  glob: train-*.tar
license:
  id: TEST
  family: research_only
  commercial_ok: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    source = Registry.load(registry_dir).get("test-dataset").sources[0]

    assert source.root == dataset_root / "data"
