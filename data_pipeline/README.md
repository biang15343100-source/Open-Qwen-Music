# Open-Qwen-Music Data Pipeline

This package prepares training corpora for Open-Qwen-Music. It contains two commands:

- `oqm-preprocess`: discover, probe, filter, deduplicate, normalize, and publish audio records;
- `oqm-annotate`: run separation, structure analysis, ASR, alignment, acoustic tags, and metadata fusion.

## Installation

```bash
pip install -e "./data_pipeline[gpu]"
```

## Preprocessing

Download the public training corpus first. This revision contains 62,417 audio
records in 827 WebDataset shards and requires approximately 815 GiB of storage.

```bash
export OQM_DATASET_ROOT=/datasets/open-qwen-music
hf download david-miller-45678/open-qwen-music-dataset-c-minimax-music3 \
  --repo-type dataset \
  --revision 454294f27d920bb3c1430566bb7f5c153ece4c27 \
  --local-dir "$OQM_DATASET_ROOT"

export OQM_WORK_DIR=/work/oqm-preprocess
export OQM_RELEASE_ROOT=/datasets/oqm-corpus

oqm-preprocess validate-registry \
  --config data_pipeline/preprocess_configs/pipeline.yaml
oqm-preprocess run \
  --config data_pipeline/preprocess_configs/pipeline.yaml

export OQM_CORPUS_RELEASE="$(oqm-preprocess release-path \
  --config data_pipeline/preprocess_configs/pipeline.yaml)"
```

The default registry scans `data/train-*.tar` under `OQM_DATASET_ROOT`. It reads the
WebDataset archives directly, so the shards do not need to be extracted. Keep corpus
selection and quality thresholds in YAML configuration.

This dataset is licensed for non-commercial academic research only. Review its
`LICENSE`, `NOTICE`, `licenses/source_policy.json`, and per-record `rights` metadata
before using or redistributing any record.

## Annotation

```bash
export ANNOT_WORK=/work/oqm-annotate

export NNODES=8
export NODE_RANK=0
oqm-annotate run \
  --config data_pipeline/configs/examples/open_qwen_music.yaml \
  --all
```

Run the annotation command on all eight nodes with a unique `NODE_RANK`. The
base configuration starts eight GPU workers per node, for 64 workers in total.

The annotation configuration reads the exact versioned directory in
`OQM_CORPUS_RELEASE`. After annotation, create the Tokenizer manifests with:

```bash
export OQM_DATA_ROOT=/datasets/oqm-training
oqm-preprocess build-tokenizer-views \
  --release "$OQM_CORPUS_RELEASE" \
  --annotations "$ANNOT_WORK/open-qwen-music/annotation.jsonl" \
  --output-dir "$OQM_DATA_ROOT/tokenizer"

oqm-preprocess build-acoustic-views \
  --tokenizer-manifest "$OQM_DATA_ROOT/tokenizer/stage34.jsonl" \
  --output-dir "$OQM_DATA_ROOT"
```

The second command creates checksummed train, validation, and test manifests for
both Acoustic VAE stages and the bandwidth refiner. It reads archive members
directly and preserves indexed-TAR offsets, so the source shards remain packed.

GPU workers use `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. Multi-node scheduling
remains the responsibility of the runtime platform.
