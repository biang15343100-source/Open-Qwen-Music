# Data Preparation

The data pipeline has three immutable boundaries: the validated corpus release,
the annotation output, and model-specific training views. Keep each boundary in a
new directory when its inputs or configuration change.

## 1. Download the community dataset

```bash
export OQM_DATASET_ROOT=/datasets/open-qwen-music

hf download david-miller-45678/open-qwen-music-dataset-c-minimax-music3 \
  --repo-type dataset \
  --local-dir "$OQM_DATASET_ROOT"
```

Review the dataset card and the per-record rights metadata before training or
redistributing derived data.

## 2. Publish a validated corpus

```bash
pip install -e "./data_pipeline[gpu]"

export OQM_WORK_DIR=/work/oqm-preprocess
export OQM_RELEASE_ROOT=/datasets/oqm-corpus

oqm-preprocess validate-registry \
  --config data_pipeline/preprocess_configs/pipeline.yaml
oqm-preprocess run \
  --config data_pipeline/preprocess_configs/pipeline.yaml

export OQM_CORPUS_RELEASE="$(oqm-preprocess release-path \
  --config data_pipeline/preprocess_configs/pipeline.yaml)"
test -f "$OQM_CORPUS_RELEASE/READY"
```

`OQM_RELEASE_ROOT` is the parent directory used by preprocessing.
`OQM_CORPUS_RELEASE` is the exact, versioned directory consumed downstream. The
`release-path` command prevents annotation from accidentally opening the parent
directory.

## 3. Produce annotations

Each annotation stage has its own data and checkpoint requirements. `index` and
the CPU metadata stages can run first. ASR, source separation, alignment, tagging,
and structure analysis require their configured model checkpoints and accelerators.
Run `oqm-annotate stages` to inspect the dependency graph.

```bash
export ANNOT_WORK=/work/oqm-annotate

export NNODES=8
export NODE_RANK=0
oqm-annotate run \
  --config data_pipeline/configs/examples/open_qwen_music.yaml \
  --all
```

Run this command on all eight nodes with a unique `NODE_RANK`. The base
configuration starts eight GPU workers per node.

The completed annotation file is
`$ANNOT_WORK/open-qwen-music/annotation.jsonl`. Do not build supervised Tokenizer
views while required annotation stages are incomplete.

## 4. Build Tokenizer training views

```bash
export OQM_DATA_ROOT=/datasets/oqm-training

oqm-preprocess build-tokenizer-views \
  --release "$OQM_CORPUS_RELEASE" \
  --annotations "$ANNOT_WORK/open-qwen-music/annotation.jsonl" \
  --output-dir "$OQM_DATA_ROOT/tokenizer"
```

The command emits:

| Artifact | Consumer |
| --- | --- |
| `stage12.jsonl` | Tokenizer stages 1a, 1b, and 2b |
| `stage34.jsonl` | Tokenizer Stage 3 head warmup, Stage 3, and Stage 4 |
| `stage4_init.jsonl` | Data-dependent Stage 4 codebook initialization |
| `ctc_vocab.json` | All Tokenizer stages |
| `VIEW.json` and `READY` | Source identity, counts, contracts, and checksums |
| `MATERIALIZATION_REQUIRED.json` | Model-dependent work that cannot be produced from metadata alone |

The base release may already contain lyrics. When `--annotations` is provided,
the builder replaces those lyrics and language fields with the completed annotation
record joined by the stable release UID.

## 5. Materialize model-dependent data

Build the audio manifests shared by the Acoustic VAE and bandwidth refiner before
starting either training workflow. The command hashes each source audio payload
and preserves indexed-TAR read hints from the Tokenizer view:

```bash
oqm-preprocess build-acoustic-views \
  --tokenizer-manifest "$OQM_DATA_ROOT/tokenizer/stage34.jsonl" \
  --output-dir "$OQM_DATA_ROOT"
```

This writes `train.jsonl`, `valid.jsonl`, and `test.jsonl` below each of
`vae-stage1/`, `vae-stage2/`, and `refiner/`. It also publishes
`ACOUSTIC_VIEW.json` and `ACOUSTIC_READY` with source and output checksums. The command
rejects duplicate sample IDs, missing splits, clipped URIs, and storage formats
that the acoustic loader cannot read.

Some products cannot be generated before their parent model is frozen:

1. Train Tokenizer stages 1a through 3 in order.
2. Use the frozen Stage 3 encoder and `stage4_init.jsonl` to materialize the
   data-dependent Stage 4 initialization artifact.
3. Train and freeze Tokenizer Stage 4.
4. Export the frozen Stage 4 Tokenizer, then build the language-model corpus with
   both frozen tokenizers:

   ```bash
   oqm-materialize-llm \
     --input "$OQM_DATA_ROOT/tokenizer/stage34.jsonl" \
     --output-dir "$OQM_DATA_ROOT/llm/corpus" \
     --tokenizer-artifact "$OQM_OUTPUT_ROOT/tokenizer/tokenizer.pt" \
     --rmvpe-checkpoint "$OQM_CHECKPOINT_ROOT/melody/rmvpe.pt" \
     --device cuda
   ```

   The command emits 25 Hz semantic `uint16` shards and 6.25 Hz melody `uint8`
   shards, preserving unvoiced ID 255. It validates the deployment sidecar,
   records both the deployment-artifact SHA-256 and the source Stage 4 checkpoint
   SHA-256, and validates the completed corpus before publishing it.
5. The output directory must not already exist. An interrupted run removes its
   temporary directory; start a new run rather than mixing shards from different
   model revisions.
6. Train the language model only after both token streams and their checkpoint
   revisions are recorded in the corpus manifest.
7. Materialize acoustic VAE latents only after freezing the VAE; train the Renderer
   against those exact latent and Tokenizer revisions.

## 6. Build the Renderer training release

Start from a local source manifest with stable sample IDs, splits, 48 kHz stereo
audio, text conditions, quality fields, and grouping identifiers. First adapt the
generic 25 Hz semantic-token release:

```bash
oqm-adapt-renderer-semantics \
  --input-dir "$OQM_DATA_ROOT/llm/corpus" \
  --source-manifest "$OQM_DATA_ROOT/renderer/source.jsonl" \
  --output-dir "$OQM_DATA_ROOT/renderer-semantics"
```

Then materialize all model-dependent Renderer inputs into one immutable release.
The built-in production adapters load the frozen Tokenizer, VAE, and text encoder
and record their identities:

```bash
oqm-materialize-renderer \
  --config configs/train/renderer.yaml \
  --source-manifest "$OQM_DATA_ROOT/renderer/source.jsonl" \
  --semantic-release "$OQM_DATA_ROOT/renderer-semantics" \
  --output-dir "$OQM_DATA_ROOT/renderer" \
  --tokenizer-checkpoint "$OQM_CHECKPOINT_ROOT/tokenizer/tokenizer.pt" \
  --vae-checkpoint "$OQM_CHECKPOINT_ROOT/vae/vae.pt" \
  --text-encoder "$OQM_MODEL_ROOT/Qwen3-Embedding-0.6B" \
  --calibration-report "$OQM_DATA_ROOT/llm/evaluation/semantic-calibration.json" \
  --semantic-top-k 384 \
  --device cuda
```

The output separates sample manifests, latent artifacts, text caches, crop
manifests, loudness manifests, and the semantic embedding into distinct
directories with `READY` metadata. The materializer generates
`renderer.resolved.yaml` in the output root with the paths and component
revisions needed by training. Pass that file to the training command in the
[Renderer training guide](renderer.md).

Changing a Tokenizer vocabulary, frame rate, codebook, or frozen checkpoint
invalidates all downstream token shards and renderer inputs. Rebuild the derived
view instead of editing an existing release.
