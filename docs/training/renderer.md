# Acoustic Renderer Training

The acoustic renderer maps 25 Hz semantic tokens and text conditions to Acoustic
VAE latents. Freeze the semantic Tokenizer, the Stage 2 VAE, and the text encoder
before creating the training release. A change to any of these artifacts requires
a new output directory and a complete rebuild.

## 1. Prepare the source manifest

Create one JSONL record per complete audio item. Every record must have a unique
`sample_id`, a `train`, `valid`, or `test` split, local 48 kHz stereo audio, a
description and lyrics condition, quality metadata, and stable recording,
performance, composition, and song group identifiers. The source manifest and
the semantic-token release must contain exactly the same sample IDs and splits.

```json
{"sample_id":"track-0001","split":"train","audio":{"path":"audio/track-0001.flac","sha256":"<sha256>"},"condition":{"description":"Acoustic folk with a calm vocal","lyrics":"Example lyrics"},"quality":{"render_broad":true},"groups":{"recording_group_id":"recording-0001","performance_group_id":"performance-0001","composition_group_id":"composition-0001","song_group_id":"song-0001"}}
```

The semantic-token producer publishes `semantic_tokens.jsonl` and `READY` in one
directory. Each token artifact is a one-dimensional `uint16` array at 25 Hz with
IDs in `[0, 32768)`, and each row binds the token artifact to the SHA-256 of its
source audio.

## 2. Adapt the semantic-token release

Normalize the generic semantic-token release into the Renderer input contract.
To reuse the semantic stream from the materialized language-model corpus, bind it
to the source manifest while adapting it:

```bash
oqm-adapt-renderer-semantics \
  --input-dir "$OQM_DATA_ROOT/llm/corpus" \
  --source-manifest "$OQM_DATA_ROOT/renderer/source.jsonl" \
  --output-dir "$OQM_DATA_ROOT/renderer-semantics"
```

The adapter checks the 25 Hz frame rate, `uint16` dtype, token range, and audio
alignment. It publishes a new `semantic_tokens.jsonl` and `READY` without
changing the token values. A producer that already
publishes generic `semantic_tokens.jsonl` plus `READY` can be passed through
`--input-dir` directly; `--source-manifest` is required for a packed
language-model corpus.

## 3. Materialize the Renderer release

The production materializer loads the frozen Tokenizer, VAE, and text encoder
directly. The semantic calibration report must describe the frozen language model
used to set the corruption rate.

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

The command validates semantic/audio alignment before encoding anything. It then
publishes the following independent artifacts:

| Directory | Contents |
| --- | --- |
| `samples/<split>/` | Parent sample manifest, random-access index, and `READY` |
| `latents/<split>/` | Standardized VAE latent artifacts, manifest, and `READY` |
| `latents/stats.json` | Training-split channel statistics and SHA-256 sidecar |
| `text/tags/` | Content-addressed description embeddings and `READY` |
| `text/short_lyrics/` | Content-addressed lyrics embeddings and `READY` |
| `crops/<split>/` | Section-aware contiguous windows targeting 90 seconds and `READY` |
| `loudness/<split>/` | Per-window loudness values and `READY` |
| `semantic_embedding/` | Frozen Tokenizer codebook and `READY` |
| `semantic_corruption/` | Codebook-neighbor tables, calibration report, and `READY` |
| `renderer.resolved.yaml` | Runtime paths, record counts, and component revisions |
| `READY` | Completion marker for the materialized release |

The final `READY` appears only after every child `READY` metadata file has been
written. The command refuses to reuse an existing output directory so publication
stays atomic.
Prefer a new versioned directory over replacing a release already used by a run.

The materializer computes channel statistics from the complete training split,
standardizes every train, validation, and test latent with those statistics, and
binds the resulting stats and cache revisions into `renderer.resolved.yaml`.

Use the generated configuration for training. Keep the checked-in recipe small:
set paths through `OQM_DATA_ROOT`, `OQM_MODEL_ROOT`, and `OQM_OUTPUT_ROOT`
instead of copying generated metadata into it.

## 4. Train with the standard launcher

The checked-in recipe trains for 29,000 optimizer updates from a fresh model and
optimizer state. The first 25,000 updates use clean semantic conditions. The next
1,000 updates ramp the configured semantic corruption from zero to its target.
The remaining updates use the full configured clean and robust sample mixture.

Run the standard launcher on every node and assign each node a unique
`NODE_RANK`:

```bash
NNODES=8 NODE_RANK=0 \
MASTER_ADDR=host0.example.net MASTER_PORT=29500 \
NPROC_PER_NODE=8 RDZV_ID=oqm-renderer \
  bash scripts/launch_torchrun.sh scripts/train.py renderer \
  --config "$OQM_DATA_ROOT/renderer/renderer.resolved.yaml"
```

The recipe uses 64 processes, one sample per process, and four gradient
accumulation steps, preserving a global parent batch of 256.

## 5. Validate the handoff

Evaluate predicted latents on the held-out Renderer set and decode them with the
frozen VAE. Check semantic timing, prompt adherence, lyrics alignment, stereo
output, loudness, and finite values. Package the selected EMA weights.
