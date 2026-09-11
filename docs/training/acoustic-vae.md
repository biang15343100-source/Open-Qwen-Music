# Acoustic VAE Training

The Acoustic VAE converts 48 kHz stereo audio into the 25 Hz latent representation used by the renderer. It is trained before the renderer and bandwidth refiner. It does not depend on the semantic tokenizer or language model.

## Data

Both stages use 1.28-second random crops of paired 48 kHz stereo audio. Each JSONL row must contain the audio path, duration, split, and integrity metadata expected by the configured dataset loader. Stage 2 uses the curated manifest under `data/vae-stage2`; its train, validation, and test source tracks must remain disjoint.

Create the manifests from the published Tokenizer audio view:

```bash
oqm-preprocess build-acoustic-views \
  --tokenizer-manifest "$OQM_DATA_ROOT/tokenizer/stage34.jsonl" \
  --output-dir "$OQM_DATA_ROOT"
```

The builder verifies that all three splits are present and records the SHA-256 of
each audio payload. See the [data-preparation guide](data-preparation.md) for the
full input and output contract.

## Stage 1: reconstruction

Stage 1 starts from a fresh model and trains the encoder and decoder with reconstruction and KL objectives. The published configuration uses 64 processes, four samples per process, and a 90,000-step schedule. `--stop-at-step` is mandatory and must name one of the configured quality gates.

```bash
NNODES=8 NODE_RANK=0 \
MASTER_ADDR=host0.example.net MASTER_PORT=29500 \
NPROC_PER_NODE=8 RDZV_ID=oqm-vae-stage1 \
  bash scripts/launch_torchrun.sh scripts/train.py vae-stage1 \
  --stop-at-step 90000
```

Set `NODE_RANK` independently on each node. For an earlier evaluation gate, replace `90000` with a value from `train.quality_gate_steps` in `configs/train/vae-stage1.yaml`. Resume an interrupted run with `--resume` and the same stop step.

## Stage 2: adversarial decoder refinement

Stage 2 starts from the selected Stage 1 VAE. It creates fresh optimizer and discriminator state and trains only the configured decoder scope. The published configuration uses 64 processes, one sample per process, and evaluates through the 5,000-step gate; this is 5,000 new Stage 2 optimizer updates.

```bash
NNODES=8 NODE_RANK=0 \
MASTER_ADDR=host0.example.net MASTER_PORT=29500 \
NPROC_PER_NODE=8 RDZV_ID=oqm-vae-stage2 \
  bash scripts/launch_torchrun.sh scripts/train.py vae-stage2 \
  --vae-checkpoint /path/to/selected-vae-stage1.pt \
  --stop-at-step 5000
```

Pass a checkpoint selected from the Stage 1 quality gates. The loader verifies its Spec-VAE architecture, STFT contract, and posterior mode before initializing Stage 2. Use `--resume` only for a checkpoint produced by the same Stage 2 run:

```bash
python scripts/train.py vae-stage2 \
  --vae-checkpoint /path/to/selected-vae-stage1.pt \
  --resume "$OQM_OUTPUT_ROOT/vae-stage2/train/last.pt" \
  --stop-at-step 5000
```

## Handoff

Select a Stage 2 checkpoint using held-out reconstruction metrics and listening tests. Before renderer cache creation or refiner training, verify 48 kHz stereo I/O, latent dimension 128, latent rate 25 Hz, finite output, bounded peaks, and split isolation. Any VAE change invalidates renderer latent caches and VAE-derived refiner pairs.
