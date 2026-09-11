# Bandwidth Refiner Training

The bandwidth refiner improves high-frequency detail after Acoustic VAE decoding. It is trained from paired coarse VAE reconstructions and 48 kHz stereo references.

## Prerequisites and data

Freeze the selected Stage 2 VAE before producing refiner pairs. Every pair must record the exact VAE identity used to produce its coarse input. The refiner does not depend on the language model or renderer, so it can train after the VAE handoff while renderer training proceeds separately.

The configured train and validation manifests contain 1.28-second paired windows. Keep source tracks disjoint across splits and retain the audio integrity fields required by the loader.

Generate the shared acoustic manifests after building the Tokenizer view:

```bash
oqm-preprocess build-acoustic-views \
  --tokenizer-manifest "$OQM_DATA_ROOT/tokenizer/stage34.jsonl" \
  --output-dir "$OQM_DATA_ROOT"
```

The training loop runs the frozen VAE on each reference crop to obtain the coarse
reconstruction, so the manifest contains the reference audio and its integrity
metadata rather than a second precomputed waveform.

## Launch

The recipe uses the final Stage 2 VAE checkpoint. The loader verifies its
architecture, STFT contract, and posterior mode. The fixed topology is 64
processes with one sample per process. The schedule runs for 1,250 optimizer
updates, and `--stop-at-step 1250` is mandatory.

Run this command on eight nodes with eight GPUs per node, setting `NODE_RANK` separately on each node:

```bash
NNODES=8 NODE_RANK=0 \
MASTER_ADDR=host0.example.net MASTER_PORT=29500 \
NPROC_PER_NODE=8 RDZV_ID=oqm-refiner \
  bash scripts/launch_torchrun.sh scripts/train.py refiner \
  --vae-checkpoint /path/to/selected-vae-stage2.pt \
  --stop-at-step 1250
```

Resume the same run with the same VAE parent and target gate:

```bash
python scripts/train.py refiner \
  --vae-checkpoint /path/to/selected-vae-stage2.pt \
  --resume "$OQM_OUTPUT_ROOT/refiner/train/last.pt" \
  --stop-at-step 1250
```

A fresh run initializes the refiner and its discriminators while keeping the
parent VAE frozen. `--resume` restores the complete state of the same run.

## Handoff

Compare refined and unrefined outputs at matched loudness. Check high-band spectral error, phase consistency, stereo image, peak level, clipping, and regressions in the low and mid bands. Select a checkpoint from the declared gates using held-out metrics and listening tests.
