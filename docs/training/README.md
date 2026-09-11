# Training Workflow

This directory documents each training component at its real handoff boundary.
The top-level order is a dependency graph rather than a single serial list.

## Execution phases

| Phase | Work | Can run concurrently | Completion gate |
| --- | --- | --- | --- |
| 1 | Download, validate, preprocess, and annotate audio | Preprocessing shards and independent annotation workers | Published corpus and annotation `READY` files |
| 2 | Train the semantic tokenizer and Acoustic VAE | Tokenizer and VAE | Frozen tokenizer checkpoint; frozen VAE checkpoint |
| 3 | Materialize semantic/melody labels and VAE latents | Token labeling, latent encoding, and text caching | Immutable manifests with checkpoint revisions and checksums |
| 4 | Train the language model, renderer, and refiner | All three, once each component's inputs are ready | Selected checkpoint for each component |
| 5 | Assemble and validate the inference bundle | Packaging checks | All artifact identities match the inference manifest |

The phase number expresses a dependency boundary, not a fixed duration. Each
component guide records its optimizer-update budget and data contract. Measure
wall-clock time on the target cluster after a short throughput run.

## Module guides

- [Data preparation](data-preparation.md)
- [Semantic tokenizer](tokenizer.md)
- [Music language model](language-model.md)
- [Acoustic VAE](acoustic-vae.md)
- [Acoustic renderer](renderer.md)
- [Bandwidth refiner](refiner.md)

## Cross-module contracts

The semantic tokenizer emits one 32,768-entry codebook at 25 Hz. Melody tokens
use a 256-entry vocabulary at 6.25 Hz, with token 255 reserved for unvoiced
frames. The renderer and VAE use 25 Hz acoustic latents. Changing any of these
contracts invalidates downstream caches and checkpoints.

Every handoff should record:

1. the source checkpoint path and SHA-256;
2. the configuration revision;
3. the data manifest and its checksum;
4. the token, latent, sample-rate, and channel contracts;
5. the exact train, validation, and test split identities.

Do not begin a downstream stage while an upstream checkpoint or materialized
dataset is still changing.
