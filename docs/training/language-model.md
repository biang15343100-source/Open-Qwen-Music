# Music Language Model Training

The language model predicts semantic music tokens from text, lyrics, and section
structure. The released recipe samples only the plain text-to-semantic sequence.
It cannot be trained from raw audio.

## Prerequisites

Freeze the semantic tokenizer first, then materialize the language-model corpus.
The corpus root configured by `data.corpus` must contain:

```text
corpus/
├── corpus.json
├── manifest.jsonl
├── semantic/
│   └── sem-00000.bin
└── melody/
    └── mel-00000.bin
```

Semantic shards use little-endian `uint16` IDs at 25 Hz. Melody shards use
`uint8` IDs at 6.25 Hz, with 255 reserved for unvoiced frames. Each manifest row
contains the token spans, split, conditions, lyrics, and section metadata. The
tokenizer revision in `corpus.json` must match the exported deployment
artifact, and its semantic extractor revision must match the frozen Stage 4
checkpoint.

Build this corpus directly from the Tokenizer `stage34.jsonl` view:

```bash
oqm-materialize-llm \
  --input "$OQM_DATA_ROOT/tokenizer/stage34.jsonl" \
  --output-dir "$OQM_DATA_ROOT/llm/corpus" \
  --tokenizer-artifact "$OQM_OUTPUT_ROOT/tokenizer/tokenizer.pt" \
  --rmvpe-checkpoint "$OQM_CHECKPOINT_ROOT/melody/rmvpe.pt" \
  --device cuda
```

The materializer expects the published 128-Mel RMVPE checkpoint and produces a
50 Hz pitch curve before converting it to 6.25 Hz melody tokens.
`--verify-rmvpe-sha256` pins the expected RMVPE file identity.
The output path must be new: the materializer refuses overwrite and resume so a
corpus cannot combine shards produced by different checkpoints.

Set both tokenizer identities before loading either LLM configuration:

```bash
export OQM_TOKENIZER_REVISION=<tokenizer-deploy-artifact-sha256>
export OQM_TOKENIZER_EXTRACTOR_REVISION=<stage-4-checkpoint-sha256>
```

`OQM_TOKENIZER_REVISION` identifies the exported deployment artifact that produced
the semantic IDs. `OQM_TOKENIZER_EXTRACTOR_REVISION` identifies the frozen Stage 4
checkpoint from which that artifact was exported. The corpus metadata must match
both values.

## Stages

The training path has two consecutive 5,000-update stages. Stage 1 starts from
the configured Qwen base model. Stage 2 initializes from the Stage 1 model
checkpoint while starting the Stage 2 optimizer and scheduler state.

```bash
python scripts/train.py llm-stage1

python scripts/train.py llm-stage2 \
  --init-from "${OQM_OUTPUT_ROOT:-./outputs}/llm/stage1/final.pt"
```

Stage 2 restores the Stage 1 sampler and data cursor, but starts a new optimizer
and scheduler. Its 5,000 stage-local updates therefore consume the next 5,000
batches rather than replaying Stage 1. The final checkpoint records a stage-local
optimizer step of 5,000 and a cumulative data step of 10,000.

The final model is written to
`${OQM_OUTPUT_ROOT:-./outputs}/llm/stage2/final.pt`.

For multi-node training:

```bash
NNODES=8 NODE_RANK=0 \
MASTER_ADDR=host0.example.net MASTER_PORT=29500 \
NPROC_PER_NODE=8 RDZV_ID=oqm-llm \
  bash scripts/launch_torchrun.sh scripts/train.py llm-stage1
```

Both stages use 64 processes, one sequence per process, and two gradient
accumulation steps, preserving the effective global batch size of 128. Run Stage
2 with the same topology after Stage 1 finishes.

## Resume after interruption

Use `--resume-from` only with a checkpoint from the same stage:

```bash
python scripts/train.py llm-stage2 \
  --resume-from "${OQM_OUTPUT_ROOT:-./outputs}/llm/stage2/last.pt"
```

The resume path restores optimizer, scheduler, sampler, and random state.

## Handoff checks

Before packaging the final checkpoint, verify that:

- the corpus tokenizer revision equals `OQM_TOKENIZER_REVISION`;
- the corpus semantic extractor revision equals
  `OQM_TOKENIZER_EXTRACTOR_REVISION`;
- semantic and materialized melody IDs satisfy their vocabulary bounds;
- train and validation manifests use stable, non-overlapping splits;
- the checkpoint reports Stage 2, stage-local step 5,000, and cumulative data step
  10,000;
- generation produces valid 25 Hz semantic streams on a held-out prompt set.
