# Semantic Tokenizer Training

The semantic Tokenizer produces one 32,768-entry codebook at 25 Hz from 24 kHz
mono audio. Its stages are sequential and every stage validates the phase recorded
in its parent checkpoint.

## Inputs and stage order

Run `oqm-preprocess build-tokenizer-views` first. Stages 1a, 1b, and 2b consume
`$OQM_DATA_ROOT/tokenizer/stage12.jsonl` with random windows up to 30 seconds.
Stage 3 head warmup, Stage 3, and Stage 4 consume
`$OQM_DATA_ROOT/tokenizer/stage34.jsonl` with complete tracks up to 300 seconds.

| Phase | Objective | Updates | Required parent |
| --- | --- | ---: | --- |
| Stage 1a | Masked acoustic pretraining | 100,000 | Fresh initialization |
| Stage 1b | Music-domain adaptation | 20,000 | Stage 1a |
| Stage 2b | Causal adaptation | 20,000 | Stage 1b |
| Stage 3 head warmup | Supervised-head warmup | 2,000 | Stage 2b |
| Stage 3 | Joint supervised training | 40,000 | Stage 3 head warmup |
| Stage 4 initialization | Data-dependent spherical k-means | 1,048,576 frames | Stage 3 |
| Stage 4 | Discrete semantic codebook training | 50,000 | Stage 4 initialization artifact |

## Distributed launch

Every Tokenizer stage uses 64 processes and one gradient accumulation step. The
per-rank batch sizes preserve each stage's configured global batch size. Launch
the stages on eight nodes with eight GPUs per node through the standard wrapper
shown in [the training guide](../../TRAINING.md#distributed-launch).

## Launch sequence

```bash
python scripts/train.py tokenizer-stage1a

python scripts/train.py tokenizer-stage1b \
  --init-from "$OQM_OUTPUT_ROOT/tokenizer/stage1a/last.pt"

python scripts/train.py tokenizer-stage2b \
  --init-from "$OQM_OUTPUT_ROOT/tokenizer/stage1b/last.pt"

python scripts/train.py tokenizer-stage3-head-warmup \
  --init-from "$OQM_OUTPUT_ROOT/tokenizer/stage2b/last.pt"

python scripts/train.py tokenizer-stage3 \
  --init-from "$OQM_OUTPUT_ROOT/tokenizer/stage3-head-warmup/last.pt"
```

Stage 4 must not start from the Stage 3 checkpoint directly. First materialize and
validate the data-dependent codebook:

```bash
python -m open_qwen_music.tokenizer.initialization \
  --config configs/train/tokenizer-stage4.yaml \
  --stage3-checkpoint "$OQM_OUTPUT_ROOT/tokenizer/stage3/last.pt"

python scripts/train.py tokenizer-stage4 \
  --init-from "$OQM_CHECKPOINT_ROOT/tokenizer/stage4-initialized.pt"
```

The initialization command loads the Stage 3 encoder, collects the configured
number of projected frames from `stage4_init.jsonl`, runs spherical k-means, and
writes an artifact containing the source checkpoint hash, manifest hash, seed,
codebook shape, and initialization parameters. Stage 4 rejects a normal Stage 3
checkpoint, an unmarked checkpoint, or an artifact with a mismatched contract.

## Resume, freeze, and export

Use `--resume-from` only with a checkpoint from the same phase. Resume restores
the optimizer, scheduler, data position, and random state. `--init-from` starts a
new optimizer and is accepted only for the declared parent phase.

After validation, export the frozen Stage 4 encoder:

```bash
oqm-tokenizer-export \
  --config configs/train/tokenizer-stage4.yaml \
  --checkpoint "$OQM_OUTPUT_ROOT/tokenizer/stage4/last.pt" \
  --output "$OQM_OUTPUT_ROOT/tokenizer/tokenizer.pt"
```

Any change to the vocabulary, 25 Hz frame contract, 32,768-entry codebook, or
frozen Tokenizer checkpoint requires rebuilding downstream semantic-token data.
