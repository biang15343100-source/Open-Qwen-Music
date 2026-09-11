# Inference

An online listening gallery is available at the
[`Open-Qwen-Music Demo`](https://huggingface.co/spaces/oqmtest1451/open-qwen-music-demo).

`scripts/infer.py` downloads and runs the complete music language model,
acoustic renderer, VAE decoder, and bandwidth refiner.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[render]"
```

## Prompt format

The input is a JSONL file with one prompt per line. The included example uses
the same conditioning prompt as `t2m-037` in the
[`Open-Qwen-Music Demo`](https://huggingface.co/spaces/oqmtest1451/open-qwen-music-demo):

```json
{"sample_id":"t2m-037","description":"A male vocalist sings gently over a steady acoustic guitar and a simple drum beat. The song has a calm, hopeful quality, with a clear structure that shifts between a verse and a chorus.","tags":{"genre":["folk","singer-songwriter"],"mood":["peaceful","calm","hopeful"],"instruments":["acoustic-guitar","drums","bass-guitar","vocals"],"vocal_gender":"male","vocal_timbre":"warm, gentle, clear","bpm":99.384,"key":"B"},"language":"en","is_instrumental":false,"sections":[{"label":"verse","lyrics":"So little time and places to see\nand we can wait so patiently No they won't catch you and me","start_sec":0.0,"end_sec":23.739999771},{"label":"chorus","lyrics":"It's all over all out of time and if you","start_sec":23.739999771,"end_sec":30.0}]}
```

A prompt can provide a free-form description, structured tags, lyrics, and
section boundaries. `sample_id` is used to name the generated files.

## Generate audio

```bash
python scripts/infer.py \
  --model oqmtest1451/open-qwen-music-weights \
  --revision 64b37b84fe6a62c5a08d3366470ca9d4f28b5e67 \
  --prompts examples/prompts.jsonl \
  --output-dir outputs/demo \
  --device cuda
```

The default language-model sampling temperature is `0.7`. The first run uses
the Hugging Face cache for the Open-Qwen-Music model bundle and the pinned
`Qwen/Qwen3-Embedding-0.6B` renderer text encoder.

For offline inference, download both the model bundle and the pinned renderer text encoder first:

```bash
hf download oqmtest1451/open-qwen-music-weights \
  --revision 64b37b84fe6a62c5a08d3366470ca9d4f28b5e67 \
  --local-dir checkpoints/open-qwen-music
hf download Qwen/Qwen3-Embedding-0.6B \
  --revision b22da495047858cce924d27d76261e96be6febc0 \
  --local-dir checkpoints/qwen3-embedding-0.6b
HF_HUB_OFFLINE=1 python scripts/infer.py \
  --model checkpoints/open-qwen-music \
  --text-encoder checkpoints/qwen3-embedding-0.6b \
  --prompts examples/prompts.jsonl \
  --output-dir outputs/demo \
  --device cuda
```

The output directory contains one 48 kHz stereo WAV file and JSON sidecar per
prompt, the generated semantic representation, renderer requests, and a run
manifest.

Text-to-music generation requires a CUDA environment with enough memory for all
model components. CPU execution is useful for configuration and interface
checks but is not practical for full inference.
