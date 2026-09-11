<div align="center">

# Open-Qwen-Music

**An independent research reproduction of the Qwen-Music method for text-to-music generation**

[📄 Paper](open_qwen_music.pdf) · [🎧 Demo](https://huggingface.co/spaces/oqmtest1451/open-qwen-music-demo) · [🤗 Model](https://huggingface.co/oqmtest1451/open-qwen-music-weights) · [📚 Dataset](https://huggingface.co/datasets/david-miller-45678/open-qwen-music-dataset-c-minimax-music3) · [⚡ Inference](INFERENCE.md) · [🛠️ Training](TRAINING.md)

</div>

> Open-Qwen-Music is an independent research reproduction of the methods
> described in the [Qwen-Music Technical Report](https://arxiv.org/abs/2607.11699).
> It is not an official Qwen release and is not affiliated with, endorsed by,
> or maintained by the Qwen team or Alibaba Cloud.
>
> The code, checkpoints, dataset, and demo artifacts published by this project
> were prepared independently and are not official Qwen-Music artifacts or
> evaluation results.

> **Project status:** Open-Qwen-Music is a work in progress. The current release
> provides complete training and inference pipelines, while model quality remains
> under active development.

This project implements a complete pipeline for generating 48 kHz stereo music
from natural-language descriptions, lyrics, and structured sections. The
repository provides model implementations, data preparation tools, training
entry points, a distributed launcher, and end-to-end inference.

## Highlights

- Generate music from text, lyrics, tags, and section-level structure.
- Train every model component with configuration-driven entry points.
- Run the published 64-GPU recipes with standard multi-node `torchrun`.
- Run the released model bundle through a single inference command.

## Quick Start

Open-Qwen-Music requires Python 3.10 or later. A CUDA-capable GPU is required for practical generation.

Install the inference dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[render]"
```

Generate audio from the included prompt file, which uses the same conditioning
prompt as `t2m-037` in the [online demo](https://huggingface.co/spaces/oqmtest1451/open-qwen-music-demo):

```bash
python scripts/infer.py \
  --model oqmtest1451/open-qwen-music-weights \
  --revision 64b37b84fe6a62c5a08d3366470ca9d4f28b5e67 \
  --prompts examples/prompts.jsonl \
  --output-dir outputs/demo \
  --device cuda
```

The first run downloads the released model bundle from Hugging Face. Each prompt produces a WAV file and a JSON metadata sidecar. See the [inference guide](INFERENCE.md) for the prompt schema, offline model download, and runtime settings.

Prefer to try the model without a local setup? Open the [online demo](https://huggingface.co/spaces/oqmtest1451/open-qwen-music-demo).

## Model Pipeline

```mermaid
flowchart LR
    A[Text, lyrics, and structure] --> B[Music language model]
    B --> C[Semantic tokens]
    C --> D[Acoustic renderer]
    D --> E[Acoustic VAE decoder]
    E --> F[Bandwidth refiner]
    F --> G[48 kHz stereo audio]
```

| Component | Purpose | Training guide |
| --- | --- | --- |
| Semantic tokenizer | Learns compact semantic music tokens | [Six-stage tokenizer workflow](docs/training/tokenizer.md) |
| Music language model | Predicts semantic tokens from text, lyrics, and structure | [Two-stage language-model workflow](docs/training/language-model.md) |
| Acoustic VAE | Compresses and reconstructs acoustic representations | [Two-stage VAE workflow](docs/training/acoustic-vae.md) |
| Acoustic renderer | Generates acoustic latents from semantic conditions | [Renderer workflow](docs/training/renderer.md) |
| Bandwidth refiner | Restores high-frequency detail in the decoded waveform | [Refiner workflow](docs/training/refiner.md) |

## Training

Install the training and data preparation dependencies:

```bash
pip install -e ".[train,render]"
pip install -e "./data_pipeline[gpu]"
```

Training is configuration-driven through `configs/train/`. Every published
recipe uses 64 GPU processes. The tokenizer and
Acoustic VAE can begin after the base corpus is ready. Language-model training
waits for the frozen tokenizer and materialized token corpus; renderer training
waits for both tokenizer labels and frozen VAE latents; refiner training waits for
the frozen VAE.

For Renderer training, adapt the semantic-token release, run
`oqm-materialize-renderer`, and pass the generated `renderer.resolved.yaml` to
the standard `scripts/launch_torchrun.sh` wrapper. The checked-in Renderer recipe
contains one data root and the final training schedule; release paths and record
counts are filled in by the materializer.

Start with the [training overview](TRAINING.md), then follow the module guide for
its data contract, stages, checkpoint handoffs, update budget, and distributed
launch command. The [data-preparation guide](docs/training/data-preparation.md)
describes the base release and each module-specific training view.

## Relationship to Qwen-Music

This implementation is inspired by the architecture and training methodology
described in the [Qwen-Music Technical Report](https://arxiv.org/abs/2607.11699)
by Xu et al. The paper is the reference for the method; its experiments and
reported results belong to the original authors.

Open-Qwen-Music uses independently prepared data, training infrastructure,
checkpoints, and demonstrations. Differences in data, scale, optimization, and
implementation may lead to results that differ from the original report. The
repository does not contain official Qwen-Music checkpoints.

## Released Resources

| Resource | Description |
| --- | --- |
| [Online demo](https://huggingface.co/spaces/oqmtest1451/open-qwen-music-demo) | Browser-based listening gallery and text-to-music examples |
| [Model weights](https://huggingface.co/oqmtest1451/open-qwen-music-weights) | Language model, semantic tokenizer, acoustic renderer, VAE, refiner, and runtime configuration |
| [Training dataset](https://huggingface.co/datasets/david-miller-45678/open-qwen-music-dataset-c-minimax-music3) | 62,417 WebDataset records with structured annotations, provenance, checksums, and per-record rights metadata |

The dataset is released for non-commercial academic research and includes source-specific terms. Review its license, notice, source policy, and per-record rights metadata before use.

## Repository Layout

```text
configs/         Training and inference configurations
src/             Tokenizer, language model, and acoustic model implementations
data_pipeline/   Corpus preprocessing and model-based annotation
scripts/         Training, inference, and distributed launchers
examples/        Ready-to-run prompt examples
tests/           Contract and implementation tests
third_party/     Dependency notices and version locks
```

## Documentation

- [Inference](INFERENCE.md): prompt format, model download, generation, and outputs
- [Training](TRAINING.md): dependency graph, module index, and distributed training
- [Module workflows](docs/training/README.md): stage-by-stage data and checkpoint handoffs
- [Data pipeline](data_pipeline/README.md): preprocessing and model-based annotation

## Future Plans

- [ ] Develop an improved music semantic tokenizer.
- [ ] Extend the Music LLM to long-form text-to-music generation and introduce Melody-CoT training.
- [ ] Introduce post-training to improve musicality, lyric alignment, and controllability.
- [ ] Build a comprehensive evaluation suite for model components and end-to-end generation.

## Development

Install the development dependencies and run the CPU-compatible checks:

```bash
pip install -e ".[dev]"
python -m ruff check src scripts data_pipeline/src tests
python -m compileall -q src scripts data_pipeline/src
python -m pytest -q tests/test_train_router.py tests/test_torchrun_launcher.py
```

## Contributing

Issues and pull requests are welcome. Keep changes focused, add tests for behavior changes, and update the corresponding configuration or guide when an interface changes.

## Citation

If you use Open-Qwen-Music in academic work, please cite the project paper:

```bibtex
@misc{yu2026openqwenmusic,
  title        = {Open-Qwen-Music: An Auditable Framework for LLM-Based Music Composition and Diffusion Rendering},
  author       = {Yangbin Yu and Mingyu Yang},
  year         = {2026},
  howpublished = {Manuscript},
  url          = {https://github.com/biang15343100-source/Open-Qwen-Music/blob/main/open_qwen_music.pdf}
}
```

Open-Qwen-Music follows the architecture and training methodology introduced in
Qwen-Music. Please also cite the original method paper when discussing that
method:

```bibtex
@misc{xu2026qwenmusic,
  title         = {Qwen-Music Technical Report},
  author        = {Jin Xu and Kangdi Wang and Ruibin Yuan and Shun Lei and
                   Xiong Wang and Xize Cheng and Xueyao Zhang and Yang Zhang and
                   Yiheng Chen and Yongqi Wang and Yue Wang and Zhifang Guo and
                   Zihan Liu and Zijian Lin and Dake Guo and Hangrui Hu and
                   Lei Xie and Linhan Ma and Wei Xue and Wenxiang Guo and
                   Xinfa Zhu and Xipin Wei and Yangze Li and Yuanjun Lv and
                   Yuxuan Wang and Yunfei Chu and Zhiyong Wu},
  year          = {2026},
  eprint        = {2607.11699},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.48550/arXiv.2607.11699},
  url           = {https://arxiv.org/abs/2607.11699}
}
```

## License

The source code is available under the [Apache License 2.0](LICENSE). Model
weights and datasets are governed by the terms on their respective Hugging Face
repository pages.

The released language model is derived from `Qwen/Qwen2.5-Omni-3B` and remains
subject to the Qwen Research License and its attribution requirements. Other
model components and adapted third-party code retain the terms and notices
listed with their releases. The use of the name “Qwen” identifies the reference
method and upstream model lineage; it does not indicate affiliation or endorsement.

## Acknowledgements

Open-Qwen-Music builds on PyTorch, Qwen, EAR_VAE2, Stable Audio Tools,
x-transformers, and the open-source audio research community. Third-party notices
are available in `third_party/`.
