# Stable Audio 3 Notice

`src/open_qwen_music/render/sa3.py` and its supporting conditioning modules adapt the transformer architecture and forward computation used by Stable Audio 3:

- Project: https://github.com/Stability-AI/stable-audio-tools
- License: MIT

Only the DiffusionTransformer and ContinuousTransformer subset required by the released renderer is retained. The interface is adapted for 128-dimensional 25 Hz latents, 25 Hz semantic tokens, text conditioning, and global loudness. Stability AI model weights, autoencoders, CLI tools, and training framework are not included.

Parts of the upstream transformer implementation derive from x-transformers:

- Project: https://github.com/lucidrains/x-transformers
- License: MIT

License texts are available in `third_party/LICENSE-STABLE_AUDIO_3.txt` and `third_party/LICENSE-XTRANSFORMERS.txt`.
