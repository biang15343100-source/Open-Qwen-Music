
SAMPLE_RATE = 24_000
FRAME_RATE = 25.0
FRAME_SAMPLES = 960
CODEBOOK_SIZE = 32_768
TOKEN_MIN = 0
TOKEN_MAX = CODEBOOK_SIZE - 1


def validate_contract(sample_rate: int, frame_rate: float, codebook_size: int) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Tokenizer input sampling rate must be {SAMPLE_RATE}; received {sample_rate}"
        )
    if frame_rate != FRAME_RATE:
        raise ValueError(
            f"Semantic token frame rate must be {FRAME_RATE}; received {frame_rate}"
        )
    if codebook_size != CODEBOOK_SIZE:
        raise ValueError(
            f"Semantic token codebook must be {CODEBOOK_SIZE}; received {codebook_size}"
        )
