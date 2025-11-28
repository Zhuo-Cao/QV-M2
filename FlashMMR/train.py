"""
FlashMMR training pipeline has been removed.

This repository now keeps inference-only code. Please run
`python FlashMMR/inference.py ...` for evaluation.
"""


def main() -> None:
    raise RuntimeError(
        "Training utilities have been removed from FlashMMR. Use FlashMMR/inference.py instead."
    )


if __name__ == "__main__":
    main()
