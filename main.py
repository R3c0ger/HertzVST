import sys

import torch

from utils import logger, get_exp_dir


def main():
    logger.info(
        "\n------------------------------------------------------------"
        f"\nEnvironment Information:"
        f"\n- Python version: {sys.version}"
        f"\n- PyTorch version: {torch.__version__}"
        f"\n- CUDA available: {torch.cuda.is_available()}"
        f"\n- CUDA version: {torch.version.cuda}"
        f"\n- GPU device: {torch.cuda.get_device_name(0)}"
        f"\n- Experiment output directory: {get_exp_dir()}"
        "\n------------------------------------------------------------\n"
    )


if __name__ == "__main__":
    main()
