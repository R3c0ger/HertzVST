from datetime import datetime
from pathlib import Path


BASE_OUTPUT_DIR = Path("./outputs")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
EXP_DIR: Path = None


def init_exp_dir(base_output_dir=BASE_OUTPUT_DIR, timestamp=TIMESTAMP) -> Path:
    """
    Initialize the experiment directory.
    This function shall be called only once at the start of the program.

    :param base_output_dir: Base directory for outputs.
    :param timestamp: Timestamp string to create a unique experiment directory.
    :return: Path to the experiment directory.
    """
    global EXP_DIR
    if EXP_DIR is not None:
        raise RuntimeError("Experiment directory already initialized!")

    experiment_dir = base_output_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    EXP_DIR = experiment_dir.resolve()  # Convert to absolute path
    print(f"Experiment directory initialized at: {EXP_DIR}")
    return EXP_DIR


def get_exp_dir() -> Path:
    """
    Get the experiment directory. Must be called after init_experiment_dir().

    :return: Path to the experiment directory.
    """
    if EXP_DIR is None:
        raise RuntimeError(
            "Experiment directory not initialized! "
            "Call init_experiment_dir() first."
        )
    return EXP_DIR


# Initialize the experiment directory at module load time
init_exp_dir()
