import logging
import os

import torch.distributed as dist

from g05.utils.logging.overwatch import initialize_overwatch_logging


def setup_logging(
    log_level: int = logging.INFO,
    is_main_process: bool = True,
    preserve_hydra_handlers: bool = True,
) -> None:
    """
    Configure the logging system for the entire codebase with Rich formatting.

    **Single source of truth**: all Rich logging configuration is delegated to
    `g05.utils.logging.overwatch.initialize_overwatch_logging()`，
    This function only handles distributed gating (is_main_process) and preserving
    the hydra file handler.

    In distributed training, only the main process outputs logs while other processes are silenced.
    This function configures the root logger so all child loggers inherit the same configuration.

    Args:
        log_level: Logging level (default INFO), only applies to main process
        is_main_process: Whether this is the main process (default True)
        preserve_hydra_handlers: Keep existing FileHandlers from Hydra (default True)

    Example:
        ```python
        # In a single-machine script
        from g05.utils.logging.logging_config import setup_logging
        setup_logging()

        # In a distributed training script
        from accelerate import PartialState
        from g05.utils.logging.logging_config import setup_logging

        distributed_state = PartialState()
        setup_logging(
            log_level=logging.INFO,
            is_main_process=distributed_state.is_main_process,
        )
        ```
    """
    root_logger = logging.getLogger()

    if not is_main_process:
        # In non-main processes, set root logger level to ERROR to silence all logs
        root_logger.setLevel(logging.ERROR)
        return

    # Save existing FileHandlers (e.g., from Hydra) before dictConfig clears them
    existing_file_handlers = []
    if preserve_hydra_handlers:
        existing_file_handlers = [
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        ]

    # Delegate to overwatch (single source of truth for Rich format)
    # force=True so even if called twice (e.g. main() + sub-script), the config
    # always ends up applied. The cost of reapplying dictConfig is negligible.
    initialize_overwatch_logging(force=True)

    # Re-attach hydra file handlers (dictConfig wiped them via disable_existing_loggers=False
    # but cleared root handlers to only keep the console Rich handler)
    for handler in existing_file_handlers:
        root_logger.addHandler(handler)

    root_logger.setLevel(log_level)


def _is_main_process() -> bool:
    """
    Best-effort check for main process without any synchronization.
    """
    # Prefer torch.distributed state if initialized.
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0

    # Fallback to environment variables commonly set by launchers.
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ.get(key, "0") in ("0", "0\n", "")

    return True


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Drop-in replacement for accelerate.logging.get_logger:
    - No implicit barriers.
    - Only the main process emits log records by default.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # if not logger.handlers and _is_main_process():
    #     handler = logging.StreamHandler()
    #     formatter = logging.Formatter(
    #         fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #         datefmt="%Y-%m-%d %H:%M:%S",
    #     )
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)

    if not _is_main_process():
        logger.propagate = False
        logger.disabled = True

    return logger
