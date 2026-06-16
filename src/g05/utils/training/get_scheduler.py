import math
from typing import Optional, Union

from diffusers.optimization import SchedulerType, Optimizer, TYPE_TO_SCHEDULER_FUNCTION
from torch.optim.lr_scheduler import LambdaLR


def _get_cosine_schedule_with_min_lr(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_min_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Cosine schedule with warmup that decays to lr * lr_min_ratio instead of 0."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = min(
            1.0,
            float(current_step - num_warmup_steps)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )
        cosine_value = max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
        return cosine_value * (1.0 - lr_min_ratio) + lr_min_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_warmup_constant_cosine_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    constant_end_ratio: float = 0.5,
    lr_min_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Warmup → constant → cosine decay schedule.

    The first ``constant_end_ratio * num_training_steps`` steps are split into a
    linear warmup phase (``num_warmup_steps``) followed by a constant-LR
    phase.  The remaining steps use cosine decay from 1.0 to *lr_min_ratio*.
    """
    constant_end = int(constant_end_ratio * num_training_steps)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < constant_end:
            return 1.0
        progress = min(
            1.0,
            float(current_step - constant_end) / float(max(1, num_training_steps - constant_end)),
        )
        cosine_value = max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
        return cosine_value * (1.0 - lr_min_ratio) + lr_min_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    lr_min_ratio: float = 0.0,
    constant_end_ratio: float = 0.5,
    **kwargs,
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        lr_min_ratio (`float`, *optional*, defaults to 0.0):
            Minimum learning rate as a ratio of the initial learning rate. Only used for cosine scheduler.
            E.g., 0.1 means the LR decays to 10% of the initial value instead of 0.
        constant_end_ratio (`float`, *optional*, defaults to 0.5):
            Fraction of total steps for warmup+constant phase. Only used for warmup_constant_cosine scheduler.
    """
    # Handle custom scheduler types not in diffusers' SchedulerType enum
    if isinstance(name, str) and name == "warmup_constant_cosine":
        if num_warmup_steps is None:
            raise ValueError(
                "warmup_constant_cosine requires `num_warmup_steps`, please provide that argument."
            )
        if num_training_steps is None:
            raise ValueError(
                "warmup_constant_cosine requires `num_training_steps`, please provide that argument."
            )
        return _get_warmup_constant_cosine_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            constant_end_ratio=constant_end_ratio,
            lr_min_ratio=lr_min_ratio,
        )

    name = SchedulerType(name)

    # Use custom cosine schedule when lr_min_ratio > 0
    if name == SchedulerType.COSINE and lr_min_ratio > 0.0:
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")
        return _get_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_min_ratio=lr_min_ratio,
            **kwargs,
        )

    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )
