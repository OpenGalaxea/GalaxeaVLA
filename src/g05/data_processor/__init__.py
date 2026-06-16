# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""Data processing layer for all transformations between dataset and batch.

Two subpackages:
- ``processor/``: per-embodiment shape meta parsing, normalization, and sample
  assembly entrypoints instantiated from the config ``processor:`` section.
- ``transforms/``: config ``_target_`` driven reversible data transforms,
  all inheriting from
  ``BaseActionStateTransform``。

Keep this separate from ``g05.models.g05.io.input_preprocessor``: that module
handles model-side text/image/action tokenization and is not part of the data layer.

This ``__init__`` only re-exports lightweight base classes. Concrete processor
and transform classes should be imported directly from their submodules, and
config ``_target_`` values should point to submodules, avoiding circular imports
with utils.
"""

from .transforms.base import BaseActionStateTransform

__all__ = ["BaseActionStateTransform"]
