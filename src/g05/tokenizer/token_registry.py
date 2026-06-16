# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase as _PreTrainedTokenizerBase

    TokenizerLike = _PreTrainedTokenizerBase
else:
    TokenizerLike = object

logger = logging.getLogger(__name__)


class TokenRegistry:
    """Sole owner of the HF tokenizer. All token registration goes through this class."""

    def __init__(
        self,
        hf_processor_path: str,
        hf_processor_class: str,
        add_loc_tokens: bool = False,
    ):
        from g05.utils.common.import_utils import get_obj_from_str

        proc_cls = get_obj_from_str(hf_processor_class)
        self.hf_processor_path = hf_processor_path
        # add_loc_tokens is currently supported only by Qwen35ProcessorWrapper.
        # Pass it through only when True to avoid giving HF AutoProcessor (PaliGemma)
        # an unknown kwarg.
        extra_kwargs = {"add_loc_tokens": True} if add_loc_tokens else {}
        self.hf_processor = proc_cls.from_pretrained(hf_processor_path, **extra_kwargs)
        self.tokenizer: TokenizerLike = self.hf_processor.tokenizer
        self._new_tokens: list[str] = []

    def register(self, tokens: list[str]) -> None:
        if not tokens:
            return
        added = self.tokenizer.add_tokens(tokens)
        self._new_tokens.extend(tokens)
        logger.info(
            f"[TokenRegistry] Registered {added} new tokens: {tokens[:5]}{'...' if len(tokens) > 5 else ''}"
        )

    def get_id(self, token_str: str) -> int | None:
        tid = self.tokenizer.convert_tokens_to_ids(token_str)
        unk_id = self.tokenizer.unk_token_id
        if tid is None or tid == unk_id:
            return None
        return tid

    def resize_model(
        self,
        model,
        base_vocab_size: int,
        pad_token_id: int | None,
        new_token_names: list[str] | None = None,
    ):
        if model is None:
            return
        target_vocab_size = len(self.tokenizer)
        try:
            current_vocab_size = model.vlm.input_proj.weight.shape[0]
            if current_vocab_size >= target_vocab_size:
                return
        except AttributeError:
            current_vocab_size = None
        model.resize_embedding(
            new_vocab_size=target_vocab_size,
            base_vocab_size=base_vocab_size,
            pad_token_id=pad_token_id,
            padded_vocab_size=None,
            new_token_names=new_token_names,
        )
        logger.info(
            f"[TokenRegistry] Resized model embedding: {current_vocab_size} → {target_vocab_size}"
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def new_tokens(self) -> list[str]:
        return list(self._new_tokens)
