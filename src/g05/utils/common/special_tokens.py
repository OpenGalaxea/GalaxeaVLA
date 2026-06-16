"""VLM-agnostic special token registry.

Centralises pad / image / bos / eos token IDs so that the processor
does not hard-code VLM-specific lookups.  Every VLM backend
(PaliGemma, Qwen2-VL, etc.) can construct a ``SpecialTokenManager``
with the tokens it needs; the processor consumes the uniform API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Tuple


@dataclass
class SpecialTokenManager:
    """Registry of special tokens used by the VLM and processor.

    Two construction paths:
      - ``for_model(model_type, tokenizer, ...)`` — primary constructor;
        registers IDs, bos/eos/pad/image strings, and chat format tokens
        in one step.  Use this in new code.
      - ``from_ids(...)`` — legacy constructor; no model_type, no chat tokens.

    Attributes:
        pad_token_id: ID of the padding token (``None`` if unknown).
        image_token_index: ID of the image placeholder token.
        tokens: Mapping from logical key (e.g. ``"bos"``, ``"eos"``,
            ``"chat_user_prefix"``) to the actual token string.
        model_type: VLM backbone identifier (``"paligemma"`` | ``"qwen35"`` | ``""``).
    """

    pad_token_id: Optional[int] = None
    image_token_index: Optional[int] = None
    tokens: Dict[str, str] = field(default_factory=dict)
    model_type: str = ""

    # Keys whose values are injected into templates as <key> placeholders.
    _TEMPLATE_KEYS: ClassVar[Tuple[str, ...]] = (
        "bos",
        "eos",
        "chat_user_prefix",
        "chat_user_suffix",
        "chat_assistant_prefix",
    )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def for_model(
        cls,
        model_type: str,
        tokenizer=None,
        pad_token_id: Optional[int] = None,
        image_token_index: Optional[int] = None,
    ) -> "SpecialTokenManager":
        """Build a complete manager for a given VLM backbone.

        Registers bos/eos/pad/image from *tokenizer* and fills chat format
        tokens according to *model_type*.  The resulting ``tokens`` dict is
        also consumed by ``clean()`` to strip these strings from decoded text.

        Args:
            model_type: ``"paligemma"`` or ``"qwen35"``.
            tokenizer: optional HF tokenizer; used to auto-discover bos/eos/pad/image.
            pad_token_id: explicit pad token ID (takes precedence over tokenizer).
            image_token_index: explicit image token ID.
        """
        tokens: Dict[str, str] = {}

        # Auto-discover bos / eos / pad / image from tokenizer.
        if tokenizer is not None:
            for key, attr in [
                ("bos", "bos_token"),
                ("eos", "eos_token"),
                ("pad", "pad_token"),
            ]:
                val = getattr(tokenizer, attr, None)
                if val is not None:
                    tokens[key] = val
            if image_token_index is not None:
                img_tok = getattr(tokenizer, "image_token", None)
                if img_tok is None:
                    try:
                        img_tok = tokenizer.convert_ids_to_tokens(image_token_index)
                    except Exception:
                        img_tok = "<image>"
                tokens["image"] = img_tok

        # Fill chat format tokens (model-specific).
        if model_type == "paligemma":
            tokens.setdefault("bos", "<bos>")
            tokens.setdefault("eos", "<eos>")
            tokens["chat_user_prefix"] = ""
            tokens["chat_user_suffix"] = "\n"
            tokens["chat_assistant_prefix"] = ""
        elif model_type == "qwen35":
            bos = tokens.get("bos", "")
            eos = tokens.get("eos", "<|endoftext|>")
            tokens["bos"] = ""  # Qwen3.5 templates do not insert BOS
            tokens["eos"] = eos
            if bos:  # instruct model: bos_token is non-empty (e.g. "<|im_start|>")
                tokens["chat_user_prefix"] = f"{bos}user\n"
                tokens["chat_user_suffix"] = "<|im_end|>\n"
                tokens["chat_assistant_prefix"] = "<|im_start|>robot\n"
            else:  # base model
                tokens["chat_user_prefix"] = ""
                tokens["chat_user_suffix"] = ""
                tokens["chat_assistant_prefix"] = ""
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        return cls(
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
            tokens=tokens,
            model_type=model_type,
        )

    @classmethod
    def from_ids(
        cls,
        pad_token_id: Optional[int] = None,
        image_token_index: Optional[int] = None,
        tokenizer: Optional[object] = None,
    ) -> "SpecialTokenManager":
        """Build a manager from explicit IDs and an optional tokenizer.

        If *tokenizer* is provided, common special tokens (bos, eos, pad,
        image) are auto-discovered from its vocabulary.
        """
        tokens: Dict[str, str] = {}
        if tokenizer is not None:
            for key, attr in [
                ("bos", "bos_token"),
                ("eos", "eos_token"),
                ("pad", "pad_token"),
            ]:
                val = getattr(tokenizer, attr, None)
                if val is not None:
                    tokens[key] = val

            if image_token_index is not None:
                img_tok = getattr(tokenizer, "image_token", None)
                if img_tok is None:
                    try:
                        img_tok = tokenizer.convert_ids_to_tokens(image_token_index)
                    except Exception:
                        img_tok = "<image>"
                tokens["image"] = img_tok

        return cls(
            pad_token_id=pad_token_id,
            image_token_index=image_token_index,
            tokens=tokens,
        )

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(self, raw_key: str) -> Optional[str]:
        """Try to resolve *raw_key* to a special-token string.

        Looks up ``raw_key`` in the registered ``tokens`` dict first,
        then tries ``<raw_key>`` (angle-bracket wrapped).  Returns
        ``None`` if no match is found.
        """
        if raw_key in self.tokens:
            return self.tokens[raw_key]
        angled = f"<{raw_key}>"
        if angled in self.tokens.values():
            return angled
        return None

    # ------------------------------------------------------------------
    # Template resolution
    # ------------------------------------------------------------------

    def resolve_template(self, template: str) -> str:
        """Replace ``<key>`` placeholders in *template* with model-specific strings.

        Handles: ``<bos>``, ``<eos>``, ``<chat_user_prefix>``,
        ``<chat_user_suffix>``, ``<chat_assistant_prefix>``.
        Unknown placeholders are left unchanged.
        """
        for key in self._TEMPLATE_KEYS:
            placeholder = f"<{key}>"
            if placeholder in template:
                template = template.replace(placeholder, self.tokens.get(key, ""))
        return template

    # ------------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Strip all registered special tokens from *text*.

        Removes both the raw key forms (``<bos>``) and any string
        values stored in ``self.tokens``.
        """
        for _key, tok_str in self.tokens.items():
            if not tok_str:
                continue
            text = text.replace(tok_str, "")
        return text
