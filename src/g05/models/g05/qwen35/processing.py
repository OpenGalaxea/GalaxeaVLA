"""
Qwen3.5 Tokenizer — zero dependency on transformers.

Uses `tokenizers` library (HuggingFace tokenizers, a standalone Rust-backed
package) to load tokenizer.json directly. Provides the interface used by
G05Policy (encode, decode, batch_decode, add_tokens, convert_tokens_to_ids,
vocab_size, special token properties, __len__, __call__).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

from tokenizers import Tokenizer
from g05.utils.hf import resolve_hf_model_path


class TokenizerOutput(dict):
    """Dict subclass that allows attribute access (like HF BatchEncoding)."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'TokenizerOutput' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self[name] = value

    def to(self, device):
        """Move all tensor values to device (HF BatchEncoding compat)."""
        import torch

        return TokenizerOutput(
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.items()}
        )


class Qwen35Tokenizer:
    """Lightweight Qwen3.5 tokenizer wrapper around ``tokenizers.Tokenizer``.

    Parameters
    ----------
    pretrained_model_path : str
        Directory containing ``tokenizer.json`` and ``tokenizer_config.json``.
    """

    def __init__(
        self,
        pretrained_model_path: str,
        add_loc_tokens: bool = False,
        local_files_only: bool = False,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | None = None,
    ):
        pretrained_model_path = resolve_hf_model_path(
            pretrained_model_path,
            allow_patterns=["tokenizer.json", "tokenizer_config.json"],
            local_files_only=local_files_only,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
        tokenizer_json = os.path.join(pretrained_model_path, "tokenizer.json")
        config_json = os.path.join(pretrained_model_path, "tokenizer_config.json")

        self._tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_json)

        with open(config_json, "r") as f:
            self._config: Dict[str, Any] = json.load(f)

        # --- sync added tokens from tokenizer_config.json ---
        added_tokens_decoder = self._config.get("added_tokens_decoder", {})
        self._base_vocab_size: int = (
            min(int(k) for k in added_tokens_decoder)
            if added_tokens_decoder
            else self._tokenizer.get_vocab_size()
        )
        if added_tokens_decoder:
            from tokenizers import AddedToken

            missing = []
            for str_id, meta in sorted(added_tokens_decoder.items(), key=lambda x: int(x[0])):
                content = meta["content"]
                if self._tokenizer.token_to_id(content) is None:
                    missing.append(AddedToken(content, special=meta.get("special", False)))
            if missing:
                self._tokenizer.add_special_tokens(missing)

        # --- (optional) add 1024 bbox coordinate tokens <loc0000>~<loc1023> ---
        # Disabled by default. Enabling shifts all action token IDs by 1024; old and
        # new checkpoints are incompatible.
        if add_loc_tokens:
            from tokenizers import AddedToken

            loc_tokens = [AddedToken(f"<loc{i:04d}>", special=True) for i in range(1024)]
            self._tokenizer.add_special_tokens(loc_tokens)
        self._add_loc_tokens: bool = add_loc_tokens

        # --- special tokens ---
        self._bos_token: Optional[str] = self._config.get("bos_token", None)
        self._eos_token: str = self._config.get("eos_token", "<|endoftext|>")
        self._pad_token: str = self._config.get("pad_token", "<|endoftext|>")
        self._unk_token: Optional[str] = self._config.get("unk_token", None)

        self._bos_token_id: Optional[int] = (
            self._tokenizer.token_to_id(self._bos_token) if self._bos_token else None
        )
        self._eos_token_id: int = self._tokenizer.token_to_id(self._eos_token)
        self._pad_token_id: int = self._tokenizer.token_to_id(self._pad_token)
        self._unk_token_id: Optional[int] = (
            self._tokenizer.token_to_id(self._unk_token) if self._unk_token else None
        )

        # compatibility flags
        self.add_bos_token: bool = self._config.get("add_bos_token", False)
        self.add_eos_token: bool = False

        self._num_user_added_tokens: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Base BPE vocabulary size (excluding added/special tokens)."""
        return self._base_vocab_size

    @property
    def bos_token(self) -> Optional[str]:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._unk_token_id

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Full vocabulary size including added tokens."""
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(
        self, token_ids: Union[List[int], "torch.Tensor"], skip_special_tokens: bool = False
    ) -> str:
        token_ids = self._to_list(token_ids)
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids_list: Union[List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = False,
    ) -> List[str]:
        if hasattr(token_ids_list, "tolist"):
            token_ids_list = token_ids_list.tolist()
        return [
            self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids_list
        ]

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)

    def add_tokens(self, new_tokens: List[str]) -> int:
        from tokenizers import AddedToken

        added = self._tokenizer.add_tokens([AddedToken(t, special=False) for t in new_tokens])
        self._num_user_added_tokens += added
        return added

    def add_special_tokens(self, special_tokens_dict: Dict[str, str]) -> int:
        from tokenizers import AddedToken

        tokens_to_add = []
        for key, value in special_tokens_dict.items():
            if isinstance(value, list):
                for v in value:
                    tokens_to_add.append(AddedToken(v, special=True))
            else:
                tokens_to_add.append(AddedToken(value, special=True))
        added = self._tokenizer.add_special_tokens(tokens_to_add)
        self._num_user_added_tokens += added
        return added

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            input_ids = encoding.ids
            attention_mask = encoding.attention_mask
        else:
            encodings = self._tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)
            input_ids = [e.ids for e in encodings]
            attention_mask = [e.attention_mask for e in encodings]

        result = TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if return_tensors == "pt":
            import torch

            if isinstance(input_ids[0], list):
                max_len = max(len(ids) for ids in input_ids)
                padded_ids = [
                    ids + [self._pad_token_id] * (max_len - len(ids)) for ids in input_ids
                ]
                padded_mask = [mask + [0] * (max_len - len(mask)) for mask in attention_mask]
                result["input_ids"] = torch.tensor(padded_ids, dtype=torch.long)
                result["attention_mask"] = torch.tensor(padded_mask, dtype=torch.long)
            else:
                result["input_ids"] = torch.tensor([input_ids], dtype=torch.long)
                result["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list(token_ids) -> List[int]:
        if hasattr(token_ids, "tolist"):
            return token_ids.tolist()
        if isinstance(token_ids, (list, tuple)):
            return list(token_ids)
        return [token_ids]


class Qwen35ProcessorWrapper:
    """Thin wrapper to match PaliGemmaProcessor interface for G05Policy.

    G05Policy expects hf_processor_class.from_pretrained(path) → object with .tokenizer attribute.
    """

    def __init__(self, tokenizer: Qwen35Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        add_loc_tokens: bool = False,
        local_files_only: bool = False,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> "Qwen35ProcessorWrapper":
        tokenizer = Qwen35Tokenizer(
            pretrained_model_path,
            add_loc_tokens=add_loc_tokens,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
        return cls(tokenizer)
