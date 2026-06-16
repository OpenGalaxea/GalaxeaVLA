# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import torch
import torch.nn.functional as F
from einops import rearrange


def _apply_repetition_penalties(
    logits,
    prev_tokens,
    repetition_penalty=None,
    presence_penalty=None,
    frequency_penalty=None,
    repetition_window=None,
    ignore_token_ids=None,
):
    if prev_tokens is None or prev_tokens.numel() == 0:
        return logits

    if repetition_window is not None and repetition_window > 0:
        prev_tokens = prev_tokens[:, -repetition_window:]

    ignore_token_ids = set(ignore_token_ids or [])
    batch_size, vocab_size = logits.shape

    for b in range(batch_size):
        tokens = prev_tokens[b]
        if ignore_token_ids:
            mask = torch.ones_like(tokens, dtype=torch.bool)
            for ignore_id in ignore_token_ids:
                mask &= tokens != ignore_id
            tokens = tokens[mask]
        if tokens.numel() == 0:
            continue

        if presence_penalty or frequency_penalty:
            counts = torch.bincount(tokens, minlength=vocab_size).to(logits.dtype)
            if presence_penalty:
                logits[b] -= (counts > 0).to(logits.dtype) * presence_penalty
            if frequency_penalty:
                logits[b] -= counts * frequency_penalty

        if repetition_penalty and repetition_penalty != 1.0:
            unique_tokens = torch.unique(tokens)
            token_logits = logits[b, unique_tokens]
            token_logits = torch.where(
                token_logits < 0,
                token_logits * repetition_penalty,
                token_logits / repetition_penalty,
            )
            logits[b, unique_tokens] = token_logits

    return logits


def _get_banned_tokens(prev_tokens, no_repeat_ngram_size):
    if prev_tokens is None or no_repeat_ngram_size is None or no_repeat_ngram_size <= 0:
        return []

    batch_size, seq_len = prev_tokens.shape
    if seq_len < no_repeat_ngram_size:
        return [[] for _ in range(batch_size)]

    banned_tokens = []
    n = no_repeat_ngram_size
    for b in range(batch_size):
        tokens = prev_tokens[b].tolist()
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            prefix = ngram[:-1]
            ngrams.setdefault(prefix, set()).add(ngram[-1])
        prefix = tuple(tokens[-(n - 1) :]) if n > 1 else tuple()
        banned_tokens.append(list(ngrams.get(prefix, set())))
    return banned_tokens


def apply_sampling_constraints(
    logits,
    temperature=1.0,
    filter_value=-float("Inf"),
    prev_tokens=None,
    repetition_penalty=None,
    presence_penalty=None,
    frequency_penalty=None,
    no_repeat_ngram_size=None,
    repetition_window=None,
    ignore_token_ids=None,
):
    """
    logits: Tensor of shape [B, V]
    returns: sampled tokens [B, 1]
    """

    assert logits.dim() == 2

    logits = logits / (temperature + 1e-8)
    logits = _apply_repetition_penalties(
        logits,
        prev_tokens=prev_tokens,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_window=repetition_window,
        ignore_token_ids=ignore_token_ids,
    )

    if no_repeat_ngram_size and no_repeat_ngram_size > 0 and prev_tokens is not None:
        banned_tokens = _get_banned_tokens(prev_tokens, no_repeat_ngram_size)
        for b, banned in enumerate(banned_tokens):
            if banned:
                logits[b, banned] = filter_value
    return logits


def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    temperature=1.0,
    filter_value=-float("Inf"),
    num_samples=1,
    prev_tokens=None,
    repetition_penalty=None,
    presence_penalty=None,
    frequency_penalty=None,
    no_repeat_ngram_size=None,
    repetition_window=None,
    ignore_token_ids=None,
):
    logits = apply_sampling_constraints(
        logits,
        temperature=temperature,
        filter_value=filter_value,
        prev_tokens=prev_tokens,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_window=repetition_window,
        ignore_token_ids=ignore_token_ids,
    )

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(logits < kth_vals, filter_value)

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0

        # scatter mask back
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, filter_value)

    probs = F.softmax(logits, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    probs = torch.nan_to_num(probs, nan=0.0)

    return torch.multinomial(probs, num_samples=num_samples)
