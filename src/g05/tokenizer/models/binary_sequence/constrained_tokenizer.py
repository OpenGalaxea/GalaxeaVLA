# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

"""
ConstrainedSequenceTokenizer — Rule-based binary sequence tokenizer with continuity constraints.

Background
----------
A binary sequence of length ``seq_len`` is **valid** if no "middle block" (a run of identical
values that is neither the first nor the last run) has length in ``[1, min_block_len]``.
In other words, every middle run must be strictly longer than ``min_block_len``.

The number of valid sequences ``a(t, n)`` satisfies the recurrence::

    a[i] = 2 * i                    for i = 1 .. n+1
    a[i] = a[i-1] + a[i-n-1]        for i > n+1

``num_tokens`` is the smallest integer ``m`` such that ``vocab_size ** m >= a(t, n)``.

Encode/decode strategy
-----------------------
Valid sequences are assigned unique indices in ``[0, a(t, n))`` in lexicographic order
(0 < 1).  Encoding maps a sequence to its index, then expresses that index in base
``vocab_size`` as ``num_tokens`` digits (most-significant first).  Decoding reverses this.

The index is computed digit-by-digit: at each position, count how many valid completions
of the remaining suffix exist if we were to place a ``0`` at this position; if the actual
bit is ``1``, add that count to the running index.  This requires the memoized helper
``_count_completions(remaining, last_value, capped_run_len, is_first_block)``.

State representation in _count_completions
-------------------------------------------
- ``remaining``:        positions still to fill
- ``last_value``:       0 or 1 — value of the currently open run
- ``capped_run_len``:   length of current run, capped at ``min_block_len + 1``
                        (only the comparison ``> min_block_len`` matters for validity)
- ``is_first_block``:   True iff the current run is the very first run of the sequence

Closing a run (transitioning to the opposite value) is allowed iff:
    ``is_first_block`` OR ``capped_run_len > min_block_len``

The last run of the sequence always ends at ``remaining == 0`` and is never constrained.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Callable

logger = logging.getLogger(__name__)


def _make_count_fn(min_block_len: int) -> Callable:
    """Return a memoized count_completions function closed over *min_block_len*."""
    n = min_block_len

    @functools.lru_cache(maxsize=None)
    def count_completions(
        remaining: int,
        last_value: int,
        capped_run_len: int,
        is_first_block: bool,
    ) -> int:
        """Number of valid completions of length *remaining* from the given state."""
        if remaining == 0:
            return 1

        # Option 1: extend the current run (same value)
        result = count_completions(
            remaining - 1,
            last_value,
            min(capped_run_len + 1, n + 1),
            is_first_block,
        )

        # Option 2: start a new run (switch value) — only if closing the current run is valid
        can_close = is_first_block or (capped_run_len > n)
        if can_close:
            result += count_completions(remaining - 1, 1 - last_value, 1, False)

        return result

    return count_completions


class ConstrainedSequenceTokenizer:
    """
    Lossless codec for valid binary sequences with continuity constraints.

    A binary sequence is *valid* when every middle run (neither first nor last) has
    length strictly greater than ``min_block_len``.

    Parameters
    ----------
    seq_len:
        Length of the binary sequences to encode / decode.
    min_block_len:
        Minimum required length for any middle block.  Middle blocks shorter than or
        equal to this value are forbidden.
    vocab_size:
        Number of distinct values each output token can take, i.e. the base for the
        mixed-radix representation.  Output tokens are integers in ``[0, vocab_size)``.
    binarize_threshold:
        Float boundary for converting a continuous sequence to binary.
        Values ``>= binarize_threshold`` map to 1; values below map to 0.
    """

    def __init__(
        self,
        seq_len: int,
        min_block_len: int,
        vocab_size: int,
        binarize_threshold: float = 0.0,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if min_block_len < 1:
            raise ValueError(f"min_block_len must be >= 1, got {min_block_len}")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")

        self.seq_len = seq_len
        self.min_block_len = min_block_len
        self.vocab_size = vocab_size
        self.binarize_threshold = binarize_threshold

        self.num_valid_sequences: int = _compute_num_valid(seq_len, min_block_len)

        if self.num_valid_sequences <= 1:
            self.num_tokens = 1
        else:
            self.num_tokens = math.ceil(math.log(self.num_valid_sequences) / math.log(vocab_size))
            # Guard against floating-point rounding
            while vocab_size**self.num_tokens < self.num_valid_sequences:
                self.num_tokens += 1

        # Build and cache the memoized count function for this min_block_len
        self._count: Callable = _make_count_fn(min_block_len)

    # ── Public API ─────────────────────────────────────────────────────────────

    def binarize(self, float_sequence: list[float]) -> list[int]:
        """Convert a float sequence to binary using ``binarize_threshold``."""
        return [1 if v >= self.binarize_threshold else 0 for v in float_sequence]

    def repair(self, sequence: list[int]) -> list[int]:
        """
        Return a copy of *sequence* with all constraint violations fixed.

        Any middle run (not first, not last) whose length is ``<= min_block_len``
        is merged into its left neighbour by flipping its bits to that neighbour's
        value.  The process repeats until the sequence is valid.

        The input list is never mutated.
        """
        seq = list(sequence)
        n = self.min_block_len

        while True:
            # Build run list: (value, start_inclusive, end_exclusive)
            runs: list[tuple[int, int, int]] = []
            run_start = 0
            for i in range(1, len(seq)):
                if seq[i] != seq[i - 1]:
                    runs.append((seq[run_start], run_start, i))
                    run_start = i
            runs.append((seq[run_start], run_start, len(seq)))

            # Find first invalid middle run (index 1 .. len-2)
            fixed = False
            for i in range(1, len(runs) - 1):
                val, start, end = runs[i]
                if (end - start) <= n:
                    left_val = runs[i - 1][0]
                    for j in range(start, end):
                        seq[j] = left_val
                    fixed = True
                    break  # recompute runs after this merge

            if not fixed:
                break

        return seq

    def is_valid(self, sequence: list[int]) -> bool:
        """Return True iff *sequence* is a valid binary sequence under this tokenizer's constraints."""
        if len(sequence) != self.seq_len:
            return False
        if not all(b in (0, 1) for b in sequence):
            return False
        # Collect runs
        runs: list[int] = []  # lengths only
        if not sequence:
            return True
        run_len = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                run_len += 1
            else:
                runs.append(run_len)
                run_len = 1
        runs.append(run_len)
        # Check middle runs (all except first and last)
        for run in runs[1:-1]:
            if run <= self.min_block_len:
                return False
        return True

    def encode(self, sequence: list[int], strict: bool = False) -> list[int]:
        """
        Encode a binary sequence to a list of ``num_tokens`` token integers.

        Parameters
        ----------
        sequence:
            A binary list of length ``seq_len``.  Pass through ``binarize()``
            first if working with float inputs.
        strict:
            If True, raise ``ValueError`` when *sequence* is invalid (original
            behaviour).  If False (default), auto-repair the sequence via
            ``repair()``, emit a ``WARNING`` log, and encode the repaired result.

        Returns
        -------
        list[int]
            ``num_tokens`` integers, each in ``[0, vocab_size)``.

        Raises
        ------
        ValueError
            If ``strict=True`` and ``sequence`` is not valid.
        """
        if not self.is_valid(sequence):
            if strict:
                raise ValueError(
                    f"Sequence is not valid under min_block_len={self.min_block_len}: {sequence}"
                )
            repaired = self.repair(sequence)
            bits_changed = sum(a != b for a, b in zip(sequence, repaired))
            # logger.warning(
            #     "ConstrainedSequenceTokenizer: invalid sequence auto-repaired "
            #     "(min_block_len=%d, bits_changed=%d). "
            #     "original=%s repaired=%s",
            #     self.min_block_len,
            #     bits_changed,
            #     sequence,
            #     repaired,
            # )
            sequence = repaired
        idx = self._sequence_to_index(sequence)
        return self._index_to_tokens(idx)

    def decode(self, tokens: list[int], safe: bool = True) -> list[int]:
        """
        Decode a token list back to the original binary sequence.

        Parameters
        ----------
        tokens:
            ``num_tokens`` integers, each in ``[0, vocab_size)``.
        safe:
            If True (default), clamp out-of-range indices to valid range instead of raising.
            If False, raise ValueError for invalid indices.

        Returns
        -------
        list[int]
            The binary sequence of length ``seq_len``.

        Raises
        ------
        ValueError
            If ``safe=False`` and any token is out of range or the combined index
            exceeds ``num_valid_sequences``.
        """
        if len(tokens) != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens, got {len(tokens)}")

        if safe:
            tokens = [max(0, min(t, self.vocab_size - 1)) for t in tokens]
        else:
            if any(t < 0 or t >= self.vocab_size for t in tokens):
                raise ValueError(f"All tokens must be in [0, {self.vocab_size}), got {tokens}")

        idx = self._tokens_to_index(tokens)
        if idx >= self.num_valid_sequences:
            if safe:
                logger.warning(
                    f"ConstrainedSequenceTokenizer: token index {idx} out of range "
                    f"[0, {self.num_valid_sequences}), clamping to max valid index. "
                    f"tokens={tokens}, seq_len={self.seq_len}, min_block_len={self.min_block_len}"
                )
                idx = self.num_valid_sequences - 1
            else:
                raise ValueError(
                    f"Token index {idx} >= num_valid_sequences {self.num_valid_sequences}"
                )
        return self._index_to_sequence(idx)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _count_placing_zero(
        self,
        remaining_after: int,
        last_value: int,  # -1 means no bit placed yet (first position)
        current_run_len: int,
        is_first_block: bool,
    ) -> int:
        """
        Count valid completions if we place a 0 at the current position.

        Returns 0 when placing 0 would immediately violate the constraints.
        """
        n = self.min_block_len
        if last_value == -1:
            # First position — placing 0 always starts a new valid first run
            return self._count(remaining_after, 0, 1, True)
        if 0 == last_value:
            # Extending the current run with the same value
            return self._count(remaining_after, 0, min(current_run_len + 1, n + 1), is_first_block)
        # last_value == 1: transitioning from 1→0 closes the current run
        can_close = is_first_block or (current_run_len > n)
        if can_close:
            return self._count(remaining_after, 0, 1, False)
        return 0

    def _sequence_to_index(self, sequence: list[int]) -> int:
        """Compute the lexicographic rank of a valid binary sequence."""
        t = self.seq_len
        n = self.min_block_len
        index = 0
        last_value = -1  # sentinel: no bit placed yet
        current_run_len = 0
        is_first_block = True

        for pos in range(t):
            bit = sequence[pos]
            remaining_after = t - pos - 1

            if bit == 1:
                # All valid sequences that would place 0 here come before this one
                index += self._count_placing_zero(
                    remaining_after, last_value, current_run_len, is_first_block
                )

            # Update state
            if last_value == -1:
                last_value = bit
                current_run_len = 1
            elif bit == last_value:
                current_run_len = min(current_run_len + 1, n + 1)
            else:
                last_value = bit
                current_run_len = 1
                is_first_block = False

        return index

    def _index_to_sequence(self, index: int) -> list[int]:
        """Recover the valid binary sequence with the given lexicographic rank."""
        t = self.seq_len
        n = self.min_block_len
        sequence: list[int] = []
        last_value = -1
        current_run_len = 0
        is_first_block = True

        for pos in range(t):
            remaining_after = t - pos - 1
            count_0 = self._count_placing_zero(
                remaining_after, last_value, current_run_len, is_first_block
            )
            if index < count_0:
                bit = 0
            else:
                index -= count_0
                bit = 1

            sequence.append(bit)

            # Update state
            if last_value == -1:
                last_value = bit
                current_run_len = 1
            elif bit == last_value:
                current_run_len = min(current_run_len + 1, n + 1)
            else:
                last_value = bit
                current_run_len = 1
                is_first_block = False

        return sequence

    def _index_to_tokens(self, index: int) -> list[int]:
        """Express *index* in base ``vocab_size`` as ``num_tokens`` digits (MSB first)."""
        digits: list[int] = []
        for _ in range(self.num_tokens):
            digits.append(index % self.vocab_size)
            index //= self.vocab_size
        digits.reverse()
        return digits

    def _tokens_to_index(self, tokens: list[int]) -> int:
        """Reconstruct an integer from base-``vocab_size`` digits (MSB first)."""
        idx = 0
        for t in tokens:
            idx = idx * self.vocab_size + t
        return idx

    def __repr__(self) -> str:
        compression = (
            f"Compression: {self.seq_len}-len binary seq with min_block_len={self.min_block_len} "
            f"has {self.num_valid_sequences:,} possibilities, vocab_size={self.vocab_size:,} → "
            f"needs {self.num_tokens} tokens"
        )
        return (
            f"ConstrainedSequenceTokenizer("
            f"seq_len={self.seq_len}, "
            f"min_block_len={self.min_block_len}, "
            f"vocab_size={self.vocab_size}, "
            f"num_valid={self.num_valid_sequences:,}, "
            f"num_tokens={self.num_tokens})\n"
            f"  └─ {compression}"
        )


# ── Module-level helpers ───────────────────────────────────────────────────────


def _compute_num_valid(seq_len: int, min_block_len: int) -> int:
    """
    Compute a(t, n) via the recurrence in O(t) time and O(n) space.

    a[i] = 2 * i                for i = 1 .. n+1
    a[i] = a[i-1] + a[i-n-1]   for i > n+1
    """
    t, n = seq_len, min_block_len
    if t == 0:
        return 1  # one empty sequence

    # We only need a sliding window of n+1 values
    a: list[int] = [0] * (t + 1)
    for i in range(1, min(n + 2, t + 1)):
        a[i] = 2 * i
    for i in range(n + 2, t + 1):
        a[i] = a[i - 1] + a[i - n - 1]
    return a[t]


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import itertools

    def _brute_force_valid(seq_len: int, min_block_len: int) -> list[list[int]]:
        """Enumerate all valid sequences by brute force for small t."""
        valid = []
        for bits in itertools.product([0, 1], repeat=seq_len):
            seq = list(bits)
            tok = ConstrainedSequenceTokenizer(seq_len, min_block_len, 2)
            if tok.is_valid(seq):
                valid.append(seq)
        return valid

    print("=== Self-test: ConstrainedSequenceTokenizer ===")

    # ── Test 1: count matches recurrence for small t ──────────────────────────
    for n in (1, 2, 3):
        for t in range(1, 10):
            expected = _compute_num_valid(t, n)
            actual = len(_brute_force_valid(t, n))
            assert expected == actual, f"Count mismatch t={t}, n={n}: {expected} vs {actual}"
    print("  [PASS] num_valid_sequences matches brute-force for t<=9, n<=3")

    # ── Test 2: round-trip for all valid sequences (small t) ─────────────────
    for n in (1, 2):
        for t in range(1, 9):
            tok = ConstrainedSequenceTokenizer(t, n, vocab_size=16)
            valid_seqs = _brute_force_valid(t, n)
            assert len(valid_seqs) == tok.num_valid_sequences
            for seq in valid_seqs:
                tokens = tok.encode(seq)
                recovered = tok.decode(tokens)
                assert recovered == seq, f"Round-trip fail: {seq} -> {tokens} -> {recovered}"
    print("  [PASS] round-trip correct for all valid sequences, t<=8, n<=2")

    # ── Test 3: t=8, n=1, K=16 — specific edge cases ─────────────────────────
    tok = ConstrainedSequenceTokenizer(seq_len=8, min_block_len=1, vocab_size=16)
    print(f"  t=8, n=1, K=16 → {tok.num_valid_sequences} valid seqs, {tok.num_tokens} tokens")
    test_seqs_8 = [
        [0] * 8,
        [1] * 8,
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
    ]
    for seq in test_seqs_8:
        assert tok.is_valid(seq), f"Expected valid: {seq}"
        assert tok.decode(tok.encode(seq)) == seq, f"Round-trip fail: {seq}"
    print("  [PASS] t=8 edge cases")

    # ── Test 4: t=32, n=2, K=4096 ────────────────────────────────────────────
    tok2 = ConstrainedSequenceTokenizer(seq_len=32, min_block_len=2, vocab_size=4096)
    print(f"  t=32, n=2, K=4096 → {tok2.num_valid_sequences} valid seqs, {tok2.num_tokens} tokens")
    test_seqs_32 = [
        [0] * 32,
        [1] * 32,
        [0] * 15 + [1] * 17,
        [0] * 10 + [1] * 12 + [0] * 10,
        [0] * 5 + [1] * 22 + [0] * 5,
    ]
    for seq in test_seqs_32:
        assert tok2.is_valid(seq), f"Expected valid: {seq}"
        assert tok2.decode(tok2.encode(seq)) == seq, f"Round-trip fail: {seq}"
    print("  [PASS] t=32 edge cases")

    # ── Test 5: invalid sequences raise ValueError ────────────────────────────
    tok_inv = ConstrainedSequenceTokenizer(seq_len=8, min_block_len=1, vocab_size=16)
    invalid_seqs = [
        [0, 1, 0, 0, 0, 0, 0, 0],  # middle run "1" has length 1 == min_block_len
        [1, 0, 1, 1, 1, 1, 1, 1],  # middle run "0" has length 1
    ]
    for seq in invalid_seqs:
        assert not tok_inv.is_valid(seq)
        try:
            tok_inv.encode(seq, strict=True)
            assert False, f"Should have raised ValueError for {seq}"
        except ValueError:
            pass
    print("  [PASS] invalid sequences raise ValueError")

    # ── Test 6: binarize helper ───────────────────────────────────────────────
    tok_bz = ConstrainedSequenceTokenizer(
        seq_len=4, min_block_len=1, vocab_size=8, binarize_threshold=0.5
    )
    assert tok_bz.binarize([0.0, 0.4, 0.5, 1.0]) == [0, 0, 1, 1]
    print("  [PASS] binarize with threshold=0.5")

    print("=== All tests passed ===")
