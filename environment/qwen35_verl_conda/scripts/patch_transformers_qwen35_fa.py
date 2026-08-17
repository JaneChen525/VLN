#!/usr/bin/env python3
"""Apply the Transformers 5.3.0 Qwen3.5/FA2 3-D position-id hotfix.

Qwen3.5 uses 3-D M-RoPE position IDs. Transformers 5.3.0 passes those IDs
to ``_is_packed_sequence()``, which mistakes them for a packed sequence and
builds invalid FlashAttention ``cu_seqlens``. The resulting out-of-bounds
access is often reported later from FSDP/NCCL during backward.

This is an idempotent, fail-closed source patch for the exact pinned release.
It corresponds to huggingface/transformers issues #44643 and #44910.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path


EXPECTED_VERSION = "5.3.0"
EXPECTED_UNPATCHED_SHA256 = (
    "2c5b839ceae298a7d995f2b8ae88cc5f8151ee00a8ddccceed1be3bd1ef9bb72"
)
OLD = """    if position_ids is None:
        return False

    increasing_position_sequences = (
"""
NEW = """    if position_ids is None:
        return False
    # Qwen3.5 supplies 3-D M-RoPE IDs [3, batch, seq]. Packed-sequence
    # detection only accepts the 2-D [batch, seq] representation.
    if position_ids.dim() > 2:
        return False

    increasing_position_sequences = (
"""


def main() -> None:
    version = importlib.metadata.version("transformers")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"This hotfix is pinned to transformers {EXPECTED_VERSION}, got {version}"
        )

    import transformers.modeling_flash_attention_utils as fa_utils

    path = Path(fa_utils.__file__).resolve()
    source = path.read_text(encoding="utf-8")
    if NEW in source:
        print(f"QWEN35_FA2_PATCH_OK already_applied path={path}")
        return
    if OLD not in source:
        raise RuntimeError(f"Expected _is_packed_sequence source not found in {path}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_UNPATCHED_SHA256:
        raise RuntimeError(
            "Refusing to patch unexpected Transformers source: "
            f"expected_sha256={EXPECTED_UNPATCHED_SHA256} actual_sha256={digest}"
        )

    path.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"QWEN35_FA2_PATCH_APPLIED path={path}")


if __name__ == "__main__":
    main()
