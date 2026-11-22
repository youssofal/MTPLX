"""Inkling 2 Mini, using the publisher's mixed-precision checkpoint."""

import os


def default_draft_depth() -> int | None:
    """An omitted depth uses the family default unless per-round choice was selected."""
    from mtplx.backends.descriptors import INKLING_DESCRIPTOR

    if os.environ.get("MTPLX_INKLING_DEPTH", "").strip().lower() == "auto":
        return None
    return INKLING_DESCRIPTOR.draft_semantics.default
