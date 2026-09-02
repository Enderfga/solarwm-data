"""The split rule — one definition, shared by every index that holds out data.

Membership is decided by a hash of the sample id ALONE, so it depends on neither file
order nor a seed anyone has to remember, and two indices built from the same corpus agree
without being built together. That last property is the reason this lives in the package
rather than in one script: a recipe and a window view that each rolled their own rule
would disagree about which clips are held out, and a clip held out of one while training
in the other is a leak nothing downstream can see.

Both consumers rank by this key and take the lowest N per owner. A window view ranks the
CLIP, never the window, so every window cut from one clip lands on the same side of the
split — windows of the same clip share almost all their frames, so splitting them across
train and test would leak the test set while every id-level overlap check stayed clean.
"""

from __future__ import annotations

import hashlib

__all__ = ["split_rank"]


def split_rank(sample_id: str) -> int:
    """A stable per-sample ordering key. Same id -> same rank, forever, everywhere."""
    return int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "big")
