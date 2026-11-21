"""A follow-up served from an Inkling 2 Mini conversation's lease below the lease's end is billed as a copy.

21 November, Bare Speed under a simulated 24 GB budget through `mtplx serve` (m1-t24-b): such a turn grew 1.15 GB,
its conversation's size, before its first chunk, where the admission priced a lease taken in place; admitted with
18 MB to spare, it was stopped with a 507 after its stream had started. Billed as a copy, the admission frees the
lease when the turn would not fit beside it, and the turn restores from the SSD tier.
"""

from types import SimpleNamespace

from mtplx.server.openai import _lease_continuation_copies


def test_an_inkling_lease_continuation_is_billed_as_a_copy(monkeypatch):
    monkeypatch.delenv("MTPLX_ONE_COPY", raising=False)
    inkling = SimpleNamespace(one_copy_store=True)
    assert _lease_continuation_copies(inkling, "near_prefix")
    assert not _lease_continuation_copies(inkling, "exact")
    assert not _lease_continuation_copies(inkling, "none")
    # The fixed-M4 verifier's runtime and every other family keep the base's bill.
    assert not _lease_continuation_copies(SimpleNamespace(qwen4_fixed_m4_compiled_verify=True), "near_prefix")
    assert not _lease_continuation_copies(SimpleNamespace(), "near_prefix")
    assert not _lease_continuation_copies(None, "near_prefix")
    monkeypatch.setenv("MTPLX_ONE_COPY", "0")
    assert not _lease_continuation_copies(inkling, "near_prefix")
