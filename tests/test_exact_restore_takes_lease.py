"""An exact follow-up takes its conversation's lease under the prefill layout (12 Nov).

Flash-Next prefills in a contiguous cache layout: ``contiguous_dense_decode`` up
to 131K tokens, ``contiguous_then_repage`` above. Under either, the exact
restore lane turned the server's ``reference_lease`` into ``clone`` unless the
coding-agent launch lane had exported MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE,
so every follow-up of a bare ``mtplx serve`` client (and of the app's non-agent
targets) was served a copy of its whole conversation beside the lease it was
copied from: +4.67 GB of footprint at 131K in window n4a4, which the admission
priced as free because the lease was the session's own. These tests run two
turns of one anonymous conversation on the tiny fixed-M4 lane under each
layout and pin: the follow-up continues in the lease's own cache, it decodes
the same tokens as the copy and as the copying store, and the switch at 0
still serves the copy.
"""

from __future__ import annotations

import pytest

from test_one_copy_snapshot_history import _snapshot_only_turns
from test_qwen4_fixed_m4_capacity_bucket import lane, pack  # noqa: F401

LAYOUTS = ["contiguous_dense_decode", "contiguous_then_repage"]


def _turns(pack, lane, *, layout: str, switch: str | None):
    lane.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", layout)
    lane.delenv("MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER", raising=False)
    if switch is None:
        lane.delenv("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", raising=False)
    else:
        lane.setenv("MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE", switch)
    return _snapshot_only_turns(pack, lane, one_copy=True, keep_live_ref=True)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_exact_follow_up_continues_in_the_answer_lease(pack, lane, layout):
    lease = _turns(pack, lane, layout=layout, switch=None)
    (_first_tokens, first_entry, first), (_tokens, _entry, second) = lease
    assert first_entry.live_ref_only
    assert second.stats.cached_tokens == len(first_entry.token_ids)
    assert second.stats.session_restore_mode == "reference_lease"
    # No copy: the follow-up decoded into the very cache the answer was banked in.
    assert second.final_state.final_trunk_cache is first.final_state.final_trunk_cache

    copy = _turns(pack, lane, layout=layout, switch="0")
    (_c_first_tokens, c_first_entry, c_first), (_c_tokens, _c_entry, c_second) = copy
    # The old path, still reachable as the rollback: a copy beside the lease.
    assert c_second.stats.cached_tokens == len(c_first_entry.token_ids)
    assert c_second.stats.session_restore_mode == "clone"
    assert c_second.final_state.final_trunk_cache is not c_first.final_state.final_trunk_cache

    # Same tokens from the lease, from the copy, and from the copying store.
    store = _snapshot_only_turns(pack, lane, one_copy=False)
    assert [t for t, _e, _r in lease] == [t for t, _e, _r in copy] == [t for t, _e, _r in store]
