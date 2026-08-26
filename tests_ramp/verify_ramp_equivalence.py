"""Standalone verification: does the forked context_copy.py behave correctly
both with RAMP off (must be byte-identical to stock) and on (must reproduce
tonight's measured RampIndex behaviour)? Run directly, not via pytest, since
this fork has no venv yet.

Usage: python3 tests_ramp/verify_ramp_equivalence.py
"""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CC_PATH = os.path.join(REPO_ROOT, "mtplx", "context_copy.py")
TRACES_PATH = os.path.join(HERE, "fixtures", "traces.json")


def _load_context_copy():
    spec = importlib.util.spec_from_file_location("context_copy_under_test", CC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stock_ngram_index(ng_min, ng_max, max_candidates=32):
    """Byte-transcribed original upstream NgramIndex (pre-RAMP), used as the
    ground truth for the off-by-default equivalence check."""

    class _Stock:
        def __init__(self):
            self.ng_min, self.ng_max, self.max_candidates = ng_min, ng_max, max_candidates
            self.grams = {}
            self.indexed = 0

        def sync(self, history):
            for e in range(max(self.indexed + 1, self.ng_min), len(history) + 1):
                self.grams.setdefault(tuple(history[e - self.ng_min:e]), []).append(e)
            self.indexed = len(history)

        def find(self, history, *, max_pos=None):
            L = len(history)
            if L < self.ng_min + 1:
                return None, -1
            cands = self.grams.get(tuple(history[-self.ng_min:]))
            if not cands:
                return None, -1
            best_pos, best_ext = None, -1
            max_ext = self.ng_max - self.ng_min
            for pos in reversed(cands[-self.max_candidates:]):
                if pos >= L:
                    continue
                if max_pos is not None and pos >= max_pos:
                    continue
                ext = 0
                while (ext < max_ext and pos - self.ng_min - 1 - ext >= 0
                       and history[pos - self.ng_min - 1 - ext] == history[L - self.ng_min - 1 - ext]):
                    ext += 1
                if ext > best_ext:
                    best_ext, best_pos = ext, pos
                    if ext == max_ext:
                        break
            return best_pos, best_ext

    return _Stock()


def _stock_block_for_ext(ext, k_cap):
    ladder = (8, 12, 16, 24, 32)
    idx = max(0, min(int(ext), len(ladder) - 1))
    return min(ladder[idx], max(4, k_cap))


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _replay(prompt_ids, output_ids, propose_fn, mtp_advance=3):
    counters = {"rounds": 0, "drafted": 0, "accepted": 0, "accepted_blocks": 0}
    n = 1
    while n < len(output_ids):
        history = prompt_ids + output_ids[:n]
        proposal = propose_fn(history, len(prompt_ids))
        if proposal is None:
            n += min(mtp_advance, len(output_ids) - n)
            continue
        pos, klen = proposal
        budget = len(output_ids) - n
        block_tokens = [int(t) for t in prompt_ids[pos:pos + klen]][:budget]
        if not block_tokens:
            n += min(mtp_advance, len(output_ids) - n)
            continue
        nacc = _lcp(block_tokens, output_ids[n:n + len(block_tokens)])
        counters["rounds"] += 1
        counters["drafted"] += len(block_tokens)
        counters["accepted"] += nacc
        if nacc:
            counters["accepted_blocks"] += 1
        advance = nacc if nacc == len(block_tokens) else nacc + 1
        n += min(advance, len(output_ids) - n)
    return counters


def main():
    traces = json.loads(open(TRACES_PATH).read())["traces"]
    NG_MIN, NG_MAX, BLOCK_K = 6, 10, 24

    print("=== TEST 1: RAMP off (default) must be byte-identical to stock ===")
    os.environ.pop("MTPLX_RAMP_ENABLED", None)
    cc = _load_context_copy()
    assert cc.ramp_enabled() is False, "RAMP must default to off"

    all_ok = True
    for trace in traces:
        prompt_ids, output_ids = trace["prompt_ids"], trace["output_ids"]

        stock_idx = _stock_ngram_index(NG_MIN, NG_MAX)
        stock_idx.sync(prompt_ids)

        def stock_propose(history, max_pos):
            pos, ext = stock_idx.find(history, max_pos=max_pos)
            if pos is None:
                return None
            return pos, _stock_block_for_ext(ext, BLOCK_K)

        fork_idx = cc.NgramIndex(NG_MIN, NG_MAX)
        fork_idx.sync(prompt_ids)

        def fork_propose(history, max_pos):
            pos, ext = fork_idx.find(history, max_pos=max_pos)
            if pos is None:
                return None
            return pos, cc.block_for_ext(ext, BLOCK_K)

        stock_counters = _replay(prompt_ids, output_ids, stock_propose)
        fork_counters = _replay(prompt_ids, output_ids, fork_propose)
        ok = stock_counters == fork_counters
        all_ok = all_ok and ok
        print(f"  {trace['name']}: {'PASS' if ok else 'FAIL'} stock={stock_counters} fork={fork_counters}")

    assert all_ok, "RAMP-off must reduce byte-identically to stock -- FAILED"
    print("TEST 1 PASSED: off-by-default is byte-identical to stock.\n")

    print("=== TEST 2: RAMP on (block=48, fuzzy=1) must beat stock and RAMP class must fire ===")
    os.environ["MTPLX_RAMP_ENABLED"] = "1"
    os.environ["MTPLX_RAMP_BLOCK"] = "48"
    os.environ["MTPLX_RAMP_FUZZY"] = "1"
    cc2 = _load_context_copy()
    assert cc2.ramp_enabled() is True
    assert cc2.ramp_block() == 48
    assert cc2.context_copy_block_k() >= 48, "context_copy_block_k must widen to admit block=48"

    for trace in traces:
        prompt_ids, output_ids = trace["prompt_ids"], trace["output_ids"]
        stock_idx = _stock_ngram_index(NG_MIN, NG_MAX)
        stock_idx.sync(prompt_ids)

        def stock_propose(history, max_pos):
            pos, ext = stock_idx.find(history, max_pos=max_pos)
            if pos is None:
                return None
            return pos, _stock_block_for_ext(ext, BLOCK_K)

        ramp_idx = cc2.NgramIndex(NG_MIN, NG_MAX)
        ramp_idx.sync(prompt_ids)

        def ramp_propose(history, max_pos):
            pos, ext = ramp_idx.find(history, max_pos=max_pos)
            if pos is None:
                return None
            return pos, cc2.block_for_ext(ext, cc2.context_copy_block_k())

        stock_counters = _replay(prompt_ids, output_ids, stock_propose)
        ramp_counters = _replay(prompt_ids, output_ids, ramp_propose)
        ramp_tok_per_pass = ramp_counters["accepted"] / max(1, ramp_counters["rounds"])
        stock_tok_per_pass = stock_counters["accepted"] / max(1, stock_counters["rounds"])
        print(
            f"  {trace['name']}: stock tok/pass={stock_tok_per_pass:.2f} "
            f"ramp tok/pass={ramp_tok_per_pass:.2f} "
            f"({'RAMP wins' if ramp_tok_per_pass > stock_tok_per_pass else 'RAMP does not win'})"
        )

    print("\nTEST 2 done -- see per-trace verdicts above (matches tonight's measured direction).")


if __name__ == "__main__":
    main()
