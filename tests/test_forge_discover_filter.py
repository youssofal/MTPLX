"""Discover-wall filter tests against real Hugging Face metadata fixtures.

The fixture rows below are captured verbatim from the live HF API on
2026-08-16 (``GET /api/models?search=MTPLX&sort=downloads&direction=-1``,
109 rows total): repo id + downloads at capture time. They pin the two
historic drop mechanisms the Discover wall shipped with:

1. Name filter: the dash-bounded, case-sensitive ``"-MTPLX-" in repo``
   check was a fossil of the retired ``<base>-MTPLX-<role>`` branding.
   Forge brands artifacts ``<base>-MTPLX`` (suffix form), so the old
   check dropped the app's own output plus every lowercase/prefix
   community variant — 36 of the 109 live MTPLX repos, including the
   #2-by-downloads community model and one of the project's own repos.

2. Windowing: rows were sliced to ``limit + offset`` at HF BEFORE the
   name filter ran, so every filtered row shrank the page and every
   repo ranked below the slice by downloads — exactly the fresh,
   low-download models — could never surface at any page size.
"""

from __future__ import annotations

from types import SimpleNamespace

from mtplx.commands import forge


# (repo_id, downloads) — captured live 2026-08-16, downloads-descending.
# Comments mark rows the old `-MTPLX-` filter dropped.
CAPTURED_LIVE_ROWS = [
    ("Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed", 12752),
    ("wang-yang/Ornith-1.0-35B-MTPLX", 9183),  # dropped: suffix form, rank #2 overall
    ("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed", 6989),
    ("OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX", 3793),  # dropped: suffix form
    ("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality", 3109),
    ("hawhyhb/Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16", 1494),  # the "Heretic" build
    ("Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed", 1441),
    ("samuelfaj/Ornstein3.6-27B-MTP-NSC-ACE-SABER-8bit-MTPLX-Optimized-Speed", 1432),
    ("SWiesmann/ThinkingCap-Qwen3.6-27B-6bit-FP16-MTPLX", 1073),  # dropped: suffix form
    ("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16", 841),
    ("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16", 747),
    ("AITRADER/Huihui-Qwen3.6-27B-abliterated-MTPLX", 672),  # dropped: suffix form
    ("ben0112/Qwen-Qwen3.8-27B-MTPLX", 593),  # dropped: suffix form
    ("Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16", 534),  # rank ~36: below the old 30-row window
    ("Shiftedx/ornith-1.0-35b-mxfp4-vision-mtplx", 513),  # dropped: lowercase
    ("wang-yang/Qwen3.6-27B-Q4-MTPLX", 469),  # dropped: suffix form
    ("nom666/Qwopus3.6-27B-Coder-MTPLX-8bit-Quality", 402),
    ("Youssofal/Qwen3.5-4B-Optimized-MTPLX", 177),  # dropped: the project's own repo
]

# Real repos with no MTPLX in the name (reachable via a user --query);
# the last row is a synthetic owner-only case: the brand must be in the
# model NAME, not just the namespace.
NON_MTPLX_ROWS = [
    "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
    "mtplx-lab/Base-Model",  # synthetic: owner-only brand does not qualify
]


def _row(repo: str, downloads: int) -> SimpleNamespace:
    return SimpleNamespace(modelId=repo, downloads=downloads, tags=[], siblings=[])


class _FakeApi:
    def __init__(self, rows, expected_search="MTPLX"):
        self.rows = rows
        self.expected_search = expected_search
        self.calls: list[dict] = []

    def list_models(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["search"] == self.expected_search
        # The scan budget must be the fixed bound, never the sliced
        # `limit + offset` window that hid low-download repos.
        assert kwargs["limit"] == forge._DISCOVER_SCAN_LIMIT
        return iter(self.rows)


def test_every_live_mtplx_repo_is_discoverable():
    for repo, _downloads in CAPTURED_LIVE_ROWS:
        assert forge.is_discoverable_repo(repo), repo


def test_repos_the_old_dash_bounded_filter_dropped_are_now_kept():
    dropped_by_old = [repo for repo, _ in CAPTURED_LIVE_ROWS if "-MTPLX-" not in repo]
    # The captured sample reproduces the bug: suffix, lowercase, and
    # double-suffix forms all failed the old check…
    assert dropped_by_old == [
        "wang-yang/Ornith-1.0-35B-MTPLX",
        "OpensourceWTF/Kimi-K3-Q2_K-t158-MTPLX",
        "SWiesmann/ThinkingCap-Qwen3.6-27B-6bit-FP16-MTPLX",
        "AITRADER/Huihui-Qwen3.6-27B-abliterated-MTPLX",
        "ben0112/Qwen-Qwen3.8-27B-MTPLX",
        "Shiftedx/ornith-1.0-35b-mxfp4-vision-mtplx",
        "wang-yang/Qwen3.6-27B-Q4-MTPLX",
        "Youssofal/Qwen3.5-4B-Optimized-MTPLX",
    ]
    # …and every one of them passes the brand-name contract now.
    for repo in dropped_by_old:
        assert forge.is_discoverable_repo(repo), repo


def test_non_mtplx_names_stay_out():
    for repo in NON_MTPLX_ROWS:
        assert not forge.is_discoverable_repo(repo), repo


def test_discover_fills_limit_after_filtering(monkeypatch):
    # 35 non-matching rows ranked ABOVE every matching one (a user
    # --query makes this shape real). The old slice-then-filter shape
    # fetched `limit` rows, filtered them all away, and returned [].
    noise = [_row(f"Qwen/Popular-Model-{i}", 100_000 - i) for i in range(35)]
    matching = [_row(repo, downloads) for repo, downloads in CAPTURED_LIVE_ROWS]
    api = _FakeApi(noise + matching, expected_search="qwen")
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)

    cards = forge.discover_models(query="qwen", limit=10, offset=0)

    assert [card["repo"] for card in cards] == [repo for repo, _ in CAPTURED_LIVE_ROWS[:10]]


def test_discover_default_query_and_full_result_set(monkeypatch):
    rows = [_row(repo, downloads) for repo, downloads in CAPTURED_LIVE_ROWS]
    api = _FakeApi(rows)
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)

    cards = forge.discover_models(query="MTPLX", limit=100, offset=0)

    # Every captured live repo surfaces, in downloads order, and the
    # cards carry the identity fields the app's DiscoveryEntry parses.
    assert [card["repo"] for card in cards] == [repo for repo, _ in CAPTURED_LIVE_ROWS]
    heretic = next(card for card in cards if "Heretic" in card["repo"])
    assert heretic["owner"] == "hawhyhb"
    assert heretic["branded_name"] == "Qwen3.6-35B-A3B-Uncensored-Heretic-MTPLX-4bit-FP16"
    assert heretic["downloads"] == 1494


def test_discover_offset_pages_over_matching_rows(monkeypatch):
    rows = [_row(f"owner/Padding-{i}", 9_999 - i) for i in range(3)]
    rows += [_row(repo, downloads) for repo, downloads in CAPTURED_LIVE_ROWS]
    api = _FakeApi(rows)
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)

    cards = forge.discover_models(query="MTPLX", limit=3, offset=2)

    # Offset counts MATCHING repos (the padding rows are not cards), so
    # page 2 starts at the third matching repo.
    assert [card["repo"] for card in cards] == [repo for repo, _ in CAPTURED_LIVE_ROWS[2:5]]



def test_cmd_discover_accepts_up_to_500_limit(monkeypatch, capsys):
    captured: list[tuple[str, int, int]] = []

    def _fake_discover(*, query: str, limit: int, offset: int = 0):
        captured.append((query, limit, offset))
        return []

    monkeypatch.setattr(forge, "discover_models", _fake_discover)
    rc = forge._cmd_discover(SimpleNamespace(query="MTPLX", limit=500, offset=0))
    assert rc == 0
    assert captured == [("MTPLX", 500, 0)]

    rc_capped = forge._cmd_discover(SimpleNamespace(query="MTPLX", limit=900, offset=0))
    assert rc_capped == 0
    assert captured[-1] == ("MTPLX", 500, 0)


def test_discover_multi_token_query_searches_mtplx_and_matches_all_tokens(monkeypatch):
    rows = [
        _row("Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed", 420),
        _row("Youssofal/MTPLX-Qwen3.8-27B-Optimized-Speed", 9000),
        _row("community/SomeOther-Inkling-Repo", 5000),  # lacks MTPLX in repo name
    ]
    api = _FakeApi(rows, expected_search="MTPLX")
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)

    assert [c["repo"] for c in forge.discover_models(query="youssofal inkling", limit=50)] == [
        "Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed"
    ]
    assert [c["repo"] for c in forge.discover_models(query="inkling2 mini", limit=50)] == [
        "Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed"
    ]


def test_discover_limit_500_reaches_official_repos_past_rank_100(monkeypatch):
    rows = [_row(f"community/Model-{i}-MTPLX", 10_000 - i) for i in range(150)]
    rows.append(_row("Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed", 120))
    api = _FakeApi(rows, expected_search="MTPLX")
    monkeypatch.setattr(forge, "_make_hf_api", lambda: api)

    cards = forge.discover_models(query="MTPLX", limit=500, offset=0)
    repos = [c["repo"] for c in cards]
    assert len(repos) == 151
    assert "Youssofal/MTPLX-Inkling-2-Mini-Bare-Speed" in repos


def test_modern_top_recommendation_ids_include_inkling2_mini():
    from mtplx import model_catalog

    for model_id in (
        "inkling2-mini-bare-speed",
        "inkling2-mini-optimized-speed",
        "inkling2-mini-optimized-quality",
    ):
        assert model_id in model_catalog._MODERN_TOP_RECOMMENDATION_IDS
