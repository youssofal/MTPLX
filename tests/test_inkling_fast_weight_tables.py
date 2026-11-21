"""Lane L1's joined scale tables are pinned by both replay sites, not snapshot into programs.

On the 4-bit packs every recorded program held the 24 joined tables (1.2 MB a layer) as constants,
twice, so the 256 MB budget held four programs a site and the other widths were evicted and recorded
again on use. ``engine_site.weight_tables`` lists the kept tables; both sites pin it before every
recording."""

from __future__ import annotations

import weakref

import mlx.core as mx

from mtplx.models.inkling.fast import attn_input, engine_site


class _Layer:
    pass


def test_weight_tables_lists_every_kept_table_and_skips_dense_layers(monkeypatch):
    table = mx.arange(8, dtype=mx.bfloat16)
    a, b = _Layer(), _Layer()
    fake = {id(a): (weakref.ref(a), (table, [0, 4])), id(b): (weakref.ref(b), (None, [0]))}
    monkeypatch.setattr(attn_input, "_joined_sb", fake)
    got = engine_site.weight_tables()
    assert len(got) == 1 and got[0] is table


def test_both_sites_pin_the_tables_before_every_recording(monkeypatch):
    pinned = []

    class _Site:
        def pin_before_each_recording(self, source):
            pinned.append(source)

    made = []

    def fake_site_for(owner, attribute, name, **kwargs):
        site = _Site()
        made.append(name)
        return site

    from mtplx.engine import sites
    from mtplx.models.inkling.fast import round_program

    monkeypatch.setattr(sites, "site_for", fake_site_for)

    class _Rt:
        pass

    rt = _Rt()
    engine_site.site(rt)
    round_program.site(rt)
    assert engine_site.weight_tables in pinned
    assert pinned.count(engine_site.weight_tables) == 2  # the step site and the round site
