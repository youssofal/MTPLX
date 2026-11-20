"""Groups, the rounding law and the bit allocation (pure arithmetic), and the
one-pass reference read on a toy model whose distributions are known."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.benchmarks.quality.recipe import GIB, Choice, allocate, choices_from_rows, recipe, stored_bytes, summary
from mtplx.benchmarks.quality.scoring import compare
from mtplx.benchmarks.quality.sensitivity import (
    Reader,
    _divergence,
    _reference_pass,
    default_groups,
    select_positions,
)


class _Toy:
    """A model whose next-token distribution depends on the current token only."""

    def __init__(self, vocab: int = 40, hidden: int = 8):
        mx.random.seed(3)
        self.table = mx.random.normal((vocab, hidden))
        self.out = mx.random.normal((hidden, vocab)) * 3
        self.language_model = self.model = self

    def __call__(self, inputs, cache=None):
        return self.table[inputs]

    def lm_head(self, hidden):
        return hidden @ self.out

    def make_cache(self):
        return None


def test_one_read_writes_server_rows_and_keeps_the_sample():
    toy = _Toy()
    token_ids = [(7 * i) % 40 for i in range(1, 30)]
    corpus = [{"id": "e", "token_ids": token_ids, "spans": [[5, 12], [20, 29]]}]
    chosen = select_positions(corpus, 6)
    rows: list[dict] = []
    text_of = lambda i: "same" if i in (3, 4) else str(i)  # noqa: E731 - two tokens sharing a text
    reference, scored = _reference_pass(Reader(toy, chunk=7), corpus, chosen, top_k=5, text_of=text_of,
                                        write_score=rows.append)

    assert scored == 16 and [p[0] for p in rows[0]["positions"]] == [*range(5, 12), *range(20, 29)]
    logits = toy.lm_head(toy.table[mx.array(token_ids)])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    for index, value, top in rows[0]["positions"]:
        row = logprobs[index - 1]
        assert value == pytest.approx(row[token_ids[index]].item(), abs=1e-4)
        assert top[text_of(token_ids[index])] == value  # the following token, always, with its own value
        best = mx.argsort(-row)[:5].tolist()
        assert {text_of(i) for i in best} <= set(top)
        if 3 in best and 4 in best and token_ids[index] not in (3, 4):
            assert top["same"] == pytest.approx(max(row[3].item(), row[4].item()), abs=1e-4)
    assert compare(rows, rows)[0].mean_kl == 0

    # The unchanged model read again: no divergence beyond the stored precision.
    floor = _divergence(Reader(toy, chunk=7), chosen, reference)
    assert floor["positions"] == 6 and floor["top1_agreement"] == 1.0 and abs(floor["kl_mean"]) < 1e-3


def test_positions_are_shared_across_entries_and_spread_over_each():
    corpus = [{"id": "long", "token_ids": list(range(1000)), "spans": [[100, 900]]},
              {"id": "short", "token_ids": list(range(300)), "spans": [[50, 250]]},
              {"id": "empty", "token_ids": [1, 2], "spans": []}]
    chosen = {entry_id: rows for entry_id, _, rows in select_positions(corpus, 100)}
    assert set(chosen) == {"long", "short"}
    assert len(chosen["long"]) == 80 and len(chosen["short"]) == 20  # in proportion to what each wrote
    assert chosen["long"][0] == 99 and chosen["long"][-1] > 880  # predictors of tokens 100 to 899
    assert chosen["long"] == sorted(set(chosen["long"]))
    capped = {entry_id: rows for entry_id, _, rows in select_positions(corpus, 100, context=200)}
    assert max(capped["long"]) < 200 and max(capped["short"]) < 200

P = "language_model.model.layers"


def test_default_groups_choose_modules_like_a_forge_recipe():
    groups = {g.name: g for g in default_groups(layers=40, experts=256, active=8, band=4)}
    first = groups["routed experts, layers 0-3"]
    assert first.routed and first.read_fraction == pytest.approx(8 / 256)
    assert first.member(f"{P}.3.mlp.switch_mlp.down_proj") and not first.member(f"{P}.4.mlp.switch_mlp.down_proj")
    assert groups["routed experts, layers 36-39"].layers == (36, 37, 38, 39)
    routers = groups["routers"]
    assert routers.member(f"{P}.7.mlp.gate") and routers.member(f"{P}.7.mlp.shared_expert_gate")
    assert not routers.member(f"{P}.7.mlp.switch_mlp.gate_proj")
    assert not groups["shared experts"].member(f"{P}.7.mlp.shared_expert_gate")
    assert groups["output head"].member("language_model.lm_head")
    assert groups["embeddings"].read_fraction == 0.0
    every = [f"{P}.0.linear_attn.in_proj_qkv", f"{P}.3.self_attn.o_proj", f"{P}.9.mlp.shared_expert.up_proj",
             "language_model.model.embed_tokens", f"{P}.12.mlp.switch_mlp.gate_proj"]
    for path in every:
        assert sum(g.member(path) for g in groups.values()) == 1, path
    assert first.overrides(8) == [{"suffix": s, "bits": 8, "group_size": 64, "layers": [0, 1, 2, 3]}
                                  for s in first.suffixes]


def _row(group, bits, kl, params, read, layers=None):
    return {"group": group, "bits": bits, "kl_mean": kl, "params": params, "params_read_per_token": read,
            "suffixes": [f"{group}.proj"], "layers": layers}


def test_the_rounding_law_extends_measured_widths():
    choice = choices_from_rows([_row("a", 4, 0.032, 10, 10)])[0]
    assert choice.kl(4) == pytest.approx(0.032)
    assert choice.kl(6) == pytest.approx(0.002)
    assert choice.kl(8) == pytest.approx(0.032 / 256)
    two = choices_from_rows([_row("b", 4, 0.032, 10, 10), _row("b", 6, 0.004, 10, 10)])[0]
    assert two.kl(6) == pytest.approx(0.004) and two.kl(8) == pytest.approx(0.004 / 16)  # from the nearest


def test_allocation_spends_bits_where_they_buy_the_most():
    rows = [
        _row("head", 4, 0.05, 500_000_000, 500_000_000),  # sensitive, read every token
        _row("experts", 4, 0.02, 30_000_000_000, 900_000_000, layers=[0, 1]),  # big, mostly unread
        _row("routers", 4, 0.01, 20_000_000, 20_000_000),  # tiny and sensitive
    ]
    choices = choices_from_rows(rows)
    narrow = summary(choices, {c.name: 4 for c in choices})
    allocate(choices, widths=[4, 6, 8], size_budget=narrow["size_gib"] * GIB + 1.5 * GIB,
             read_budget=narrow["read_gib_per_token"] * GIB + 0.5 * GIB)
    bits = {c.name: c.bits for c in choices}
    assert bits["routers"] == 8 and bits["head"] == 8 and bits["experts"] == 4
    built = recipe(choices, name="t", description="d", rounding="clip")
    # The body is the width most weights are at (the experts' 4 bits); only the other groups are overrides.
    assert built["body_bits"] == 4 and built["allocation"] == bits and built["rounding"] == "clip"
    assert built["module_overrides"] == [{"suffix": "head.proj", "bits": 8, "group_size": 64},
                                         {"suffix": "routers.proj", "bits": 8, "group_size": 64}]


def test_choices_come_from_the_rows_of_the_pack_rounding():
    rows = [_row("a", 4, 0.03, 10, 10), {**_row("a", 4, 0.02, 10, 10), "rounding": "clip"},
            {"group": "floor (no weight changed)", "kl_mean": 1e-5}]
    assert choices_from_rows(rows)[0].kl(4) == pytest.approx(0.03)
    assert choices_from_rows(rows, "clip")[0].kl(4) == pytest.approx(0.02)


def test_matching_a_uniform_pack_never_spends_more_and_predicts_no_worse(tmp_path, capsys):
    import json

    from mtplx.benchmarks.quality import recipe as recipe_cli

    rows = [_row("head", 4, 0.05, 500_000_000, 500_000_000), _row("routers", 4, 0.01, 20_000_000, 20_000_000),
            _row("experts a", 4, 0.004, 3_000_000_000, 90_000_000, layers=[0]),
            _row("experts b", 4, 0.04, 3_000_000_000, 90_000_000, layers=[1])]
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "recipe.json"
    assert recipe_cli.main([str(source), "--match-uniform", "6", "--rounding", "plain", "--name", "t",
                            "--out", str(out)]) == 0
    built = json.loads(out.read_text())
    assert built["rounding"] == "plain"
    uniform = summary(choices_from_rows(rows), {r["group"]: 6 for r in rows})
    assert built["predicted"]["size_gib"] <= uniform["size_gib"] + 1e-9
    assert built["predicted"]["read_gib_per_token"] <= uniform["read_gib_per_token"] + 1e-9
    assert built["predicted"]["predicted_kl"] <= uniform["predicted_kl"]
    assert "| uniform 6-bit, plain rounding |" in capsys.readouterr().out
    # More bytes per token allowed: never a worse prediction, never past the larger read budget.
    roomier = tmp_path / "roomier.json"
    assert recipe_cli.main([str(source), "--match-uniform", "6", "--read-scale", "1.2", "--rounding", "plain",
                            "--name", "t", "--out", str(roomier)]) == 0
    more = json.loads(roomier.read_text())["predicted"]
    assert more["predicted_kl"] <= built["predicted"]["predicted_kl"]
    assert more["read_gib_per_token"] <= 1.2 * uniform["read_gib_per_token"] + 1e-9
    # A larger pack as well: never a worse prediction than either, never past either larger budget.
    larger = tmp_path / "larger.json"
    assert recipe_cli.main([str(source), "--match-uniform", "6", "--size-scale", "1.2", "--read-scale", "1.2",
                            "--rounding", "plain", "--name", "t", "--out", str(larger)]) == 0
    most = json.loads(larger.read_text())["predicted"]
    assert most["predicted_kl"] <= more["predicted_kl"]
    assert most["size_gib"] <= 1.2 * uniform["size_gib"] + 1e-9
    assert most["read_gib_per_token"] <= 1.2 * uniform["read_gib_per_token"] + 1e-9


def test_a_matched_pack_can_keep_a_shipped_packs_exceptions(tmp_path, capsys):
    import json

    from mtplx.benchmarks.quality import recipe as recipe_cli

    rows = [_row("head", 4, 0.05, 500_000_000, 500_000_000), _row("routers", 4, 0.01, 20_000_000, 20_000_000),
            _row("experts", 4, 0.02, 3_000_000_000, 90_000_000, layers=[0])]
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "recipe.json"
    assert recipe_cli.main([str(source), "--match-uniform", "6", "--except", "routers=8", "--rounding", "plain",
                            "--name", "t", "--out", str(out)]) == 0
    shipped = summary(choices_from_rows(rows), {"head": 6, "routers": 8, "experts": 6})
    predicted = json.loads(out.read_text())["predicted"]
    assert predicted["size_gib"] <= shipped["size_gib"] + 1e-9
    assert predicted["read_gib_per_token"] <= shipped["read_gib_per_token"] + 1e-9
    assert predicted["predicted_kl"] <= shipped["predicted_kl"]
    assert "| the matched pack: 6-bit, routers=8, plain rounding |" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        recipe_cli.main([str(source), "--match-uniform", "6", "--except", "router=8", "--rounding", "plain",
                         "--name", "t", "--out", str(out)])


def test_budgets_that_cannot_hold_the_narrowest_width_are_refused():
    choices = [Choice("a", 1000, 1000, {4: 0.1}, ["a"], None)]
    with pytest.raises(ValueError):
        allocate(choices, widths=[4, 8], size_budget=stored_bytes(1000, 4) - 1, read_budget=1e9)
