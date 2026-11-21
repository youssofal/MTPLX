"""A replayed program runs no stage function, so its site counts again the stage paths its record
call took (the pack's own product, an unfused launch, the reference attention), not what the
pack's format would allow. Review 2, finding 11."""

from types import SimpleNamespace

from mtplx.models.inkling.fast import engine_site, routes


def test_a_capture_sees_the_counts_inside_it_and_they_still_count():
    before = routes.fast_route_counts().get("fallback.test_capture.x", 0)
    with routes.capture_counts() as outer:
        routes.count("fallback.test_capture.x")
        with routes.capture_counts() as inner:
            routes.count("unfused.test_capture.y", 2)
            routes.count("rows_proof.test_capture.z")
    assert outer == {"fallback.test_capture.x": 1, "unfused.test_capture.y": 2, "rows_proof.test_capture.z": 1}
    assert inner == {"unfused.test_capture.y": 2, "rows_proof.test_capture.z": 1}
    assert routes.stage_paths(outer) == {"fallback.test_capture.x": 1, "unfused.test_capture.y": 2}
    assert routes.fast_route_counts()["fallback.test_capture.x"] == before + 1


class _Ticket:
    def __init__(self, kind, outputs=None):
        self.kind, self.native, self.outputs, self.reason = kind, kind == "native", outputs, None

    def finish(self, outputs):
        pass


class _Replay:
    def __init__(self, kinds):
        self.kinds = list(kinds)
        self.last_decline = None

    def declined(self):
        return False

    def plan(self, geometry, family=None):
        return True

    def begin(self, geometry, inputs, **kw):
        return _Ticket(self.kinds.pop(0), outputs=["replayed"])

    def refuse(self, reason):
        pass


def test_a_native_replay_counts_the_stage_paths_its_record_call_took(monkeypatch):
    replay = _Replay(["record", "native", "native"])
    monkeypatch.setattr(engine_site, "site", lambda rt: replay)
    monkeypatch.setattr(engine_site, "_unfit", lambda spec: None)
    monkeypatch.setattr(engine_site.engine_runtime, "mode", lambda: "auto")
    monkeypatch.setattr(engine_site, "identity", lambda rt: "pack")
    monkeypatch.setattr(engine_site, "round_inputs", lambda state, ids: (state, ["inputs"]))
    monkeypatch.setattr(engine_site, "flatten", lambda *parts: list(parts))
    monkeypatch.setattr(engine_site, "unflatten", lambda outputs, n: ("logits", "captures", "deltas"))
    engine_site._PATHS.clear()

    def stages_ran(model, ids, state, stages=None):
        # What the stages counted while the program was recorded: a head whose first-use test
        # failed (the pack's own product), an unfused output launch, a one-off proof.
        routes.count("fallback.head.packed_rows")
        routes.count("unfused.bc_output.packed_4bit", 4)
        routes.count("rows_proof.head.one_at_a_time")
        return "logits", "captures", "deltas"

    monkeypatch.setattr(engine_site, "decode_panel", stages_ran)
    decoder = SimpleNamespace(model=SimpleNamespace(capture_ids=(3,)), rt=None)
    ids = SimpleNamespace(shape=(4,))
    engine_site.decode(decoder, ids, "state", None, "a,bc,d,e")  # records
    assert engine_site.last_route()["route"] == "recording"
    with routes.capture_counts() as replayed:
        engine_site.decode(decoder, ids, "state", None, "a,bc,d,e")  # replays: no stage runs
    assert engine_site.last_route()["route"] == "native"
    assert replayed == {"engine.native.w4": 1, "fallback.head.packed_rows": 1, "unfused.bc_output.packed_4bit": 4}
    engine_site.reset_route()
    assert engine_site.last_route()["route"] is None
