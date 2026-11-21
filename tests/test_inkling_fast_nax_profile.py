"""The relative-position profile ``rel @ rel_basis`` as lane F1's launch (fast/nax_profile.py), GPU.

Every panel of 1-8 and 16 rows (24 heads of 16 relative values), with a bf16 and a float32 basis of
1,024 columns, must equal the library's product bit for bit; a float16 stream and the switch keep
the library's launch. On a GPU with matrix units (generation 17 and up) the launch must be proven
and taken; on one without (M1 to M4) it must decline before it is built, counted, and the library's
product gives the same bits.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


def _bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(mx.uint16), mx.contiguous(b).view(mx.uint16)).item())


def _route_taken(key, box):
    """The launch where this GPU has matrix units; elsewhere the counted decline, nothing decided."""
    from mtplx.models.inkling.fast import nax_profile
    from mtplx.models.inkling.fast.library_gemm import gpu_generation

    if gpu_generation() >= 17:
        assert nax_profile._VERDICT[key] is True
        assert not box.get("fallback.nax_profile.no_matrix_units")
    else:
        assert key not in nax_profile._VERDICT
        assert box.get("fallback.nax_profile.no_matrix_units", 0) >= 1, box


def _operands(rows, basis_dtype, seed):
    k1, k2 = mx.random.split(mx.random.key(seed))
    rel = (mx.random.normal((1, rows, 24, 16), key=k1) * 3).astype(mx.bfloat16)
    basis = (mx.random.normal((16, 1024), key=k2) * 0.3).astype(getattr(mx, basis_dtype))
    mx.eval(rel, basis)
    return rel, basis


@needs
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8, 16])
@pytest.mark.parametrize("basis_dtype", ["bfloat16", "float32"])
def test_profile_equals_the_library(rows, basis_dtype, monkeypatch):
    from mtplx.models.inkling.fast import nax_profile

    from mtplx.models.inkling.fast.routes import capture_counts

    monkeypatch.setattr(nax_profile, "_VERDICT", {})
    rel, basis = _operands(rows, basis_dtype, 3 * rows)
    want = rel @ basis.astype(rel.dtype)
    with capture_counts() as box:
        got = nax_profile.profile(rel, basis, lambda: rel @ basis.astype(rel.dtype))
    assert _bits(got, want)
    _route_taken((rows * 24, 16, 1024, str(basis.dtype)), box)


@needs
def test_attend_profiles_take_the_launch(monkeypatch):
    from mtplx.models.inkling.fast import attend, nax_profile

    from mtplx.models.inkling.fast.routes import capture_counts

    monkeypatch.setattr(nax_profile, "_VERDICT", {})
    rel, basis = _operands(4, "bfloat16", 1)
    with capture_counts() as box:
        got = attend.profiles(rel, basis)
    assert _bits(got, rel @ basis)
    _route_taken((96, 16, 1024, str(mx.bfloat16)), box)


@needs
def test_no_matrix_units_declines_before_building(monkeypatch):
    """An M1 to M4 GPU (rehearsed here by its generation): the counted decline, the library's bits,
    and no kernel built or decided."""
    from mtplx.models.inkling.fast import nax_profile
    from mtplx.models.inkling.fast.routes import capture_counts

    monkeypatch.setattr(nax_profile, "_VERDICT", {})
    monkeypatch.setattr(nax_profile, "gpu_generation", lambda: 16)
    monkeypatch.setattr(nax_profile, "_kernel", lambda: (_ for _ in ()).throw(AssertionError("built")))
    rel, basis = _operands(3, "bfloat16", 5)
    with capture_counts() as box:
        got = nax_profile.profile(rel, basis, lambda: rel @ basis.astype(rel.dtype))
    assert _bits(got, rel @ basis.astype(rel.dtype))
    assert box.get("fallback.nax_profile.no_matrix_units") == 1
    assert not nax_profile._VERDICT


@needs
def test_float16_and_switch_keep_the_library(monkeypatch):
    from mtplx.models.inkling.fast import nax_profile

    monkeypatch.setattr(nax_profile, "_VERDICT", {})
    rel, basis = _operands(1, "float16", 2)
    rel16 = rel.astype(mx.float16)
    assert _bits(nax_profile.profile(rel16, basis, lambda: rel16 @ basis), rel16 @ basis)
    monkeypatch.setenv("MTPLX_INKLING_NAX_PROFILE", "0")
    rel, basis = _operands(1, "bfloat16", 4)
    nax_profile.profile(rel, basis, lambda: rel @ basis)
    assert not nax_profile._VERDICT
