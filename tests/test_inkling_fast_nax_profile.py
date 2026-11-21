"""The relative-position profile ``rel @ rel_basis`` as lane F1's launch (fast/nax_profile.py), GPU.

Every panel of 1-8 and 16 rows (24 heads of 16 relative values), with a bf16 and a float32 basis of
1,024 columns, must equal the library's product bit for bit; a float16 stream and the switch keep
the library's launch.
"""

import pytest

mx = pytest.importorskip("mlx.core")

needs = pytest.mark.skipif(not mx.metal.is_available(), reason="needs a Metal GPU")


def _bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(
        mx.array_equal(mx.contiguous(a).view(mx.uint16), mx.contiguous(b).view(mx.uint16)).item())


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

    monkeypatch.setattr(nax_profile, "_VERDICT", {})
    rel, basis = _operands(rows, basis_dtype, 3 * rows)
    want = rel @ basis.astype(rel.dtype)
    got = nax_profile.profile(rel, basis, lambda: rel @ basis.astype(rel.dtype))
    assert _bits(got, want)
    assert nax_profile._VERDICT[(rows * 24, 16, 1024, str(basis.dtype))] is True


@needs
def test_attend_profiles_take_the_launch(monkeypatch):
    from mtplx.models.inkling.fast import attend, nax_profile

    monkeypatch.setattr(nax_profile, "_VERDICT", {})
    rel, basis = _operands(4, "bfloat16", 1)
    assert _bits(attend.profiles(rel, basis), rel @ basis)
    assert nax_profile._VERDICT[(96, 16, 1024, str(mx.bfloat16))] is True


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
