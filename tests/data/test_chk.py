import pytest

np = pytest.importorskip('numpy')
chk_module = pytest.importorskip('aiida_supercon.data.chk')
ChkData = chk_module.ChkData


def test_get_u_without_disentangled():
    class Dummy:
        have_disentangled = False
        Uml = [np.eye(2, dtype=np.complex128)]
        n_kpts = 1
        n_wann = 2

    U = ChkData.get_U(Dummy())
    assert len(U) == 1
    assert np.array_equal(U[0], np.eye(2))
