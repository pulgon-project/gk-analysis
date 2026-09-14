import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from ase.build import bulk


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def bcc_fe_atoms():
    """A tiny, cubic bcc Fe cell -- known space group Im-3m (229)."""
    return bulk("Fe", "bcc", a=2.87, cubic=True)


@pytest.fixture
def poscar_path(tmp_path, bcc_fe_atoms):
    import ase.io

    path = tmp_path / "POSCAR"
    ase.io.write(path, bcc_fe_atoms, format="vasp")
    return str(path)


@pytest.fixture
def write_flux_file(tmp_path):
    """Factory fixture: write a synthetic LAMMPS-style flux file (header + data) to tmp_path."""

    def _write(data, name="flux.dat", header="# step temp jx jy jz"):
        path = tmp_path / name
        with open(path, "w") as fh:
            fh.write(header + "\n")
            np.savetxt(fh, data)
        return str(path)

    return _write


@pytest.fixture(autouse=True, scope="session")
def _warm_up_numba():
    """Pay the one-time numba JIT compilation cost for compute_cov_contrib once, up front."""
    from gk_analysis.uncertainty_tools import compute_cov_contrib

    compute_cov_contrib(np.zeros((2, 3)))


@pytest.fixture
def chdir_tmp_path(tmp_path, monkeypatch):
    """Some GreenKubo_run methods write plot files into the CWD as a side effect."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
