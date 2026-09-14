import builtins

import pytest

pytest.importorskip("spglib")
pytest.importorskip("pymatgen")

from ase.build import bulk

from gk_analysis.struct.symmetry import Symmetry


# ---------------------------------------------------------------------------
# get_space_group: known bug (see BUGS.md)
#
# Symmetry.get_space_group() passes atoms.positions (Cartesian) to
# spglib.get_spacegroup(), which expects fractional/scaled coordinates. For
# a non-identity cell this silently returns the WRONG space group. The
# tests below pin the actual (buggy) output for well-known structures --
# NOT the textbook space group, which is what a correct implementation
# would return (also asserted, via spglib directly, for contrast).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,name,correct_symbol,correct_number",
    [
        ("bcc", "Fe", "Im-3m", "229"),
        ("fcc", "Cu", "Fm-3m", "225"),
    ],
)
def test_get_space_group_is_wrong_for_cubic_cells(kind, name, correct_symbol, correct_number):
    import spglib

    atoms = bulk(name, kind, cubic=True)

    # what a correct call (fractional coordinates) actually returns:
    correct_cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    correct = spglib.get_spacegroup(correct_cell, symprec=1e-5)
    assert correct.split("(")[0].strip() == correct_symbol
    assert correct.split("(")[1].split(")")[0] == correct_number

    # what Symmetry actually returns, using Cartesian positions:
    sym = Symmetry(atoms)
    assert sym.space_group_symbol != correct_symbol
    assert sym.space_group_number != correct_number


def test_get_space_group_diamond_si_primitive_cell_is_wrong():
    import spglib

    atoms = bulk("Si", "diamond", a=5.43)

    correct_cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    correct = spglib.get_spacegroup(correct_cell, symprec=1e-5)
    assert correct.split("(")[0].strip() == "Fd-3m"

    sym = Symmetry(atoms)
    assert sym.space_group_symbol != "Fd-3m"


# ---------------------------------------------------------------------------
# pulgon_tools_wip: weak/optional dependency handling
# ---------------------------------------------------------------------------


def test_axial_and_cyclic_point_group_return_none_when_pulgon_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pulgon_tools_wip"):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    with caplog.at_level("WARNING"):
        sym = Symmetry(atoms)

    assert sym.axial_point_group is None
    assert sym.cyclic_point_group is None
    assert any("pulgon_tools_wip not installed" in msg for msg in caplog.messages)


def test_axial_and_cyclic_point_group_not_skipped_when_pulgon_available(caplog):
    # Symmetry imports these specific submodules, not just the top-level
    # package -- importorskip the exact names it needs so this test skips
    # cleanly on an environment with an incompatible pulgon_tools_wip
    # version (missing these submodules) instead of failing.
    pytest.importorskip("pulgon_tools_wip.detect_point_group")
    pytest.importorskip("pulgon_tools_wip.detect_generalized_translational_group")

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    with caplog.at_level("WARNING"):
        Symmetry(atoms)

    assert not any("pulgon_tools_wip not installed" in msg for msg in caplog.messages)
