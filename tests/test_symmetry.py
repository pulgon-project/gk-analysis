import builtins

import pytest

pytest.importorskip("spglib")
pytest.importorskip("pymatgen")

from ase.build import bulk

from gk_analysis.struct.symmetry import Symmetry


# ---------------------------------------------------------------------------
# get_space_group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,name,expected_symbol,expected_number",
    [
        ("bcc", "Fe", "Im-3m", "229"),
        ("fcc", "Cu", "Fm-3m", "225"),
    ],
)
def test_get_space_group_matches_textbook_value_for_cubic_cells(
    kind, name, expected_symbol, expected_number
):
    atoms = bulk(name, kind, cubic=True)

    sym = Symmetry(atoms)
    # get_space_group() doesn't strip the raw spglib symbol, which carries a
    # trailing space before "(<number>)" (e.g. "Im-3m ") -- not part of this fix.
    assert sym.space_group_symbol.strip() == expected_symbol
    assert sym.space_group_number == expected_number


def test_get_space_group_diamond_si_primitive_cell():
    atoms = bulk("Si", "diamond", a=5.43)

    sym = Symmetry(atoms)
    assert sym.space_group_symbol.strip() == "Fd-3m"
    assert sym.space_group_number == "227"


# ---------------------------------------------------------------------------
# pulgon_tools: weak/optional dependency handling
# ---------------------------------------------------------------------------


def test_axial_and_cyclic_point_group_return_none_when_pulgon_missing(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pulgon_tools"):
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    with caplog.at_level("WARNING"):
        sym = Symmetry(atoms)

    assert sym.axial_point_group is None
    assert sym.cyclic_point_group is None
    assert any("pulgon_tools not installed" in msg for msg in caplog.messages)


def test_axial_and_cyclic_point_group_not_skipped_when_pulgon_available(caplog):
    # Symmetry imports these specific submodules, not just the top-level
    # package -- importorskip the exact names it needs so this test skips
    # cleanly on an environment without a compatible pulgon_tools install
    # (missing these submodules) instead of failing.
    pytest.importorskip("pulgon_tools.detect_point_group")
    pytest.importorskip("pulgon_tools.detect_generalized_translational_group")

    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    with caplog.at_level("WARNING"):
        Symmetry(atoms)

    assert not any("pulgon_tools not installed" in msg for msg in caplog.messages)
