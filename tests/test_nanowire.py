import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("spglib")  # struct/nanowire.py imports struct/symmetry.py unconditionally
pytest.importorskip("pymatgen")

from ase import Atoms
from ase.data import atomic_numbers
from ase.data.vdw_alvarez import vdw_radii

from gk_analysis.struct.nanowire import Nanowire


def _single_atom_wire(symbol="Ar", cell=(20.0, 20.0, 3.0), position=None):
    if position is None:
        position = [cell[0] / 2, cell[1] / 2, cell[2] / 2]
    atoms = Atoms(symbol, positions=[position], cell=cell, pbc=[False, False, True])
    return Nanowire(atoms)


def test_is_orthorhombic():
    nw = _single_atom_wire()
    assert nw.is_orthorhombic() is True

    tilted = Atoms("Ar", positions=[[0, 0, 0]], cell=[[3, 0, 0], [1, 3, 0], [0, 0, 3]], pbc=True)
    assert Nanowire(tilted).is_orthorhombic() is False


def test_get_orientation_detects_periodic_axis():
    cell = (20.0, 20.0, 3.0)
    nw = _single_atom_wire(cell=cell)
    orientation = nw.get_orientation()
    assert list(orientation) == [False, False, True]

    cell_x = (3.0, 20.0, 20.0)
    nw_x = _single_atom_wire(cell=cell_x, position=[1.5, 10.0, 10.0])
    orientation_x = nw_x.get_orientation()
    assert list(orientation_x) == [True, False, False]


def test_single_atom_volume_area_diameter_hand_computed():
    """For a single atom (no neighbors, so no overlap correction), the
    occupied volume is exactly the vdW sphere volume."""
    cell = (20.0, 20.0, 3.0)
    nw = _single_atom_wire("Ar", cell=cell)

    r = vdw_radii[atomic_numbers["Ar"]]
    expected_volume = (4.0 / 3.0) * np.pi * r**3
    assert nw.get_volume() == pytest.approx(expected_volume)

    expected_area = expected_volume / cell[2]
    assert nw.get_area() == pytest.approx(expected_area)

    expected_diameter = 2 * np.sqrt(expected_area / np.pi)
    assert nw.get_diameter() == pytest.approx(expected_diameter)


def test_get_num_neighbors_counts_self_and_close_atoms():
    cell = (20.0, 20.0, 3.0)
    atoms = Atoms(
        "Ar2",
        positions=[[10.0, 10.0, 1.0], [10.0, 10.0, 2.5]],
        cell=cell,
        pbc=[False, False, True],
    )
    nw = Nanowire(atoms)
    # periodic along z with length 3.0, so the mic distance between the two
    # atoms is min(1.5, 3.0 - 1.5) = 1.5
    neighbors = nw.get_num_neighbors(cutoff=2.0)
    assert list(neighbors) == [2, 2]  # includes self (distance 0 <= cutoff)

    neighbors_tight = nw.get_num_neighbors(cutoff=1.0)
    assert list(neighbors_tight) == [1, 1]  # only self


def test_to_coordinate_center_and_to_vacuum_center():
    cell = (20.0, 20.0, 3.0)
    nw = _single_atom_wire("Ar", cell=cell, position=[15.0, 5.0, 1.5])

    centered = nw.to_coordinate_center(copy=True)
    assert np.allclose(centered.get_center_of_mass(), [0.0, 0.0, 0.0])
    # copy=True must not mutate the original
    assert not np.allclose(nw.positions[0], centered.positions[0])

    nw.to_vacuum_center()
    assert np.allclose(nw.positions[0], [10.0, 10.0, 1.5])


def test_rotate_to_z_reorients_periodic_axis():
    cell = (3.0, 20.0, 20.0)
    nw = _single_atom_wire("Ar", cell=cell, position=[1.5, 10.0, 10.0])
    assert list(nw.get_orientation()) == [True, False, False]

    nw.rotate_to_z()
    assert list(nw.get_orientation(center_coordinates=True)) == [False, False, True]


def test_add_vacuum_extends_non_periodic_directions():
    cell = (20.0, 20.0, 3.0)
    nw = _single_atom_wire("Ar", cell=cell)
    nw.add_vacuum(5.0)

    cell_lengths = nw.cell.lengths()
    assert cell_lengths[0] == pytest.approx(25.0)
    assert cell_lengths[1] == pytest.approx(25.0)
    assert cell_lengths[2] == pytest.approx(3.0)
