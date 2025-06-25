# Copyright 2025 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from ase.atoms import Atoms
from ase.data.vdw_alvarez import vdw_radii
import numpy as np
import ase.io
from gk_eval.struct.symmetry import Symmetry


class Nanowire(Atoms):
    def __init__(
        self,
        atoms: Atoms = None,
        miller: np.ndarray = None,
        facets: dict = None,
        **kwargs,
    ):
        if isinstance(atoms, str):
            atoms = ase.io.read(atoms)

        if atoms is not None:
            kwargs["positions"] = atoms.positions
            kwargs["numbers"] = atoms.numbers
            kwargs["cell"] = atoms.cell
        super().__init__(**kwargs)
        self._init_variables()

        self.miller = miller
        self.facets = facets

    def _init_variables(self):
        self.diameter = None
        self.area = None
        self.surface_area = None
        self.volume = None
        self.orientation = None
        self.vacuum_volume = None
        self.vacuum_dirs = None
        self.vacuum_dist = None
        self.symmetry = None

    def to_coordinate_center(self, copy=False):
        """
        Translates the positions to the 0 point of the coordinate system.

        Args:
            copy (bool, optional): If True, a copy of the atoms object will be returned. Defaults to False.

        Returns:
            Atoms: The atoms object translated to the center of mass.
        """
        if copy:
            atoms = self.copy()
        else:
            atoms = self
        atoms.positions -= self.get_center_of_mass()
        # atoms.translate(np.array([0, -0.5, 0]) * self.cell.lengths())
        return atoms

    def to_vacuum_center(self, copy=False):
        """
        Translates the atoms to the center of the vacuum.
        Will only work reliably for orthorhombic cells

        Parameters:
            copy (bool): If True, the realigned atoms object will be returned as a copy and not set internally.

        Returns:
            Atoms: The atoms object translated to the center of the vacuum.
        """
        if copy:
            atoms = self.copy()
        else:
            atoms = self
        atoms = self.to_coordinate_center(copy=copy)
        atoms.translate(np.array([0.5, 0.5, 0.5]) * self.cell.lengths())
        # for i in range(3):
        #     shift = atoms.positions[:,i] // self.cell.lengths()[i]
        #     print(shift)
        #     # if len(neg_values) > 0:
        #     atoms.positions[:,i] -= self.cell.lengths()[i] * shift

        return atoms

    def rotate_axis(self, vec1, vec2):
        """
        Rotates the nanowire by a specified angle around a given rotation axis.

        Parameters:
            vec1 (numpy.ndarray): The first vector defining the rotation axis.
            vec2 (numpy.ndarray): The second vector defining the rotation axis.

        Returns:
            None
        """
        rotation_axis = np.cross(vec1, vec2)
        # the ONE place where the angle has to be given in degrees
        rotation_angle = np.arccos(np.dot(vec1, vec2)) / 2 / np.pi * 360
        self.rotate(rotation_angle, rotation_axis, rotate_cell=True)

    def rotate_to_z(self):
        """
        Rotates the nanowire to align it with the z-axis.

        This method does not take any parameters and does not return any values.
        """
        if self.orientation is None:
            self.get_orientation()

        init_dir = self.orientation
        final_dir = 2
        if not self.orientation[final_dir]:
            vec1 = np.zeros(3)
            vec2 = np.zeros(3)
            vec1[init_dir] = 1
            vec2[final_dir] = 1
            self.rotate_axis(vec1, vec2)
            # because ASE does not switch the vectors
            newcell = self.cell.copy()
            newcell[final_dir] = newcell[init_dir]
            newcell[init_dir] = -self.cell[final_dir]
            self.set_cell(newcell)
            self.positions[:, init_dir] = -self.positions[:, init_dir]

            # update orientation
            self.get_orientation()

    def rotate_part_around_orientation(self, indices, angle):
        """
        Rotates a subset of the nanowire by a specified angle around the orientation of the nanowire.

        Parameters:
            indices (numpy.ndarray): The indices of the atoms to rotate.
            angle (float): The angle in radians to rotate the atoms by.

        Returns:
            None
        """
        orientation = self.get_orientation()
        rot_mat = np.eye(3)
        direcs = np.where(np.logical_not(orientation))[0]
        rot_mat[direcs[0], direcs[1]] = -np.sin(angle)
        rot_mat[direcs[1], direcs[0]] = np.sin(angle)
        rot_mat[direcs, direcs] = np.cos(angle)
        self.to_coordinate_center()
        self.positions[indices] = np.dot(self.positions[indices], rot_mat)
        self.to_vacuum_center()

    def get_diameter(self):
        """
        Calculate the diameter of the nanowire.

        This method assumes that the nanowire is roughly circular and calculates the diameter based on its area.

        Returns:
            float: The diameter of the nanowire.
        """

        # assumption: roughly circular
        area = self.get_area()
        self.diameter = float(2 * np.sqrt(area / np.pi))
        return self.diameter

    def get_area(self):
        """
        Calculate the area of the nanowire from the volume and periodic direction.

        Returns:
            float: The area of the nanowire.
        """
        if self.volume is None:
            self.detect_occupied_volume()
        if self.orientation is None:
            self.get_orientation()
        return float(self.volume / self.cell[self.orientation, self.orientation])

    def get_volume(self):
        """
        Get the volume of the nanowire.

        Returns:
            float: The volume of the nanowire.
        """
        if self.volume is None:
            self.detect_occupied_volume()
        return self.volume

    def get_surface_area(self):
        """
        Get the surface area of the nanowire.

        Returns:
            float: The surface area of the nanowire.
        """
        if self.surface_area is None:
            self.detect_occupied_volume()
        return self.surface_area

    def get_symmetry(self, symprec=1e-5):
        """
        Get the symmetry of the nanowire. Includes symmetries from pulgon and spglib.

        Returns:
            Symmetry: The symmetry of the nanowire as an object.
        """
        if self.symmetry is None:
            self._detect_symmetry(symprec=symprec)
        return self.symmetry

    def _detect_symmetry(self, symprec=1e-5):
        """
        Detect the symmetry of the nanowire.

        Parameters:
            symprec (float, optional): The tolerance for symmetry finding. Defaults to 1e-5.

        Returns:
            None
        """
        self.symmetry = Symmetry(self, symprec=symprec)

    def is_orthorhombic(self):
        """
        Check if the cell of the atoms is orthorhombic.

        Returns:
            bool: True if the cell is orthorhombic, False otherwise.
        """
        if np.allclose(self.cell.angles(), [90, 90, 90]):
            orthorhombic = True
        else:
            orthorhombic = False
        return orthorhombic

    def get_extremal_cartesian_coordinates(self):
        """
        Calculate the extremal Cartesian coordinates of the atoms in the system including their van der Waals radii.

        Returns:
            tuple: A tuple containing two arrays:
                - min_cart_coors (numpy.ndarray): An array of shape (n_atoms, 3) containing the minimum Cartesian coordinates of each atom.
                - max_cart_coors (numpy.ndarray): An array of shape (n_atoms, 3) containing the maximum Cartesian coordinates of each atom.
        """
        radii = vdw_radii[self.numbers]
        min_cart_coors = np.min(self.positions.T - radii, axis=1)
        max_cart_coors = np.max(self.positions.T + radii, axis=1)
        return min_cart_coors, max_cart_coors

    def add_vacuum(self, amount):
        """
        Add vacuum to the nanowire.

        Parameters:
            amount (float): The amount of vacuum to add.

        Returns:
            None
        """
        self.to_vacuum_center()
        orientation = np.diag(np.logical_not(self.get_orientation()))

        self.cell[orientation] += amount
        self.to_vacuum_center()

    def detect_occupied_volume(self):
        """
        Detects the occupied volume of the atoms in the system based on their van der Waals radii.

        Returns:
            float: The total volume occupied by the atoms.
        """
        numbers = self.numbers
        volume = 0
        surface_area = 0

        # when using the mic, this should require no special consideration regarding the orientation
        distances = self.get_all_distances(mic=True)

        radii = vdw_radii[numbers]
        for aid in range(len(self)):
            radius1 = radii[aid]

            # add the sphere volume
            volume += (4 / 3) * np.pi * (radius1**3)

            # compute and subtract the overlapping volume
            other_radii = vdw_radii[numbers[aid + 1 :]]
            other_distances = distances[aid, aid + 1 :]
            overlapping = other_distances < radius1 + other_radii
            other_radii = other_radii[overlapping]
            other_distances = other_distances[overlapping]
            overlap_volume = np.sum(
                np.pi
                / 12
                / other_distances
                * (radius1 + other_radii - other_distances) ** 2
                * (
                    other_distances**2
                    + 2 * other_distances * (radius1 + other_radii)
                    - 3 * (radius1 - other_radii) ** 2
                )
            )
            volume -= overlap_volume

            # same for the surface area
            # unfortunately, it does not quite work like that...
            # the error is not huge though
            atom_surface_area = 4 * np.pi * radius1**2
            overlap_surface_area = np.sum(
                np.pi
                / other_distances
                * (
                    radius1
                    * (other_radii - radius1 + other_distances)
                    * (other_radii + radius1 - other_distances)
                    # + other_radii
                    # * (radius1 - other_radii + other_distances)
                    # * (radius1 + other_radii - other_distances)
                )
            )

            surface_area += atom_surface_area - overlap_surface_area
        # exit()
        self.volume = volume
        self.vacuum_volume = volume - self.get_volume()
        self.surface_area = surface_area
        return volume

    def get_orientation(self, center_coordinates=False):
        """
        Retrieves the orientation of the nanowire based on vacuum.

        This function does not take any parameters.

        Returns:
            a vector specifying the nanowire orientation
        """
        orhorhombic = self.is_orthorhombic()
        if not center_coordinates:
            saved_pos = self.positions.copy()
        self.to_coordinate_center()
        if orhorhombic:
            minpos, maxpos = self.get_extremal_cartesian_coordinates()
            center_cell = np.diag(self.cell.array) / 2
            # the assumption here is that the vdW radii of the atoms stick out of the periodic cell direction
            vacuum_dirs1 = -center_cell < minpos
            vacuum_dirs2 = center_cell > maxpos
            if np.all(vacuum_dirs1 != vacuum_dirs2):
                raise ValueError(
                    "Confusion during nanowire orientation calculation. Does the structure not contain vacuum?"
                )
            self.vacuum_dirs = vacuum_dirs1
            self.orientation = np.logical_not(vacuum_dirs1)
            self.vacuum_dist = 2 * center_cell - (maxpos - minpos)
            self.vacuum_dist[self.orientation] = 0
            # print(self.positions)
            # print(vacuum_dirs1, vacuum_dirs2)
            # print(center_cell)
            # print(minpos, maxpos)
            # self.set_pbc(self.orientation.copy())
            if np.sum(self.orientation) != 1:
                # TODO: improve this so that it fails less frequently
                print(
                    "WARNING: orientation is ambiguous. Naively assuming z orientation"
                )
                self.orientation = [False, False, True]
                # from ase.visualize import view
                # view(self)

        else:
            # raise NotImplementedError(
            #     "getting nanowire orientation is not implemented for non-orthorhombic cells"
            # )
            print(
                "WARNING: getting nanowire orientation is not implemented for non-orthorhombic cells. Assuming z orientation"
            )
            self.orientation = [False, False, True]

        if not center_coordinates:
            self.set_positions(saved_pos)

        return self.orientation

    def create_supercell(self, supercell):
        """
        Creates a supercell of the nanowire by repeating the atoms along the specified orientation.

        Parameters:
            supercell (int or list of int): The number of times to repeat the atoms along the periodic direction of the nanowire.

        Returns:
            ase.Atoms: The supercell of the nanowire.
        """
        if self.orientation is None:
            self.get_orientation()
        vec = np.ones(3)
        vec[self.orientation] = supercell
        supercell = self.repeat(vec)
        return supercell

    def get_num_neighbors(self, cutoff=5.0):
        """
        Returns the number of neighbors for each atom in the nanowire based on a cutoff.

        Parameters:
            cutoff (float): The cutoff distance for determining neighbor atoms.

        Returns:
            np.array: The number of neighbors for each atom in the nanowire.
        """

        dists = self.get_all_distances(mic=True)
        neighbors = np.sum(dists <= cutoff, axis=0)
        return neighbors

    def get_CN_ASANN(self):
        """
        Returns the coordination number for each atom using the ASANN approach

        Returns:
            np.array: The coordination number for each atom in the nanowire.
        """
        from src.dockonsurf.ASANN import coordination_numbers

        asann_CNs, asann_radii, asann_edges, vectors = coordination_numbers(
            self.positions, pbc=True, cell_vectors=self.cell.array
        )
        return asann_CNs
