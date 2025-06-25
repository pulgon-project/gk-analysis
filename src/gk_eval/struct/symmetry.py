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

import spglib
from ase import Atoms
from pymatgen.core import Molecule
import logging


class Symmetry:
    def __init__(self, atoms: Atoms = None, symprec=1e-5, **kwargs):
        self.atoms = atoms
        self.symprec = symprec
        self._init_variables()

        if self.atoms is not None:
            self.detect_symmetry(self.atoms)

    def _init_variables(self):
        self.space_group_symbol = None
        self.space_group_number = None
        self.axial_point_group = None
        self.cyclic_point_group = None

        self.line_group_analyzer = None
        self.cyclic_group_analyzer = None

    def detect_symmetry(self, atoms):
        self.atoms = atoms
        self.get_space_group()
        self.get_axial_point_group()
        self.get_cyclic_point_group()
        return self.space_group_symbol, self.axial_point_group, self.cyclic_point_group

    def get_axial_point_group(self):
        # only import the pulgon tools if needed
        try:
            from pulgon_tools_wip.detect_point_group import LineGroupAnalyzer

            mol = Molecule(species=self.atoms.numbers, coords=self.atoms.positions)
            self.line_group_analyzer = LineGroupAnalyzer(mol, tolerance=self.symprec)
            self.axial_point_group = self.line_group_analyzer.get_pointgroup()
        except ImportError:
            self.axial_point_group = None
            logging.warning("pulgon_tools_wip not installed")
        except Exception as e:
            logging.warning("WARNING axial point group detection failed: ", e)

        return self.axial_point_group

    def get_cyclic_point_group(self):
        try:
            from pulgon_tools_wip.detect_generalized_translational_group import (
                CyclicGroupAnalyzer,
            )

            self.cyclic_group_analyzer = CyclicGroupAnalyzer(
                self.atoms, tolerance=self.symprec
            )
            self.cyclic_point_group, self.cyclic_monomers = (
                self.cyclic_group_analyzer.get_cyclic_group()
            )
        except ImportError:
            self.cyclic_point_group = None
            self.cyclic_monomers = None
            logging.warning("pulgon_tools_wip not installed")
        except Exception as e:
            logging.warning("WARNING cyclic point group detection failed: ", e)

        return self.cyclic_point_group

    def get_space_group(self):

        cell = (
            self.atoms.cell.array,
            self.atoms.positions,
            self.atoms.get_atomic_numbers(),
        )

        space_group = spglib.get_spacegroup(cell, symprec=self.symprec)
        self.space_group_symbol = space_group.split("(")[0]
        self.space_group_number = space_group.split("(")[1].split(")")[0]

        return self.space_group_number

    def print_symmetry(self):
        print(f"space group: {self.space_group_symbol} {self.space_group_number}")
        print(f"axial point group: {self.axial_point_group}")
        print(f"cyclic point group: {self.cyclic_point_group}")
