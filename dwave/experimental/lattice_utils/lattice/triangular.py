# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Generator, Hashable
from pathlib import Path

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import dimod

from dwave.experimental.lattice_utils.lattice.lattice import Lattice
from dwave.experimental.lattice_utils.lattice.embedded_lattice import EmbeddedLattice

__all__ = ['Triangular', 'DimerizedTriangular']


class Triangular(Lattice):
    """Triangular lattice class.

    This class represents a 2D triangular lattice, where each node is connected
    to its six nearest neighbors (except at boundaries, if not periodic).

    Args:
        dimensions: Two-element tuple giving the number of spins in the y and x
            dimensions.
        periodic: Two-element tuple indicating whether the lattice is periodic
            in the y and x dimensions.
        data_root: A string or Path to the root directory for storing lattice
            data. orbit_type: Method for determining qubit and coupler orbits.
            Must be one of "global", "standard", "singleton", or "explicit". See
            ``initialize_orbits`` for details.
        qubit_orbits: Explicit qubit orbit labels, used only when
            ``orbit_type == "explicit"``. Must have length equal to the number
            of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when
            ``orbit_type == "explicit"``. Must have length equal to the number
            of edges in the lattice.
        halve_boundary_couplers: A boolean indicating whether to assign half the
            coupling strength to boundary couplers.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int, int],
        periodic: tuple[bool, bool] = (True, False),
        data_root: str | Path,
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
        halve_boundary_couplers: bool = False,
    ):
        if len(dimensions) != 2:
            raise ValueError(f"Triangular requires dimensions of length 2, got {len(dimensions)}.")

        self.geometry_name: str = "Triangular"
        self.halve_boundary_couplers: bool = halve_boundary_couplers
        self.num_spins = dimensions[0] * dimensions[1]
        self.sublattice: NDArray | None = None
        self.integer_coords: list[tuple[int, int]] | None = None
        self.xy_coords: list[tuple[float, float]] | None = None
        self.xy_size: tuple[float, float] | None = None
        super().__init__(
            dimensions=dimensions,
            periodic=periodic,
            data_root=data_root,
            orbit_type=orbit_type,
            qubit_orbits=qubit_orbits,
            coupler_orbits=coupler_orbits,
        )
        if self.periodic[0] and self.dimensions[0] % 3 != 0:
            raise ValueError(
                "For Triangular with periodic[0]=True, dimensions[0] must be divisible by 3."
            )
        if self.periodic[1] and self.dimensions[1] % 3 != 0:
            raise ValueError(
                "For Triangular with periodic[1]=True, dimensions[1] must be divisible by 3."
            )

    def coordinates(self, node: int) -> tuple[int, int]:
        """Return the coordinates of a node in the lattice given its index.

        Node indices are ordered by traversing the y direction first.

        Args:
            node: The index of the node for which to return coordinates.

        Returns:
            A tuple (y, x) representing the coordinates of the node in the lattice.
        """
        length_y = self.dimensions[0]
        return node % length_y, node // length_y

    def make_nominal_bqm(self) -> dimod.BQM:
        """Construct the nominal triangular lattice BQM.

        If ``halve_boundary_couplers`` is True, couplers that are on the boundary
        of the lattice are assigned a coupling strength of 0.5 instead of 1.0.

        Returns:
            A dimod.BQM representing the nominal triangular lattice.
        """
        graph = self._make_networkx_graph()
        bqm = dimod.BQM(vartype="SPIN")

        for v in range(self.num_spins):
            bqm.add_variable(v)
        for u, v in self.edge_list:
            if not self.halve_boundary_couplers or graph.degree[u] == 6 or graph.degree[v] == 6:
                bqm.add_quadratic(u, v, 1.0)
            else:
                bqm.add_quadratic(u, v, 0.5)

        return bqm

    def generate_edges(self) -> Generator[tuple[int, int]]:
        """Yield edges for the triangular lattice and initialize coordinate attributes.

        y is the first dimension, x is the second. Edges are straight along
        the y dimension, so boundary must be staggered in the x dimension, if
        not periodic.

        Returns:
            A generator of tuples, where each tuple represents an edge between
            two spins in the lattice.
        """
        length_y, length_x = self.dimensions

        graph = nx.Graph()
        for x in range(length_x):
            for y in range(length_y):
                graph.add_node((y, x))

        for x in range(length_x):
            for y in range(length_y):
                # Do y couplers
                if y < length_y - 1 or self.periodic[0]:
                    graph.add_edge((y, x), ((y + 1) % length_y, x))

                if x < length_x - 1 or self.periodic[1]:

                    # Do up-up couplers
                    graph.add_edge((y, x), (y, (x + 1) % length_x))
                    # Do up-down couplers
                    if y > 0 or self.periodic[0]:
                        graph.add_edge((y, x), ((y - 1) % length_y, (x + 1) % length_x))

        num_nodes = graph.number_of_nodes()
        relabeling = {self.coordinates(v): v for v in range(num_nodes)}
        graph = nx.relabel_nodes(graph, relabeling)

        self.sublattice = np.array([(v - (v // length_y)) % 3 for v in range(num_nodes)])

        self.integer_coords = [
            (self.coordinates(v)[1], (self.coordinates(v)[0])) for v in range(num_nodes)
        ]
        self.xy_coords = [
            (
                self.integer_coords[v][0] * 3**0.5 / 2,
                self.integer_coords[v][0] / 2 + self.integer_coords[v][1],
            )
            for v in range(num_nodes)
        ]
        self.xy_size = (length_y, length_x * 3**0.5 / 2)  # Size as though periodic.

        yield from sorted([tuple(sorted(e)) for e in graph.edges])


class DimerizedTriangular(EmbeddedLattice):
    """Dimerized triangular lattice class.

    This class represents a dimerized version of the 2D triangular lattice,
    where each node in the logical lattice is represented by a chain of two spins
    in the physical lattice.

    Args:
        dimensions: Two-element tuple giving the number of spins in the y and x
            dimensions.
        periodic: Two-element tuple indicating whether the lattice is periodic
            in the y and x dimensions.
        data_root: A string or Path to the root directory for storing lattice data.
        orbit_type: Method for determining qubit and coupler orbits. Must be one of "global",
            "standard", "singleton", or "explicit". See ``initialize_orbits`` for details.
        qubit_orbits: Explicit qubit orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of edges in the lattice.
        halve_boundary_couplers: A boolean indicating whether to assign half the
            coupling strength to boundary couplers in the logical lattice.
        chain_strength: The strength of the couplings within each chain.
        logical_lattice: Optional logical lattice instance to embed. If not
            provided, a ``Triangular`` lattice is constructed from the other
            initialization arguments.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int, int],
        periodic: tuple[bool, bool] = (True, False),
        data_root: str | Path,
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
        halve_boundary_couplers: bool = False,
        chain_strength: float = 2,
        logical_lattice: Lattice | None = None,
    ):
        if len(dimensions) != 2:
            raise ValueError(
                f"DimerizedTriangular requires dimensions of length 2, got {len(dimensions)}."
            )
        chain_nodes = {v: (v, v + np.prod(dimensions)) for v in range(np.prod(dimensions))}
        self.geometry_name: str = "DimerizedTriangular"
        self.num_spins = 2 * int(np.prod(dimensions))
        if logical_lattice is None:
            logical_lattice = Triangular(
                dimensions=dimensions,
                periodic=periodic,
                data_root=data_root,
                orbit_type=orbit_type,
                qubit_orbits=qubit_orbits,
                coupler_orbits=coupler_orbits,
                halve_boundary_couplers=halve_boundary_couplers,
            )

        super().__init__(
            logical_lattice=logical_lattice,
            chain_nodes=chain_nodes,
            dimensions=dimensions,
            periodic=periodic,
            data_root=data_root,
            orbit_type=orbit_type,
            qubit_orbits=qubit_orbits,
            coupler_orbits=coupler_orbits,
            chain_strength=chain_strength,
        )
        self.halve_boundary_couplers: bool = self.logical_lattice.halve_boundary_couplers

    def get_chain_connectivity(
        self,
        u: Hashable,
        v: Hashable | None = None,
    ) -> tuple[tuple[int, int]]:
        """Return the connectivity for a chain or edge in the logical lattice.

        Args:
            u: The first node in the logical edge.
            v: The second node in the logical edge. If None, this is treated as
                a chain edge (u == v).
        Returns:
            A tuple of tuples, where each inner tuple represents a pair of indices
            in the chainscorresponding to u and v that should be connected. For
            a chain edge (u == v or v is None), this will return pairs of indices
            within the same chain. For a logical edge (u != v), this will return
            pairs of indices between the two chains.
        """
        if u == v or v is None:
            # Interior chain connectivity.
            # Generic version: add all possible edges.
            return ((0, 1),)

        # Connectivity between two edges.
        # Triangular version
        uy, ux = self.logical_lattice.coordinates(u)
        vy, vx = self.logical_lattice.coordinates(v)

        if ux == vx:  # straight up.
            if uy > vy or (uy == 0 and vy == self.dimensions[0] - 1 and self.periodic[0]):
                return ((0, 1),)
            return ((1, 0),)

        # x-edge, i.e. tilted.
        if ux == vx - 1 or vx == 0:  # (ux == self.dimensions[1] - 1 and self.periodic[1]):
            if uy == vy:
                return ((1, 0),)
            return ((0, 1),)

        if uy == vy:
            return ((0, 1),)
        return ((1, 0),)
