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

from collections.abc import Iterator, Hashable

import networkx as nx
import numpy as np
from numpy.typing import NDArray
import dimod

from dwave.experimental.lattice_utils.lattice.lattice import Lattice
from dwave.experimental.lattice_utils.lattice.embedded_lattice import EmbeddedLattice

__all__ = ['Triangular', 'DimerizedTriangular']

# For triangular, really for triangular AFM.  Will include explicit dimerized
# version in square lattice.
class Triangular(Lattice):
    """FM or AFM, like Chain."""

    def __init__(self, **kwargs):
        periodic = kwargs.pop("periodic", (True, False))
        self.geometry_name: str = "Triangular"
        self.halve_boundary_couplers: bool = kwargs.pop("halve_boundary_couplers", False)
        self.num_spins = kwargs["dimensions"][0] * kwargs["dimensions"][1]
        self.sublattice: NDArray | None = None
        self.integer_coords: list[tuple[int, int]] | None = None
        self.xy_coords: list[tuple[float, float]] | None = None
        self.xy_size: tuple[float, float] | None = None
        super().__init__(periodic=periodic, **kwargs)
        assert self.periodic[0] is False or self.dimensions[0] % 3 == 0
        assert self.periodic[1] is False or self.dimensions[1] % 3 == 0

    def coordinates(self, node: int) -> tuple[int, int]:
        """Get y,x coordinates, traversing y first."""
        Ly = self.dimensions[0]
        return node % Ly, node // Ly

    def make_nominal_bqm(self, **kwargs) -> dimod.BQM:
        """Accommodate the possibility of halving boundary couplers."""
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

    def generate_edges(self) -> Iterator[tuple[int, int]]:
        """Yield edges for the lattice and initialize coordinate attributes.
        
        y is the first dimension, x is the second.  Edges are straight along
        the y dimension, so boundary must be  staggered in the x dimension, if
        not periodic.
        """
        length_y, length_x  = self.dimensions

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

    def __init__(self, logical_lattice_class: Lattice = Triangular, **kwargs):
        chain_nodes = {
            v: (v, v + np.prod(kwargs["dimensions"]))
            for v in range(np.prod(kwargs["dimensions"]))
        }
        self.geometry_name: str  = "DimerizedTriangular"
        self.num_spins = 2 * int(np.prod(kwargs["dimensions"]))
        logical_lattice_kwargs = kwargs.copy()
        logical_lattice_kwargs.update({"ignore_embedding": True})
        super().__init__(
            logical_lattice_class=logical_lattice_class,
            logical_lattice_kwargs=logical_lattice_kwargs,
            chain_nodes=chain_nodes,
            **kwargs
        )
        self.halve_boundary_couplers: bool = self.logical_lattice.halve_boundary_couplers

    def get_chain_connectivity(
        self,
        u: Hashable,
        v: Hashable | None = None,
    ) -> Iterator[tuple[int, int]]:
        """Should also work for chains!  These can be thought of as self-loops."""
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
        if (ux == vx - 1 or vx == 0):  # (ux == self.dimensions[1] - 1 and self.periodic[1]):
            if uy == vy:
                return ((1, 0),)
            return ((0, 1),)

        if uy == vy:
            return ((0, 1),)
        return ((1, 0),)
