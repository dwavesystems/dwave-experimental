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


from itertools import combinations, product
from numbers import Integral
from collections.abc import Generator, Hashable
from pathlib import Path

import dimod
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice import Lattice

__all__ = ['EmbeddedLattice']


class EmbeddedLattice(Lattice):
    """Embed a logical lattice onto a physical lattice using chains.

    Logical nodes are represented by chains of physical spins. Subclasses can
    specialize ``get_chain_connectivity`` to describe how spins within a chain,
    and between neighboring logical chains, should be connected.

    For example, a dimer-style embedding might map different logical couplings to
    different physical index pairs. In a 3D dimer class, x-, y-, and z-couplings
    could return ``((1, 1),)``, ``((0, 0),)``, and ``((0, 1), (1, 0))``,
    respectively. A chain coupling, where the logical edge is ``(u, u)``, could
    return ``((0, 1),)``.

    Args:
        logical_lattice: The logical lattice instance to embed.
        chain_nodes: Mapping from logical nodes to their physical chains.
    """

    def __init__(
        self,
        *,
        logical_lattice: Lattice,
        chain_nodes: dict[int, tuple[int, Integral]],
        dimensions: tuple[int, ...],
        data_root: str | Path | None = None,
        periodic: tuple[bool, ...] | None = None,
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
        chain_strength: float = 2,
    ):
        if not isinstance(logical_lattice, Lattice):
            raise TypeError("logical_lattice must be a Lattice instance.")

        self.logical_lattice = logical_lattice
        if hasattr(self.logical_lattice, "logical_lattice"):
            raise NotImplementedError("Nested embedded lattices not supported.")

        if data_root is None:
            data_root = logical_lattice.data_root

        self.chain_nodes = chain_nodes
        self.chain_coupling = -chain_strength

        if not hasattr(self, "num_spins"):
            self.num_spins = sum(len(c) for c in chain_nodes.values())

        if periodic is None:
            periodic = self.logical_lattice.periodic

        super().__init__(
            dimensions=dimensions,
            data_root=data_root,
            periodic=periodic,
            orbit_type=orbit_type,
            qubit_orbits=qubit_orbits,
            coupler_orbits=coupler_orbits,
        )

    def get_chain_connectivity(
        self,
        u: Hashable,
        v: Hashable | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Get the connectivity for a given edge in the logical lattice.

        Args:
            u: The first node in the logical edge.
            v: The second node in the logical edge. If None, this is treated as
                a chain edge (u == v).
        Returns:
            A tuple of tuples, where each inner tuple represents a pair of indices
            in the chains corresponding to u and v that should be connected. For
            a chain edge (u == v or v is None), this will return pairs of indices
            within the same chain. For a logical edge (u != v), this will return
            pairs of indices between the two chains.
        """
        # Interior chain connectivity. Generic version: add all possible edges.
        if u == v or v is None:
            return tuple(combinations(range(len(self.chain_nodes[u])), 2))

        # Connectivity between two edges. Generic version: add all possible edges.
        return tuple(product(range(len(self.chain_nodes[u])), range(len(self.chain_nodes[v]))))

    def generate_edges(self) -> Generator[tuple[Hashable, Hashable]]:
        """Yield physical edges for the embedded lattice.

        Returns:
            A generator of tuples, where each tuple represents an edge between
            two spins in the physical lattice.
        """
        logical_bqm = self.logical_lattice.make_bqm()

        # Now embed it.  First make embedded spins and connect the chains.
        for v in logical_bqm.variables:
            for edge in self.get_chain_connectivity(v):
                yield self.chain_nodes[v][edge[0]], self.chain_nodes[v][edge[1]]

        # Next, connect the chains together
        for u, v in self.logical_lattice.edge_list:
            u_chain = self.chain_nodes[u]
            v_chain = self.chain_nodes[v]
            for edge in self.get_chain_connectivity(u, v):
                yield u_chain[edge[0]], v_chain[edge[1]]

    def make_bqm(self, **kwargs) -> dimod.BQM:
        """Construct the physical BQM for this embedded lattice.

        Overrides the base class ``make_bqm`` for the

        Args:
            kwargs: Keyword arguments to pass to the logical lattice's
                `make_bqm` method.

        Returns:
            A dimod.BQM representing the embedded logical BQM.
        """
        if hasattr(self, "fixed_seed"):
            self.logical_lattice.fixed_seed = self.fixed_seed
            kwargs.pop("seed", None)

        return self.embed_bqm(self.logical_lattice.make_bqm(**kwargs))

    def embed_bqm(self, logical_bqm: dimod.BQM) -> dimod.BQM:
        """Embed a logical BQM onto the physical lattice.

        This is a lattice-aware alternative to ``dwave.embedding.embed_bqm``.
        The standard implementation treats each chain as an unodered set of
        physical qubits and routes interactions across whatever target edges
        happen to be available. Here, chains are ordered tuples and the
        physical edges used for each logical interaction are chosen
        deterministically by ``get_chain_connectivity``, so that the position of
        a qubit within its chain carries geometric meaining (e.g. in dimerized
        lattices, index 0 vs. 1 corresponds to a specific sublattice). This
        allows for more structured embeddings that can be tailored to the
        geometry of the logical lattice and the physics of the problem.

        Chain couplings are fixed at ``self.chain_coupling`` rather than
        computed by a chain-strength heuristic.

        Args:
            logical_bqm: A dimod.BQM representing the BQM defined on the logical
                variable space of the embedded lattice.

        Returns:
            A dimod.BQM representing the embedded BQM defined on the physical
            variable space of the embedded lattice.
        """
        # First make embedded spins and connect the chains.
        embedded_bqm = dimod.BQM(vartype="SPIN")
        embedded_variables = np.concatenate(list(self.chain_nodes.values()))
        embedded_variables.sort()

        for v in embedded_variables:
            embedded_bqm.add_variable(v)
        for v in logical_bqm.variables:
            if logical_bqm.degree(v) > 0:  # If the degree is zero we won't add any chain couplings.
                for embedded_v in self.chain_nodes[v]:
                    embedded_bqm.add_linear(
                        embedded_v,
                        logical_bqm.linear[v] / len(self.chain_nodes[v]),
                    )
                for edge in self.get_chain_connectivity(v):
                    embedded_bqm.add_quadratic(
                        self.chain_nodes[v][edge[0]],
                        self.chain_nodes[v][edge[1]],
                        self.chain_coupling,
                    )

        # Next, connect the chains together
        for u, v in self.logical_lattice.edge_list:
            u_chain = self.chain_nodes[u]
            v_chain = self.chain_nodes[v]
            bias_uv = logical_bqm.quadratic[u, v]
            edges = self.get_chain_connectivity(u, v)
            for x, y in edges:
                embedded_bqm.add_quadratic(u_chain[x], v_chain[y], bias_uv / len(edges))

        return embedded_bqm

    def unembed_bqm(self, embedded_bqm: dimod.BQM) -> dimod.BQM:
        """Unembed an embedded BQM back onto the logical variable space.

        Args:
            embedded_bqm: A dimod.BQM representing the BQM defined on the physical
                variable space of the embedded lattice.

        Returns:
            A dimod.BQM representing the unembedded logical BQM.
        """
        logical_bqm = dimod.BQM(vartype="SPIN")
        for v in range(self.logical_lattice.num_spins):
            logical_bqm.add_variable(v)

        which_spin = np.zeros(self.num_spins).astype(int)
        for spin, chain in self.chain_nodes.items():
            which_spin[np.array(chain)] = spin

        for v in embedded_bqm.variables:
            logical_bqm.add_linear(which_spin[v], embedded_bqm.linear[v])

        for u, v in embedded_bqm.quadratic:
            if which_spin[u] != which_spin[v]:
                bias_uv = embedded_bqm.quadratic[u, v]
                logical_bqm.add_quadratic(which_spin[u], which_spin[v], bias_uv)

        return logical_bqm

    def unembed_sampleset(self, sampleset: dimod.SampleSet) -> dimod.SampleSet:
        """Unembed a SampleSet using majority vote with random tie-breaking.

        Args:
            sampleset: A dimod.SampleSet representing samples in the physical
                variable space.

        Returns:
            A dimod.SampleSet representing the unembedded logical samples.
        """
        sample_array = dimod.as_samples(sampleset)[0].T

        voted_samples = np.asarray(
            [
                np.sum(sample_array[self.chain_nodes[v], :], axis=0)
                for v in range(len(self.chain_nodes))
            ]
        )
        voted_samples = np.sign(voted_samples + np.random.rand(*voted_samples.shape) - 0.5).T

        return dimod.SampleSet.from_samples(voted_samples, vartype=dimod.SPIN, energy=0)

    def embed_sample(self, sample: NDArray) -> NDArray:
        """Embed a logical sample onto the physical lattice.

        Args:
            sample: A NumPy array representing a sample in the logical variable space.

        Returns
            A NumPy array representing the embedded physical sample.
        """
        ret = np.zeros(self.num_spins)
        for spin, chain in self.chain_nodes.items():
            ret[np.array(chain)] = sample[spin]

        return ret

    def unembed_sample(self, sample: NDArray) -> NDArray:
        """Unembed a physical sample using majority vote with random tie-breaking.

        Args:
            sample: A NumPy array representing a sample in the physical variable space.

        Returns:
            A NumPy array representing the unembedded logical sample.
        """
        ret = np.zeros(self.logical_lattice.num_spins)
        for spin, chain in self.chain_nodes.items():
            ret[spin] = np.sign(np.sum(sample[np.array(chain)]) + np.random.rand() - 0.5)

        return ret
