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

"""What do we want here?

- A class that handles embedded models.
- Perhaps called FixedEmbeddingModel
- Should have a function called get_chain_connections, which takes an edge and
returns a list or tuple of adjacent indices.  For example, in the 3D dimer class,
x,y, and z-couplings would return
((1,1))
((0,0))
((0,1),(1,0))
respectively.
A chain coupling (edge=(u,v) where u==v) can, in this case, return ((0,1)).
"""
from itertools import combinations, product
from numbers import Integral
from typing import Any
from collections.abc import Iterator, Hashable

import dimod
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice import Lattice

__all__ = ['EmbeddedLattice']

class EmbeddedLattice(Lattice):
    """Specifics should depend on the embedding.  Should this have a logical
    model as an attribute? Let's try."""
    def __init__(
        self,
        logical_lattice_class: Lattice,
        logical_lattice_kwargs: dict[str, Any],
        chain_nodes: dict[int, tuple[int, Integral]],
        **kwargs,
    ):
        self.logical_lattice: Lattice = logical_lattice_class(**logical_lattice_kwargs)
        self.chain_nodes: dict[tuple[int, Integral]] = chain_nodes
        self.chain_coupling: float = -kwargs.pop("chain_strength", 2)
        kwargs.setdefault("periodic", self.logical_lattice.periodic)
        super().__init__(**kwargs)

    def get_chain_connectivity(self, u, v=None):
        """Should also work for chains!  These can be thought of as self-loops."""
        # Interior chain connectivity. Generic version: add all possible edges.
        if u == v or v is None:
            return tuple(combinations(range(len(self.chain_nodes[u])), 2))

        # Connectivity between two edges. Generic version: add all possible edges.
        return tuple(product(range(len(self.chain_nodes[u])), range(len(self.chain_nodes[v]))))

    def generate_edges(self) -> Iterator[tuple[Hashable, Hashable]]:
        """Yield physical edges for the embedded lattice."""
        logical_bqm = self.logical_lattice.make_nominal_bqm()

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

    def make_nominal_bqm(self, **kwargs) -> dimod.BQM:
        """Construct and embed the nominal BQM."""
        if hasattr(self, "fixed_seed"):
            self.logical_lattice.fixed_seed = self.fixed_seed
            kwargs.pop("seed", None)

        return self.embed_bqm(self.logical_lattice.make_nominal_bqm(**kwargs))

    def embed_bqm(self, logical_bqm: dimod.BQM) -> dimod.BQM:
        """Embed a logical BQM onto the physical lattice."""
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
        """Unembed an embedded BQM back onto the logical variable space."""
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
                bias_uv =  embedded_bqm.quadratic[u, v]
                logical_bqm.add_quadratic(which_spin[u], which_spin[v], bias_uv)

        return logical_bqm

    def unembed_sampleset(self, sampleset: dimod.SampleSet) -> dimod.SampleSet:
        """Unembed a SampleSet using majority vote with random tie-breaking."""
        sample_array = dimod.as_samples(sampleset)[0].T

        voted_samples = np.asarray(
            [
                np.sum(sample_array[self.chain_nodes[v], :], axis=0)
                for v in range(len(self.chain_nodes))
            ]
        )
        voted_samples = np.sign(voted_samples + np.random.rand(*voted_samples.shape)).T

        return dimod.SampleSet.from_samples(voted_samples, vartype=dimod.SPIN, energy=0)

    def embed_sample(self, sample: NDArray) -> NDArray:
        """Embed a logical sample onto the physical lattice."""
        ret = np.zeros(self.num_spins)
        for spin, chain in self.chain_nodes.items():
            ret[np.array(chain)] = sample[spin]

        return ret

    def unembed_sample(self, sample: NDArray) -> NDArray:
        """Unembed a physical sample using majority vote with random tie-breaking."""
        ret = np.zeros(self.logical_lattice.num_spins)
        for spin, chain in self.chain_nodes.items():
            ret[spin] = np.sign(np.sum(sample[np.array(chain)]) + np.random.rand() - 0.5)

        return ret
