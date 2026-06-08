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

import os
from pathlib import Path
from collections.abc import Generator, Hashable
from abc import ABC, abstractmethod
import warnings

import dimod
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice.orbits import get_orbits
from dwave.experimental.lattice_utils.lattice.optimize import optimize

__all__ = ['Lattice']


class Lattice(ABC):
    """An abstract base class for representing lattice geometries used in lattice-utils experiments.

    Subclasses are resonsible for defining the lattice geometry itself. In particular,
    a subclass must:

    - Implement the ``generate_edges`` method, which yields the edges of the lattice as pairs
    - Initialize the ``self.num_spins`` attribute in the constructor, which is used by the base class
    - set any geometry-specific identifiers such as ``self.geometry_name``

    Args:
        dimensions: Tuple specifying the size of the lattice in each dimension.
        data_root: Root directory for loading and saving lattice data such as embeddings and orbits.
        periodic: Tuple indicating whether each dimension is periodic (True) or open (False).
        orbit_type: Method for determining qubit and coupler orbits. Must be one of "global",
            "standard", "singleton", or "explicit". See ``initialize_orbits`` for details.
        qubit_orbits: Explicit qubit orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of edges in the lattice.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int, ...],
        data_root: str | Path,
        periodic: tuple[bool, ...] | None = None,
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
    ):
        self.dimensions = dimensions
        self.data_root = Path(data_root)

        self.periodic = periodic if periodic is not None else tuple(False for _ in dimensions)
        if len(self.periodic) != len(self.dimensions):
            raise ValueError(
                f"periodic and dimensions must have the same length: "
                f"got {len(self.periodic)} and {len(self.dimensions)}."
            )

        self.edge_list: list[tuple[Hashable, Hashable]] = list(self.generate_edges())

        if not hasattr(self, "num_spins"):
            raise AttributeError(f"{type(self).__name__} subclass must initialize self.num_spins")

        self.num_edges: int = len(self.edge_list)
        self.orbit_type: str = orbit_type
        self.initialize_orbits(qubit_orbits, coupler_orbits)

    @abstractmethod
    def generate_edges(self) -> Generator[tuple[Hashable, Hashable]]:
        """Yield the edges for this lattice."""

    def embed_lattice(
        self,
        sampler: dimod.Sampler,
        try_to_load: bool = True,
        timeout: int = 10,
        max_number_of_embeddings: int | None = None,
        min_number_of_embeddings: int = 1,
        exclude_qubits: list | None = None,
        **kwargs,
    ) -> None:
        """Find or load embeddings onto the sampler graph.

        Args:
            sampler: Sampler whose hardware graph is used as the target for embedding.
            try_to_load: If True, attempt to load embeddings from disk before
                trying to find them.
            timeout: Time limit for the embedding search, in seconds.
            max_number_of_embeddings: Maximum number of embeddings to search for.
            min_number_of_embeddings: Minimum number of embeddings required to save.
            exclude_qubits: Qubits to remove from the sampler graph before searching
                for embeddings.
        """
        if exclude_qubits is None:
            exclude_qubits = []

        graph_bqm = dimod.to_networkx_graph(self.make_nominal_bqm())
        graph_sampler = sampler.to_networkx_graph()
        graph_sampler.remove_nodes_from(exclude_qubits)

        if try_to_load:
            try:
                self._load_embeddings(sampler)
                return
            except FileNotFoundError:
                warnings.warn("No embedding file found.")

        embedding_dicts = find_multiple_embeddings(
            graph_bqm,
            graph_sampler,
            max_num_emb=max_number_of_embeddings,
            embedder_kwargs={'timeout': timeout},
        )
        if not embedding_dicts:
            raise ValueError("No embeddings found")

        embeddings = np.stack([list(emb.values()) for emb in embedding_dicts])
        if len(embeddings) >= min_number_of_embeddings and np.prod(embeddings.shape):
            self._save_embeddings(sampler, embeddings)

    def make_nominal_bqm(self) -> dimod.BQM:
        """Construct a default nominal BQM coupling strength values set to +1.

        Args:
            **kwargs: additional keyword arguments forwarded to subclass implementations.
                Subclasses may use these to modify the construction of the nominal BQM.

        Returns:
            A binary quadratic model representing the lattice with uniform
            coupling strength.
        """
        bqm = dimod.BQM(vartype="SPIN")
        for v in range(self.num_spins):
            bqm.add_variable(v)
        for u, v in self.edge_list:
            bqm.add_quadratic(u, v, 1.0)

        return bqm

    def initialize_orbits(
        self,
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
    ) -> None:
        """Initialize qubit and coupler orbits.

        Orbit assignments are determined according to ``self.orbit_type``:

        -``global``: Put all the couplers in one orbit and all the qubits in one
        orbit. Exception: for embedded lattices, put all logical couplers in one
        orbit and all chain couplers in another.
        -``standard``: Load previously computed automorphism-based orbits, or
            compute them and save them if unavailable.
        -``singleton``: Put each qubit and coupler in its own orbit.
        -``explicit``: Use the orbit assignments provided via ``qubit_orbits``
            and ``coupler_orbits``.

        Args:
            qubit_orbits: Explicit qubit orbit labels, used only when
                ``self.orbit_type == "explicit"``. Must have length ``self.num_spins``.
            coupler_orbits: Explicit coupler orbit labels. Used only when
                ``self.orbit_type == "explicit"``. Must have length ``self.num_edges``.
        """
        if self.orbit_type == "global":
            self.qubit_orbits = np.zeros(self.num_spins, dtype=int)

            if hasattr(self, "logical_lattice"):
                which_chain = {v: key for key, val in self.chain_nodes.items() for v in val}
                self.coupler_orbits = np.zeros(self.num_edges, dtype=int)

                for i, (u, v) in enumerate(self.edge_list):
                    if which_chain[u] == which_chain[v]:
                        self.coupler_orbits[i] = 1
            else:
                self.coupler_orbits = np.zeros(self.num_edges, dtype=int)

        elif self.orbit_type == "standard":
            try:
                self._load_orbits()
            except FileNotFoundError:
                # calculating orbits
                bqm = self.make_nominal_bqm()
                self.qubit_orbits, self.coupler_orbits = get_orbits(bqm, self.edge_list)
                self._save_orbits()

        elif self.orbit_type == "singleton":
            self.qubit_orbits = np.arange(self.num_spins)
            self.coupler_orbits = np.arange(self.num_edges)

        elif self.orbit_type == "explicit":
            if qubit_orbits is not None and coupler_orbits is not None:
                if len(qubit_orbits) != self.num_spins:
                    raise ValueError(
                        f"qubit_orbits must have length {self.num_spins}, got {len(qubit_orbits)}."
                    )
                if len(coupler_orbits) != self.num_edges:
                    raise ValueError(
                        f"coupler_orbits must have length {self.num_edges}, "
                        f"got {len(coupler_orbits)}."
                    )
            self.qubit_orbits = qubit_orbits
            self.coupler_orbits = coupler_orbits
        else:
            raise ValueError(
                f'Unknown orbit type {self.orbit_type}. '
                'Must be "global", "standard", "singleton", or "explicit".'
            )

    def _get_path(
        self,
        kind: str,
        sampler_name: str | None = None,
        extra_subdir: str | Path | None = None,
    ) -> Path:
        """Construct a standarized file path for embedding or orbit data."""
        if kind not in {"embedding", "orbits"}:
            raise ValueError("kind must be provided as either `embedding` or `orbits`")

        class_subdir = Path(self.geometry_name)
        if extra_subdir is not None:
            class_subdir = class_subdir / extra_subdir

        base_dir = self.data_root / "lattice_data" / kind / class_subdir
        if sampler_name is not None:
            base_dir = base_dir / sampler_name

        filename = f"{self._get_size_pathstring()}.txt"
        return base_dir / filename

    def _make_filename(self, kind: str, sampler: dimod.Sampler | None = None) -> Path:
        """Construct a data filename for the specified sampler and data type."""
        if sampler is None:
            return self._get_path(kind)

        if type(sampler).__name__ == "MockDWaveSampler":
            return self._get_path(kind, sampler_name="MockDWaveSampler")
        return self._get_path(kind, sampler_name=sampler.solver.name)

    def _save_embeddings(self, sampler: dimod.Sampler, embeddings: NDArray) -> None:
        """Save embedding data to disk."""
        cache_filename = self._make_filename("embedding", sampler=sampler)
        os.makedirs(cache_filename.parent, exist_ok=True)
        np.savetxt(cache_filename, embeddings, fmt="%d")

    def _load_embeddings(self, sampler: str) -> None:
        """Load embedding data."""
        filename = self._make_filename("embedding", sampler=sampler)
        self.embedding_list = np.atleast_2d(np.loadtxt(filename, dtype=int))

    def _save_orbits(self) -> None:
        """Save qubit and coupler orbits to disk."""
        cache_filename = self._make_filename("orbits")
        cache_dir = cache_filename.parent / cache_filename.stem
        os.makedirs(cache_dir, exist_ok=True)
        np.savetxt(cache_dir / "qubit_orbits.txt", self.qubit_orbits, fmt="%d")
        np.savetxt(cache_dir / "coupler_orbits.txt", self.coupler_orbits, fmt="%d")

    def _load_orbits(self) -> None:
        """Load qubit and coupler orbits."""
        cache_filename = self._make_filename("orbits")
        cache_dir = cache_filename.parent / cache_filename.stem

        self.qubit_orbits = np.loadtxt(cache_dir / "qubit_orbits.txt", dtype=int)
        self.coupler_orbits = np.loadtxt(cache_dir / "coupler_orbits.txt", dtype=int)

    def _get_instance_pathstring(self) -> str:
        """Construct an instance-specific pathstring.

        Generic version.  Let more complex classes, including inputs that are
        processor-dependent, redefine their pathstrings.  This will incorporate
        periodic dimensions, if available.
        """
        return type(self).__name__ + "/" + self._get_size_pathstring()

    def _get_size_pathstring(self) -> str:
        """Construct a size-specific pathstring including dimensions and periodicity."""
        return "size" + "x".join(f"{dim}{'p'*p}" for dim, p in zip(self.dimensions, self.periodic))

    def _make_networkx_graph(self) -> nx.Graph:
        """Construct a NetworkX graph reprensetation of the lattice."""
        graph = nx.Graph()
        for v in range(self.num_spins):
            graph.add_node(v)
        for u, v in self.edge_list:
            graph.add_edge(u, v)

        return graph

    def _optimize(self, bqm: dimod.BQM, **kwargs) -> tuple[float, NDArray, str]:
        return optimize(lattice=self, bqm=bqm, **kwargs)
