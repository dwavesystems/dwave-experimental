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
from collections.abc import Hashable
from numbers import Integral

import dimod
#import matplotlib.pyplot as plt
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice.orbits import get_orbits
from dwave.experimental.lattice_utils.lattice.optimize import optimize

__all__ = ['Lattice']

class Lattice():
    """Base class for instances in LatQA."""

    def __init__(self, **kwargs):

        self.dimensions: tuple[int, ...] = kwargs["dimensions"]
        self.lattice_data_root: Path = kwargs.get("lattice_data_root", Path.cwd() / "lattice_data")

        self.periodic: tuple[bool, ...] = kwargs.get("periodic", [False] * len(self.dimensions))
        self.edge_list: list[tuple[Hashable, Hashable]] = list(self.generate_edges())
        if len(self.edge_list) > 0:
            self.num_spins: Integral = np.max(np.asarray(self.edge_list)) + 1

        self.num_edges: int = len(self.edge_list)
        self.orbit_type: str = kwargs.get("orbit_type", "singleton")
        self.initialize_orbits(
            kwargs.get("qubit_orbits"),
            kwargs.get("coupler_orbits"),
        )

    def embed_lattice(
        self,
        sampler: dimod.Sampler,
        try_to_load: bool = True,
        timeout: int = 10,
        data_root: str | Path | None = None,
        max_number_of_embeddings: int | None = None,
        min_number_of_embeddings: int = 1,
        exclude_qubits: list = [],
        **kwargs,
    ) -> None:
        """Find or load embeddings onto the sampler graph.

        Args:
            sampler: Sampler whose hardware graph is used as the target for embedding.
            try_to_load: If True, attempt to load embeddings from disk before
                trying to find them.
            timeout: Time limit for the embedding search, in seconds.
            data_root: Root directory for loading and saving embedding data.
            max_number_of_embeddings: Maximum number of embeddings to search for.
            min_number_of_embeddings: Minimum number of embeddings required to save.
            exclude_qubits: Qubits to remove from the sampler graph before searching
                for embeddings.
        """
        graph_bqm = dimod.to_networkx_graph(self.make_nominal_bqm())
        graph_sampler = sampler.to_networkx_graph()
        graph_sampler.remove_nodes_from(exclude_qubits)

        if try_to_load:
            try:
                self._load_embeddings(sampler, data_root)
                filename = self._make_filename(
                    "embedding",
                    data_root=data_root,
                    sampler=sampler,
                )
                print(f"Loaded embedding from file {filename}")
                return
            except FileNotFoundError:
                pass

        embedding_dicts = find_multiple_embeddings(
            graph_bqm,
            graph_sampler,
            max_num_emb=max_number_of_embeddings,
            embedder_kwargs={'timeout':timeout}
        )
        if not embedding_dicts:
            raise ValueError("No Embeddings Found")

        embeddings = np.stack([list(emb.values()) for emb in embedding_dicts])
        if len(embeddings) >= min_number_of_embeddings and np.prod(embeddings.shape):
            self._save_embeddings(sampler, embeddings, data_root=data_root)

        return

    def make_nominal_bqm(self, **kwargs) -> dimod.BQM:
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
        -``explicit``: use the orbit assignments provided via ``qubit_orbits``
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
                if hasattr(self.logical_lattice, "logical_lattice"):
                    raise NotImplementedError  # Nested embedded lattices not supported.
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
                print('Calculating orbits...')
                bqm = self.make_nominal_bqm()
                self.qubit_orbits, self.coupler_orbits = get_orbits(bqm, self.edge_list)
                self._save_orbits()

        elif self.orbit_type == "singleton":
            self.qubit_orbits = np.arange(self.num_spins)
            self.coupler_orbits = np.arange(self.num_edges)

        elif self.orbit_type == "explicit":
            if qubit_orbits is not None and coupler_orbits is not None:
                assert len(qubit_orbits) == self.num_spins
                assert len(coupler_orbits) == self.num_edges
            self.qubit_orbits = qubit_orbits
            self.coupler_orbits = coupler_orbits
        else:
            raise ValueError(
                f'Unknown orbit type {self.orbit_type}.' \
                'Must be "global", "standard", "singleton", or "explicit".'
            )

    def _get_path(
        self,
        root: Path | None,
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
        if root is None:
            root = Path(__file__).parent.parent / "data"

        if sampler_name is None:
            path = Path(root) / kind / class_subdir / self._get_size_pathstring()
        else:
            path = Path(root) / kind / class_subdir / sampler_name / self._get_size_pathstring()

        return path.with_suffix(".txt")

    def _make_filename(
        self,
        kind: str,
        sampler: dimod.Sampler | None = None,
        data_root: str | Path | None = None,
    ) -> Path:
        """Construct a data filename for the specified sampler and data type."""
        if data_root is None:
            data_root = self.lattice_data_root
        if sampler is None:
            return self._get_path(data_root, kind)

        if type(sampler).__name__ == "MockDWaveSampler":
            return self._get_path(data_root, kind, sampler_name="MockDWaveSampler")
        return self._get_path(data_root, kind, sampler_name=sampler.solver.name)

    def _save_embeddings(
        self,
        sampler: dimod.Sampler,
        embeddings: NDArray,
        data_root: str | Path | None = None,
    ) -> None:
        """Save embedding data to disk."""
        cache_filename = self._make_filename("embedding", sampler=sampler, data_root=data_root)
        os.makedirs(cache_filename.parent, exist_ok=True)
        np.savetxt(cache_filename, embeddings, fmt="%d")
        print(f"Saved {len(embeddings)} embeddings to file {cache_filename}")

    def _load_embeddings(self, sampler: str, data_root: str | Path | None = None, **kwargs) -> None:
        """Load embedding data."""
        filename = self._make_filename("embedding", sampler=sampler, data_root=data_root)
        self.embedding_list = np.atleast_2d(np.loadtxt(filename, dtype=int))

    def _save_orbits(self, data_root: str | Path | None = None) -> None:
        """Save qubit and coupler orbits to disk."""
        cache_filename = self._make_filename("orbits", data_root=data_root)
        cache_dir = cache_filename.parent / cache_filename.stem
        os.makedirs(cache_dir, exist_ok=True)
        np.savetxt(cache_dir / "qubit_orbits.txt", self.qubit_orbits, fmt="%d")
        np.savetxt(cache_dir / "coupler_orbits.txt", self.coupler_orbits, fmt="%d")
        print(f"Saved orbits to folder {cache_dir}")

    def _load_orbits(self, data_root: str | Path | None = None, **kwargs) -> None:
        """Load qubit and coupler orbits."""
        cache_filename = self._make_filename("orbits", data_root=data_root)
        cache_dir = cache_filename.parent / cache_filename.stem

        self.qubit_orbits = np.loadtxt(cache_dir / "qubit_orbits.txt", dtype=int)
        self.coupler_orbits = np.loadtxt(cache_dir / "coupler_orbits.txt", dtype=int)
        print(f'Loaded orbits from {cache_dir}')

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
