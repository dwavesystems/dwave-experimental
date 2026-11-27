# Copyright 2026 D-Wave
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

import hashlib
import json
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator, Hashable
from pathlib import Path
from typing import Any

import dimod
import networkx as nx
import numpy as np
from dwave.system.testing import MockDWaveSampler
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice.optimize import optimize
from dwave.experimental.lattice_utils.lattice.orbits import get_orbits
from dwave.experimental.lattice_utils._database import decode_array, encode_array, open_database

__all__ = ['Lattice']


class Lattice(ABC):
    """An abstract base class for representing lattice geometries used in lattice-utils experiments.

    Subclasses are responsible for defining the lattice geometry itself. In particular,
    a subclass must:

    - Implement the ``generate_edges`` method, which yields the edges of the lattice as pairs
    - Initialize the ``self.num_spins`` attribute in the constructor, which is
        used by the base class
    - set any geometry-specific identifiers such as ``self.geometry_name``

    Args:
        dimensions: Tuple specifying the size of the lattice in each dimension.
        data_root: Root directory for file-based lattice data such as orbits.
            Embeddings are stored separately in ``data/lattice_utils.sqlite3``
            under the parent of ``data_root``.
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
        reference_energy_sampler: dimod.Sampler | None = None,
        reference_energy_sampler_kwargs: dict[str, Any] | None = None,
    ):
        self.dimensions = dimensions
        self.data_root = Path(data_root)
        self.reference_energy_sampler = reference_energy_sampler
        self.reference_energy_sampler_kwargs = reference_energy_sampler_kwargs

        self.periodic = periodic if periodic is not None else tuple(False for _ in dimensions)
        if len(self.periodic) != len(self.dimensions):
            raise ValueError(
                f"periodic and dimensions must have the same length: "
                f"got {len(self.periodic)} and {len(self.dimensions)}."
            )
        if not hasattr(self, "num_spins"):
            raise AttributeError(f"{type(self).__name__} subclass must initialize self.num_spins")

        self.edge_list: list[tuple[Hashable, Hashable]] = list(self.generate_edges())
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
    ) -> None:
        """Find or load embeddings onto the sampler graph.

        Check cached results with matching qubit exclusions first. Within each
        group, start with results that contain more embeddings. If a maximum is
        specified, count only up to that limit and prefer a result with exactly
        that many embeddings over one that needs trimming. Use the first cached
        result with at least ``min_number_of_embeddings`` compatible embeddings.

        Args:
            sampler: Sampler whose hardware graph is used to validate cached
                embeddings and find new ones.
            try_to_load: If True, try to reuse cached embeddings before searching.
                If False, skip the lookup. Newly found embeddings are saved either way.
            timeout: Time limit in seconds for each attempt to find a new embedding.
            max_number_of_embeddings: Maximum number of embeddings to find or load.
                If None, no maximum is imposed.
            min_number_of_embeddings: Minimum number of compatible embeddings needed
                to reuse a cached result. If a new search finds fewer than this
                number, raise ``ValueError``.
            exclude_qubits: Qubits to remove from the sampler graph before validating
                cached embeddings or searching for new ones.
        """
        if exclude_qubits is None:
            exclude_qubits = []
        if (
            max_number_of_embeddings is not None
            and max_number_of_embeddings < min_number_of_embeddings
        ):
            raise ValueError("max_number_of_embeddings must be at least min_number_of_embeddings.")

        graph_bqm = dimod.to_networkx_graph(self.make_bqm())
        graph_sampler = sampler.to_networkx_graph()
        graph_sampler.remove_nodes_from(exclude_qubits)

        if try_to_load:
            try:
                self._load_embeddings(
                    graph_sampler,
                    exclude_qubits=exclude_qubits,
                    min_number_of_embeddings=min_number_of_embeddings,
                    max_number_of_embeddings=max_number_of_embeddings,
                )
            except FileNotFoundError:
                warnings.warn("No usable cached embeddings found, computing new embeddings.")
            else:
                # Save the filtered subset under its own ID
                if self.embedding_set_id is None:
                    self._save_embeddings(
                        self.embedding_list,
                        sampler=sampler,
                        exclude_qubits=exclude_qubits,
                    )
                return

        embedding_dicts = find_multiple_embeddings(
            graph_bqm,
            graph_sampler,
            max_num_emb=max_number_of_embeddings,
            embedder_kwargs={'timeout': timeout},
        )
        num_embeddings = len(embedding_dicts)
        if num_embeddings < min_number_of_embeddings:
            raise ValueError(
                f"Found {num_embeddings} embedding(s) for {type(self).__name__}"
                f"(dimensions={self.dimensions}) on {type(sampler).__name__}, "
                f"but at least {min_number_of_embeddings} are required. "
                f"Try increasing timeout, reducing the lattice size, or using a "
                f"different solver/yield."
            )

        self.embedding_list = np.asarray(
            [[embedding[v] for v in range(self.num_spins)] for embedding in embedding_dicts]
        )
        self._save_embeddings(self.embedding_list, sampler=sampler, exclude_qubits=exclude_qubits)

    def make_bqm(self) -> dimod.BQM:
        """Construct a BQM with coupling strength values set to +1.

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

            if hasattr(self, "chain_nodes"):
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
                bqm = self.make_bqm()
                self.qubit_orbits, self.coupler_orbits = get_orbits(bqm, self.edge_list)
                self._save_orbits()

        elif self.orbit_type == "singleton":
            self.qubit_orbits = np.arange(self.num_spins)
            self.coupler_orbits = np.arange(self.num_edges)

        elif self.orbit_type == "explicit":
            if qubit_orbits is None or coupler_orbits is None:
                raise ValueError(
                    'orbit_type "explicit" requires both qubit_orbits and coupler_orbits.'
                )
            if len(qubit_orbits) != self.num_spins:
                raise ValueError(
                    f"qubit_orbits must have length {self.num_spins}, got {len(qubit_orbits)}."
                )
            if len(coupler_orbits) != self.num_edges:
                raise ValueError(
                    f"coupler_orbits must have length {self.num_edges}, got {len(coupler_orbits)}."
                )
            self.qubit_orbits = qubit_orbits
            self.coupler_orbits = coupler_orbits
        else:
            raise ValueError(
                f'Unknown orbit type {self.orbit_type}. '
                'Must be "global", "standard", "singleton", or "explicit".'
            )

    def make_networkx_graph(self) -> nx.Graph:
        """Construct a NetworkX graph representation of the lattice.

        Returns:
            A NetworkX graph where nodes correspond to spins and edges correspond to couplers.
        """
        graph = nx.Graph()
        for v in range(self.num_spins):
            graph.add_node(v)
        for u, v in self.edge_list:
            graph.add_edge(u, v)

        return graph

    def optimize(self, bqm: dimod.BQM) -> tuple[float, NDArray, str]:
        """Return the lowest energy sample by optimizing the BQM using the reference sampler.

        Returns:
            A tuple containing the best energy found, the corresponding sample as a
            NumPy array, and a string indicating the optimization method used.
        """
        return optimize(
            lattice=self,
            bqm=bqm,
            sampler=self.reference_energy_sampler,
            sampler_kwargs=self.reference_energy_sampler_kwargs,
        )

    def _get_path(
        self,
        kind: str,
        sampler_name: str | None = None,
        extra_subdir: str | Path | None = None,
    ) -> Path:
        """Construct a standardized file path for embedding or orbit data."""
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

        if isinstance(sampler, MockDWaveSampler):
            return self._get_path(kind, sampler_name="MockDWaveSampler")
        return self._get_path(kind, sampler_name=sampler.solver.name)

    def _lattice_key(self) -> str:
        identity = [
            1,
            int(self.num_spins),
            sorted(sorted(map(int, edge)) for edge in self.edge_list),
        ]
        return hashlib.sha256(json.dumps(identity).encode("ascii")).hexdigest()

    def _save_embeddings(
        self,
        embeddings: NDArray,
        *,
        sampler: dimod.Sampler,
        exclude_qubits: list | None = None,
    ) -> None:
        """Save an ordered embedding set and record its database ID."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.num_spins:
            raise ValueError("Invalid embedding array.")
        if not np.issubdtype(embeddings.dtype, np.integer):
            raise ValueError("Embeddings must contain integer labels.")

        sampler_name = (
            "MockDWaveSampler"
            if isinstance(sampler, MockDWaveSampler)
            else sampler.solver.name
        )
        excluded_qubits = sorted({int(qubit) for qubit in (exclude_qubits or [])})

        with open_database(self.data_root.parent / "data" / "lattice_utils.sqlite3") as connection:
            cursor = connection.execute(
                "INSERT INTO embedding_sets "
                "(lattice_key, geometry_name, dimensions, periodic, sampler_name, "
                "exclude_qubits, num_embeddings, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._lattice_key(),
                    self.geometry_name,
                    json.dumps(self.dimensions),
                    json.dumps(self.periodic),
                    sampler_name,
                    json.dumps(excluded_qubits),
                    len(embeddings),
                    encode_array(embeddings),
                ),
            )
            embedding_set_id = cursor.lastrowid

        self.embedding_set_id = embedding_set_id

    def _load_embeddings(
        self,
        hardware_graph: nx.Graph,
        *,
        exclude_qubits: list | None = None,
        min_number_of_embeddings: int = 1,
        max_number_of_embeddings: int | None = None,
    ) -> None:
        """Load embeddings from the SQLite cache that are compatible with the given hardware graph."""
        with open_database(self.data_root.parent / "data" / "lattice_utils.sqlite3") as connection:
            excluded = json.dumps(sorted({
                int(qubit) for qubit in (exclude_qubits or [])
            }))
            records = connection.execute(
                "SELECT embedding_set_id, data FROM embedding_sets "
                "WHERE lattice_key = :lattice_key "
                "ORDER BY (exclude_qubits = :excluded) DESC, "
                "MIN(num_embeddings, COALESCE(:maximum, num_embeddings)) DESC, "
                "(num_embeddings = :maximum) DESC, "
                "embedding_set_id DESC",
                {
                    "lattice_key": self._lattice_key(),
                    "excluded": excluded,
                    "maximum": max_number_of_embeddings,
                },
            )

            for embedding_set_id, encoded_embeddings in records:
                embeddings = decode_array(encoded_embeddings)
                if embeddings.ndim != 2 or embeddings.shape[1] != self.num_spins:
                    raise ValueError("Invalid cached embedding array.")
                if not np.issubdtype(embeddings.dtype, np.integer):
                    raise ValueError("Cached embeddings must contain integer labels.")
                if len(embeddings) < min_number_of_embeddings:
                    continue

                selected = []
                for index, row in enumerate(embeddings):
                    if len(set(row)) != self.num_spins:
                        continue
                    if not all(qubit in hardware_graph for qubit in row):
                        continue
                    if not all(
                        hardware_graph.has_edge(row[source], row[target])
                        for source, target in self.edge_list
                    ):
                        continue

                    selected.append(index)
                    if (
                        max_number_of_embeddings is not None
                        and len(selected) >= max_number_of_embeddings
                    ):
                        break

                if selected and len(selected) >= min_number_of_embeddings:
                    all_emb_kept = len(selected) == len(embeddings)
                    self.embedding_list = (embeddings if all_emb_kept else embeddings[selected])
                    self.embedding_set_id = embedding_set_id if all_emb_kept else None
                    return

        raise FileNotFoundError("No cached set has enough usable embeddings.")

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
