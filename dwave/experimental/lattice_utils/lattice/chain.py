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

from collections.abc import Generator
from pathlib import Path
from typing import Any

from numpy.typing import NDArray
import dimod

from dwave.experimental.lattice_utils.lattice import Lattice

__all__ = ['Chain']


class Chain(Lattice):
    """One-dimensional chain lattice.

    This class represents a 1D chain of spins, where each spin is connected to
    its nearest neighbors. The chain can be periodic (forming a ring) or
    non-periodic (open chain) based on the ``periodic`` parameter.

    Args:
        dimensions: One-element tuple giving the number of spins in the chain.
        periodic: One-element tuple indicating whether the chain is periodic.
        data_root: A string or Path to the root directory for storing lattice data.
        orbit_type: A string specifying the type of orbits to compute for the
            lattice.
        qubit_orbits: Explicit qubit orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of edges in the lattice.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int],
        data_root: str | Path,
        periodic: tuple[bool] = (True,),
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
        reference_energy_sampler: dimod.Sampler | None = None,
        reference_energy_sampler_kwargs: dict[str, Any] | None = None,
    ):
        self.geometry_name = "Chain"
        self.num_spins = dimensions[0]
        if len(dimensions) != 1:
            raise ValueError(f"Chain requires dimensions of length 1, got {len(dimensions)}.")

        super().__init__(
            dimensions=dimensions,
            periodic=periodic,
            data_root=data_root,
            orbit_type=orbit_type,
            qubit_orbits=qubit_orbits,
            coupler_orbits=coupler_orbits,
            reference_energy_sampler=reference_energy_sampler,
            reference_energy_sampler_kwargs=reference_energy_sampler_kwargs,
        )

    def generate_edges(self) -> Generator[tuple[int, int]]:
        """Yield edges for a 1D chain lattice.

        Returns:
            A generator of tuples, where each tuple represents an edge between
            two spins in the chain.
        """
        n = self.dimensions[0]
        for i in range(n - 1):
            yield (i, i + 1)

        if self.periodic[0] and n > 1:
            yield (n - 1, 0)
