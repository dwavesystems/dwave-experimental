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

from collections.abc import Generator
from pathlib import Path

import dimod
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice.lattice import Lattice
from dwave.experimental.lattice_utils.lattice.spin_glass import SpinGlassMixin
from dwave.experimental.lattice_utils.lattice.embedded_lattice import EmbeddedLattice

__all__ = ['Cubic', 'DimerizedCubic', 'CubicSpinGlass', 'DimerizedCubicSpinGlass']


class Cubic(Lattice):
    """Three-dimensional cubic lattice with nearest-neighbor couplings.

    This class represents a 3D cubic lattice, where each node is connected to
    its six nearest neighbors (except at boundaries, if not periodic).

    Spins are indexed by x + y * length_x + z * length_x * length_y.
    Each axis may have open or periodic boundaries; the default is periodic
    only along z.

    Args:
        dimensions: Number of logical sites along the x, y, and z axes.
        periodic: Whether each corresponding axis has periodic boundaries.
        data_root: Root directory for lattice data.
        orbit_type: Method for determining qubit and coupler orbits.
            Must be one of "global", "standard", "singleton", or "explicit". See
            ``initialize_orbits`` for details.
        qubit_orbits: Explicit qubit orbit labels, used only when
            ``orbit_type == "explicit"``. Must have length equal to the number
            of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when
            ``orbit_type == "explicit"``. Must have length equal to the number
            of edges in the lattice.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int, int, int],
        periodic: tuple[bool, bool, bool] = (False, False, True),
        data_root: str | Path,
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
    ):
        if len(dimensions) != 3:
            raise ValueError(f"Cubic requires dimensions of length 3, got {len(dimensions)}.")

        self.geometry_name = "Cubic"
        self.num_spins = int(np.prod(dimensions))

        super().__init__(
            dimensions=dimensions,
            periodic=periodic,
            data_root=data_root,
            orbit_type=orbit_type,
            qubit_orbits=qubit_orbits,
            coupler_orbits=coupler_orbits,
        )

    def generate_edges(self) -> Generator[tuple[int, int]]:
        """Yield edges for a cubic lattice with the given dimensions and periodicity.

        Returns:
            A generator of pairs of integer spin labels. Each label is the
            linear index of a lattice coordinate ``(x, y, z)``.
        """
        length_x, length_y, length_z = self.dimensions

        def index(x: int, y: int, z: int) -> int:
            """Return the linear index for lattice coordinates ``(x, y, z)``."""
            return x + y * length_x + z * length_x * length_y

        # x-direction edges
        for z in range(length_z):
            for y in range(length_y):
                for x in range(length_x):
                    if x < length_x - 1:
                        yield index(x, y, z), index(x + 1, y, z)
                    elif x == length_x - 1 and self.periodic[0] and length_x > 2:
                        yield index(x, y, z), index(0, y, z)

        # y-direction edges
        for z in range(length_z):
            for y in range(length_y):
                for x in range(length_x):
                    if y < length_y - 1:
                        yield index(x, y, z), index(x, y + 1, z)
                    elif y == length_y - 1 and self.periodic[1] and length_y > 2:
                        yield index(x, y, z), index(x, 0, z)

        # z-direction edges
        for z in range(length_z):
            for y in range(length_y):
                for x in range(length_x):
                    if z < length_z - 1:
                        yield index(x, y, z), index(x, y, z + 1)
                    elif z == length_z - 1 and self.periodic[2] and length_z > 2:
                        yield index(x, y, z), index(x, y, 0)


class CubicSpinGlass(SpinGlassMixin, Cubic):
    """Three-dimensional cubic lattice with seeded spin-glass couplings."""


class DimerizedCubic(EmbeddedLattice):
    """Dimerized cubic lattice with two spins per unit cell.

    This class represents a dimerized version of the 3D cubic lattice,
    where each node in the logical lattice is represented by a chain of two spins
    in the physical lattice. If a logical lattice is not provided, a logical
    lattice is constructed with ``singleton`` orbit type, since the orbits of
    the logical lattice are not relevant to the physical lattice.

    Args:
        dimensions: Number of logical sites along the x, y, and z axes.
        periodic: Whether each corresponding axis has periodic boundaries.
        data_root: Root directory for lattice data.
        orbit_type: Method for determining qubit and coupler orbits. Must be one of "global",
            "standard", "singleton", or "explicit". See ``initialize_orbits`` for details.
        qubit_orbits: Explicit qubit orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of edges in the lattice.
        chain_strength: The strength of the couplings within each chain.
        logical_lattice: Optional logical lattice instance to embed. If not
            provided, a ``Cubic`` lattice is constructed from the other
            initialization arguments.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int, int, int],
        periodic: tuple[bool, bool, bool] = (False, False, True),
        data_root: str | Path,
        orbit_type: str = "standard",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
        chain_strength: float = 2,
        logical_lattice: Lattice | None = None,
    ):
        if len(dimensions) != 3:
            raise ValueError(
                f"DimerizedCubic requires dimensions of length 3, got {len(dimensions)}."
            )

        self.geometry_name = "DimerizedCubic"
        self.num_spins = 2 * int(np.prod(dimensions))

        if logical_lattice is None:
            logical_lattice = Cubic(
                dimensions=dimensions,
                periodic=periodic,
                data_root=data_root,
                orbit_type="singleton",
            )

        chain_nodes = {v: (v, v + self.num_spins // 2) for v in range(self.num_spins // 2)}

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

    def get_chain_connectivity(
        self,
        u: int,
        v: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return the connectivity for a chain or edge in the logical lattice.

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
        if v is None or u == v:  # Internal chain connectivity
            return ((0, 1),)

        length_x, length_y, _ = self.dimensions
        if (u - v) % length_x:
            return ((0, 0),)  # x coupling
        if (u - v) % (length_x * length_y):
            return ((1, 1),)  # y coupling
        return (0, 1), (1, 0)  # z coupling


class DimerizedCubicSpinGlass(DimerizedCubic):
    """Dimerized cubic lattice with seeded spin-glass couplings.

    Constructs a cubic spin-glass logical lattice with singleton orbits. For
    each seed, its logical BQM is mapped onto the two-spin chains of the physical
    dimerized lattice.

    Args:
        dimensions: Numbers of logical sites along the x, y, and z axes.
        periodic: Whether each corresponding axis has periodic boundaries.
        data_root: Root directory for lattice data.
        orbit_type: Method for determining qubit and coupler orbits. Must be one of "global",
            "standard", "singleton", or "explicit". See ``initialize_orbits`` for details.
        qubit_orbits: Explicit qubit orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of spins in the lattice.
        coupler_orbits: Explicit coupler orbit labels, used only when ``orbit_type == "explicit"``.
            Must have length equal to the number of edges in the lattice.
        chain_strength: The strength of the couplings within each chain.
        precision: 1 for bimodal ±1 couplings, or a positive even integer
            specifying the number of uniformly distributed signed values.
    """

    def __init__(
        self,
        *,
        dimensions: tuple[int, int, int],
        periodic: tuple[bool, bool, bool] = (False, False, True),
        data_root: str | Path,
        orbit_type: str = "singleton",
        qubit_orbits: NDArray | None = None,
        coupler_orbits: NDArray | None = None,
        chain_strength: float = 2,
        precision: int = 1,
    ):
        logical_lattice = CubicSpinGlass(
            dimensions=dimensions,
            periodic=periodic,
            data_root=data_root,
            orbit_type="singleton",
            precision=precision,
        )

        super().__init__(
            dimensions=dimensions,
            periodic=periodic,
            data_root=data_root,
            orbit_type=orbit_type,
            qubit_orbits=qubit_orbits,
            coupler_orbits=coupler_orbits,
            chain_strength=chain_strength,
            logical_lattice=logical_lattice,
        )

    def make_bqm(self, seed: int = 0) -> dimod.BQM:
        """Construct the physical spin-glass BQM for the given disorder seed."""
        return self.embed_bqm(self.logical_lattice.make_bqm(seed=seed))

    def _get_instance_pathstring(self) -> str:
        """Return a string representing the path to the instance data for this lattice."""
        return f"{super()._get_instance_pathstring()}/{self.logical_lattice.glass_string}"
