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

from __future__ import annotations
from typing import Any

from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
import dimod
from numpy.typing import NDArray

__all__ = ['optimize', 'optimize_increasing_sa_sweeps']


def optimize(
    lattice: Lattice,
    bqm: dimod.BQM,
    sa_kwargs: dict[str, Any] | None = None,
) -> tuple[float, NDArray, str]:
    """Return the best sample found by optimizing the BQM using simulated annealing.

    For ordinary lattices, this function applies simulated annealing directly to
    the BQM.

    For embedded lattices, this function first unembeds the BQM to get the logical
    BQM, optimizes the logical BQM, and then embeds the resulting sample back into
    the physical lattice. The energy of the embedded sample is then optimized using
    simulated annealing.

    Args:
        lattice: Lattice instance defining how the optimization should be performed.
            If the lattice is an EmbeddedLattice, the logical lattice will be
            optimized and the resulting sample will be embedded back into the
            physical lattice.
        bqm: The binary quadratic model to optimize.
        sa_kwargs: Optional keyword arguments to pass to the simulated annealing
            sampler, such as ``num_reads`` and ``num_sweeps``.

    Returns:
        A tuple containing the best energy found, the corresponding sample as a
        NumPy array, and a string indicating the optimization method used.
    """
    if sa_kwargs is None:
        sa_kwargs = {}

    # If the lattice is embedded, we should optimize the logical lattice
    if hasattr(lattice, "logical_lattice"):
        _, logical_sample, _ = optimize(
            lattice.logical_lattice, lattice.unembed_bqm(bqm), sa_kwargs=sa_kwargs
        )
        embedded_sample = lattice.embed_sample(logical_sample)
        embedded_energy = bqm.energy(embedded_sample)

        return optimize_increasing_sa_sweeps(bqm, embedded_energy, embedded_sample)

    # If no special case, just use SA.
    return optimize_increasing_sa_sweeps(bqm, sa_kwargs=sa_kwargs)


def optimize_increasing_sa_sweeps(
    bqm: dimod.BQM,
    reference_energy: float = np.inf,
    reference_sample: NDArray | None = None,
    sa_kwargs: dict[str, Any] | None = None,
) -> tuple[float, NDArray, str]:
    """Optimize a BQM with simulated annealing and increasing sweep counts.

    Args:
        bqm: The binary quadratic model to optimize.
        reference_energy: An initial energy to compare against. If the best energy
            found by SA is not better than this, the function will return without
            increasing the number of sweeps.
        reference_sample: An initial sample corresponding to the reference energy.
        sa_kwargs: Optional keyword arguments to pass to the simulated annealing
            sampler, such as ``num_reads`` and ``num_sweeps``. The ``num_sweeps``
            value will be overridden by this function as it increases exponentially.

    Returns:
        A tuple containing the best energy found, the corresponding sample as a
        NumPy array, and a string indicating the optimization method used.
    """
    sa = SimulatedAnnealingSampler()

    if sa_kwargs is None:
        sa_kwargs = {}
    num_sweeps = sa_kwargs.get("num_sweeps", 256)
    num_reads = sa_kwargs.get("num_reads", 256)

    while True:
        sample_set = sa.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
        energies = sample_set.data_vectors["energy"]
        best = np.argmin(energies)
        best_energy = energies[best]

        if best_energy < reference_energy:
            reference_energy = best_energy
            reference_sample = sample_set.record[best][0]
            num_sweeps *= 2
            if num_sweeps > 1e3:
                break
        else:
            break

    return reference_energy, reference_sample, "sa_exponential"
