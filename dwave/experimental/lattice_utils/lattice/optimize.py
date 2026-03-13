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
import time
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
    """Should take the lattice, from which we can infer the appropriate action."""
    if sa_kwargs is None:
        sa_kwargs = {}

    # If the lattice is embedded, we should optimize the logical lattice
    if hasattr(lattice, "logical_lattice"):
        _, logical_sample, _ = optimize(
            lattice.logical_lattice,
            lattice.unembed_bqm(bqm),
            sa_kwargs=sa_kwargs
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
    """Run SA with exponentially increasing sweep counts until no improvement is achieved."""
    start = time.time()
    sa = SimulatedAnnealingSampler()

    if sa_kwargs is None:
        sa_kwargs = {}
    num_sweeps = sa_kwargs.get("num_sweeps", 256)
    num_reads = sa_kwargs.get("num_reads", 256)

    while True:
        print(f"Running SA with {num_sweeps} sweeps;", end=" ")
        sample_set = sa.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps)
        energies = sample_set.data_vectors["energy"]
        best = np.argmin(energies)
        best_energy = energies[best]
        print(f"best energy is {best_energy}. ")

        if best_energy < reference_energy:
            reference_energy = best_energy
            reference_sample = sample_set.record[best][0]
            num_sweeps *= 2
            if num_sweeps > 1e3:
                break
        else:
            break

    end = time.time()
    print(f"took {end - start:2f}s")
    return reference_energy, reference_sample, "sa_exponential"
