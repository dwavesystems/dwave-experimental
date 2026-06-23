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

__all__ = ['optimize', 'ExponentialBackoffSimulatedAnnealingSampler']


class ExponentialBackoffSimulatedAnnealingSampler(dimod.Sampler):
    """SA sampler that doubles num_sweeps until energy stops improving or a cap is hit.

    Starts at ``min_num_sweeps`` and doubles after each round that improves the
    best energy, stopping when no improvement is found or ``max_num_sweeps`` is
    exceeded.

    Args:
        max_num_sweeps: Upper bound on the number of sweeps per SA call. Once
            ``num_sweeps`` exceeds this value, the backoff loop terminates.
        min_num_sweeps: Initial number of sweeps for the first SA call.
    """

    properties = None
    parameters = None

    def __init__(self, max_num_sweeps=1024, min_num_sweeps=256):
        self.sampler = SimulatedAnnealingSampler()
        self.max_num_sweeps = max_num_sweeps
        self.min_num_sweeps = min_num_sweeps
        self.properties = self.sampler.properties.copy()
        self.parameters = self.sampler.parameters.copy()

    def sample(self, bqm, **parameters):
        num_sweeps = parameters.pop("num_sweeps", self.min_num_sweeps)
        num_reads = parameters.pop("num_reads", 256)

        best_energy = np.inf
        best_sampleset = None

        while num_sweeps <= self.max_num_sweeps:
            ss = self.sampler.sample(bqm, num_sweeps=num_sweeps, num_reads=num_reads, **parameters)
            energy = ss.first.energy

            if energy < best_energy:
                best_energy = energy
                best_sampleset = ss
                num_sweeps *= 2
            else:
                break

        best_sampleset.info["num_sweeps_exit"] = num_sweeps
        return best_sampleset


def optimize(
    lattice: Lattice,
    bqm: dimod.BQM,
    sampler: dimod.Sampler | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
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
        sampler: A dimod Sampler to use for optimization of the reference energy.
            If None, a default ExponentialBackoffSimulatedAnnealingSampler will
            be used.
        sampler_kwargs: Optional keyword arguments to pass to the provided
            sampler, such as ``num_reads`` and ``num_sweeps`` in the case of a
            SA sampler.

    Returns:
        A tuple containing the best energy found, the corresponding sample as a
        NumPy array, and a string indicating the optimization method used.
    """
    if sampler_kwargs is None:
        sampler_kwargs = {}

    if sampler is None:
        sampler = ExponentialBackoffSimulatedAnnealingSampler()

    reference_energy = np.inf
    reference_sample = None

    # If the lattice is embedded, we should optimize the logical lattice
    if hasattr(lattice, "logical_lattice"):
        _, logical_sample, _ = optimize(
            lattice.logical_lattice,
            lattice.unembed_bqm(bqm),
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
        )
        reference_sample = lattice.embed_sample(logical_sample)
        reference_energy = bqm.energy(reference_sample)

    sampleset = sampler.sample(bqm, **sampler_kwargs)
    best = sampleset.first

    if best.energy < reference_energy:
        sample = np.array([best.sample[v] for v in bqm.variables])
        return best.energy, sample, type(sampler).__name__

    return reference_energy, reference_sample, type(sampler).__name__
