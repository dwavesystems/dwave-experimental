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

"""For triangular order parameters"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import dimod
from dimod import BQM

from dwave.experimental.lattice_utils.observable.observable import Observable

__all__ = ['TriangularOP']

class TriangularOP(Observable):
    """For triangular lattices.  Unembeds if possible."""
    def evaluate(
        self,
        experiment: Experiment,
        bqm: BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:

        if hasattr(experiment.inst, "logical_lattice"):
            # If the lattice is an embedded lattice
            lbqm = experiment.inst.unembed_bqm(bqm)

            # unembed the sample set.
            lss = experiment.inst.unembed_sampleset(sample_set)
            triangular_sublattice = experiment.inst.logical_lattice.sublattice
        else:
            lbqm, lss = bqm, sample_set
            triangular_sublattice = experiment.inst.sublattice

        sample_array = dimod.as_samples(lss)[0]

        for edge in lbqm.quadratic:
            assert triangular_sublattice[edge[0]] != triangular_sublattice[edge[1]]

        sublattice_mags = np.zeros((sample_array.shape[0], 3), dtype=float)
        for sublattice in range(3):
            sublattice_mags[:, sublattice] = np.mean(
                sample_array[:, triangular_sublattice == sublattice], axis=1
            )

        angles = np.array(np.exp([0.0, 1.0j * 4 * np.pi / 3, 1.0j * 2 * np.pi / 3])).T
        op = np.matmul(sublattice_mags, angles).ravel() / np.sqrt(3)

        return op
