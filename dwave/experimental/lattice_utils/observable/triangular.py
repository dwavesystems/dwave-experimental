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
import numpy as np
from numpy.typing import NDArray
import dimod
from dimod import BQM

from dwave.experimental.lattice_utils.observable.observable import Observable

__all__ = ['TriangularOP']


class TriangularOP(Observable):
    """A class for calculating the order parameter of triangular lattices."""

    def evaluate(
        self,
        experiment: Experiment,
        bqm: BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        """Calculate the triangular lattice order parameter.

        This observable uses the three-sublattice complex order parameter described in
        `King et al. (2023) <https://doi.org/10.3389/fcomp.2023.1238988>_`.

        Args:
            experiment: The experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: The samples on which to compute the order parameter.

        Returns:
            A numpy array containing the order parameter values for each sample.
        """

        # If the lattice is an embedded lattice then the BQM and sampleset must be unembedded.
        if hasattr(experiment.inst, "logical_lattice"):
            lbqm = experiment.inst.unembed_bqm(bqm)

            lss = experiment.inst.unembed_sampleset(sample_set)
            triangular_sublattice = experiment.inst.logical_lattice.sublattice
        else:
            lbqm, lss = bqm, sample_set
            triangular_sublattice = experiment.inst.sublattice

        sample_array = dimod.as_samples(lss)[0]

        for u, v in lbqm.quadratic:
            if triangular_sublattice[u] == triangular_sublattice[v]:
                raise ValueError(
                    "Invalid triangular sublattice assignment: edge "
                    f"({u}, {v}) connects nodes in the same sublattice"
                )

        sublattice_mags = np.zeros((sample_array.shape[0], 3), dtype=float)
        for sublattice in range(3):
            sublattice_mags[:, sublattice] = np.mean(
                sample_array[:, triangular_sublattice == sublattice], axis=1
            )

        angles = np.array(np.exp([0.0, 1.0j * 4 * np.pi / 3, 1.0j * 2 * np.pi / 3])).T
        order_parameter = np.matmul(sublattice_mags, angles).ravel() / np.sqrt(3)

        return order_parameter
