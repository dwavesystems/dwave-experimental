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
import dimod
from dimod import BQM, SampleSet
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.observable.observable import Observable

__all__ = ['KinkKinkCorrelator']


class KinkKinkCorrelator(Observable):
    """For 1D chains."""

    def __init__(self):
        super().__init__()

    def evaluate(self, experiment: Experiment, bqm: BQM, sample_set: SampleSet) -> NDArray:
        """Compute the kink-kink correlator for 1D spin chains.

        Args:
            experiment: The experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: The samples on which to compute the kink-kink correlator.

        Returns:
            A numpy array containing the kink-kink correlator values for each sample.
        """
        samples = dimod.as_samples(sample_set)[0]

        shifted_samples = np.roll(samples, 1, axis=1)
        kink_mask = shifted_samples * samples == np.sign(experiment.param["energy_scale"])
        chain_length = kink_mask.shape[-1]
        kink_mask = np.reshape(kink_mask, (-1, chain_length))
        kink_density = np.mean(kink_mask)

        kink_kink_correlator = np.zeros((kink_mask.shape[-1],))

        mean_kink = np.mean(kink_mask)
        for distance in range(1, chain_length):
            shifted_kink_mask = np.roll(kink_mask, distance, axis=1)
            kink_kink_correlator[distance] = np.mean(kink_mask * shifted_kink_mask) - mean_kink ** 2

        kink_kink_correlator /= kink_density ** 2

        return kink_kink_correlator
