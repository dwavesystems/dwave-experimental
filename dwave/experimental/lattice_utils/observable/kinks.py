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
        """Compute the kink-kink correlator for 1D spin chains."""
        sample_array = dimod.as_samples(sample_set)[0]

        S = np.roll(sample_array, 1, axis=1)
        K = np.multiply(S, sample_array) == np.sign(experiment.param["energy_scale"])
        L = K.shape[-1]
        K = np.reshape(K, (-1, L))
        kink_density = np.mean(K)

        CKK = np.zeros((K.shape[-1],))

        for R in range(1, L):
            KR = np.roll(K, R, axis=1)
            CKK[R] = np.mean(np.multiply(K, KR)) - np.power(np.mean(K), 2)

        CKK /= kink_density**2

        return CKK
