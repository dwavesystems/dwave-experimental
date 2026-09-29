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

from __future__ import annotations

from typing import TYPE_CHECKING

import dimod
import numpy as np
from dimod import BQM

from dwave.experimental.lattice_utils.observable.observable import Observable

if TYPE_CHECKING:
    from dwave.experimental.lattice_utils.experiment.experiment import Experiment


class EdwardsAndersonOverlap(Observable):
    """A class for calculating the Edwards-Anderson overlap order parameter."""

    def evaluate(
        self,
        experiment: Experiment,
        bqm: BQM,
        sample_set: dimod.SampleSet,
    ) -> tuple[float, float, float]:
        """Calculate the Edwards-Anderson overlap order parameter.

        Args:
            experiment: The experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: The samples on which to compute the order parameter.

        Returns:
            A tuple ``(mean_abs_q, mean_q2, mean_q4)`` containing the mean
            absolute Edwards-Anderson overlap, mean squared overlap, and mean
            fourth-power overlap, respectively, over all unique pairs of samples.
        """
        if len(sample_set) < 2:
            raise ValueError("EdwardsAndersonOverlap requires at least two samples.")
        if len(sample_set.variables) == 0:
            raise ValueError(
                "EdwardsAndersonOverlap requires samples containing at least one spin."
            )

        sample_array = dimod.as_samples(sample_set)[0].astype(int)

        overlap = sample_array @ sample_array.T / sample_array.shape[1]
        overlap_vector = overlap[np.triu_indices_from(overlap, k=1)]

        mean_abs_q = np.mean(np.abs(overlap_vector))
        mean_q2 = np.mean(overlap_vector**2)
        mean_q4 = np.mean(overlap_vector**4)

        return mean_abs_q, mean_q2, mean_q4
