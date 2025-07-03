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

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from dimod.typing import Variable, Bias
from dwave.system import DWaveSampler

__all__ = ['shim_flux_biases']


def shim_flux_biases(h: Mapping[Variable, Bias],
                     J: Mapping[tuple[Variable, Variable], Bias],
                     embedding: Mapping[Variable, Sequence[Variable]],
                     sampler: DWaveSampler,
                     sampling_params: Optional[dict[str, Any]] = None,
                     flux_biases: Optional[list[Bias]] = None,
                     learning_rate: float = 1e-5,
                     num_steps: int = 20,
                     ) -> list[Bias]:

    if any(h.values()):
        raise ValueError("Zero linear biases required for shimming")

    if flux_biases is None:
        flux_biases = [0] * sampler.properties['num_qubits']

    if sampling_params is None:
        sampling_params = {}
    sampling_params.update(answer_mode="raw", auto_scale=False)

    # TODO: implement shimming
    for step in range(num_steps):
        sampler.sample_ising(h, J, flux_biases=flux_biases, **sampling_params)

    return flux_biases
