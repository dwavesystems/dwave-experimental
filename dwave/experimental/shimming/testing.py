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

from typing import Optional

import dimod
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.temperatures import fluxbias_to_h
from dwave.system.testing import MockDWaveSampler

__all__ = ["ShimmingMockSampler"]


class ShimmingMockSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with sensitivity to flux_biases.

    We modify the DWaveMockSampler routine so that the sampling distribution is
    sensitive to flux_biases (linear fields are modified in proportion to
    flux_biases). Under a freeze-out approximation flux_bias offset can be interpretted
    as a scaled offset on the Ising model field h, the scale is set to match a typical
    single-qubit freezeout value. The classical sampler for the resulting Ising
    models is chosen as a form of simulated annealing to which is appropriate for
    basic tests of shimming functionality.

    The sampled distribution is not a function of most QPU parameters, if an unsupported
    parameter is used these will raise warnings. Warnings can be suppressed by use of the
    mocked_parameters class variable.

    flux_biases_baseline can be added as a list of length
    self.properties['num_qubits'] to model a static processor noise.
    By default a baseline of 1e-5 Phi0 is used,
    in this way a shimming routine can be seen to systematically correct for this
    baseline.

    See the shimming tutorial and dwave-experimental tests for examples of usage.
    """

    def __init__(self, flux_biases_baseline: Optional[list[float]] = None, **kwargs):
        kwargs.setdefault("topology_type", "zephyr")
        kwargs.setdefault("topology_shape", [6, 4])

        kwargs.setdefault("substitute_sampler", SimulatedAnnealingSampler())
        kwargs.setdefault(
            "substitute_kwargs",
            {
                "beta_range": [0, 3],
                "beta_schedule_type": "linear",
                "num_sweeps": 100,
                "randomize_order": True,
                "proposal_acceptance_criteria": "Gibbs",
            },
        )
        super().__init__(**kwargs)
        self.parameters["x_polarizing_schedules"] = ["parameters"]
        num_qubits = self.properties["num_qubits"]
        if flux_biases_baseline is None:
            self.flux_biases_baseline = [1e-5] * num_qubits
        else:
            self.flux_biases_baseline = flux_biases_baseline
        self.sampler_type = "mock"
        self.mocked_parameters.add("flux_biases")
        self.mocked_parameters.add("x_polarizing_schedules")

    def sample(self, bqm, **kwargs):
        """Sample with flux_biases transformed to Ising model linear biases."""

        flux_biases = kwargs.pop("flux_biases", None)
        if self.flux_biases_baseline is not None:
            if flux_biases is None:
                flux_biases = self.flux_biases_baseline
            else:
                flux_biases = [
                    sum(fbs) for fbs in zip(flux_biases, self.flux_biases_baseline)
                ]

        if flux_biases is None:
            ss = self.substitute_sampler.sample(
                bqm=bqm, **kwargs
            )  # super() is too fastidious w.r.t. kwargs
        else:
            _bqm = bqm.change_vartype("SPIN", inplace=False)
            flux_to_h_factor = fluxbias_to_h()

            for v in _bqm.variables:
                bias = _bqm.get_linear(v)
                _bqm.set_linear(v, bias + flux_to_h_factor * flux_biases[v])

            ss = self.substitute_sampler.sample(
                bqm=_bqm, **kwargs
            )  # super() is too fastidious w.r.t. kwargs

            ss.change_vartype(bqm.vartype)

            ss = dimod.SampleSet.from_samples_bqm(ss, bqm)  # energy of bqm, not _bqm

        return ss
