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

import unittest
import unittest.mock
import dimod
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system.testing import MockDWaveSampler

from dwave.system.temperatures import fluxbias_to_h
from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alphaPhi


class ShimmingMockSampler(MockDWaveSampler):
    """Replace the MockSampler by an MCMC sampler with sensitivity to flux_biases.

    We modify the MockSampler routine so that the sampling distribution is
    sensitive to flux_biases (linear fields are modified in proportion to
    flux_biases). Translation of flux_biases into Ising model linear fields
    uses a conversion factor appropriate to single-qubit freezeout.

    flux_biases_baseline can be added as a list of length
    self.properties['num_qubits'].
    This is zero by default, when non-zero the tutorial routines shim away
    the offset by analogy with the noise shimming in QPU solvers.

    Irrelevant warning messages on unsupported (QPU) parameters are suppressed.

    Replacing the default MockSampler sampler routine with Block Gibbs we
    allow a more realistic susceptibility.
    The default topology is chosen to match defect-free Advantage processor
    architectures.
    """

    def __init__(
        self,
        topology_type="zephyr",
        topology_shape=[3],
        flux_biases_baseline=None,
        use_SA=False,
    ):
        if use_SA:
            substitute_sampler = SimulatedAnnealingSampler()
            substitute_kwargs = {
                "beta_range": [0, 3],
                "beta_schedule_type": "linear",
                "num_sweeps": 100,
                "randomize_order": True,
                "proposal_acceptance_criteria": "Gibbs",
            }
        else:
            substitute_sampler = None
            substitute_kwargs = None
        super().__init__(
            topology_type=topology_type,
            topology_shape=topology_shape,
            substitute_sampler=substitute_sampler,
            substitute_kwargs=substitute_kwargs,
        )
        num_qubits = self.properties["num_qubits"]
        if flux_biases_baseline is None:
            self.flux_biases_baseline = [1e-5] * num_qubits
        else:
            self.flux_biases_baseline = flux_biases_baseline
        self.sampler_type = "mock"
        # Added to suppress warnings (not mocked, but irrelevant to tutorial)
        self.mocked_parameters.add("flux_drift_compensation")
        self.mocked_parameters.add("auto_scale")
        self.mocked_parameters.add("readout_thermalization")
        self.mocked_parameters.add("annealing_time")

    def sample(self, bqm, **kwargs):
        """Sample with flux_biases transformed to Ising model linear biases."""

        # Extract flux biases from kwargs (if provided)
        flux_biases = kwargs.pop("flux_biases", None)
        if self.flux_biases_baseline is not None:
            if flux_biases is None:
                flux_biases = self.flux_biases_baseline
            else:
                flux_biases = [
                    sum(fbs) for fbs in zip(flux_biases, self.flux_biases_baseline)
                ]

        # Adjust the BQM to include flux biases
        if flux_biases is None:
            ss = super().sample(bqm=bqm, **kwargs)
        else:
            _bqm = bqm.change_vartype("SPIN", inplace=False)
            flux_to_h_factor = fluxbias_to_h()

            for v in _bqm.variables:
                bias = _bqm.get_linear(v)
                _bqm.set_linear(v, bias + flux_to_h_factor * flux_biases[v])

            ss = super().sample(bqm=_bqm, **kwargs)

            ss.change_vartype(bqm.vartype)

            ss = dimod.SampleSet.from_samples_bqm(ss, bqm)  # energy of bqm, not _bqm

        return ss


class FluxBiases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler = ShimmingMockSampler()

    def test_sampler_called(self):
        with unittest.mock.patch.object(self.sampler, "sample") as m:
            bqm = dimod.BinaryQuadraticModel("SPIN").from_ising({0: 1}, {})
            fb, fbh, mh = shim_flux_biases(bqm, self.sampler)
            m.assert_called()

        self.assertIsInstance(fb, list)
        self.assertEqual(len(fb), self.sampler.properties["num_qubits"])
        self.assertIsInstance(fbh, dict)
        self.assertIsInstance(mh, dict)
        self.assertSetEqual(set(mh.keys()), set(fbh.keys()))
        self.assertSetEqual(set(mh.keys()), set(bqm.variables))

    def test_flux_params(self):
        """Check parameters in = parameters out for empty learning_schedule or convergence test"""
        nv = 10
        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {i: 1 for i in range(nv)}, {}
        )
        sampler = ShimmingMockSampler()

        val = 1.1
        sampling_params = {
            "num_reads": 1,
            "flux_biases": [val] * sampler.properties["num_qubits"],
        }

        # Defaults, with initialization
        fb, fbh, mh = shim_flux_biases(bqm, sampler, sampling_params=sampling_params)
        self.assertTrue(all(x == y for x, y in zip(fb, sampling_params["flux_biases"])))
        self.assertEqual(sum(x != val for x in fb), nv)
        self.assertEqual(nv, len(fbh))
        self.assertEqual(nv, len(mh))

        # Check shimmed_variables selection works
        sampling_params = {
            "num_reads": 1,
            "flux_biases": [val] * sampler.properties["num_qubits"],
        }
        shimmed_variables = list(range(nv)[::2])
        fb, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            shimmed_variables=shimmed_variables,
        )
        self.assertTrue(all(x == y for x, y in zip(fb, sampling_params["flux_biases"])))
        self.assertEqual(sum(x != val for x in fb), len(shimmed_variables))
        self.assertEqual(nv // 2, len(shimmed_variables))

        # No movement if no updates:
        sampling_params = {
            "num_reads": 1,
            "flux_biases": [val] * sampler.properties["num_qubits"],
        }
        fb, fbh, mh = shim_flux_biases(
            bqm, sampler, sampling_params=sampling_params, learning_schedule=[]
        )  # , shimmed_variables, learning_schedule, convergence_test, symmetrize_experiments
        self.assertTrue(all(x == y for x, y in zip(fb, sampling_params["flux_biases"])))
        self.assertTrue(all(x == val for x in fb))
        # No movement if converged:
        fb, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            convergence_test=lambda x, y: True,
        )
        self.assertTrue(all(x == y for x, y in zip(fb, sampling_params["flux_biases"])))
        self.assertTrue(all(x == val for x in fb))

        # Symmetrized experiment, twice as many magnetizations:
        for symmetrize_experiments in [True, False]:
            shimmed_variables = [1]
            learning_schedule = [1, 1 / 2]
            fb, fbh, mh = shim_flux_biases(
                bqm,
                sampler,
                sampling_params=sampling_params,
                learning_schedule=learning_schedule,
                shimmed_variables=shimmed_variables,
            )
            self.assertNotIn(0, fbh)
            self.assertTrue(len(learning_schedule), len(fbh[1]))
            self.assertTrue(
                len(learning_schedule), len(mh[1]) // (1 + int(symmetrize_experiments))
            )

    def test_qubit_freezeout_alphaPhi(self):
        x = qubit_freezeout_alphaPhi()
        y = qubit_freezeout_alphaPhi(2, 1, 1, 1)
        self.assertNotEqual(x, y)
        self.assertEqual(1, y)
