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
from dwave.samplers import SteepestDescentSampler

from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alpha_phi
from dwave.experimental.shimming.testing import ShimmingMockSampler


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

        sampler = ShimmingMockSampler(substitute_sampler=SteepestDescentSampler())

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
            self.assertEqual(len(learning_schedule) + 1, len(fbh[1]))
            self.assertTrue(
                len(learning_schedule), len(mh[1]) // (1 + int(symmetrize_experiments))
            )

    def test_qubit_freezeout_alpha_phi(self):
        x = qubit_freezeout_alpha_phi()
        y = qubit_freezeout_alpha_phi(2, 1, 1, 1)
        self.assertNotEqual(x, y)
        self.assertEqual(1, y)
