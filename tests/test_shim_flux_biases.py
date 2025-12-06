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
import math

import dimod
from dwave.samplers import SteepestDescentSampler
from itertools import product

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
                symmetrize_experiments=symmetrize_experiments,
            )
            self.assertNotIn(0, fbh)
            self.assertEqual(len(learning_schedule) + 1, len(fbh[1]))
            num_signed_experiments = 1 + int(symmetrize_experiments)
            self.assertEqual(
                len(learning_schedule) * num_signed_experiments, len(mh[1])
            )
            shimmed_variables = [1, 2]
            sampling_params_updates = [{"num_reads": 4}, {}, {"num_reads": 1}]
            num_experiments = len(sampling_params_updates) * num_signed_experiments
            fb, fbh, mh = shim_flux_biases(
                bqm,
                sampler,
                sampling_params=sampling_params,
                learning_schedule=learning_schedule,
                shimmed_variables=shimmed_variables,
                sampling_params_updates=sampling_params_updates,
                symmetrize_experiments=symmetrize_experiments,
            )
            self.assertNotIn(0, fbh)
            self.assertEqual(len(learning_schedule) + 1, len(fbh[1]))
            self.assertEqual(
                len(learning_schedule) * num_experiments,
                len(mh[1]),
            )
        # Check num_steps:
        for num_steps in [0, 4]:
            bqm = dimod.BinaryQuadraticModel("SPIN").from_ising({0: 1}, {})

            flux_biases, fbh, mh = shim_flux_biases(bqm, sampler, num_steps=num_steps)
            self.assertEqual(len(fbh[0]), num_steps + 1)
            self.assertEqual(len(mh[0]), num_steps * 2)
        # Check beta, alpha:
        res = []

        for alpha, beta_hypergradient in product([1e-6, 1e-7], [0.45, 0.004]):
            flux_biases0, fbh, mh = shim_flux_biases(
                bqm,
                sampler,
                beta_hypergradient=beta_hypergradient,
                alpha=alpha,
                symmetrize_experiments=False,
            )
            self.assertTrue(all(m == -1.0 for m in mh[0]))
            # No agreement due to update difference
            for r in res:
                self.assertFalse(all(math.isclose(a, b) for a, b in zip(fbh[0], r)))
            res.append(fbh[0])

        # Agreement (sampler is deterministic)
        _, fbh, _ = shim_flux_biases(
            bqm,
            sampler,
            beta_hypergradient=beta_hypergradient,
            alpha=alpha,
            symmetrize_experiments=False,
        )
        self.assertTrue(all(math.isclose(a, b) for a, b in zip(fbh[0], res[-1])))

    def test_symmetry_detection(self):

        sampler = ShimmingMockSampler(substitute_sampler=SteepestDescentSampler())
        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {sampler.nodelist[0]: 0}, {}
        )
        nq = sampler.properties["num_qubits"]
        sampling_params = {
            "flux_biases": [0] * nq,
            "x_polarizing_schedules": [[[0.0, 0.0], [1.0, 0.0]]] * 6,
        }
        _, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            num_steps=1,
            symmetrize_experiments=True,
        )

        self.assertTrue(
            len(fbh[0]) - 1 == len(mh[0]) == 1,
            "Should detect symmetry, 1 experiment per iteration",
        )

        # NB: Parameters are not checked for validity beyond their impact on symmetry break:
        bqmB = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {sampler.nodelist[0]: 1}, {}
        )
        _, fbh, mh = shim_flux_biases(
            bqmB,
            sampler,
            sampling_params=sampling_params,
            num_steps=1,
            symmetrize_experiments=True,
        )
        self.assertTrue(
            len(fbh[0]) - 1 == len(mh[0]) // 2 == 1,
            "Should detect asymmetry, 2 experiment per iteration",
        )

        sampling_params = {
            "flux_biases": [0] * nq,
            "x_polarizing_schedules": [[[0.0, 0.0], [1.0, 0.0]]] * 6,
            "initial_state": {0: 1},
        }
        _, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            num_steps=1,
            symmetrize_experiments=True,
        )
        self.assertTrue(
            len(fbh[0]) - 1 == len(mh[0]) // 2 == 1,
            "Should detect asymmetry, 2 experiment per iteration",
        )

        sampling_params = {
            "flux_biases": [1] * nq,
            "x_polarizing_schedules": [[[0.0, 0.0], [1.0, 0.0]]] * 6,
        }
        _, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            num_steps=1,
            symmetrize_experiments=True,
        )
        self.assertTrue(
            len(fbh[0]) - 1 == len(mh[0]) // 2 == 1,
            "Should detect asymmetry, 2 experiment per iteration",
        )

        sampling_params = {
            "flux_biases": [0] * nq,
            "x_polarizing_schedules": [[[0.0, 0.0], [1.0, 1.0]]] * 6,
        }
        _, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            num_steps=1,
            symmetrize_experiments=True,
        )
        self.assertTrue(
            len(fbh[0]) - 1 == len(mh[0]) // 2 == 1,
            "Should detect asymmetry, 2 experiment per iteration",
        )

    def test_qubit_freezeout_alpha_phi(self):
        x = qubit_freezeout_alpha_phi()
        y = qubit_freezeout_alpha_phi(2, 1, 1, 1)
        self.assertNotEqual(x, y)
        self.assertEqual(1, y)
