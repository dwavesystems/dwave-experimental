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

from copy import deepcopy
import unittest
import unittest.mock
import math

import dimod
from dwave.samplers import SteepestDescentSampler
from itertools import product

from dwave.experimental.multicolor_anneal import make_tds_x_schedules
from dwave.experimental.shimming import (
    shim_flux_biases,
    shim_tds_flux_biases,
    qubit_freezeout_alpha_phi,
)
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

    def test_flux_biases_params(self):
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
                sampling_params=deepcopy(sampling_params),
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
            exp_weights_per_update = {
                v: [1 / num_experiments] * num_experiments for v in shimmed_variables
            }
            fb2, fbh2, mh2 = shim_flux_biases(
                bqm,
                sampler,
                sampling_params=deepcopy(sampling_params),
                learning_schedule=learning_schedule,
                shimmed_variables=shimmed_variables,
                sampling_params_updates=sampling_params_updates,
                symmetrize_experiments=symmetrize_experiments,
                exp_weights_per_update=exp_weights_per_update,
            )
            self.assertTrue(all(math.isclose(a, b) for a, b in zip(fb, fb2)))
            self.assertTrue(
                all(
                    math.isclose(fbh[v][i], fbh2[v][i])
                    for v in fbh
                    for i in range(len(fbh[v]))
                )
            )
            self.assertTrue(
                all(
                    math.isclose(mh[v][i], mh2[v][i])
                    for v in mh
                    for i in range(len(mh[v]))
                )
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
            "x_polarizing_schedule": [[0.0, 0.0], [1.0, 0.0]],
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
            "x_polarizing_schedule": [[0.0, 0.0], [1.0, 0.0]],
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
            "x_polarizing_schedule": [[0.0, 0.0], [1.0, 0.0]],
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
            "x_polarizing_schedule": [[0.0, 0.0], [1.0, 1.0]],
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

    def _tds_setup(self):
        """Return a (sampler, bqm, target_lines, detector_lines,
        line_assignments, sampling_params) tuple for exercising
        :func:`shim_tds_flux_biases`."""
        sampler = ShimmingMockSampler(substitute_sampler=SteepestDescentSampler())
        edge = sampler.edgelist[0]
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {edge: -1})

        # Replace assignment by get_properties(qpu) when client available.
        target_c = 0.37
        n_lines = 3
        polarizing_line_info = {
            "minPolarizingTimeStep": 0.02,
            "depolarizationAnnealScheduleRequiredDelay": 2.0,
        }
        exp_feature_line_info = [
            {
                "annealingLine": i,
                "minAnnealingTimeStep": 0.01,
                "holdOvershootFor": 0.02,
                "minCOvershoot": -7.0,
                "maxCOvershoot": 8.0,
                "maxC": 3.0,
                "minC": -2.0,
                "scheduleDelayStep": 1e-06,
                "qubits": [edge[0]] if i == 0 else [edge[1]] if i == 1 else [],
            }
            for i in range(n_lines)
        ]
        exp_feature_info = [polarizing_line_info, exp_feature_line_info]

        line_assignments = {
            q: l
            for l, efi_l in enumerate(exp_feature_line_info)
            for q in efi_l["qubits"]
        }
        target_lines = {line_assignments[edge[0]]}
        detector_lines = {line_assignments[edge[1]]}
        x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
            exp_feature_info=exp_feature_info,
            target_lines=target_lines,
            target_c=target_c,
            detector_lines=detector_lines,
            source_lines=set(),
        )
        sampling_params = {
            "num_reads": 16,
            "x_anneal_schedules": x_anneal_schedules,
            "x_polarizing_schedule": x_polarizing_schedule,
            "x_schedule_delays": [0.0] * n_lines,
        }
        return (
            sampler,
            bqm,
            target_lines,
            detector_lines,
            line_assignments,
            sampling_params,
            exp_feature_line_info,  # line info part.
            target_c,
        )

    def test_shim_tds_flux_biases_basic_functionality(self):
        # See examples/ for more practical use case.

        # sampler can be replaced by DWaveSampler() when client available
        (
            sampler,
            bqm,
            target_lines,
            detector_lines,
            line_assignments,
            sampling_params,
            exp_feature_line_info,
            target_c,
        ) = self._tds_setup()

        cases = [
            ("explicit_sampling_params", sampling_params),
            ("default_sampling_params", None),
        ]
        for case_name, sp in cases:
            with self.subTest(case=case_name):
                flux_biases, fb_history, mag_history = shim_tds_flux_biases(
                    bqm,
                    sampler,
                    target_lines,
                    detector_lines,
                    line_assignments,
                    sampling_params=sp,
                    num_steps=2,
                    symmetrize_experiments=False,
                    exp_feature_line_info=exp_feature_line_info,
                    target_c=target_c,
                )

                self.assertIsInstance(flux_biases, list)
                self.assertEqual(len(flux_biases), sampler.properties["num_qubits"])
                self.assertSetEqual(set(fb_history.keys()), set(bqm.variables))
                self.assertSetEqual(set(mag_history.keys()), set(bqm.variables))
                # ``use_target_variables`` is True: two experiments per step.
                self.assertTrue(all(len(fb_history[v]) == 3 for v in bqm.variables))
                self.assertTrue(all(len(mag_history[v]) == 4 for v in bqm.variables))

    def test_shim_tds_flux_biases_detector_only(self):
        """When only detector variables are shimmed, no target/detector
        alternation is performed and a single experiment runs per step."""
        (
            sampler,
            bqm,
            target_lines,
            detector_lines,
            line_assignments,
            sampling_params,
            exp_feature_line_info,
            target_c,
        ) = self._tds_setup()
        shimmed_variables = {
            v for v in bqm.variables if line_assignments[v] in detector_lines
        }

        _, fb_history, mag_history = shim_tds_flux_biases(
            bqm,
            sampler,
            target_lines,
            detector_lines,
            line_assignments,
            sampling_params=sampling_params,
            num_steps=2,
            symmetrize_experiments=False,
            shimmed_variables=shimmed_variables,
        )

        self.assertSetEqual(set(fb_history.keys()), shimmed_variables)
        # Single experiment per step: fb_history len == num_steps + 1,
        # mag_history len == num_steps.
        self.assertTrue(all(len(fb_history[v]) == 3 for v in shimmed_variables))
        self.assertTrue(all(len(mag_history[v]) == 2 for v in bqm.variables))
