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

import math
import unittest
import unittest.mock
from copy import deepcopy
from itertools import product

import dimod
from dwave.experimental.multicolor_anneal import make_tds_x_schedules
from dwave.experimental.shimming import (
    shim_flux_biases,
    shim_tds_flux_biases,
    qubit_freezeout_alpha_phi,
)
from dwave.experimental.shimming.flux_biases import (
    shim_linewise_flux_biases,
    _extend_history,
    _neutralize_line_schedule,
)
from dwave.experimental.shimming.testing import ShimmingMockSampler
from dwave.samplers import SteepestDescentSampler


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
        """With ``sampling_params_in_place=False`` the caller's ``sampling_params``
        is left unmutated; verified against a pre-call deep copy. Also checks the
        no-movement cases (empty learning_schedule or convergence test)."""
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
        sp_before = deepcopy(sampling_params)
        fb, fbh, mh = shim_flux_biases(
            bqm, sampler, sampling_params=sampling_params, sampling_params_in_place=False
        )
        self.assertEqual(sampling_params, sp_before)
        self.assertEqual(sum(x != val for x in fb), nv)
        self.assertEqual(nv, len(fbh))
        self.assertEqual(nv, len(mh))

        # Check shimmed_variables selection works
        sampling_params = {
            "num_reads": 1,
            "flux_biases": [val] * sampler.properties["num_qubits"],
        }
        shimmed_variables = list(range(nv)[::2])
        sp_before = deepcopy(sampling_params)
        fb, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            shimmed_variables=shimmed_variables,
            sampling_params_in_place=False,
        )
        self.assertEqual(sampling_params, sp_before)
        self.assertEqual(sum(x != val for x in fb), len(shimmed_variables))
        self.assertEqual(nv // 2, len(shimmed_variables))

        # No movement if no updates:
        sampling_params = {
            "num_reads": 1,
            "flux_biases": [val] * sampler.properties["num_qubits"],
        }
        sp_before = deepcopy(sampling_params)
        fb, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            learning_schedule=[],
            sampling_params_in_place=False,
        )
        self.assertEqual(sampling_params, sp_before)
        self.assertTrue(all(x == val for x in fb))

        # No movement if converged:
        sp_before = deepcopy(sampling_params)
        fb, fbh, mh = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=sampling_params,
            convergence_test=lambda x, y: True,
            sampling_params_in_place=False,
        )
        self.assertEqual(sampling_params, sp_before)
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
                v: [1] * num_experiments for v in shimmed_variables
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
            len(fbh[0]) - 1 == len(mh[0]) == 1,
            "Flux biases do not trigger symmetrization by default",
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

    def test_flux_bias_symmetry_detection(self):
        sampler = ShimmingMockSampler(substitute_sampler=SteepestDescentSampler())
        shimmed = sampler.nodelist[0]
        unshimmed = sampler.nodelist[1]
        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {shimmed: 0, unshimmed: 0}, {}
        )
        flux_biases = [0] * sampler.properties["num_qubits"]
        flux_biases[shimmed] = 0.2
        flux_biases[unshimmed] = 0.3
        sampling_params = {
            "flux_biases": flux_biases,
            "x_polarizing_schedule": [[0.0, 0.0], [1.0, 0.0]],
        }

        original_sample = sampler.sample
        sampled_flux_biases = []

        def record_sample(*args, **kwargs):
            sampled_flux_biases.append(kwargs["flux_biases"].copy())
            return original_sample(*args, **kwargs)

        with unittest.mock.patch.object(sampler, "sample", side_effect=record_sample):
            shim_flux_biases(
                bqm,
                sampler,
                sampling_params=deepcopy(sampling_params),
                shimmed_variables={shimmed},
                learning_schedule=[0],
                flux_biases_break_symmetry=True,
            )

        self.assertEqual(len(sampled_flux_biases), 2)
        self.assertEqual(
            [sample[unshimmed] for sample in sampled_flux_biases], [-0.3, 0.3]
        )
        self.assertEqual([sample[shimmed] for sample in sampled_flux_biases], [0.2, 0.2])

        sampled_flux_biases.clear()
        with unittest.mock.patch.object(sampler, "sample", side_effect=record_sample):
            shim_flux_biases(
                bqm,
                sampler,
                sampling_params=deepcopy(sampling_params),
                shimmed_variables={shimmed},
                learning_schedule=[0],
            )

        self.assertEqual(len(sampled_flux_biases), 1)
        self.assertEqual(sampled_flux_biases[0][unshimmed], 0.3)

        flux_biases[unshimmed] = 0
        sampled_flux_biases.clear()
        sampling_params["flux_biases"] = flux_biases
        with unittest.mock.patch.object(sampler, "sample", side_effect=record_sample):
            shim_flux_biases(
                bqm,
                sampler,
                sampling_params=deepcopy(sampling_params),
                shimmed_variables={shimmed},
                learning_schedule=[0],
                flux_biases_break_symmetry=True,
            )

        self.assertEqual(len(sampled_flux_biases), 1)

    def test_qubit_freezeout_alpha_phi(self):
        x = qubit_freezeout_alpha_phi()
        y = qubit_freezeout_alpha_phi(2, 1, 1, 1)
        self.assertNotEqual(x, y)
        self.assertEqual(1, y)

    def test_neutralize_line_schedule_starts_at_zero(self):
        schedule = [[0.0, 0.0], [2.0, 0.0], [34.0, 0.0]]

        self.assertEqual(
            _neutralize_line_schedule(schedule, -2.0, no_polarizing_schedule=True),
            [[0.0, 0.0], [2.0, -2.0], [34.0, -2.0]],
        )

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
                    exp_feature_line_info,
                    sampling_params=sp,
                    num_steps=2,
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
            exp_feature_line_info,
            sampling_params=sampling_params,
            num_steps=2,
            shimmed_variables=shimmed_variables,
        )

        self.assertSetEqual(set(fb_history.keys()), shimmed_variables)
        # Single experiment per step: fb_history len == num_steps + 1,
        # mag_history len == num_steps.
        self.assertTrue(all(len(fb_history[v]) == 3 for v in shimmed_variables))
        self.assertTrue(all(len(mag_history[v]) == 2 for v in bqm.variables))

    def test_shim_tds_flux_biases_alternating_skips_empty_line_pairs(self):
        (
            sampler,
            bqm,
            target_lines,
            detector_lines,
            line_assignments,
            sampling_params,
            exp_feature_line_info,
            _,
        ) = self._tds_setup()
        target_lines.add(2)
        shimmed_variables = {next(v for v, line in line_assignments.items() if line == 0)}

        flux_biases, fb_history, mag_history = shim_tds_flux_biases(
            bqm,
            sampler,
            target_lines,
            detector_lines,
            exp_feature_line_info,
            sampling_params=sampling_params,
            num_steps=1,
            shimmed_variables=shimmed_variables,
            td_shim_type="alternating",
            by_line_pair=True,
        )

        self.assertIsInstance(flux_biases, list)
        self.assertSetEqual(set(fb_history), shimmed_variables)
        self.assertSetEqual(set(mag_history), shimmed_variables)
        self.assertEqual(len(fb_history[next(iter(shimmed_variables))]), 2)

    def test_shim_tds_flux_biases_alternating_skips_pairs_without_target_variables(self):
        """A pair whose target line has no shimmed variables is skipped rather
        than degenerating into a detector-only shim, so each detector is shimmed
        once per pair with a populated target line."""
        (
            sampler,
            bqm,
            target_lines,
            detector_lines,
            line_assignments,
            sampling_params,
            exp_feature_line_info,
            _,
        ) = self._tds_setup()
        target_lines.add(2)  # Line 2 has no qubits.
        num_steps = 1

        with unittest.mock.patch.object(
            sampler, "sample", wraps=sampler.sample
        ) as sample:
            _, fb_history, _ = shim_tds_flux_biases(
                bqm,
                sampler,
                target_lines,
                detector_lines,
                exp_feature_line_info,
                sampling_params=sampling_params,
                num_steps=num_steps,
                td_shim_type="alternating",
                by_line_pair=True,
            )

        # Only the populated (0, 1) pair runs, with two experiments per step.
        self.assertEqual(sample.call_count, 2 * num_steps)
        self.assertSetEqual(set(fb_history), set(bqm.variables))
        self.assertTrue(all(len(h) == num_steps + 1 for h in fb_history.values()))

    def test_shim_tds_flux_biases_detector_only_explicit(self):
        """td_shim_type='detector_only' shims only detector-line variables with
        a single experiment per step."""
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
        original_schedules = deepcopy(sampling_params["x_anneal_schedules"])

        _, fb_history, mag_history = shim_tds_flux_biases(
            bqm,
            sampler,
            target_lines,
            detector_lines,
            exp_feature_line_info,
            sampling_params=sampling_params,
            num_steps=2,
            td_shim_type="detector_only",
        )
        self.assertEqual(sampling_params["x_anneal_schedules"], original_schedules)
        detector_vars = {
            v for v in bqm.variables if line_assignments[v] in detector_lines
        }
        self.assertSetEqual(set(fb_history.keys()), detector_vars)
        self.assertTrue(all(len(fb_history[v]) == 3 for v in detector_vars))
        self.assertTrue(all(len(mag_history[v]) == 2 for v in bqm.variables))

    def test_shim_tds_flux_biases_by_line_quench(self):
        """td_shim_type='by_line_quench' shims every occupied line independently
        via :func:`shim_linewise_flux_biases`."""
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

        flux_biases, fb_history, mag_history = shim_tds_flux_biases(
            bqm,
            sampler,
            target_lines,
            detector_lines,
            exp_feature_line_info,
            sampling_params=sampling_params,
            td_shim_type="by_line_quench",
        )
        self.assertIsInstance(flux_biases, list)
        self.assertEqual(len(flux_biases), sampler.properties["num_qubits"])
        self.assertSetEqual(set(fb_history.keys()), set(bqm.variables))
        self.assertSetEqual(set(mag_history.keys()), set(bqm.variables))
        for v in bqm.variables:
            # One extra flux-bias entry (initial condition) vs magnetizations.
            self.assertEqual(len(fb_history[v]), len(mag_history[v]) + 1)

    def test_shim_tds_flux_biases_sequential(self):
        """td_shim_type='sequential' runs a line-wise quench followed by a
        detector-only shim, extending the detector-variable histories."""
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

        _, fb_history, mag_history = shim_tds_flux_biases(
            bqm,
            sampler,
            target_lines,
            detector_lines,
            exp_feature_line_info,
            sampling_params=sampling_params,
            num_steps=2,
            td_shim_type="sequential",
        )
        self.assertSetEqual(set(fb_history.keys()), set(bqm.variables))
        detector_vars = {
            v for v in bqm.variables if line_assignments[v] in detector_lines
        }
        target_vars = {v for v in bqm.variables if line_assignments[v] in target_lines}
        # Detector variables are shimmed line-wise AND in the detector-only
        # stage, so their histories are longer than target-only histories.
        for dv in detector_vars:
            for tv in target_vars:
                self.assertGreater(len(fb_history[dv]), len(fb_history[tv]))

    def test_shim_tds_flux_biases_invalid_td_shim_type(self):
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
        with self.assertRaises(ValueError):
            shim_tds_flux_biases(
                bqm,
                sampler,
                target_lines,
                detector_lines,
                exp_feature_line_info,
                sampling_params=sampling_params,
                td_shim_type="not_a_real_mode",
            )

    def test_shim_linewise_flux_biases(self):
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
        num_steps = 2
        flux_biases, fb_history, mag_history = shim_linewise_flux_biases(
            bqm,
            sampler,
            exp_feature_line_info,
            sampling_params=sampling_params,
            num_steps=num_steps,
        )
        self.assertIsInstance(flux_biases, list)
        self.assertEqual(len(flux_biases), sampler.properties["num_qubits"])
        # Each variable is shimmed once, on the line it is assigned to.
        self.assertSetEqual(set(fb_history.keys()), set(bqm.variables))
        self.assertSetEqual(set(mag_history.keys()), set(bqm.variables))
        for v in bqm.variables:
            self.assertEqual(len(fb_history[v]), num_steps + 1)
            self.assertEqual(len(mag_history[v]), num_steps)

    def test_extend_history(self):
        target = {"a": [1, 2], "b": [3]}
        source = {"a": [9], "c": [7, 8]}
        _extend_history(target, source)
        self.assertEqual(target["a"], [1, 2, 9])  # existing key extended
        self.assertEqual(target["b"], [3])  # untouched
        self.assertEqual(target["c"], [7, 8])  # new key added

    def test_shim_flux_biases_input_validation(self):
        sampler = ShimmingMockSampler(substitute_sampler=SteepestDescentSampler())
        nq = sampler.properties["num_qubits"]
        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {0: 0, 1: 0}, {}
        )

        with self.subTest(case="flux_biases_wrong_length"):
            with self.assertRaises(ValueError):
                shim_flux_biases(
                    bqm, sampler, sampling_params={"flux_biases": [0] * (nq - 1)}
                )

        with self.subTest(case="empty_shimmed_variables"):
            with self.assertRaises(ValueError):
                shim_flux_biases(bqm, sampler, shimmed_variables=[])

        with self.subTest(case="invalid_shimmed_variables"):
            with self.assertRaises(ValueError):
                shim_flux_biases(bqm, sampler, shimmed_variables=[nq + 1])

        with self.subTest(case="flux_biases_in_updates"):
            with self.assertRaises(ValueError):
                shim_flux_biases(
                    bqm,
                    sampler,
                    sampling_params_updates=[{"flux_biases": [0] * nq}],
                )

        with self.subTest(case="beta_out_of_range"):
            with self.assertRaises(ValueError):
                shim_flux_biases(bqm, sampler, beta_hypergradient=1.5)

        with self.subTest(case="alpha_non_positive"):
            with self.assertRaises(ValueError):
                shim_flux_biases(bqm, sampler, alpha=-1.0)

        with self.subTest(case="exp_weights_key_mismatch"):
            with self.assertRaises(ValueError):
                shim_flux_biases(
                    bqm,
                    sampler,
                    shimmed_variables=[0, 1],
                    exp_weights_per_update={0: [1.0], 2: [1.0]},
                )

        with self.subTest(case="exp_weights_length_mismatch"):
            # With symmetrize off and no updates there is a single experiment,
            # so weight sequences of length two are rejected.
            with self.assertRaises(ValueError):
                shim_flux_biases(
                    bqm,
                    sampler,
                    symmetrize_experiments=False,
                    shimmed_variables=[0, 1],
                    exp_weights_per_update={0: [1.0, 1.0], 1: [1.0, 1.0]},
                )

    def test_shim_tds_flux_biases_input_validation(self):
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

        with self.subTest(case="empty_detector_lines"):
            with self.assertRaises(ValueError):
                shim_tds_flux_biases(
                    bqm,
                    sampler,
                    target_lines,
                    set(),
                    exp_feature_line_info,
                    sampling_params=sampling_params,
                )

        with self.subTest(case="target_detector_not_disjoint"):
            with self.assertRaises(ValueError):
                shim_tds_flux_biases(
                    bqm,
                    sampler,
                    target_lines,
                    target_lines,
                    exp_feature_line_info,
                    sampling_params=sampling_params,
                )

        with self.subTest(case="shimmed_variables_not_subset"):
            with self.assertRaises(ValueError):
                shim_tds_flux_biases(
                    bqm,
                    sampler,
                    target_lines,
                    detector_lines,
                    exp_feature_line_info,
                    sampling_params=sampling_params,
                    td_shim_type="detector_only",
                    shimmed_variables={sampler.properties["num_qubits"] + 1},
                )

        with self.subTest(case="detector_only_rejects_target_variable"):
            target_qubit = next(
                v for v, l in line_assignments.items() if l in target_lines
            )
            with self.assertRaises(ValueError):
                shim_tds_flux_biases(
                    bqm,
                    sampler,
                    target_lines,
                    detector_lines,
                    exp_feature_line_info,
                    sampling_params=sampling_params,
                    td_shim_type="detector_only",
                    shimmed_variables={target_qubit},
                )

    def test_shim_tds_flux_biases_sequential_cross_role_shimmed_variables(self):
        """``"sequential"`` accepts a cross-role ``shimmed_variables`` (target and
        detector): used in full for the line-wise stage, then narrowed to the
        detector subset for the detector-only second stage."""
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
        target_qubit = next(
            v for v, l in line_assignments.items() if l in target_lines
        )
        detector_qubit = next(
            v for v, l in line_assignments.items() if l in detector_lines
        )

        _, fb_history, mag_history = shim_tds_flux_biases(
            bqm,
            sampler,
            target_lines,
            detector_lines,
            exp_feature_line_info,
            sampling_params=sampling_params,
            num_steps=2,
            td_shim_type="sequential",
            shimmed_variables={target_qubit, detector_qubit},
        )
        self.assertIn(target_qubit, fb_history)
        self.assertIn(detector_qubit, fb_history)
        # Target shimmed only in the line-wise stage; detector also in stage two.
        self.assertGreater(
            len(fb_history[detector_qubit]), len(fb_history[target_qubit])
        )

    def test_shim_linewise_flux_biases_default_sampling_params(self):
        """``sampling_params=None`` falls back to internally built defaults."""
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
        num_steps = 1

        flux_biases, fb_history, mag_history = shim_linewise_flux_biases(
            bqm,
            sampler,
            exp_feature_line_info,
            num_steps=num_steps,
        )
        self.assertIsInstance(flux_biases, list)
        self.assertEqual(len(flux_biases), sampler.properties["num_qubits"])
        self.assertSetEqual(set(fb_history.keys()), set(bqm.variables))
        for v in bqm.variables:
            self.assertEqual(len(fb_history[v]), num_steps + 1)
