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

import random
import unittest
import unittest.mock

import networkx as nx

from dwave.experimental.multicolor_anneal import (
    SOLVER_FILTER,
    ExpFeatureInfo,
    ScheduleError,
    get_properties,
    get_solver_name,
    line_assignments_from_feature_info,
    make_tds_graph,
    make_tds_intervals,
    make_tds_x_anneal_schedules,
    make_tds_x_polarizing_schedule,
    make_tds_x_schedules,
    parse_exp_feature_line_info,
    qubit_to_Advantage2_annealing_line,
    standardize_schedule_endpoints,
    verify_schedules,
)
from dwave.graphs import zephyr_coordinates
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler


class PropertiesCheckMixin:

    polarizing_line_properties = ['minPolarizingTimeStep',
        'depolarizationAnnealScheduleRequiredDelay'] 
    # Pending addition: 'polarizationChangeAnnealScheduleRequiredDelay'

    annealing_line_properties = [
        'annealingLine', 'minAnnealingTimeStep', 'holdOvershootFor',
        'minCOvershoot', 'maxCOvershoot', 'maxC', 'minC',
        'scheduleDelayStep', 'qubits'
    ]

    def validate_exp_feature_info(self, data):
        self.assertIsInstance(data, ExpFeatureInfo)
        self.assertEqual(len(data), 2)
        polarizing_line_info, annealing_line_info = data
        self.assertIsInstance(polarizing_line_info, dict)
        for p in self.polarizing_line_properties:
            self.assertIn(p, polarizing_line_info)
        self.assertIsInstance(annealing_line_info, list)
        self.assertGreater(len(annealing_line_info), 0)
        n_lines = len(annealing_line_info)
        for i in range(n_lines):
            for p in self.annealing_line_properties:
                self.assertIn(p, annealing_line_info[i])
            self.assertEqual(annealing_line_info[i]['annealingLine'], i)
            self.assertGreater(len(annealing_line_info[i]['qubits']), 0)


class MCA(unittest.TestCase, PropertiesCheckMixin):

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_sampler_properties(self):
        n_lines = 6
        n_qubits = 100
        polarizing_line_info = {'minPolarizingTimeStep': 0.02,
                                'depolarizationAnnealScheduleRequiredDelay': 2.0,
                                'polarizationChangeAnnealScheduleRequiredDelay': 1.0}
        annealing_line_info = [{'annealingLine': i,
                 'minAnnealingTimeStep': 0.01,
                 'depolarizationAnnealScheduleRequiredDelay': 2.0,
                 'holdOvershootFor': 0.02,
                 'minCOvershoot': -7.0,
                 'maxCOvershoot': 8.0,
                 'maxC': 3.0,
                 'minC': -2.0,
                 'scheduleDelayStep': 1e-06,
                 'qubits': list(range(i*100, (i+1)*100))} for i in range(n_lines)]
        info = [polarizing_line_info, annealing_line_info]

        with unittest.mock.MagicMock() as sampler:
            sampler.solver.edges = [(0,1)]
            sampler.solver.sample_qubo.return_value.result.return_value = \
                dict(x_get_multicolor_annealing_exp_feature_info=info)

            exp_feature_info = get_properties(sampler)

            self.assertEqual(len(exp_feature_info), 2)
            lines = exp_feature_info[1]

            self.assertEqual(len(lines), n_lines)
            self.assertTrue(all(lines[i]['annealingLine'] == i for i in range(n_lines)))
            self.assertTrue(all(len(lines[i]['qubits']) == n_qubits for i in range(n_lines)))

            self.validate_exp_feature_info(exp_feature_info)

    @unittest.mock.patch('dwave.experimental.fast_reverse_anneal.api.Client')
    def test_default_solver_name(self, client):
        class Solver:
            name = "mock-solver"

        client.from_config.return_value.__enter__.return_value.get_solver.return_value = (
            Solver()
        )

        solver_name = get_solver_name()

        self.assertEqual(solver_name, Solver.name)


class LiveSmokeTests(unittest.TestCase, PropertiesCheckMixin):

    @classmethod
    def setUpClass(cls):
        try:
            cls.sampler = DWaveSampler(solver=SOLVER_FILTER)
        except:
            raise unittest.SkipTest('Multicolor annealing solver not available.')

    @classmethod
    def tearDownClass(cls):
        cls.sampler.close()

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_get_parameters_from_sampler(self):
        exp_feature_info = get_properties(self.sampler)
        self.validate_exp_feature_info(exp_feature_info)

    def test_get_parameters_from_name(self):
        exp_feature_info = get_properties(get_solver_name())
        self.validate_exp_feature_info(exp_feature_info)

    def test_6_line_accuracy(self):
        annealing_lines = get_properties(self.sampler)[1]
        topology_type = self.sampler.properties["topology"]["type"]
        if len(annealing_lines) == 6 and topology_type == "zephyr":
            shape = self.sampler.properties["topology"]["shape"]
            for al_idx, al in enumerate(annealing_lines):
                self.assertTrue(
                    all(
                        qubit_to_Advantage2_annealing_line(n, shape) == al_idx
                        for n in al["qubits"]
                    )
                )


class UtilsTestWithoutClient(unittest.TestCase):

    def test_qubit_to_Advantage2_annealing_line(self):
        shape = [3, 2]
        qpu = MockDWaveSampler(topology_type="zephyr", topology_shape=shape)
        # 0th qubit is always line 0:
        assignments = {
            n: qubit_to_Advantage2_annealing_line(n, shape) for n in qpu.nodelist
        }

        self.assertSetEqual(
            set(assignments.values()),
            set(range(6)),
            "All 6 lines should be represented",
        )

        self.assertEqual(assignments[0], 1, "Zeroth qubit should be line 1")
        test_node = qpu.nodelist[-1]  # could be any
        test_nodeC = zephyr_coordinates(*shape).linear_to_zephyr(test_node)
        self.assertEqual(
            assignments[test_node],
            qubit_to_Advantage2_annealing_line(test_nodeC, shape),
            "Coordinates are handled correctly",
        )
        self.assertEqual(
            qubit_to_Advantage2_annealing_line(test_nodeC),
            qubit_to_Advantage2_annealing_line(test_nodeC, shape),
            "shape is optional when a Zephyr coordinate tuple is given",
        )

    def test_line_assignments_from_feature_info(self):
        polarizing_line_info = {
            "minPolarizingTimeStep": 0.02,
            "depolarizationAnnealScheduleRequiredDelay": 2.0,
            "polarizationChangeAnnealScheduleRequiredDelay": 1.0,
        }
        annealing_line_info = [
            {"annealingLine": 0, "qubits": [2, 6, 9]},
            {"annealingLine": 1, "qubits": [0, 4]},
            {"annealingLine": 2, "qubits": [1, 3, 5, 7]},
        ]
        exp_feature_info = (polarizing_line_info, annealing_line_info)

        assignments = line_assignments_from_feature_info(exp_feature_info)

        self.assertEqual(
            assignments,
            {2: 0, 6: 0, 9: 0, 0: 1, 4: 1, 1: 2, 3: 2, 5: 2, 7: 2},
        )
        # Every qubit maps to the line whose ``qubits`` list contains it.
        self.assertTrue(
            all(
                n in annealing_line_info[line]["qubits"]
                for n, line in assignments.items()
            )
        )
        # Empty annealing-line list yields an empty mapping.
        self.assertEqual(line_assignments_from_feature_info(({}, [])), {})

    def test_tds_graph(self):
        target_graph = nx.from_edgelist([(0, 1)])
        G, node_to_tds = make_tds_graph(target_graph)
        for n in node_to_tds:
            if type(n) is tuple:
                self.assertEqual(node_to_tds[n], n[0])
            else:
                self.assertEqual("target", node_to_tds[n])
        self.assertEqual(G.number_of_edges(), 5)
        self.assertEqual(G.number_of_nodes(), 6)
        detected_nodes = [1]
        G, node_to_tds = make_tds_graph(
            target_graph,
            detected_nodes=detected_nodes,
        )
        self.assertEqual(G.number_of_edges(), 4)
        self.assertEqual(G.number_of_nodes(), 5)
        sourced_nodes = []
        G, node_to_tds = make_tds_graph(target_graph, sourced_nodes=sourced_nodes)
        self.assertEqual(G.number_of_edges(), 3)
        self.assertEqual(G.number_of_nodes(), 4)

    def test_make_tds_x_schedules(self):
        random.seed(0)
        n_lines = 6  # Must be at least 3.
        target_c = round(random.random(), 2)  # target_c to 2 d.p.
        minCOvershoot = -7.0
        maxCOvershoot = 8.0
        depolarization_time_scale = (
            3.0  # Choose as a machine number to avoid precision issues.
        )

        polarizing_line_info = {
            "minPolarizingTimeStep": 0.02,
            "depolarizationAnnealScheduleRequiredDelay": 2.0,
            "polarizationChangeAnnealScheduleRequiredDelay": 1.0,
        }
        annealing_line_info = [
            {
                "annealingLine": i,
                "minAnnealingTimeStep": 0.01,
                "holdOvershootFor": 0.02,
                "minCOvershoot": minCOvershoot,
                "maxCOvershoot": maxCOvershoot,
                "maxC": 3.0,
                "minC": -2.0,
                "scheduleDelayStep": 1e-06,
                "qubits": list(range(i * 100, (i + 1) * 100)),
            }
            for i in range(n_lines)
        ]
        exp_feature_info = [polarizing_line_info, annealing_line_info]

        all_lines = list(range(n_lines))
        random.shuffle(all_lines)
        split_a = random.randint(1, n_lines - 3)
        split_b = random.randint(split_a + 1, n_lines - 2)
        split_c = random.randint(split_b + 1, n_lines - 1)
        target_lines = set(all_lines[:split_a])
        detector_lines = set(all_lines[split_a:split_b])
        source_lines = set(all_lines[split_b:split_c])
        # Other lines are unused, but should not cause issues.

        x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
            exp_feature_info=exp_feature_info,
            target_lines=target_lines,
            target_c=target_c,
            detector_lines=detector_lines,
            source_lines=source_lines,
        )

        with self.subTest(scenario="default_endpoints"):
            self.assertEqual(len(x_anneal_schedules), n_lines)
            self.assertEqual(x_polarizing_schedule[0], [0.0, 1])
            self.assertEqual(x_polarizing_schedule[-1][1], 0)

            # All schedules must have aligned endpoints.
            # (Values are copied, should not be subject to rounding errors)
            end_time = x_anneal_schedules[0][-1][0]
            self.assertTrue(all(s[-1][0] == end_time for s in x_anneal_schedules))
            self.assertEqual(x_polarizing_schedule[-1][0], end_time)

        with self.subTest(scenario="target_c_held"):
            # Target holds the requested C value.
            for target_line in target_lines:
                self.assertIn(target_c, [v for _, v in x_anneal_schedules[target_line]])

        with self.subTest(scenario="overshoot_enabled"):
            # Source and detector include overshoot values when enabled.
            for source_line in source_lines:
                source_values = [v for _, v in x_anneal_schedules[source_line]]
                self.assertIn(maxCOvershoot, source_values)
                self.assertIn(minCOvershoot, source_values)
            for detector_line in detector_lines:
                detector_values = [v for _, v in x_anneal_schedules[detector_line]]
                self.assertIn(minCOvershoot, detector_values)
                self.assertIn(maxCOvershoot, detector_values)

        with self.subTest(scenario="overshoot_disabled"):
            # Disabling overshoot should remove overshoot plateau values.
            x_anneal_no_overshoot, _ = make_tds_x_schedules(
                exp_feature_info=exp_feature_info,
                target_lines=target_lines,
                target_c=target_c,
                detector_lines=detector_lines,
                source_lines=source_lines,
                use_overshoot=False,
            )
            for source_line in source_lines:
                self.assertNotIn(
                    minCOvershoot, [v for _, v in x_anneal_no_overshoot[source_line]]
                )
                self.assertNotIn(
                    maxCOvershoot, [v for _, v in x_anneal_no_overshoot[source_line]]
                )
            for detector_line in detector_lines:
                self.assertNotIn(
                    minCOvershoot, [v for _, v in x_anneal_no_overshoot[detector_line]]
                )
                self.assertNotIn(
                    maxCOvershoot, [v for _, v in x_anneal_no_overshoot[detector_line]]
                )

        with self.subTest(scenario="overshoot_independent"):
            # Overshoot can be enabled on detector lines independently of source.
            x_anneal_mixed, _ = make_tds_x_schedules(
                exp_feature_info=exp_feature_info,
                target_lines=target_lines,
                target_c=target_c,
                detector_lines=detector_lines,
                source_lines=source_lines,
                use_overshoot={"source": False, "detector": True},
            )
            for source_line in source_lines:
                source_values = [v for _, v in x_anneal_mixed[source_line]]
                self.assertNotIn(minCOvershoot, source_values)
                self.assertNotIn(maxCOvershoot, source_values)
            for detector_line in detector_lines:
                detector_values = [v for _, v in x_anneal_mixed[detector_line]]
                self.assertIn(minCOvershoot, detector_values)
                self.assertIn(maxCOvershoot, detector_values)

        with self.subTest(scenario="negative_polarization"):
            # Sign controls the initial polarization direction.
            _, x_polarizing_neg = make_tds_x_schedules(
                exp_feature_info=exp_feature_info,
                target_lines=target_lines,
                target_c=target_c,
                detector_lines=detector_lines,
                source_lines=source_lines,
                sign_polarization=-1,
            )
            self.assertEqual(x_polarizing_neg[0][1], -1)
            self.assertEqual(x_polarizing_neg[1][1], -1)
            self.assertEqual(x_polarizing_neg[2][1], 0)
            self.assertEqual(x_polarizing_neg[-1][1], 0)

        with self.subTest(scenario="custom_depolarization_time_scale"):
            # Changing anneal step size should change the target preparation times.
            _, x_polarizing_custom = make_tds_x_schedules(
                exp_feature_info=exp_feature_info,
                target_lines=target_lines,
                target_c=target_c,
                detector_lines=detector_lines,
                source_lines=source_lines,
                depolarization_time_scale=depolarization_time_scale,
            )
            target_times = [t for t, _ in x_polarizing_custom][:3]  # First 3 points
            self.assertListEqual(
                [0.0, 2 * depolarization_time_scale, 3 * depolarization_time_scale],
                target_times,
            )

    @staticmethod
    def _annealing_line_info(n_lines, min_step=0.01, minC=-2.0, maxC=3.0):
        return [
            {
                "annealingLine": i,
                "minAnnealingTimeStep": min_step,
                "holdOvershootFor": 0.02,
                "minCOvershoot": -7.0,
                "maxCOvershoot": 8.0,
                "maxC": maxC,
                "minC": minC,
                "scheduleDelayStep": 1e-06,
                "qubits": list(range(i * 100, (i + 1) * 100)),
            }
            for i in range(n_lines)
        ]

    def test_make_tds_x_anneal_schedules_defaults(self):
        n_lines = 4
        minC = -2.0
        annealing_line_info = self._annealing_line_info(n_lines, minC=minC)

        # Default operation: target_lines and detector_lines now default to
        # empty, so only exp_feature_line_info is required.
        x_anneal_schedules = make_tds_x_anneal_schedules(annealing_line_info)
        self.assertEqual(len(x_anneal_schedules), n_lines)
        end_time = x_anneal_schedules[0][-1][0]
        for schedule in x_anneal_schedules:
            self.assertEqual(schedule[0], [0.0, 0.0])
            # All schedules share a common, aligned endpoint.
            self.assertEqual(schedule[-1][0], end_time)
            # Unassigned lines are prepared to minC and then held.
            self.assertTrue(all(c in (0.0, minC) for _, c in schedule))

        # Roles can still be assigned explicitly.
        target_c = 0.5
        x_anneal_schedules = make_tds_x_anneal_schedules(
            annealing_line_info,
            target_lines={0},
            detector_lines={1},
            source_lines={2},
            target_c=target_c,
        )
        self.assertIn(target_c, [c for _, c in x_anneal_schedules[0]])

    def test_make_tds_x_anneal_schedules_quench_step_sizes(self):
        n_lines = 3
        min_step = 0.01
        minC, maxC = -2.0, 3.0
        annealing_line_info = self._annealing_line_info(
            n_lines, min_step=min_step, minC=minC, maxC=maxC
        )

        def detector_quench_gap(schedule):
            # Time between the detector reaching minC and then maxC at quench.
            for (t0, c0), (t1, c1) in zip(schedule, schedule[1:]):
                if c0 == minC and c1 == maxC:
                    return t1 - t0
            raise AssertionError("No detector quench step found")

        with self.subTest(scenario="default_step"):
            # The default quench step equals the per-line minimum time step.
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info, detector_lines={0}, use_overshoot=False
            )
            self.assertAlmostEqual(detector_quench_gap(schedule[0]), min_step)

        with self.subTest(scenario="scalar"):
            # A scalar is applied to all lines.
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info,
                detector_lines={0},
                use_overshoot=False,
                quench_step_sizes=0.05,
            )
            self.assertAlmostEqual(detector_quench_gap(schedule[0]), 0.05)

        with self.subTest(scenario="per_line_dict"):
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info,
                detector_lines={0},
                use_overshoot=False,
                quench_step_sizes={line: 0.05 for line in range(n_lines)},
            )
            self.assertAlmostEqual(detector_quench_gap(schedule[0]), 0.05)

        with self.subTest(scenario="below_min_step_rejected"):
            with self.assertRaises(ValueError):
                make_tds_x_anneal_schedules(
                    annealing_line_info,
                    detector_lines={0},
                    use_overshoot=False,
                    quench_step_sizes=min_step / 2,
                )

    def test_make_tds_x_anneal_schedules_detected_target(self):
        n_lines = 4
        min_step = 0.01
        minC, maxC = -2.0, 3.0
        minCOvershoot, maxCOvershoot = -7.0, 8.0
        target_c = 0.5
        annealing_line_info = self._annealing_line_info(
            n_lines, min_step=min_step, minC=minC, maxC=maxC
        )

        def held_value(schedule):
            # A detected target is ramped up, held flat, then quenched; the held
            # value is the C reached by two consecutive equal-value points.
            for (_, c0), (_, c1) in zip(schedule, schedule[1:]):
                if c0 == c1 and c0 != 0.0:
                    return c0
            raise AssertionError("No held value found")

        with self.subTest(scenario="prepared_and_read_out"):
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info,
                detected_target_lines={0},
                target_c=target_c,
            )[0]
            values = [c for _, c in schedule]
            # Prepared to target_c (held), like a target line.
            self.assertAlmostEqual(held_value(schedule), target_c)
            self.assertIn(target_c, values)
            # Read out (quenched) to maxC, like a detector line.
            self.assertEqual(schedule[-1][1], maxC)

        with self.subTest(scenario="overshoot_enabled"):
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info,
                detected_target_lines={0},
                target_c=target_c,
                use_overshoot=True,
            )[0]
            values = [c for _, c in schedule]
            self.assertIn(maxCOvershoot, values)
            self.assertEqual(schedule[-1][1], maxC)

        with self.subTest(scenario="overshoot_disabled"):
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info,
                detected_target_lines={0},
                target_c=target_c,
                use_overshoot=False,
            )[0]
            values = [c for _, c in schedule]
            self.assertNotIn(maxCOvershoot, values)
            self.assertNotIn(minCOvershoot, values)
            self.assertEqual(schedule[-1][1], maxC)

        with self.subTest(scenario="default_target_c_is_minC"):
            # Without target_c, detected targets are held at minC.
            schedule = make_tds_x_anneal_schedules(
                annealing_line_info,
                detected_target_lines={0},
            )[0]
            self.assertAlmostEqual(held_value(schedule), minC)

        with self.subTest(scenario="per_line_target_c"):
            # target_c may vary by line.
            per_line_c = {0: 0.25, 1: -0.5}
            schedules = make_tds_x_anneal_schedules(
                annealing_line_info,
                detected_target_lines={0, 1},
                target_c=per_line_c,
            )
            self.assertAlmostEqual(held_value(schedules[0]), per_line_c[0])
            self.assertAlmostEqual(held_value(schedules[1]), per_line_c[1])

        with self.subTest(scenario="disjoint_roles_required"):
            # A line cannot be both a detected target and another role.
            for conflicting in ("detector_lines", "target_lines", "source_lines"):
                with self.assertRaises(ValueError):
                    make_tds_x_anneal_schedules(
                        annealing_line_info,
                        detected_target_lines={0},
                        target_c=target_c,
                        **{conflicting: {0}},
                    )

        with self.subTest(scenario="invalid_line_index"):
            with self.assertRaises(ValueError):
                make_tds_x_anneal_schedules(
                    annealing_line_info,
                    detected_target_lines={n_lines},
                    target_c=target_c,
                )

    def test_make_tds_x_schedules_empty_lines(self):
        random.seed(0)
        n_lines = 6  # Must be at least 3.
        target_c = round(random.random(), 2)  # target_c to 2 d.p.

        polarizing_line_info = {
            "minPolarizingTimeStep": 0.02,
            "depolarizationAnnealScheduleRequiredDelay": 2.0,
            "polarizationChangeAnnealScheduleRequiredDelay": 1.0,
        }
        annealing_line_info = [
            {
                "annealingLine": i,
                "minAnnealingTimeStep": 0.01,
                "holdOvershootFor": 0.02,
                "minCOvershoot": -7.0,
                "maxCOvershoot": 8.0,
                "maxC": 3.0,
                "minC": -2.0,
                "scheduleDelayStep": 1e-06,
                "qubits": list(range(i * 100, (i + 1) * 100)),
            }
            for i in range(n_lines)
        ]
        exp_feature_info = [polarizing_line_info, annealing_line_info]

        all_lines = set(range(n_lines))
        scenarios = {
            "empty_target_lines": (set(), {0, 1}, {2, 3}),
            "empty_detector_lines": ({0, 1}, set(), {2, 3}),
            "empty_source_lines": ({0, 1}, {2, 3}, set()),
            "all_empty": (set(), set(), set()),
        }
        for scenario, (target_lines, detector_lines, source_lines) in scenarios.items():
            with self.subTest(scenario=scenario):
                x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
                    exp_feature_info=exp_feature_info,
                    target_lines=target_lines,
                    target_c=target_c,
                    detector_lines=detector_lines,
                    source_lines=source_lines,
                )
                self.assertEqual(len(x_anneal_schedules), n_lines)

    def test_make_tds_intervals(self):
        with self.subTest(scenario="default"):
            (
                polarization_interval,
                polarized_interval,
                depolarization_interval,
                depolarized_interval,
                quench_time,
            ) = make_tds_intervals()
            self.assertTupleEqual(polarization_interval, (0.0, 0.0))
            self.assertTupleEqual(polarized_interval, (0.0, 2.0))
            self.assertTupleEqual(depolarization_interval, (4.0, 6.0))
            self.assertTupleEqual(depolarized_interval, (8.0, 10.0))
            self.assertEqual(quench_time, 30.0)

        with self.subTest(scenario="custom_parameters"):
            (
                polarization_interval,
                polarized_interval,
                depolarization_interval,
                depolarized_interval,
                quench_time,
            ) = make_tds_intervals(
                post_preparation_delay=11.0,
                buffering_time_scale=3.0,
                depolarizing_time_scale=5.0,
                anneal_preparation_time_scale=7.0,
                skip_explicit_polarization=False,
            )
            self.assertTupleEqual(polarization_interval, (0.0, 5.0))
            self.assertTupleEqual(polarized_interval, (8.0, 15.0))
            self.assertTupleEqual(depolarization_interval, (18.0, 23.0))
            self.assertTupleEqual(depolarized_interval, (26.0, 33.0))
            self.assertEqual(quench_time, 44.0)

    def test_make_tds_x_polarizing_schedule(self):
        schedule = make_tds_x_polarizing_schedule((2.0, 5.0), (8.0, 10.0))
        self.assertListEqual(
            schedule, [[0.0, 0], [2.0, 0], [5.0, 1], [8.0, 1], [10.0, 0]]
        )

        schedule_neg = make_tds_x_polarizing_schedule(
            (2.0, 5.0), (8.0, 10.0), sign_polarization=-1
        )
        self.assertListEqual(
            schedule_neg, [[0.0, 0], [2.0, 0], [5.0, -1], [8.0, -1], [10.0, 0]]
        )

        with self.assertRaises(ValueError):
            make_tds_x_polarizing_schedule((3.0, 3.0), (8.0, 10.0))

    def test_standardize_schedule_endpoints(self):
        x_anneal = [
            [[0.0, 0.0], [2.0, 1.0]],
            [[0.0, 0.0], [3.0, -1.0]],
        ]
        x_polarizing = [[0.0, 1], [1.5, 1], [2.5, 0]]

        anneal_std, polarizing_std = standardize_schedule_endpoints(
            x_anneal,
            x_polarizing,
            post_pwl_delay=0.4,
            decimals=0,
        )
        self.assertEqual(anneal_std[0][-1], [4.0, 1.0])
        self.assertEqual(anneal_std[1][-1], [4.0, -1.0])
        self.assertEqual(polarizing_std[-1], [4.0, 0])

        with self.assertRaises(ValueError):
            standardize_schedule_endpoints(x_anneal, post_pwl_delay=-1.0)

    def test_qubit_to_Advantage2_annealing_line_num_lines(self):
        shape = [3, 2]
        qpu = MockDWaveSampler(topology_type="zephyr", topology_shape=shape)

        lines_6 = {qubit_to_Advantage2_annealing_line(n, shape, 6) for n in qpu.nodelist}
        self.assertSetEqual(lines_6, set(range(6)), "6-line scheme uses lines 0-5")

        lines_12 = {
            qubit_to_Advantage2_annealing_line(n, shape, 12) for n in qpu.nodelist
        }
        self.assertSetEqual(lines_12, set(range(12)), "12-line scheme uses lines 0-11")

        with self.assertRaises(ValueError):
            qubit_to_Advantage2_annealing_line(qpu.nodelist[0], shape, num_lines=7)

    def test_make_tds_graph_invalid_nodes(self):
        target_graph = nx.from_edgelist([(0, 1)])
        with self.assertRaises(ValueError):
            make_tds_graph(target_graph, detected_nodes=[5])
        with self.assertRaises(ValueError):
            make_tds_graph(target_graph, sourced_nodes=[5])

    def test_make_tds_graph_single_pass_iterables(self):
        # A single-pass iterable (generator) must not be exhausted by internal
        # validation before the decorator edges are built.
        target_graph = nx.from_edgelist([(0, 1)])
        tds_graph, node_to_tds = make_tds_graph(
            target_graph,
            detected_nodes=(n for n in [0, 1]),
            sourced_nodes=(n for n in [0, 1]),
        )
        self.assertIn(("detector", 0), tds_graph)
        self.assertIn(("source", 1), tds_graph)
        self.assertEqual(node_to_tds[("detector", 0)], "detector")
        self.assertEqual(node_to_tds[("source", 1)], "source")

    def test_make_tds_x_polarizing_schedule_variants(self):
        # Zero sign produces a flat zero schedule retaining the interval pattern.
        schedule_zero = make_tds_x_polarizing_schedule(
            (2.0, 5.0), (8.0, 10.0), sign_polarization=0
        )
        self.assertListEqual(
            schedule_zero, [[0.0, 0], [2.0, 0], [5.0, 0], [8.0, 0], [10.0, 0]]
        )

        # With default arguments the schedule starts polarized at time 0 and
        # relaxes over the default depolarization interval.
        schedule = make_tds_x_polarizing_schedule()
        self.assertListEqual(schedule, [[0.0, 1], [4.0, 1], [6.0, 0]])

        # A depolarization interval that does not start after polarization ends
        # is rejected.
        with self.assertRaises(ValueError):
            make_tds_x_polarizing_schedule((2.0, 5.0), (3.0, 6.0))

        # A depolarization interval supplied without a polarization interval
        # defaults to starting in a polarized state at time 0.
        schedule_depol_only = make_tds_x_polarizing_schedule(
            depolarization_interval=(4.0, 6.0)
        )
        self.assertListEqual(schedule_depol_only, [[0.0, 1], [4.0, 1], [6.0, 0]])

    def test_standardize_schedule_endpoints_decimals(self):
        # ``decimals`` rounds the common endpoint up to the given precision.
        # ``end_buffer=0`` isolates the rounding from the terminal-hold guarantee.
        x_anneal = [[[0.0, 0.0], [2.33, 1.0]]]

        anneal_std, _ = standardize_schedule_endpoints(
            x_anneal, post_pwl_delay=0.0, decimals=1, end_buffer=0.0
        )
        self.assertEqual(anneal_std[0][-1], [2.4, 1.0])

        anneal_std, _ = standardize_schedule_endpoints(
            x_anneal, post_pwl_delay=0.0, decimals=0, end_buffer=0.0
        )
        self.assertEqual(anneal_std[0][-1], [3.0, 1.0])

    def test_make_tds_x_schedules_bound_options(self):
        n_lines = 4
        minC, maxC = -2.0, 3.0
        minCOvershoot, maxCOvershoot = -7.0, 8.0
        annealing_line_info = self._annealing_line_info(
            n_lines, minC=minC, maxC=maxC
        )
        exp_feature_info = [
            {
                "minPolarizingTimeStep": 0.02,
                "depolarizationAnnealScheduleRequiredDelay": 2.0,
                "polarizationChangeAnnealScheduleRequiredDelay": 1.0,
            },
            annealing_line_info,
        ]
        common = dict(
            exp_feature_info=exp_feature_info,
            target_lines={0},
            target_c=0.5,
            detector_lines={1},
            source_lines={2},
        )

        with self.subTest(option="symmetrize_c_bounds"):
            schedules, _ = make_tds_x_schedules(**common, symmetrize_c_bounds=True)
            values = [c for s in schedules for _, c in s]
            # minC/maxC become symmetric at min(|minC|, maxC) = 2.0, and the
            # overshoot bounds at min(|minCOvershoot|, maxCOvershoot) = 7.0.
            self.assertEqual(max(values), 7.0)
            self.assertEqual(min(values), -7.0)

        with self.subTest(option="use_01_c_range"):
            schedules, _ = make_tds_x_schedules(**common, use_01_c_range=True)
            values = [c for s in schedules for _, c in s]
            self.assertEqual(min(values), 0.0)
            self.assertEqual(max(values), 1.0)

        with self.subTest(option="use_common_bounds"):
            schedules, _ = make_tds_x_schedules(**common, use_common_bounds=True)
            self.assertEqual(len(schedules), n_lines)

    def test_parse_exp_feature_line_info_common_time_step(self):
        # Lines have different minimum time steps; the common step must satisfy
        # every line, i.e. it is the largest per-line minimum, not the smallest.
        annealing_line_info = self._annealing_line_info(3)
        annealing_line_info[0]["minAnnealingTimeStep"] = 0.01
        annealing_line_info[1]["minAnnealingTimeStep"] = 0.05
        annealing_line_info[2]["minAnnealingTimeStep"] = 0.02

        min_time_steps = parse_exp_feature_line_info(
            annealing_line_info,
            use_common_bounds=True,
            symmetrize_c_bounds=False,
        )[4]

        self.assertTrue(all(step == 0.05 for step in min_time_steps.values()))

    def test_exp_feature_info_namedtuple(self):
        info = ExpFeatureInfo({"minPolarizingTimeStep": 0.02}, [{"annealingLine": 0}])
        self.assertEqual(info.polarizing_line_info, {"minPolarizingTimeStep": 0.02})
        self.assertEqual(info.annealing_line_info, [{"annealingLine": 0}])
        # Still behaves as a 2-tuple for backward compatibility.
        self.assertEqual(info[0], info.polarizing_line_info)
        self.assertEqual(len(info), 2)

    def test_qubit_to_Advantage2_annealing_line_requires_shape(self):
        # An integer qubit label needs a shape to resolve Zephyr coordinates.
        with self.assertRaises(ValueError):
            qubit_to_Advantage2_annealing_line(42)
        # A coordinate tuple does not need a shape.
        self.assertIsInstance(
            qubit_to_Advantage2_annealing_line((0, 0, 0, 0, 0)), int
        )

    def test_make_tds_x_anneal_schedules_all_target_lines(self):
        # Every line is a target: the non-target duration check must not be
        # evaluated over an empty set (regression for min() on empty sequence).
        annealing_line_info = self._annealing_line_info(2)
        schedules = make_tds_x_anneal_schedules(
            annealing_line_info,
            target_lines={0, 1},
            target_c=0.5,
        )
        self.assertEqual(len(schedules), 2)

    def test_make_tds_x_anneal_schedules_partial_quench_step_sizes(self):
        # A dict covering only the quenched line is accepted; other lines keep
        # their own minimum step.
        minC, maxC = -2.0, 3.0
        annealing_line_info = self._annealing_line_info(
            3, min_step=0.01, minC=minC, maxC=maxC
        )
        schedule = make_tds_x_anneal_schedules(
            annealing_line_info,
            detector_lines={0},
            use_overshoot=False,
            quench_step_sizes={0: 0.05},
        )
        self.assertEqual(len(schedule), 3)
        gaps = [
            t1 - t0
            for (t0, c0), (t1, c1) in zip(schedule[0], schedule[0][1:])
            if c0 == minC and c1 == maxC
        ]
        self.assertTrue(any(abs(gap - 0.05) < 1e-9 for gap in gaps))

    def test_make_tds_x_anneal_schedules_large_hold_overshoot(self):
        # A holdOvershootFor reaching before the preparation interval must not
        # produce a backwards (non-monotonic) schedule.
        annealing_line_info = self._annealing_line_info(2)
        for line in annealing_line_info:
            line["holdOvershootFor"] = 29.0
        schedules = make_tds_x_anneal_schedules(
            annealing_line_info,
            source_lines={0},
            detector_lines={1},
        )
        for schedule in schedules:
            times = [t for t, _ in schedule]
            self.assertEqual(times, sorted(times))

    def test_make_tds_x_polarizing_schedule_requires_ordered_intervals(self):
        # Supplying only a polarization interval still validates ordering
        # against the default depolarization interval.
        with self.assertRaises(ValueError):
            make_tds_x_polarizing_schedule(polarization_interval=(0.0, 5.0))

    def test_standardize_schedule_endpoints_idempotent(self):
        x_anneal = [[[0.0, 0.0], [2.0, 1.0]]]
        once, _ = standardize_schedule_endpoints(x_anneal, post_pwl_delay=1.0)
        twice, _ = standardize_schedule_endpoints(once, post_pwl_delay=1.0)
        self.assertEqual(once, twice)

    def test_standardize_schedule_endpoints_end_buffer(self):
        x_anneal = [[[0.0, 0.0], [2.0, 1.0]]]
        # A negative end_buffer is rejected.
        with self.assertRaises(ValueError):
            standardize_schedule_endpoints(x_anneal, end_buffer=-1.0)
        # end_buffer of 0 is accepted.
        result, _ = standardize_schedule_endpoints(
            x_anneal, post_pwl_delay=0.0, end_buffer=0.0
        )
        self.assertEqual(len(result), 1)

    def test_verify_schedules_min_time_step_spacing(self):
        info = [
            {"minPolarizingTimeStep": 0.02},
            [{"minAnnealingTimeStep": 0.01}],
        ]
        # Consecutive points closer than the minimum time step are rejected.
        bad = [[[0.0, 0.0], [0.005, 1.0], [1.0, 1.0]]]
        with self.assertRaises(ScheduleError):
            verify_schedules(info, x_anneal_schedules=bad, check_rounding=False)
        # A compliant schedule passes.
        good = [[[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]]]
        verify_schedules(info, x_anneal_schedules=good, check_rounding=False)

    def test_verify_schedules_unsynchronized_endpoints(self):
        info = [
            {"minPolarizingTimeStep": 0.02},
            [{"minAnnealingTimeStep": 0.01}, {"minAnnealingTimeStep": 0.01}],
        ]
        # Anneal lines ending at different times are rejected.
        bad = [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [2.0, 1.0]]]
        with self.assertRaises(ScheduleError):
            verify_schedules(info, x_anneal_schedules=bad, check_rounding=False)
        # A polarizing schedule ending at a different time is rejected.
        # Anneal lines hold flat across the depolarization window so only the
        # endpoint mismatch is exercised.
        anneal = [[[0.0, 0.0], [0.5, 1.0], [2.0, 1.0]], [[0.0, 0.0], [0.5, 1.0], [2.0, 1.0]]]
        polarizing = [[0.0, 1], [1.0, 1], [1.5, 0]]
        with self.assertRaises(ScheduleError):
            verify_schedules(
                info,
                x_anneal_schedules=anneal,
                x_polarizing_schedule=polarizing,
                check_rounding=False,
            )
        # Synchronized endpoints pass.
        polarizing_sync = [[0.0, 1], [1.0, 1], [1.5, 0], [2.0, 0]]
        verify_schedules(
            info,
            x_anneal_schedules=anneal,
            x_polarizing_schedule=polarizing_sync,
            check_rounding=False,
        )

    def test_verify_schedules_depolarization_constancy(self):
        info = [
            {
                "minPolarizingTimeStep": 0.02,
                "depolarizationAnnealScheduleRequiredDelay": 2.0,
            },
            [{"minAnnealingTimeStep": 0.01}],
        ]
        polarizing = [[0.0, 1], [4.0, 1], [6.0, 0], [10.0, 0]]
        # Anneal varies within [6, 8] (the depolarization settling window) but
        # is flat across the depolarization interval itself.
        bad = [[[0.0, 0.5], [6.5, 0.5], [7.0, 0.9], [8.0, 0.9], [10.0, 0.9]]]
        with self.assertRaises(ScheduleError):
            verify_schedules(
                info,
                x_anneal_schedules=bad,
                x_polarizing_schedule=polarizing,
                check_rounding=False,
            )
        # Holding constant across the window passes.
        good = [[[0.0, 0.5], [10.0, 0.5]]]
        verify_schedules(
            info,
            x_anneal_schedules=good,
            x_polarizing_schedule=polarizing,
            check_rounding=False,
        )

    def test_verify_schedules_polarization_change_constancy(self):
        info = [
            {
                "minPolarizingTimeStep": 0.02,
                "depolarizationAnnealScheduleRequiredDelay": 2.0,
                "polarizationChangeAnnealScheduleRequiredDelay": 1.0,
            },
            [{"minAnnealingTimeStep": 0.01}],
        ]
        # Polarization onset 0 -> 1 over (0, 3); padded window (-1, 4).
        polarizing = [[0.0, 0], [3.0, 1], [4.0, 1], [6.0, 0], [10.0, 0]]
        bad = [[[0.0, 0.5], [3.5, 0.9], [10.0, 0.9]]]
        with self.assertRaises(ScheduleError):
            verify_schedules(
                info,
                x_anneal_schedules=bad,
                x_polarizing_schedule=polarizing,
                check_rounding=False,
            )
        good = [[[0.0, 0.5], [10.0, 0.5]]]
        verify_schedules(
            info,
            x_anneal_schedules=good,
            x_polarizing_schedule=polarizing,
            check_rounding=False,
        )


