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

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from dwave.experimental.multicolor_anneal import (
    get_properties, get_solver_name, SOLVER_FILTER,
    qubit_to_Advantage2_annealing_line, make_tds_graph,
    make_tds_intervals, make_tds_x_polarizing_schedule,
    make_tds_x_schedules, standardize_schedule_endpoints,
)
from dwave_networkx import zephyr_coordinates


class PropertiesCheckMixin:

    properties = [
        'annealingLine', 'minAnnealingTimeStep', 'minPolarizingTimeStep',
        'depolarizationAnnealScheduleRequiredDelay', 'holdOvershootFor',
        'minCOvershoot', 'maxCOvershoot', 'maxC', 'minC',
        'scheduleDelayStep', 'qubits'
    ]

    def validate_annealing_lines_properties(self, data):
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        n_lines = len(data)
        for i in range(n_lines):
            for p in self.properties:
                self.assertIn(p, data[i])
            self.assertEqual(data[i]['annealingLine'], i)
            self.assertGreater(len(data[i]['qubits']), 0)


class MCA(unittest.TestCase, PropertiesCheckMixin):

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_sampler_properties(self):
        n_lines = 6
        n_qubits = 100
        info = [{'annealingLine': i,
                 'minAnnealingTimeStep': 0.01,
                 'minPolarizingTimeStep': 0.02,
                 'depolarizationAnnealScheduleRequiredDelay': 2.0,
                 'holdOvershootFor': 0.02,
                 'minCOvershoot': -7.0,
                 'maxCOvershoot': 8.0,
                 'maxC': 3.0,
                 'minC': -2.0,
                 'scheduleDelayStep': 1e-06,
                 'qubits': list(range(i*100, (i+1)*100))} for i in range(n_lines)]

        with unittest.mock.MagicMock() as sampler:
            sampler.solver.edges = [(0,1)]
            sampler.solver.sample_qubo.return_value.result.return_value = \
                dict(x_get_multicolor_annealing_exp_feature_info=info)

            lines = get_properties(sampler)

            self.assertEqual(len(lines), n_lines)
            self.assertTrue(all(lines[i]['annealingLine'] == i for i in range(n_lines)))
            self.assertTrue(all(len(lines[i]['qubits']) == n_qubits for i in range(n_lines)))

            self.validate_annealing_lines_properties(lines)

        
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
        lines = get_properties(self.sampler)
        self.validate_annealing_lines_properties(lines)

    def test_get_parameters_from_name(self):
        lines = get_properties(get_solver_name())
        self.validate_annealing_lines_properties(lines)

    def test_6_line_accuracy(self):
        annealing_lines = get_properties(self.sampler)
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
        n_lines = 6 # Must be atleast 3.
        target_c = (random.random() * 100) / 100  # target_c to 2 s.f.
        minCOvershoot = -7.0
        maxCOvershoot = 8.0
        depolarization_time_scale = 3.0  # Choose as a machine number to avoid precision issues.

        exp_feature_info = [
            {'annealingLine': i,
             'minAnnealingTimeStep': 0.01,
             'minPolarizingTimeStep': 0.02,
             'depolarizationAnnealScheduleRequiredDelay': 2.0,
             'holdOvershootFor': 0.02,
             'minCOvershoot': minCOvershoot,
             'maxCOvershoot': maxCOvershoot,
             'maxC': 3.0,
             'minC': -2.0,
             'scheduleDelayStep': 1e-06,
             'qubits': list(range(i*100, (i+1)*100))} for i in range(n_lines)]

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

        self.assertEqual(len(x_anneal_schedules), n_lines)
        self.assertEqual(x_polarizing_schedule[0], [0.0, 1])
        self.assertEqual(x_polarizing_schedule[-1][1], 0)

        # All schedules must have aligned endpoints.
        # (Values are copied, should not be subject to rounding errors)
        end_time = x_anneal_schedules[0][-1][0]
        self.assertTrue(all(s[-1][0] == end_time for s in x_anneal_schedules))
        self.assertEqual(x_polarizing_schedule[-1][0], end_time)

        # Target holds the requested C value.
        for target_line in target_lines:
            self.assertIn(target_c, [v for _, v in x_anneal_schedules[target_line]])
        # Source and detector include overshoot values when enabled.
        for source_line in source_lines:
            source_values = [v for _, v in x_anneal_schedules[source_line]]
            self.assertIn(maxCOvershoot, source_values)
            self.assertIn(minCOvershoot, source_values)
        for detector_line in detector_lines:
            detector_values = [v for _, v in x_anneal_schedules[detector_line]]
            self.assertIn(minCOvershoot, detector_values)
            self.assertIn(maxCOvershoot, detector_values)

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
            self.assertNotIn(minCOvershoot, [v for _, v in x_anneal_no_overshoot[source_line]])
            self.assertNotIn(maxCOvershoot, [v for _, v in x_anneal_no_overshoot[source_line]])
        for detector_line in detector_lines:
            self.assertNotIn(minCOvershoot, [v for _, v in x_anneal_no_overshoot[detector_line]])
            self.assertNotIn(maxCOvershoot, [v for _, v in x_anneal_no_overshoot[detector_line]])

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
        self.assertEqual(x_polarizing_neg[-1][1], 0)

        # Changing anneal step size should change the target preparation times.
        _, x_polarizing_schedule = make_tds_x_schedules(
            exp_feature_info=exp_feature_info,
            target_lines=target_lines,
            target_c=target_c,
            detector_lines=detector_lines,
            source_lines=source_lines,
            depolarization_time_scale=depolarization_time_scale,
        )
        target_times = [t for t, _ in x_polarizing_schedule][:3]  # First 3 points
        self.assertListEqual(
            [0.0, 2 * depolarization_time_scale, 3 * depolarization_time_scale],
            target_times,
        )

    def test_make_tds_intervals(self):
        polarized_interval, depolarization_interval, depolarized_interval, quench_time = (
            make_tds_intervals()
        )
        self.assertTupleEqual(polarized_interval, (0.0, 2.0))
        self.assertTupleEqual(depolarization_interval, (4.0, 6.0))
        self.assertTupleEqual(depolarized_interval, (8.0, 10.0))
        self.assertEqual(quench_time, 30.0)

        polarized_interval, depolarization_interval, depolarized_interval, quench_time = (
            make_tds_intervals(
                post_preparation_delay=11.0,
                depolarization_time_scale=3.0,
                depolarizing_time_scale=5.0,
                anneal_preparation_time_scale=7.0,
            )
        )
        self.assertTupleEqual(polarized_interval, (0.0, 7.0))
        self.assertTupleEqual(depolarization_interval, (10.0, 15.0))
        self.assertTupleEqual(depolarized_interval, (18.0, 25.0))
        self.assertEqual(quench_time, 36.0)

    def test_make_tds_x_polarizing_schedule(self):
        schedule = make_tds_x_polarizing_schedule((2.0, 5.0))
        self.assertListEqual(schedule, [[0.0, 1], [2.0, 1], [5.0, 0]])

        schedule_neg = make_tds_x_polarizing_schedule((2.0, 5.0), sign_polarization=-1)
        self.assertListEqual(schedule_neg, [[0.0, -1], [2.0, -1], [5.0, 0]])

        with self.assertRaises(ValueError):
            make_tds_x_polarizing_schedule((3.0, 3.0))

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
