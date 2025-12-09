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

import networkx as nx

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from dwave.experimental.multicolor_anneal import (
    get_properties, get_solver_name, SOLVER_FILTER,
    qubit_to_Advantage2_annealing_line, make_tds_graph,
)
from dwave_networkx import zephyr_coordinates


class PropertiesCheckMixin:

    properties = [
        'annealingLine', 'minTimeStep', 'depolarizationAnnealScheduleRequiredDelay',
        'holdOvershootFor', 'minCOvershoot', 'maxCOvershoot', 'maxC', 'minC',
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
                 'minTimeStep': 0.01,
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
