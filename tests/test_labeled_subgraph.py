# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import unittest

import networkx as nx
from dwave.graphs import (
    zephyr_graph,
    pegasus_graph,
    chimera_graph,
    chimera_two_color,
    pegasus_four_color,
    zephyr_four_color,
)

from dwave.experimental.embedding_methods.labeled_subgraph import (
    find_labeled_subgraph,
    node_labels_by_coloring,
    node_labels_by_orientation,
    node_labels_by_quotient,
)


class TestNodeLabelHelpers(unittest.TestCase):
    """Basic tests for helper labeling functions."""

    def test_node_labels_by_orientation_zephyr_matches_orientation_axis(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_orientation(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[0])

    def test_node_labels_by_orientation_chimera_matches_orientation_axis(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_orientation(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[2])

    def test_node_labels_by_coloring_preserves_original_linear_labels(self):
        graph = zephyr_graph(2, 2, coordinates=False)

        labels = node_labels_by_coloring(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        self.assertTrue(set(labels.values()).issubset({0, 1, 2, 3}))

    def test_node_labels_by_coloring_chimera_matches_per_node_two_color(self):
        # Regression test: coloring must be computed per node, not once for the
        # whole graph.
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_coloring(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], chimera_two_color(node))
        # Sanity check that the coloring is not degenerate (i.e. not all nodes
        # share one color).
        self.assertEqual(set(labels.values()), {0, 1})

    def test_node_labels_by_coloring_pegasus_matches_per_node_four_color(self):
        graph = pegasus_graph(3, coordinates=True)

        labels = node_labels_by_coloring(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], pegasus_four_color(node))

    def test_node_labels_by_coloring_zephyr_matches_per_node_four_color(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_coloring(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], zephyr_four_color(node))

    def test_node_labels_by_coloring_as_str_converts_labels_to_strings(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_coloring(graph, as_str=True)

        self.assertEqual(set(labels), set(graph.nodes()))
        self.assertTrue(all(isinstance(v, str) for v in labels.values()))
        self.assertEqual(set(labels.values()), {"0", "1"})

    def test_node_labels_by_coloring_falls_back_to_greedy_color_without_family(self):
        # A plain graph without D-Wave family metadata should use
        # nx.greedy_color as a fallback.
        graph = nx.cycle_graph(6)

        labels = node_labels_by_coloring(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        expected = nx.greedy_color(graph)
        self.assertEqual(labels, expected)

    def test_node_labels_by_quotient_remaps_zephyr_boundaries(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_quotient(
            graph, expand_boundary_search=True, as_str=False
        )

        self.assertEqual(labels[(0, 0, 0, 0, 0)], (0, 1, 0, 0))
        self.assertEqual(labels[(0, 4, 0, 0, 0)], (0, 3, 0, 0))
        self.assertEqual(labels[(0, 2, 0, 0, 0)], (0, 2, 0, 0))

    def test_node_labels_by_quotient_chimera_drops_qubit_index(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_quotient(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[:3])

    def test_node_labels_by_orientation_as_str_converts_labels_to_strings(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_orientation(graph, as_str=True)

        self.assertEqual(set(labels), set(graph.nodes()))
        self.assertTrue(all(isinstance(v, str) for v in labels.values()))

    def test_node_labels_by_orientation_falls_back_to_greedy_color_without_family(self):
        graph = nx.cycle_graph(6)

        labels = node_labels_by_orientation(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        self.assertEqual(set(labels.values()), {0, 1})

    def test_node_labels_by_orientation_raises_for_non_bipartite_fallback(self):
        graph = nx.cycle_graph(5)

        with self.assertRaises(ValueError):
            node_labels_by_orientation(graph, as_str=False)

    def test_node_labels_by_quotient_pegasus_drops_odd_qubit_bit(self):
        graph = pegasus_graph(3, coordinates=True)

        labels = node_labels_by_quotient(graph, as_str=False)

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            u, w, k, z = node
            self.assertEqual(labels[node], (u, w, k // 2, z))

    def test_node_labels_by_quotient_zephyr_without_boundary_expansion(self):
        graph = zephyr_graph(2, 2, coordinates=True)

        labels = node_labels_by_quotient(
            graph, expand_boundary_search=False, as_str=False
        )

        self.assertEqual(set(labels), set(graph.nodes()))
        for node in graph.nodes():
            self.assertEqual(labels[node], node[:2] + node[3:])

    def test_node_labels_by_quotient_as_str_converts_labels_to_strings(self):
        graph = chimera_graph(2, t=2, coordinates=True)

        labels = node_labels_by_quotient(graph, as_str=True)

        self.assertEqual(set(labels), set(graph.nodes()))
        self.assertTrue(all(isinstance(v, str) for v in labels.values()))

    def test_node_labels_by_quotient_raises_for_unrecognized_family(self):
        graph = nx.path_graph(4)

        with self.assertRaises(ValueError):
            node_labels_by_quotient(graph)

    def test_find_labeled_subgraph_returns_identity_on_simple_graph(self):
        source = nx.path_graph(2)
        target = nx.path_graph(2)
        node_labels = ({0: "0", 1: "1"}, {0: "0", 1: "1"})

        embedding = find_labeled_subgraph(
            source, target, node_labels=node_labels, timeout=1
        )

        self.assertIsInstance(embedding, dict)
        self.assertEqual(set(embedding.keys()), set(source.nodes()))
        self.assertEqual(set(embedding.values()), set(target.nodes()))


class TestFindLabeledSubgraphLabelingDispatch(unittest.TestCase):
    """Tests for the automatic node-labeling dispatch in find_labeled_subgraph."""

    def test_orientation_labeling_method_computes_labels_internally(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = chimera_graph(2, t=2, coordinates=True)

        embedding = find_labeled_subgraph(
            source, target, labeling_method="orientation", timeout=1
        )

        self.assertEqual(set(embedding.keys()), set(source.nodes()))

    def test_coloring_labeling_method_computes_labels_internally(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = chimera_graph(2, t=2, coordinates=True)

        embedding = find_labeled_subgraph(
            source, target, labeling_method="coloring", timeout=1
        )

        self.assertEqual(set(embedding.keys()), set(source.nodes()))

    def test_quotient_labeling_method_computes_labels_internally(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = chimera_graph(2, t=2, coordinates=True)

        embedding = find_labeled_subgraph(
            source, target, labeling_method="quotient", timeout=1
        )

        self.assertEqual(set(embedding.keys()), set(source.nodes()))

    def test_quotient_labeling_method_raises_for_mismatched_families(self):
        source = chimera_graph(2, t=2, coordinates=True)
        target = zephyr_graph(2, 2, coordinates=True)

        with self.assertRaises(ValueError):
            find_labeled_subgraph(source, target, labeling_method="quotient")

    def test_quotient_labeling_method_raises_for_unsupported_family(self):
        source = nx.path_graph(4)
        source.graph["family"] = "unsupported"
        target = nx.path_graph(4)
        target.graph["family"] = "unsupported"

        with self.assertRaises(ValueError):
            find_labeled_subgraph(source, target, labeling_method="quotient")

    def test_unknown_labeling_method_raises(self):
        source = nx.path_graph(2)
        target = nx.path_graph(2)

        with self.assertRaises(ValueError):
            find_labeled_subgraph(source, target, labeling_method="unknown")
