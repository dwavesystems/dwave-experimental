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

import dimod
import numpy as np
import networkx as nx

from dwave.system.testing import MockDWaveSampler
from dwave_networkx import chimera_graph, pegasus_graph, zephyr_graph

from dwave.experimental.automorphism import *


@dimod.testing.load_sampler_bqm_tests(AutomorphismComposite(dimod.ExactSolver()))
class TestAutomorphismComposite(unittest.TestCase):
    def test_instantiation(self):
        for factory in [
            dimod.ExactSolver,
            dimod.RandomSampler,
            dimod.SimulatedAnnealingSampler,
            dimod.NullSampler,
        ]:
            sampler = AutomorphismComposite(factory())

            dimod.testing.assert_sampler_api(sampler)
            dimod.testing.assert_composite_api(sampler)

    def test_NullSampler_composition(self):
        # Check NullSampler() works, this was a reported bug.
        sampler = AutomorphismComposite(dimod.NullSampler())
        sampleset = sampler.sample_ising({"a": 1}, {}, num_automorphisms=1)

        self.assertTrue(len(sampleset) == 0)
        sampleset = sampler.sample_ising({"a": 1}, {}, num_automorphisms=2)

        self.assertTrue(len(sampleset) == 0)

    def test_empty_bqm_composition(self):
        # Check NullSampler() works, this was a reported bug.

        sampler = AutomorphismComposite(dimod.RandomSampler())
        bqm = dimod.BinaryQuadraticModel("SPIN")
        sampleset = sampler.sample(bqm, num_automorphisms=1)
        self.assertEqual(len(sampleset.variables), 0)

    def test_concatenation_stripping(self):
        # Check samplesets are not stripped of information
        # under default operation

        # Simple sampler with an info field.
        # When multiple samplesets needn't be recombined, this should
        # be maintained
        class RichSampler(dimod.NullSampler):
            def sample_ising(self, *args, **kwargs):
                ss = super().sample_ising(*args, **kwargs)
                ss.info["hello"] = "world"
                return ss

        sampler = AutomorphismComposite(RichSampler())

        sampleset = sampler.sample_ising({0: 1}, {})
        self.assertTrue(hasattr(sampleset, "info"))

    def test_sampleset_size(self):
        # Check num_reads and num_automorphisms combine
        # for anticipated number of samples.

        sampler = AutomorphismComposite(dimod.RandomSampler())
        for num_automorphisms in [1, 2]:
            for num_reads in [1, 3]:
                with self.subTest(f"{num_reads} {num_automorphisms}"):
                    sampleset = sampler.sample_ising(
                        {0: 1},
                        {},
                        num_automorphisms=num_automorphisms,
                        num_reads=num_reads,
                    )
                    self.assertEqual(
                        sum(sampleset.record.num_occurrences),
                        num_reads * num_automorphisms,
                    )

    def test_empty(self):
        # Check that empty BQMs are handled
        sampler = AutomorphismComposite(dimod.ExactSolver())
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        sampleset = sampler.sample(bqm, num_automorphisms=3)

        self.assertEqual(sampleset.record.sample.shape, (0, 0))
        self.assertIs(sampleset.vartype, bqm.vartype)

    def test_ground_state(self):
        num_variables = 10

        bqm = dimod.BQM({v: 1 for v in range(num_variables)}, {}, 0, "SPIN")

        class Sampler:
            def sample(self, bqm):
                sample = {v: +1 if bias < 0 else -1 for v, bias in bqm.linear.items()}
                return dimod.SampleSet.from_samples_bqm(sample, bqm)

        sampler = AutomorphismComposite(Sampler())
        sampleset = sampler.sample(bqm, num_automorphisms=num_variables)

        self.assertTrue((sampleset.record.sample == -1).all())

    def test_async(self):
        class SampleSet:
            is_done = False

            def done(self):
                return self.is_done

            def relabel_variables(self, mapping, inplace=True):
                # Non blocking.
                pass

            @property
            def record(self):
                raise Exception("boom")

            @property
            def variables(self):
                raise Exception("boom")

        class Sampler:
            def __init__(self):
                self.count = 0

            def sample(self, bqm):
                self.count += 1
                return SampleSet()

        sampler = AutomorphismComposite(Sampler())

        # No exception because nothing has been resolved
        sampleset = sampler.sample_ising({"a": 1}, {})

        # We can test whether it's done
        self.assertFalse(sampleset.done())
        SampleSet.is_done = True
        self.assertTrue(sampleset.done())

        # Resolving raises the exception
        with self.assertRaisesRegex(Exception, "boom"):
            sampleset.resolve()

    def test_seed(self):
        """Test mirrors SpinReversalTransform, spins should be reordere w.h.p"""
        bqm = dimod.BQM(1000, "SPIN")

        class Sampler:
            def sample(self, bqm):
                return dimod.SampleSet.from_samples_bqm(
                    [(-1) ** i for i in range(bqm.num_variables)], bqm
                )

        ss1 = AutomorphismComposite(Sampler(), seed=42).sample(bqm)
        ss2 = AutomorphismComposite(Sampler(), seed=42).sample(bqm)
        ss3 = AutomorphismComposite(Sampler(), seed=35).sample(bqm)
        self.assertTrue((ss1.record == ss2.record).all())
        self.assertFalse((ss1.record == ss3.record).all())

    def test_variable_order(self):

        class AlternatingSampler:
            """Return the same solution, but in alternating order."""

            def __init__(self):
                self.rng = np.random.default_rng(42)

            def sample(self, bqm):

                order = list(bqm.variables)
                self.rng.shuffle(order)

                sample = [
                    [-1 if bqm.linear[v] >= 0 else +1 for v in order],
                    [+1 if bqm.linear[v] >= 0 else -1 for v in order],
                ]

                return dimod.SampleSet.from_samples_bqm(
                    (sample, order), bqm, sort_labels=False
                )

        sampler = AutomorphismComposite(AlternatingSampler(), seed=42)

        bqm = dimod.BinaryQuadraticModel(
            {"a": -1, "b": 1, "c": -1, "d": -1.4}, {"ab": -2}, 0, "SPIN"
        )

        sampleset = sampler.sample(bqm, num_automorphisms=40)

        # there should only be two unique samples after SRTs
        self.assertEqual(len(sampleset.aggregate()), 2)

    def test_variable_order2(self):
        class GroundStateSampler:
            @staticmethod
            def sample(bqm):
                return dimod.ExactSolver().sample(bqm).lowest()

        bqm = dimod.BinaryQuadraticModel(
            {},
            {(1, 0): 0, (4, 3): 0, (2, 0): -1, (2, 1): 1, (5, 3): -1, (5, 4): 1},
            0,
            "SPIN",
        )

        sampler = AutomorphismComposite(GroundStateSampler(), seed=42)

        sampleset = sampler.sample(bqm, num_automorphisms=3)

        self.assertEqual(len(sampleset.aggregate()), 4)

    def test_variable_order3(self):
        class GroundStateSampler:
            @staticmethod
            def sample(bqm):
                return dimod.ExactSolver().sample(bqm).lowest()

        bqm = dimod.BinaryQuadraticModel({}, {(1, 0): 1, (2, 3): 1}, 0, "SPIN")

        sampler = AutomorphismComposite(GroundStateSampler(), seed=42)

        sampleset = sampler.sample(bqm, num_automorphisms=10)

        self.assertEqual(len(sampleset.aggregate()), 4)

    def test_propagation_of_info(self):
        # NB: info is not propagated when num_automorphisms is
        # greater than 1, as a best general aggregation method is not obvious.
        class InfoRichSampler:
            @staticmethod
            def sample(bqm):
                ss = dimod.ExactSolver().sample(bqm).lowest()
                ss.info["has_some"] = True
                return ss

        sampler = AutomorphismComposite(InfoRichSampler())

        bqm = dimod.BinaryQuadraticModel({0: 1}, {(0, 1): 1}, 0, "SPIN")

        sampleset = sampler.sample(bqm, num_automorphisms=1)

        self.assertTrue(hasattr(sampleset, "info"))
        self.assertEqual(sampleset.info, {"has_some": True})

    def test_mappings_argument(self):
        # All 1 ground state
        class Sampler:
            def sample(self, bqm):
                return dimod.SampleSet.from_samples_bqm(
                    [-(1**i) for i in range(bqm.num_variables)], bqm
                )

        num_var = 10
        bqm = dimod.BinaryQuadraticModel({i: -1 for i in range(num_var)}, {}, 0, "SPIN")

        sampler = Sampler()
        ss = sampler.sample(bqm)
        samples = ss.record.sample
        sampler = AutomorphismComposite(sampler)
        mappings = [{v: v for v in bqm.variables}]
        ss = sampler.sample(bqm, mappings=mappings)
        self.assertTrue(
            np.all(ss.record.sample == samples),
            "Neutral mappings leaves result unpermuted.",
        )

        ss = sampler.sample(bqm, mappings=[], num_automorphisms=0)
        self.assertTrue(
            np.all(ss.record.sample == samples),
            "Empty mappings also allows pass through "
            "(just like num_spin_reversals=0).",
        )

        mapping = {v: vr for v, vr in zip(bqm.variables, reversed(bqm.variables))}
        mappings = [mapping]
        ss = sampler.sample(bqm, mappings=mappings)

        self.assertTrue(
            np.all(ss.record.sample == (-1) ** num_var * samples),
            "Flip-all mappings inverts the order",
        )

        with self.subTest("srt shape"):
            mappings3 = [mapping.copy(), mapping, mapping]
            num_automorphisms = len(mappings3)
            mappings3[0].update({k: k for k in (0, num_var - 1)})
            ss = sampler.sample(bqm, mappings=mappings3)
            self.assertEqual(
                np.sum(ss.record.num_occurrences), len(mappings3), "Apply 3 mappings"
            )
            self.assertTrue(
                np.all(
                    (-1) ** m[v] == s[idx]
                    for m, s in zip(mappings3, ss.record.sample)
                    for idx, v in enumerate(ss.variables)
                )
            )

            with self.assertRaises(ValueError):
                # Inconsistent arguments
                ss = sampler.sample(
                    bqm, mappings=mappings, num_automorphisms=num_automorphisms + 1
                )

    def test_inheritance_and_u_vector(self):
        qpu = MockDWaveSampler(topology_type="chimera", topology_shape=[1, 2, 3])
        G = (
            nx.Graph()
        )  # to_networkx_graph() does not preserve node order, and can't be used unfortunately.
        G.add_nodes_from(qpu.nodelist)
        G.add_edges_from(qpu.edgelist)
        h = {i: 1 for i in qpu.nodelist}
        J = {e: 1 for e in qpu.edgelist}

        sampler1 = AutomorphismComposite(qpu)
        ss = sampler1.sample_ising(h, J, num_automorphisms=1, num_reads=3)
        self.assertEqual(np.sum(ss.record.num_occurrences), 3)
        unconstrained = dimod.RandomSampler()
        sampler2 = AutomorphismComposite(unconstrained, G=G)
        self.assertTrue(
            all(
                np.all(npa1 == npa2)
                for l1, l2 in zip(
                    sampler1.generators_u_vector, sampler2.generators_u_vector
                )
                for npa1, npa2 in zip(l1, l2)
            )
        )
        ss = sampler2.sample_ising(h, J, num_automorphisms=3, num_reads=1)
        self.assertEqual(np.sum(ss.record.num_occurrences), 3)
        idx_to_node = {idx: n for idx, n in enumerate(qpu.nodelist)}
        sampler3 = AutomorphismComposite(
            unconstrained,
            generators_u_vector=sampler2.generators_u_vector,
            idx_to_node=idx_to_node,
        )

        self.assertTrue(
            all(
                np.all(npa1 == npa2)
                for l1, l2 in zip(
                    sampler3.generators_u_vector, sampler2.generators_u_vector
                )
                for npa1, npa2 in zip(l1, l2)
            )
        )

        ss = sampler3.sample_ising(h, J, num_automorphisms=2, num_reads=2)
        self.assertEqual(np.sum(ss.record.num_occurrences), 4)

    def test_nontrivial_sampling(self):
        class Sampler:
            # First 2 qubits are (+1), other qubits are (-1)
            def sample(self, bqm, num_reads=1):
                return dimod.SampleSet.from_samples_bqm(
                    np.tile(np.arange(bqm.num_variables), (num_reads, 1)), bqm
                )

        m, n, t = [1, 2, 3]  # Chimera shape
        nodelist = list(
            range(1, m * n * t * 2)
        )  # Missing first (vertical) node labeled 0
        qpu = MockDWaveSampler(
            topology_type="chimera", topology_shape=[m, n, t], nodelist=nodelist
        )

        bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {n: (-1) ** n for n in qpu.nodelist}, {e: 0 for e in qpu.edgelist}
        )
        num_automorphisms = 10
        generators_listtuple = [
            ({1: 2, 2: 1}, 2),
            ({2 * t + i: 2 * t + (i % t) for i in range(t)}, t),
        ]  # Swap of yielded verticals on first Chimera cell, and rotation of verticals on last Chimera cell
        for sampler in [
            AutomorphismComposite(qpu),
            AutomorphismComposite(qpu, generators_listtuple=generators_listtuple),
        ]:
            # Uniformly sample all Chimera automorphisms with Schreir-Sims (default)
            # Every qubit should move with high probability
            ss = sampler.sample(bqm, num_reads=1, num_automorphisms=num_automorphisms)
            self.assertEqual(
                sum(ss.record.sample @ np.array([(-1) ** v for v in ss.variables])),
                (-bqm.num_variables * num_automorphisms),
                "All automorphisms should yield ground states -sign(h)",
            )

    def test_listtuples_to_arrays(self):
        # Commuting subset of automorphisms on a Chimera[1,2,3]
        m = 1
        n = 2
        t = 3
        generators_listtuple = [
            ({1: 2, 2: 1}, 2),
            ({2 * t + i: 2 * t + (i % t) for i in range(t)}, t),
        ]
        node_to_idx = {n: idx for idx, n in enumerate(range(1, m * n * t * 2))}
        generators_u_vector = listtuple_to_arrays(generators_listtuple, node_to_idx)
        self.assertEqual(len(generators_listtuple), len(generators_u_vector))
        res = [
            [np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
            [
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ],
        ]
        self.assertTrue(
            all(
                np.all(g1 == g2)
                for l1, l2 in zip(res, generators_u_vector)
                for g1, g2 in zip(l1, l2)
            )
        )

    def test_sample_automorphisms_listtuple(self):
        sigma = 3
        nodes = ["a", "b"]
        generator = {"a": "b", "b": "a"}
        generators_listtuple = [(generator, 2)]
        # significance test on Bernouilli, 4 sigma:
        mapping = None
        num_automorphisms = 100
        total_not_identity = 0
        for seed in range(num_automorphisms):
            # seed = None
            mapping = sample_automorphisms_listtuple(
                generators_listtuple=generators_listtuple, prng=seed, mapping=mapping
            )
            total_not_identity += int(mapping == generator)  # 1 of two options.
        self.assertLess(
            total_not_identity,
            num_automorphisms / 2 + sigma * np.sqrt(num_automorphisms / 4),
            "The seeds are set so this test will not fail for convenience of automated testing."
            "Seed randomization also passes with high probability per commented line.",
        )
        self.assertGreater(
            total_not_identity,
            num_automorphisms / 2 - sigma * np.sqrt(num_automorphisms / 4),
            "The seeds are set so this test will not fail for convenience of automated testing."
            "Seed randomization also passes with high probability per commented line.",
        )

        pass

    def test_dnx_graph_generators(self):
        dnx_shapes = {
            "chimera": [(1, 1, 1), (2, 2, 4), (3, 1, 3)],
            "zephyr": [(1, 1), (2, 3)],
            "pegasus": [(2,), (3,), (4,)],
        }
        for dnx_type, shapes in dnx_shapes.items():
            if dnx_type == "chimera":
                make_graph = chimera_graph
                make_generators = chimera_generators
            elif dnx_type == "pegasus":
                make_graph = pegasus_graph
                make_generators = pegasus_generators
            elif dnx_type == "zephyr":
                make_graph = zephyr_graph
                make_generators = zephyr_generators
            else:
                raise ValueError("Unknown dnx graph type")
            for shape in shapes:
                G = make_graph(*shape)
                rep = schreier_rep(G)
                generators_listtuple = make_generators(
                    *shape, generator_type="redundant"
                )
                for g, d in generators_listtuple:
                    self.assertEqual(type(d), int)
                    self.assertGreater(
                        d, 1
                    )  # Can be as large as 't' for shore permutations.
                    self.assertSetEqual(
                        set(g.keys()),
                        set(g.values()),
                        "Every automorphism defines a 1:1 mapping on a subspace",
                    )
                gen_sample_space = np.prod([d for _, d in generators_listtuple])
                self.assertEqual(
                    rep.num_automorphisms,
                    gen_sample_space,
                    "Uniform sampling anticipated",
                )
                generators_listtuple2 = make_generators(
                    *shape, generator_type="strongest"
                )
                self.assertLessEqual(
                    len(generators_listtuple2), len(generators_listtuple)
                )

    def test_reduce_generator_by_node_set(self):
        # Chimera cell (biclique)
        t = 2 + np.random.randint(3)
        gens = chimera_generators(1, 1, t, "redundant")
        self.assertEqual(len(gens), 2 * t - 1)
        node_set = {(0, 0, u, k) for u in range(2) for k in range(t)}
        gens_p = [reduce_generator_by_node_set(gen, node_set) for gen, _ in gens]
        self.assertTrue(
            all(g1[0] == g2 for g1, g2 in zip(gens, gens_p)),
            "With full set of nodes, generators should be unchanged",
        )

        # Vertical generators survive, reflection generator survives but reduced in size (by 2), final horizontal generator is empty:
        node_set_p = node_set.difference({(0, 0, 1, t - 1)})
        gens_p = [reduce_generator_by_node_set(gen[0], node_set_p) for gen in gens]
        self.assertTrue(
            all(set(gen.keys()) == set(gen.values()) for gen in gens_p),
            "Returned generators should be closed",
        )

        modified_idxs = {0, t}  # Reflection, and rotation on horizontals
        self.assertEqual(gens_p[t], {}, "Rotation on all horizontals not viable")
        self.assertEqual(
            len(gens_p[0]), 2 * (t - 1), "All v to h matching except the last"
        )
        self.assertEqual(
            set(gens_p[0].keys()),
            set({(0, 0, u, k) for u in range(2) for k in range(t - 1)}),
            "Reflection excludes (0,0,0,t-1), because partner is missing",
        )
        self.assertTrue(
            all(
                gens[idx][0] == gens_p[idx]
                for idx in range(0, 2 * t - 2)
                if idx not in modified_idxs
            ),
            "All but first and last generators should be unchanged",
        )
