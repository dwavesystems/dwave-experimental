# Copyright 2026 D-Wave
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

import tempfile
import unittest
from unittest import mock

import dimod
import numpy as np
from dwave.samplers import SteepestDescentSolver
from dwave.system.testing import MockDWaveSampler

from dwave.experimental.lattice_utils import lattice
from tests.test_lattice_utils._helpers import _make_embedded_chain, _make_triangular


class TestChain(unittest.TestCase):
    def test_periodic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(6,), periodic=(True,), data_root=tmpdir)
            self.assertEqual(chain.num_spins, 6)
            self.assertEqual(chain.num_edges, 6)
            self.assertIn((5, 0), chain.edge_list)

    def test_non_periodic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(6,), periodic=(False,), data_root=tmpdir)
            self.assertEqual(chain.num_spins, 6)
            self.assertEqual(chain.num_edges, 5)
            self.assertNotIn((5, 0), chain.edge_list)

    def test_single_node_periodic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(1,), periodic=(True,), data_root=tmpdir)
            self.assertEqual(chain.num_spins, 1)
            self.assertEqual(chain.num_edges, 0)

    def test_geometry_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(6,), periodic=(True,), data_root=tmpdir)
            self.assertEqual(chain.geometry_name, "Chain")

    def test_default_periodic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), data_root=tmpdir)
            self.assertTrue(chain.periodic[0])

    def test_edge_list_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(5,), periodic=(False,), data_root=tmpdir)
            for u, v in chain.edge_list:
                self.assertLess(u, v)

    def test_bqm_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            self.assertEqual(len(bqm.variables), 4)
            self.assertEqual(len(bqm.quadratic), 3)
            for u, v in chain.edge_list:
                self.assertAlmostEqual(bqm.quadratic[(u, v)], 1.0)

    def test_bqm_vartype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bqm = lattice.Chain(dimensions=(3,), data_root=tmpdir).make_bqm()
            self.assertEqual(bqm.vartype, dimod.SPIN)


class TestLattice(unittest.TestCase):
    def test_orbit_singleton(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), orbit_type="singleton", data_root=tmpdir)
            np.testing.assert_array_equal(chain.qubit_orbits, np.arange(4))
            np.testing.assert_array_equal(chain.coupler_orbits, np.arange(chain.num_edges))

    def test_orbit_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), orbit_type="global", data_root=tmpdir)
            np.testing.assert_array_equal(chain.qubit_orbits, np.zeros(4, dtype=int))
            np.testing.assert_array_equal(
                chain.coupler_orbits, np.zeros(chain.num_edges, dtype=int)
            )

    def test_orbit_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(True,),
                orbit_type="explicit",
                qubit_orbits=np.array([0, 0, 1, 1]),
                coupler_orbits=np.array([0, 0, 1, 1]),
                data_root=tmpdir,
            )
            np.testing.assert_array_equal(chain.qubit_orbits, [0, 0, 1, 1])

    def test_unknown_orbit_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                lattice.Chain(
                    dimensions=(4,), periodic=(True,), orbit_type="bogus", data_root=tmpdir
                )

    def test_get_path_invalid_kind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            with self.assertRaises(ValueError):
                chain._get_path(None, "invalid")

    def test_standard_orbit_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(True,),
                orbit_type="standard",
                data_root=tmpdir,
            )
            self.assertIsNotNone(chain.qubit_orbits)
            self.assertIsNotNone(chain.coupler_orbits)
            # Second instantiation should load from disk
            chain2 = lattice.Chain(
                dimensions=(4,),
                periodic=(True,),
                orbit_type="standard",
                data_root=tmpdir,
            )
            np.testing.assert_array_equal(chain.qubit_orbits, chain2.qubit_orbits)

    def test_embed_no_embeddings_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = MockDWaveSampler()

            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings",
                return_value=[],
            ):
                with self.assertRaises(ValueError):
                    chain.embed_lattice(sampler, try_to_load=False, timeout=1)

    def test_embed_load_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = MockDWaveSampler()

            embeddings = np.array([[0, 1, 2, 3]])
            chain._save_embeddings(sampler, embeddings)

            chain.embed_lattice(sampler, try_to_load=True)
            np.testing.assert_array_equal(chain.embedding_list, embeddings)

    def test_embed_find_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = MockDWaveSampler()

            emb_dict = {i: i for i in range(4)}
            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings",
                return_value=[emb_dict],
            ):
                chain.embed_lattice(sampler, try_to_load=False, timeout=1)
            # Verify embedding was found and saved
            emb_path = chain._get_path("embedding", sampler_name="MockDWaveSampler")
            self.assertTrue(emb_path.exists())

    def test_embed_excluded_qubits_reuses_filtered_unrestricted_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), data_root=tmpdir)
            sampler = MockDWaveSampler()
            chain._save_embeddings(sampler, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))

            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings"
            ) as search:
                chain.embed_lattice(sampler, exclude_qubits=[1])

            search.assert_not_called()
            np.testing.assert_array_equal(chain.embedding_list, [[4, 5, 6, 7]])

    def test_embed_excluded_qubits_saves_and_loads_separate_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), data_root=tmpdir)
            sampler = MockDWaveSampler()
            unrestricted = np.array([[0, 1, 2, 3]])
            restricted = np.array([[10, 11, 12, 13]])
            chain._save_embeddings(sampler, unrestricted)

            with (
                mock.patch(
                    "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings",
                    return_value=[dict(enumerate(restricted[0]))],
                ) as search,
                self.assertWarns(UserWarning),
            ):
                chain.embed_lattice(sampler, exclude_qubits=[3, 1, 3])

            search.assert_called_once()
            np.testing.assert_array_equal(chain.embedding_list, restricted)
            np.testing.assert_array_equal(
                np.atleast_2d(np.loadtxt(chain._embedding_cache_filename(sampler), dtype=int)),
                unrestricted,
            )
            specific_cache = chain._embedding_cache_filename(sampler, [1, 3])
            self.assertTrue(specific_cache.exists())
            self.assertNotEqual(specific_cache, chain._embedding_cache_filename(sampler))

            another_chain = lattice.Chain(dimensions=(4,), data_root=tmpdir)
            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings"
            ) as search:
                another_chain.embed_lattice(sampler, exclude_qubits=[1, 3])

            search.assert_not_called()
            np.testing.assert_array_equal(another_chain.embedding_list, restricted)

    def test_embed_minimum_checked_after_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), data_root=tmpdir)
            sampler = MockDWaveSampler()
            chain._save_embeddings(sampler, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
            found = [dict(enumerate(row)) for row in ([8, 9, 10, 11], [12, 13, 14, 15])]

            with (
                mock.patch(
                    "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings",
                    return_value=found,
                ) as search,
                self.assertWarns(UserWarning),
            ):
                chain.embed_lattice(
                    sampler,
                    exclude_qubits=[1],
                    min_number_of_embeddings=2,
                    max_number_of_embeddings=2,
                )

            search.assert_called_once()
            np.testing.assert_array_equal(chain.embedding_list, [[8, 9, 10, 11], [12, 13, 14, 15]])


class TestLatticeOptimize(unittest.TestCase):
    """Tests for the ``Lattice.optimize`` instance method (vs. the free function
    ``lattice.optimize`` covered by ``TestOptimizeFunction``)."""

    def test_plain_lattice_private_optimize_default_sampler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(False,),
                data_root=tmpdir,
                reference_energy_sampler_kwargs={"num_sweeps": 256, "num_reads": 16},
            )
            bqm = chain.make_bqm()
            energy, sample, method = chain.optimize(bqm)

            self.assertEqual(energy, -3)
            self.assertEqual(bqm.energy(sample), energy)
            self.assertEqual(method, "ExponentialSweepsSimulatedAnnealingSampler")

    def test_plain_lattice_private_optimize_custom_sampler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(False,),
                data_root=tmpdir,
                reference_energy_sampler=dimod.ExactSolver(),
            )
            bqm = chain.make_bqm()
            energy, sample, method = chain.optimize(bqm)

            self.assertEqual(energy, -3)
            self.assertEqual(bqm.energy(sample), energy)
            self.assertEqual(method, "ExactSolver")

    def test_optimize_with_custom_sampler_steepest_descent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(False,),
                data_root=tmpdir,
                reference_energy_sampler=SteepestDescentSolver(),
                reference_energy_sampler_kwargs={"initial_states": np.array([[1, -1, 1, -1]])},
            )
            bqm = chain.make_bqm()
            energy, sample, method = chain.optimize(bqm)

            self.assertEqual(energy, -3)
            self.assertAlmostEqual(bqm.energy(sample), energy)
            self.assertEqual(method, "SteepestDescentSolver")

    def test_optimize_with_custom_exponential_sweeps_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = lattice.ExponentialSweepsSimulatedAnnealingSampler(
                max_num_sweeps=512, min_num_sweeps=64
            )
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(False,),
                data_root=tmpdir,
                reference_energy_sampler=sampler,
                reference_energy_sampler_kwargs={"num_reads": 16},
            )
            bqm = chain.make_bqm()
            energy, sample, method = chain.optimize(bqm)

            self.assertEqual(energy, -3)
            self.assertAlmostEqual(bqm.energy(sample), energy)
            self.assertEqual(method, "ExponentialSweepsSimulatedAnnealingSampler")
            self.assertEqual(sampler.max_num_sweeps, 512)
            self.assertEqual(sampler.min_num_sweeps, 64)


class TestTriangular(unittest.TestCase):
    def test_basic_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tri = lattice.Triangular(
                dimensions=(3, 3),
                periodic=(True, False),
                data_root=tmpdir,
                orbit_type="singleton",
                halve_boundary_couplers=False,
            )
            self.assertEqual(tri.num_spins, 9)
            self.assertGreater(tri.num_edges, 0)
            self.assertEqual(tri.geometry_name, "Triangular")

    def test_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tri = _make_triangular(tmpdir, 3, 3)
            y, x = tri.coordinates(0)
            self.assertEqual(y, 0)
            self.assertEqual(x, 0)
            y, x = tri.coordinates(4)
            self.assertEqual(y, 1)
            self.assertEqual(x, 1)

    def test_halve_boundary_couplers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tri = _make_triangular(
                tmpdir, 3, 3, periodic=(False, False), halve_boundary_couplers=True
            )
            bqm = tri.make_bqm()
            graph = tri.make_networkx_graph()
            for u, v in tri.edge_list:
                expected = 1.0 if (graph.degree[u] == 6 or graph.degree[v] == 6) else 0.5
                self.assertAlmostEqual(bqm.quadratic[(u, v)], expected)

    def test_periodicity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tri = _make_triangular(tmpdir, 3, 3, periodic=(False, True))
            self.assertFalse(tri.periodic[0])
            self.assertTrue(tri.periodic[1])


class TestDimerizedTriangular(unittest.TestCase):
    def test_basic_construction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = lattice.DimerizedTriangular(
                dimensions=(3, 3), periodic=(True, False), orbit_type="singleton", data_root=tmpdir
            )
            self.assertEqual(dt.geometry_name, "DimerizedTriangular")
            self.assertIsNotNone(dt.logical_lattice)
            self.assertEqual(dt.num_spins, 18)

    def test_chain_connectivity_self(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = lattice.DimerizedTriangular(dimensions=(3, 3), data_root=tmpdir)
            cc = dt.get_chain_connectivity(0)
            self.assertEqual(cc, ((0, 1),))

    def test_chain_connectivity_cases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = lattice.DimerizedTriangular(dimensions=(3, 3), data_root=tmpdir)
            cases = [
                ((0,), ((0, 1),)),
                ((0, 1), ((1, 0),)),
                ((0, 3), ((1, 0),)),
            ]
            for args, expected in cases:
                with self.subTest(args=args):
                    self.assertEqual(dt.get_chain_connectivity(*args), expected)


class TestCubic(unittest.TestCase):
    def test_open_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cube = lattice.Cubic(
                dimensions=(2, 2, 3),
                periodic=(False, False, False),
                data_root=tmpdir,
            )
            self.assertEqual(cube.num_spins, 12)
            self.assertEqual(cube.num_edges, 20)
            self.assertNotIn((8, 0), cube.edge_list)

    def test_default_z_periodic_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cube = lattice.Cubic(dimensions=(2, 2, 3), data_root=tmpdir)
            self.assertEqual(cube.num_spins, 12)
            self.assertEqual(cube.num_edges, 24)
            self.assertIn((8, 0), cube.edge_list)
            self.assertEqual(len({frozenset(edge) for edge in cube.edge_list}), cube.num_edges)


class TestCubicSpinGlass(unittest.TestCase):
    def test_seeded_couplings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for precision in (1, 4, 256):
                with self.subTest(precision=precision):
                    glass = lattice.CubicSpinGlass(
                        dimensions=(2, 2, 3),
                        data_root=tmpdir,
                        precision=precision,
                    )
                    bqm = glass.make_bqm(seed=7)
                    couplings = np.array([bqm.quadratic[edge] for edge in glass.edge_list])

                    self.assertEqual(bqm, glass.make_bqm(seed=7))
                    self.assertNotEqual(bqm, glass.make_bqm(seed=8))
                    self.assertEqual(bqm.vartype, dimod.SPIN)
                    self.assertEqual(set(bqm.variables), set(range(glass.num_spins)))
                    self.assertEqual(len(bqm.quadratic), glass.num_edges)
                    self.assertTrue(all(bias == 0 for bias in bqm.linear.values()))

                    if precision == 1:
                        self.assertTrue(set(couplings).issubset({-1, 1}))
                    else:
                        half = precision // 2
                        self.assertTrue(np.all(np.abs(couplings) > 0))
                        self.assertTrue(np.all(np.abs(couplings) <= 1))
                        np.testing.assert_allclose(
                            np.abs(couplings) * half,
                            np.rint(np.abs(couplings) * half),
                        )

    def test_invalid_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for precision in (0, 3, -2):
                with self.subTest(precision=precision), self.assertRaises(ValueError):
                    lattice.CubicSpinGlass(
                        dimensions=(2, 2, 2),
                        data_root=tmpdir,
                        precision=precision,
                    )
            for precision in (True, 2.5):
                with self.subTest(precision=precision), self.assertRaises(TypeError):
                    lattice.CubicSpinGlass(
                        dimensions=(2, 2, 2),
                        data_root=tmpdir,
                        precision=precision,
                    )


class TestDimerizedCubicSpinGlass(unittest.TestCase):
    def test_physical_bqm_preserves_logical_couplings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            glass = lattice.DimerizedCubicSpinGlass(
                dimensions=(2, 2, 2),
                periodic=(False, False, False),
                data_root=tmpdir,
                precision=4,
                chain_strength=2,
            )
            logical = glass.logical_lattice.make_bqm(seed=7)
            physical = glass.make_bqm(seed=7)

            self.assertEqual(glass.num_spins, 16)
            self.assertEqual(glass.num_edges, 24)  # 8 chains + 4 x + 4 y + 8 z
            self.assertEqual(
                {frozenset(edge) for edge in physical.quadratic},
                {frozenset(edge) for edge in glass.edge_list},
            )
            self.assertEqual(physical.quadratic[0, 8], -2)  # chain
            self.assertEqual(physical.quadratic[0, 1], logical.quadratic[0, 1])  # x
            self.assertEqual(physical.quadratic[8, 10], logical.quadratic[0, 2])  # y
            for edge in ((0, 12), (8, 4)):  # two halves of the z coupling
                self.assertEqual(physical.quadratic[edge], logical.quadratic[0, 4] / 2)
            self.assertEqual(physical, glass.make_bqm(seed=7))


class TestEmbeddedLattice(unittest.TestCase):
    def test_embed_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = lattice.DimerizedTriangular(dimensions=(3, 3), data_root=tmpdir)
            logical_sample = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])
            embedded = dt.embed_sample(logical_sample)
            self.assertEqual(len(embedded), dt.num_spins)
            # Each chain should have the same value
            for spin, chain in dt.chain_nodes.items():
                for node in chain:
                    self.assertEqual(embedded[node], logical_sample[spin])

    def test_unembed_sample(self):
        chain_nodes = {0: (0, 1, 2), 1: (3, 4, 5)}
        with tempfile.TemporaryDirectory() as tmpdir:
            embedded = lattice.EmbeddedLattice(
                logical_lattice=lattice.Chain(
                    dimensions=(len(chain_nodes),),
                    periodic=(False,),
                    data_root=tmpdir,
                ),
                chain_nodes=chain_nodes,
                dimensions=(sum(len(chain) for chain in chain_nodes.values()),),
                periodic=(False,),
            )
            physical_sample = np.array([1, 1, -1, -1, -1, 1])
            logical = embedded.unembed_sample(physical_sample)
            np.testing.assert_array_equal(logical, np.array([1, -1]))

    def test_unembed_sample_breaks_ties_randomly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedded = _make_embedded_chain({0: (0, 1), 1: (2, 3)}, tmpdir)
            physical_sample = np.array([1, -1, 1, -1])
            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.embedded_lattice.np.random.rand",
                side_effect=[0.9, 0.1],
            ):
                logical = embedded.unembed_sample(physical_sample)
            np.testing.assert_array_equal(logical, np.array([1, -1]))

    def test_unembed_sampleset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            embedded = _make_embedded_chain({0: (0, 1), 1: (2, 3)}, tmpdir)
            samples = np.array(
                [
                    [1, 1, -1, -1],
                    [1, -1, 1, -1],
                ]
            )
            ss = dimod.SampleSet.from_samples(samples, vartype=dimod.SPIN, energy=0)
            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.embedded_lattice.np.random.rand",
                return_value=np.array([[0.2, 0.9], [0.2, 0.1]]),
            ):
                result = embedded.unembed_sampleset(ss)
            np.testing.assert_array_equal(dimod.as_samples(result)[0], np.array([[1, -1], [1, -1]]))

    def test_connectivity_generic_self(self):
        chain_nodes = {0: (10, 11, 12), 1: (20, 21, 22)}
        with tempfile.TemporaryDirectory() as tmpdir:
            el = _make_embedded_chain(chain_nodes, tmpdir)
            cc = lattice.EmbeddedLattice.get_chain_connectivity(el, 0)
            self.assertEqual(cc, ((0, 1), (0, 2), (1, 2)))
            self.assertEqual(
                {tuple(chain_nodes[0][index] for index in edge) for edge in cc},
                {(10, 11), (10, 12), (11, 12)},
            )

    def test_nested_embedded_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_chain = lattice.Chain(
                dimensions=(2,),
                periodic=(False,),
                data_root=tmpdir,
            )
            embedded_once = lattice.EmbeddedLattice(
                logical_lattice=base_chain,
                chain_nodes={0: (0, 1), 1: (2, 3)},
                dimensions=(4,),
                periodic=(False,),
            )
            with self.assertRaises(NotImplementedError):
                lattice.EmbeddedLattice(
                    logical_lattice=embedded_once,
                    chain_nodes={0: (0, 1), 1: (2, 3), 2: (4, 5), 3: (6, 7)},
                    dimensions=(8,),
                    periodic=(False,),
                )


class TestOrbits(unittest.TestCase):
    def test_reindex_basic(self):
        mapping = {"a": 5, "b": 5, "c": 10}
        result = lattice.reindex(mapping)
        self.assertEqual(result, {'a': 0, 'b': 0, 'c': 1})

    def test_signed_bqm_symmetry(self):
        bqm = dimod.BQM(vartype="SPIN")
        bqm.add_variable(0, 0.5)
        bqm.add_variable(1, -0.3)
        bqm.add_quadratic(0, 1, 1.0)
        signed = lattice.make_signed_bqm(bqm)
        self.assertAlmostEqual(signed.linear["p0"], 0.5)
        self.assertAlmostEqual(signed.linear["m0"], -0.5)


class TestOptimizeFunction(unittest.TestCase):
    """Tests for the free function ``lattice.optimize`` (vs. the ``Lattice.optimize``
    instance method covered by ``TestLatticeOptimize``)."""

    def test_plain_lattice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            energy, sample, _ = lattice.optimize(
                chain, bqm, sampler_kwargs={"num_sweeps": 256, "num_reads": 16}
            )
            self.assertEqual(energy, -3)
            self.assertEqual(bqm.energy(sample), energy)

    def test_embedded_lattice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dt = lattice.DimerizedTriangular(dimensions=(3, 3), data_root=tmpdir)
            bqm = dt.make_bqm()
            energy, sample, _ = lattice.optimize(
                dt, bqm, sampler_kwargs={"num_sweeps": 256, "num_reads": 16}
            )
            self.assertEqual(len(sample), dt.num_spins)
            self.assertEqual(bqm.energy(sample), energy)
            self.assertTrue(set(sample).issubset({-1, 1}))


if __name__ == "__main__":
    unittest.main()
