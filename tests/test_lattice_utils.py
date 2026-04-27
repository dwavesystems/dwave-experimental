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

import lzma
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dimod
import numpy as np

from dwave.experimental.lattice_utils.utils import (
    bootstrap,
    confidence_interval,
    generate_bootstrap_indices,
)
from dwave.experimental.lattice_utils.lattice.chain import Chain
from dwave.experimental.lattice_utils.lattice.triangular import DimerizedTriangular, Triangular
from dwave.experimental.lattice_utils.lattice.embedded_lattice import EmbeddedLattice
from dwave.experimental.lattice_utils.lattice.orbits import make_signed_bqm, reindex
from dwave.experimental.lattice_utils.lattice.optimize import optimize
from dwave.experimental.lattice_utils.observable.observable import (
    BitpackedSpins,
    CouplerCorrelation,
    CouplerFrustration,
    QubitMagnetization,
    ReferenceEnergy,
    SampleEnergy,
    get_reference_energy_path,
)
from dwave.experimental.lattice_utils.observable.kinks import KinkKinkCorrelator
from dwave.experimental.lattice_utils.observable.triangular import TriangularOP
from dwave.experimental.lattice_utils.experiment.samplercall import SamplerCall
from dwave.experimental.lattice_utils.experiment.experiment import Experiment
from dwave.experimental.lattice_utils.experiment.fast_anneal_experiment import FastAnnealExperiment


def _make_triangular(
    ly=3,
    lx=3,
    periodic=(True, False),
    orbit_type="singleton",
    halve_boundary_couplers=False,
):
    return Triangular(
        dimensions=(ly, lx),
        periodic=periodic,
        orbit_type=orbit_type,
        halve_boundary_couplers=halve_boundary_couplers,
    )


def _make_mock_sampler(num_qubits=128, nodelist=None, solver_name="TestSolver"):
    """Create a minimal mock sampler resembling DWaveSampler."""
    sampler = mock.MagicMock(spec=dimod.Sampler)
    type(sampler).__name__ = "DWaveSampler"
    if nodelist is None:
        nodelist = list(range(num_qubits))
    sampler.nodelist = nodelist
    sampler.properties = {"num_qubits": num_qubits}
    sampler.solver = mock.MagicMock()
    sampler.solver.name = solver_name
    return sampler


def _make_sync_sampler(n_cols=128, solver_name="TestSolver"):
    """Sampler whose sample() immediately returns all-ones raw data (done=True).

    Mimics the async response interface used by DWaveSampler: .done() and
    .samples() -> 2-D ndarray of shape (num_reads, n_cols).
    """
    class _Response:
        def done(self):
            return True

        def samples(self, sorted_by=None):
            return np.ones((10, n_cols), dtype=float)

    s = mock.MagicMock()
    type(s).__name__ = "DWaveSampler"
    s.solver.name = solver_name
    s.nodelist = list(range(n_cols))
    s.properties = {"num_qubits": n_cols}
    s.sample.return_value = _Response()
    return s


def _make_mock_experiment(
    inst,
    energy_scale=1.0,
    run_index=0,
    num_random_instances=1,
    extra_params=None
):
    """Return a lightweight mock Experiment with .inst and .param."""
    exp = mock.MagicMock()
    exp.inst = inst
    exp.param = {"energy_scale": energy_scale, "num_random_instances": num_random_instances}
    exp.run_index = run_index
    if extra_params:
        exp.param.update(extra_params)
    return exp


def _make_embedded_chain(chain_nodes):
    return EmbeddedLattice(
        logical_lattice_class=Chain,
        logical_lattice_kwargs={
            "dimensions": (len(chain_nodes),),
            "periodic": (False,),
            "ignore_embedding": True,
        },
        chain_nodes=chain_nodes,
        dimensions=(sum(len(chain) for chain in chain_nodes.values()),),
        periodic=(False,),
    )


class TestUtils(unittest.TestCase):
    def test_bootstrap_all_nan_skipnan(self):
        result = bootstrap(np.array([np.nan, np.nan]), repetitions=5, skipnan=True)
        self.assertEqual(len(result), 5)
        for val in result:
            self.assertTrue(np.isnan(val))

    def test_bootstrap_skipnan_false(self):
        result = bootstrap(np.array([1.0, 2.0, np.nan]), repetitions=5, skipnan=False)
        self.assertEqual(len(result), 5)

    def test_bootstrap_custom_function(self):
        result = bootstrap(np.arange(20), repetitions=10, bootstrap_function=np.mean, seed=0)
        self.assertEqual(len(result), 10)

    def test_bootstrap_seed_reproducibility(self):
        r1 = bootstrap(np.arange(10), repetitions=20, seed=123)
        r2 = bootstrap(np.arange(10), repetitions=20, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_generate_bootstrap_indices_correct_count(self):
        indices = list(generate_bootstrap_indices(10, 5, seed=0))
        self.assertEqual(len(indices), 5)
        for idx in indices:
            self.assertEqual(len(idx), 10)
            self.assertTrue(np.all(idx >= 0))
            self.assertTrue(np.all(idx < 10))

    def test_confidence_interval_width(self):
        arr = np.arange(1000)
        _, low1, high1 = confidence_interval(arr, width=0.5)
        _, low2, high2 = confidence_interval(arr, width=0.99)
        self.assertGreater(low2 + high2, low1 + high1)


class TestChain(unittest.TestCase):
    def test_periodic(self):
        chain = Chain(dimensions=(6,), periodic=(True,))
        self.assertEqual(chain.num_spins, 6)
        self.assertEqual(chain.num_edges, 6)
        self.assertIn((5, 0), chain.edge_list)

    def test_non_periodic(self):
        chain = Chain(dimensions=(6,), periodic=(False,))
        self.assertEqual(chain.num_spins, 6)
        self.assertEqual(chain.num_edges, 5)
        self.assertNotIn((5, 0), chain.edge_list)

    def test_single_node_periodic(self):
        chain = Chain(dimensions=(1,), periodic=(True,))
        self.assertEqual(chain.num_spins, 1)
        self.assertEqual(chain.num_edges, 0)

    def test_two_node_periodic(self):
        chain = Chain(dimensions=(2,), periodic=(True,))
        self.assertEqual(chain.num_edges, 2)

    def test_geometry_name(self):
        chain = Chain(dimensions=(6,), periodic=(True,))
        self.assertEqual(chain.geometry_name, "Chain")


class TestLattice(unittest.TestCase):
    def test_default_periodic(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        self.assertFalse(chain.periodic[0])

    def test_edge_list_sorted(self):
        chain = Chain(dimensions=(5,), periodic=(False,))
        for u, v in chain.edge_list:
            self.assertLess(u, v)

    def test_bqm_structure(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        self.assertEqual(len(bqm.variables), 4)
        self.assertEqual(len(bqm.quadratic), 3)
        for u, v in chain.edge_list:
            self.assertAlmostEqual(bqm.quadratic[(u, v)], 1.0)

    def test_bqm_vartype(self):
        bqm = Chain(dimensions=(3,), periodic=(True,)).make_nominal_bqm()
        self.assertEqual(bqm.vartype, dimod.SPIN)

    def test_orbit_singleton(self):
        chain = Chain(dimensions=(4,), periodic=(True,), orbit_type="singleton")
        np.testing.assert_array_equal(chain.qubit_orbits, np.arange(4))
        np.testing.assert_array_equal(chain.coupler_orbits, np.arange(chain.num_edges))

    def test_orbit_global(self):
        chain = Chain(dimensions=(4,), periodic=(True,), orbit_type="global")
        np.testing.assert_array_equal(chain.qubit_orbits, np.zeros(4, dtype=int))
        np.testing.assert_array_equal(chain.coupler_orbits, np.zeros(chain.num_edges, dtype=int))

    def test_orbit_explicit(self):
        chain = Chain(
            dimensions=(4,),
            periodic=(True,),
            orbit_type="explicit",
            qubit_orbits=np.array([0, 0, 1, 1]),
            coupler_orbits=np.array([0, 0, 1, 1]),
        )
        np.testing.assert_array_equal(chain.qubit_orbits, [0, 0, 1, 1])

    def test_unknown_orbit_type(self):
        with self.assertRaises(ValueError):
            Chain(dimensions=(4,), periodic=(True,), orbit_type="bogus")

    def test_get_path_invalid_kind(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        with self.assertRaises(ValueError):
            chain._get_path(None, "invalid")

    def test_standard_orbit_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(
                dimensions=(4,),
                periodic=(True,),
                orbit_type="standard",
                lattice_data_root=Path(tmpdir),
            )
            self.assertIsNotNone(chain.qubit_orbits)
            self.assertIsNotNone(chain.coupler_orbits)
            # Second instantiation should load from disk
            chain2 = Chain(
                dimensions=(4,),
                periodic=(True,),
                orbit_type="standard",
                lattice_data_root=Path(tmpdir),
            )
            np.testing.assert_array_equal(chain.qubit_orbits, chain2.qubit_orbits)

    def test_nested_embedded_raises(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        # Fake a nested embedded lattice
        dt.logical_lattice.logical_lattice = mock.MagicMock()
        dt.orbit_type = "global"
        with self.assertRaises(NotImplementedError):
            dt.initialize_orbits()

    def test_embed_no_embeddings_found(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = mock.MagicMock()
        type(sampler).__name__ = "MockDWaveSampler"
        sampler.to_networkx_graph.return_value = chain._make_networkx_graph()

        with mock.patch(
            "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings",
            return_value=[],
        ):
            with self.assertRaises(ValueError):
                chain.embed_lattice(sampler, try_to_load=False, timeout=1)

    def test_embed_load_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,), lattice_data_root=Path(tmpdir))
            sampler = mock.MagicMock()
            type(sampler).__name__ = "MockDWaveSampler"
            sampler.to_networkx_graph.return_value = chain._make_networkx_graph()

            embeddings = np.array([[0, 1, 2, 3]])
            chain._save_embeddings(sampler, embeddings, data_root=Path(tmpdir))

            chain.embed_lattice(sampler, try_to_load=True, data_root=Path(tmpdir))
            np.testing.assert_array_equal(chain.embedding_list, embeddings)

    def test_embed_find_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,), lattice_data_root=Path(tmpdir))
            sampler = mock.MagicMock()
            type(sampler).__name__ = "MockDWaveSampler"
            sampler.to_networkx_graph.return_value = chain._make_networkx_graph()

            emb_dict = {i: i for i in range(4)}
            with mock.patch(
                "dwave.experimental.lattice_utils.lattice.lattice.find_multiple_embeddings",
                return_value=[emb_dict],
            ):
                chain.embed_lattice(sampler, try_to_load=False, timeout=1,
                                    data_root=Path(tmpdir))
            # Verify embedding was found and saved
            emb_path = chain._get_path(Path(tmpdir), "embedding", sampler_name="MockDWaveSampler")
            self.assertTrue(emb_path.exists())


class TestTriangular(unittest.TestCase):
    def test_basic_construction(self):
        tri = _make_triangular(3, 3)
        self.assertEqual(tri.num_spins, 9)
        self.assertGreater(tri.num_edges, 0)
        self.assertEqual(tri.geometry_name, "Triangular")

    def test_coordinates(self):
        tri = _make_triangular(3, 3)
        y, x = tri.coordinates(0)
        self.assertEqual(y, 0)
        self.assertEqual(x, 0)
        y, x = tri.coordinates(4)
        self.assertEqual(y, 1)
        self.assertEqual(x, 1)

    def test_halve_boundary_couplers(self):
        tri = _make_triangular(3, 3, periodic=(False, False), halve_boundary_couplers=True)
        bqm = tri.make_nominal_bqm()
        graph = tri._make_networkx_graph()
        for u, v in tri.edge_list:
            expected = 1.0 if (graph.degree[u] == 6 or graph.degree[v] == 6) else 0.5
            self.assertAlmostEqual(bqm.quadratic[(u, v)], expected)

    def test_periodicity(self):
        tri = _make_triangular(3, 3, periodic=(False, True))
        self.assertFalse(tri.periodic[0])
        self.assertTrue(tri.periodic[1])

class TestDimerizedTriangular(unittest.TestCase):
    def test_basic_construction(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        self.assertEqual(dt.geometry_name, "DimerizedTriangular")
        self.assertIsNotNone(dt.logical_lattice)
        self.assertEqual(dt.num_spins, 18)

    def test_chain_connectivity_self(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        cc = dt.get_chain_connectivity(0)
        self.assertEqual(cc, ((0, 1),))

    def test_chain_connectivity_cases(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        cases = [
            ((0,), ((0, 1),)),
            ((0, 1), ((1, 0),)),
            ((0, 3), ((1, 0),)),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                self.assertEqual(dt.get_chain_connectivity(*args), expected)


class TestEmbeddedLattice(unittest.TestCase):
    def test_embed_sample(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        logical_sample = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])
        embedded = dt.embed_sample(logical_sample)
        self.assertEqual(len(embedded), dt.num_spins)
        # Each chain should have the same value
        for spin, chain in dt.chain_nodes.items():
            for node in chain:
                self.assertEqual(embedded[node], logical_sample[spin])

    def test_unembed_sample(self):
        embedded = _make_embedded_chain({0: (0, 1, 2), 1: (3, 4, 5)})
        physical_sample = np.array([1, 1, -1, -1, -1, 1])
        logical = embedded.unembed_sample(physical_sample)
        np.testing.assert_array_equal(logical, np.array([1, -1]))

    def test_unembed_sample_breaks_ties_randomly(self):
        embedded = _make_embedded_chain({0: (0, 1), 1: (2, 3)})
        physical_sample = np.array([1, -1, 1, -1])
        with mock.patch(
            "dwave.experimental.lattice_utils.lattice.embedded_lattice.np.random.rand",
            side_effect=[0.9, 0.1],
        ):
            logical = embedded.unembed_sample(physical_sample)
        np.testing.assert_array_equal(logical, np.array([1, -1]))

    def test_unembed_sampleset(self):
        embedded = _make_embedded_chain({0: (0, 1), 1: (2, 3)})
        samples = np.array([
            [1, 1, -1, -1],
            [1, -1, 1, -1],
        ])
        ss = dimod.SampleSet.from_samples(samples, vartype=dimod.SPIN, energy=0)
        with mock.patch(
            "dwave.experimental.lattice_utils.lattice.embedded_lattice.np.random.rand",
            return_value=np.array([[0.2, 0.9], [0.2, 0.1]]),
        ):
            result = embedded.unembed_sampleset(ss)
        np.testing.assert_array_equal(dimod.as_samples(result)[0], np.array([[1, -1], [1, -1]]))

    def test_connectivity_generic_self(self):
        # Use a simple embedded lattice with chain_nodes of length 3
        chain_nodes = {0: (10, 11, 12), 1: (20, 21, 22)}
        el = _make_embedded_chain(chain_nodes)
        # Generic self-connectivity: all combinations within the chain
        cc = EmbeddedLattice.get_chain_connectivity(el, 0)
        self.assertEqual(cc, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(
            {tuple(chain_nodes[0][index] for index in edge) for edge in cc},
            {(10, 11), (10, 12), (11, 12)},
        )


class TestOrbits(unittest.TestCase):
    def test_reindex_basic(self):
        mapping = {"a": 5, "b": 5, "c": 10}
        result = reindex(mapping)
        self.assertEqual(result, {'a': 0, 'b': 0, 'c': 1})

    def test_signed_bqm_symmetry(self):
        bqm = dimod.BQM(vartype="SPIN")
        bqm.add_variable(0, 0.5)
        bqm.add_variable(1, -0.3)
        bqm.add_quadratic(0, 1, 1.0)
        signed = make_signed_bqm(bqm)
        self.assertAlmostEqual(signed.linear["p0"], 0.5)
        self.assertAlmostEqual(signed.linear["m0"], -0.5)


class TestOptimize(unittest.TestCase):
    def test_plain_lattice(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        energy, sample, _ = optimize(chain, bqm, sa_kwargs={"num_sweeps": 256, "num_reads": 16})
        self.assertEqual(energy, -3.0)
        self.assertEqual(bqm.energy(sample), energy)

    def test_embedded_lattice(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        bqm = dt.make_nominal_bqm()
        energy, sample, _ = optimize(dt, bqm, sa_kwargs={"num_sweeps": 256, "num_reads": 16})
        self.assertEqual(len(sample), dt.num_spins)
        self.assertEqual(bqm.energy(sample), energy)
        self.assertTrue(set(sample).issubset({-1, 1}))


class TestObservables(unittest.TestCase):
    def test_qubit_magnetization(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        samples = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]])
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(chain)
        result = QubitMagnetization().evaluate(exp, bqm, ss)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])

    def test_coupler_correlation(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        exp = _make_mock_experiment(chain)
        alt = np.tile([1, -1, 1, -1], (4, 1))
        ss_alt = dimod.SampleSet.from_samples_bqm(alt, bqm)
        np.testing.assert_array_equal(
            CouplerCorrelation().evaluate(exp, bqm, ss_alt), -np.ones(chain.num_edges)
        )

    def test_coupler_frustration(self):
        # All aligned (corr=1) -> frustration = 1.0
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        samples = np.ones((4, 4))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(chain)
        np.testing.assert_array_almost_equal(
            CouplerFrustration().evaluate(exp, bqm, ss), np.ones(chain.num_edges)
        )

    def test_sample_energy(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        # All-ones: energy = sum of J for 3 edges = 3.0
        samples = np.ones((1, 4))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp_pos = _make_mock_experiment(chain, energy_scale=1.0)
        np.testing.assert_array_almost_equal(
            SampleEnergy().evaluate(exp_pos, bqm, ss), [3.0]
        )

    def test_bitpacked_spins(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        bqm = chain.make_nominal_bqm()
        samples = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(chain)
        packed, shape = BitpackedSpins().evaluate(exp, bqm, ss)
        self.assertEqual(shape, (2, 4))
        # Unpack and verify round-trip
        unpacked = np.unpackbits(packed)[:shape[0] * shape[1]].reshape(shape)
        np.testing.assert_array_equal(unpacked, np.equal(samples, 1))

    def test_reference_energy_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.txt"
            chain = Chain(dimensions=(4,), periodic=(False,))
            bqm = chain.make_nominal_bqm()
            sample = np.array([1, -1, 1, -1])
            obs = ReferenceEnergy()
            obs.save(path, -3.0, sample, "SA")

            exp = _make_mock_experiment(chain)
            energy, loaded_sample, method = obs.load(exp, bqm, path)
            self.assertAlmostEqual(energy, -3.0)
            self.assertEqual(method, "SA")
            np.testing.assert_array_equal(loaded_sample, sample)

    def test_reference_energy_evaluate_generates_and_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(False,), lattice_data_root=Path(tmpdir))
            bqm = chain.make_nominal_bqm()
            obs = ReferenceEnergy()

            path1 = Path(tmpdir) / "ref_inst.txt"
            energy1 = obs.evaluate(None, bqm, None, path=path1, inst=chain)
            self.assertTrue(path1.exists())
            # Second call loads from cache — same value
            energy1b = obs.evaluate(None, bqm, None, path=path1)
            self.assertAlmostEqual(energy1, energy1b)

            exp = _make_mock_experiment(chain, run_index=0, num_random_instances=1)
            path2 = Path(tmpdir) / "ref_exp.txt"
            energy2 = obs.evaluate(exp, bqm, None, path=path2)
            self.assertTrue(path2.exists())
            self.assertAlmostEqual(energy1, energy2)

    def test_reference_energy_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.txt"
            chain = Chain(dimensions=(4,), periodic=(False,))
            bqm = chain.make_nominal_bqm()
            obs = ReferenceEnergy()
            exp = _make_mock_experiment(chain)

            bad_sample = np.ones(4)
            obs.save(path, bqm.energy(bad_sample), bad_sample, "SA")

            better = np.array([1, -1, 1, -1])
            obs.update(exp, bqm, better, path=path)
            energy, _, _ = obs.load(exp, bqm, path)
            self.assertAlmostEqual(energy, bqm.energy(better))

            # Attempting to update with a worse sample raises ValueError
            with self.assertRaises(ValueError):
                obs.update(exp, bqm, bad_sample, path=path)

    def test_reference_energy_path(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        bqm = chain.make_nominal_bqm()
        exp = _make_mock_experiment(chain)

        with self.assertRaises(NotImplementedError):
            get_reference_energy_path(experiment=None, bqm=None)

        path = get_reference_energy_path(experiment=exp, bqm=bqm)
        self.assertTrue(str(path).endswith(".txt"))

        # Via dummy data dict (experiment=None)
        dummy = {"run_index": 0, "num_random_instances": 1, "inst": chain}
        path2 = get_reference_energy_path(bqm=bqm, dummy_experiment_data_dict=dummy)
        self.assertTrue(str(path2).endswith(".txt"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path3 = get_reference_energy_path(experiment=exp, bqm=bqm, root=tmpdir)
            self.assertIn(tmpdir, str(path3))


class TestKinks(unittest.TestCase):
    def test_all_aligned(self):
        chain = Chain(dimensions=(6,), periodic=(True,))
        bqm = chain.make_nominal_bqm()
        samples = np.ones((10, 6))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(chain)
        result = KinkKinkCorrelator().evaluate(exp, bqm, ss)
        # All neighbors aligned -> every site is a "kink" (K=1 everywhere)
        np.testing.assert_array_equal(result, np.zeros(6))

    def test_mixed_pattern(self):
        chain = Chain(dimensions=(6,), periodic=(True,))
        bqm = chain.make_nominal_bqm()
        # [1,1,-1,-1,1,1]: kink at sites 2,4 (domain walls)
        samples = np.tile([1, 1, -1, -1, 1, 1], (20, 1))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(chain)
        result = KinkKinkCorrelator().evaluate(exp, bqm, ss)
        expected = np.array([0.0, -0.25, 0.125, -0.25, 0.125, -0.25])
        np.testing.assert_array_almost_equal(result, expected)


class TestTriangularOP(unittest.TestCase):
    def test_uniform_state_vanishes(self):
        tri = _make_triangular(3, 3, periodic=(True, False))
        bqm = tri.make_nominal_bqm()
        samples = np.ones((5, 9))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(tri)
        result = TriangularOP().evaluate(exp, bqm, ss)
        # Uniform spins: equal sublattice mags cancel
        np.testing.assert_array_almost_equal(np.abs(result), np.zeros(5), decimal=10)

    def test_evaluate_embedded(self):
        dt = DimerizedTriangular(dimensions=(3, 3), periodic=(True, False), orbit_type="singleton")
        bqm = dt.make_nominal_bqm()
        # Uniform embedded spins
        samples = np.ones((5, dt.num_spins))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
        exp = _make_mock_experiment(dt)
        result = TriangularOP().evaluate(exp, bqm, ss)
        np.testing.assert_array_almost_equal(np.abs(result), np.zeros(5), decimal=10)


class TestSamplerCall(unittest.TestCase):
    def test_defaults(self):
        sc = SamplerCall(run_index=0)
        self.assertEqual(sc.run_index, 0)
        self.assertIsNone(sc.bqm)
        self.assertEqual(sc.shimdata, {})
        self.assertEqual(sc.nominal_bqms, [])
        self.assertEqual(sc.sampler_params, {})

    def test_with_values(self):
        bqm = dimod.BQM(vartype="SPIN")
        sc = SamplerCall(
            run_index=5,
            bqm=bqm,
            shimdata={"total_iterations": 1},
            nominal_bqms=[bqm],
            sampler_params={"num_reads": 100},
        )
        self.assertEqual(sc.run_index, 5)
        self.assertIs(sc.bqm, bqm)
        self.assertEqual(sc.shimdata["total_iterations"], 1)


class TestExperiment(unittest.TestCase):
    def test_default_params(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler)
        self.assertEqual(exp.param["energy_scale"], 1.0)
        self.assertEqual(exp.param["num_reads"], 100)
        self.assertIs(exp.inst, chain)

    def test_results_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            self.assertEqual(exp.experiment_results_root, Path(tmpdir).resolve())

    def test_data_path_with_schedule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir, anneal_time=5.0)
            exp.param["anneal_schedule"] = [(0, 1), (5, 0.5)]
            del exp.param["anneal_time"]
            exp.apply_param({"energy_scale": 1.0, "anneal_schedule": [(0, 1), (5, 0.5)]})
            self.assertIn("asched", str(exp.data_path))

    def test_apply_param_unknown_sampler_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            type(sampler).__name__ = "UnknownSampler"
            exp = Experiment(chain, sampler, results_root=tmpdir)
            with self.assertRaises(TypeError):
                exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})

    def test_apply_param_no_anneal_or_schedule_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            del exp.param["anneal_time"]
            with self.assertRaises(ValueError):
                exp.apply_param({"energy_scale": 1.0})

    def test_spin_reversal_disabled(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler)
        self.assertIsNone(exp._get_spin_reversal_transform())

    def test_spin_reversal_enabled_with_seed(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler(num_qubits=8)
        exp = Experiment(chain, sampler)
        exp.param["spin_reversal_transform"] = True
        exp.param["spin_reversal_transform_seed"] = 42
        srt1 = exp._get_spin_reversal_transform()
        srt2 = exp._get_spin_reversal_transform()
        self.assertEqual(srt1, srt2)

    def test_initial_shim_no_embeddings(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler)
        exp.already_initialized = False
        shimdata = exp._make_initial_shim()
        self.assertEqual(shimdata["total_iterations"], 0)
        self.assertNotIn("flux_biases", shimdata)

    def test_initial_shim_with_embeddings(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        chain.embedding_list = np.array([[0, 1, 2, 3]])
        sampler = _make_mock_sampler(num_qubits=128)
        exp = Experiment(chain, sampler)
        shimdata = exp._make_initial_shim()
        self.assertIn("flux_biases", shimdata)
        self.assertEqual(len(shimdata["flux_biases"]), 128)

    def test_initial_shim_with_preset_flux_biases(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        chain.embedding_list = np.array([[0, 1, 2, 3]])
        sampler = _make_mock_sampler(num_qubits=128)
        fb = np.ones(128) * 0.01
        exp = Experiment(chain, sampler)
        exp.param["flux_biases"] = fb
        shimdata = exp._make_initial_shim()
        np.testing.assert_array_almost_equal(shimdata["flux_biases"], fb)

    def test_load_shim_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            exp.run_index = 1

            shimdata = {"total_iterations": 5, "flux_biases": np.zeros(10)}
            data = {"shimdata": shimdata}
            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            with lzma.open(fn, "wb") as f:
                pickle.dump(data, f)

            loaded = exp._load_shim()
            self.assertEqual(loaded["total_iterations"], 5)

    def test_load_shim_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            exp.run_index = 1
            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            fn.touch()

            with self.assertRaises(FileNotFoundError):
                exp._load_shim()

    def test_load_shim_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            exp.run_index = 1
            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            fn.write_bytes(b"not a valid lzma file")

            with self.assertRaises(OSError):
                exp._load_shim()

    def test_load_shim_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            exp.run_index = 1

            # No file exists at all - patch getsize to not fail early
            with mock.patch("os.path.getsize", return_value=100):
                with self.assertRaises(FileNotFoundError):
                    exp._load_shim()

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            exp.run_index = 0
            data = {"QubitMagnetization": np.zeros(4)}
            exp._save_results(data)
            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            self.assertTrue(fn.exists())

            with lzma.open(fn, "rb") as f:
                loaded = pickle.load(f)
            np.testing.assert_array_equal(loaded["QubitMagnetization"], np.zeros(4))

    def test_save_with_filename_and_run_index_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            with self.assertRaises(ValueError):
                exp._save_results({}, run_index=0, filename="test.pkl.lzma")

    def test_save_custom_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir)
            exp.data_path = Path(tmpdir)
            data = {"x": 1}
            exp._save_results(data, filename="custom.pkl.lzma")
            self.assertTrue((Path(tmpdir) / "custom.pkl.lzma").exists())

    def test_apply_param_sets_run_index_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir, anneal_time=1.0)
            exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})
            self.assertEqual(exp.run_index, 0)

    def test_apply_param_resumes_from_existing_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir, anneal_time=1.0)
            exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})
            for i in range(3):
                fn = exp.data_path / f"iter{i:05d}.pkl.lzma"
                fn.parent.mkdir(parents=True, exist_ok=True)
                with lzma.open(fn, "wb") as f:
                    pickle.dump({}, f)

            exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})
            self.assertEqual(exp.run_index, 3)

    def test_load_results_ignore_shim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir, anneal_time=1.0)
            exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})
            fn = exp.data_path / "iter00000.pkl.lzma"
            fn.parent.mkdir(parents=True, exist_ok=True)
            with lzma.open(fn, "wb") as f:
                pickle.dump({"value": 0, "shimdata": {}}, f)

            results = exp.load_results(num_iterations=1, ignore_shim=True)
            self.assertNotIn("shimdata", results[0])

    def test_load_results_starting_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir, anneal_time=1.0)
            exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})
            for i in range(10):
                fn = exp.data_path / f"iter{i:05d}.pkl.lzma"
                fn.parent.mkdir(parents=True, exist_ok=True)
                with lzma.open(fn, "wb") as f:
                    pickle.dump({"value": i, "shimdata": {}}, f)

            results = exp.load_results(num_iterations=3, starting_iteration=2)
            self.assertEqual(len(results), 3)

    def test_load_results_corrupted_lzma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(True,))
            sampler = _make_mock_sampler()
            exp = Experiment(chain, sampler, results_root=tmpdir, anneal_time=1.0)
            exp.apply_param({"energy_scale": 1.0, "anneal_time": 1.0})
            fn = exp.data_path / "iter00000.pkl.lzma"
            fn.parent.mkdir(parents=True, exist_ok=True)
            fn.write_bytes(b"corrupted data")

            with self.assertRaises(lzma.LZMAError):
                exp.load_results(num_iterations=1)

    def test_generate_data_type_conversions(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler)
        sc = SamplerCall(run_index=0)
        sc.shimdata = {"total_iterations": 1, "flux_biases": np.zeros(4)}

        results = {
            "QubitMagnetization": np.array([0.1, 0.2, 0.3, 0.4]),
            "Complex": np.array([1 + 2j, 3 + 4j]),
            "ListData": [1, 2, 3],
        }
        savedata = exp._generate_data_to_save(sc, results)
        self.assertEqual(savedata["QubitMagnetization"].dtype, np.float32)
        self.assertEqual(savedata["Complex"].dtype, np.complex64)
        self.assertEqual(savedata["shimdata"]["total_iterations"], 1)

    def test_make_bqm_no_embeddings(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler, energy_scale=0.5)
        sc = SamplerCall(run_index=0)
        sc.nominal_bqms = [chain.make_nominal_bqm()]
        sc.shimdata = {"total_iterations": 0}
        bqm = exp._make_bqm(sc)
        for u, v in chain.edge_list:
            self.assertAlmostEqual(bqm.quadratic[(u, v)], 0.5)

    def test_make_bqm_with_embeddings(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        chain.embedding_list = np.array([[0, 1, 2, 3]])
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler, energy_scale=1.0)
        sc = SamplerCall(run_index=0)
        sc.nominal_bqms = [chain.make_nominal_bqm()]
        sc.shimdata = {
            "total_iterations": 0,
            "relative_coupler_strength": np.ones((1, chain.num_edges)),
        }
        sc.spin_reversal_transform = None

        bqm = exp._make_bqm(sc)
        self.assertGreater(len(bqm.quadratic), 0)

    def test_run_iteration_basic(self):
        """run_iteration() exercises the full pipeline: build call, sample, parse, shim, save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(False,), lattice_data_root=Path(tmpdir))
            exp = Experiment(chain, _make_sync_sampler(), results_root=tmpdir,
                             anneal_time=1.0, max_iterations=1)
            chain._load_embeddings = mock.MagicMock()
            finished = exp.run_iteration([{"energy_scale": 1.0, "anneal_time": 1.0}])

            self.assertFalse(finished)
            result_files = list(exp.data_path.glob("iter*.pkl.lzma"))
            self.assertEqual(len(result_files), 1)
            with lzma.open(result_files[0], "rb") as f:
                data = pickle.load(f)
            self.assertIn("QubitMagnetization", data)
            self.assertIn("CouplerCorrelation", data)
            self.assertIn("shimdata", data)
            self.assertEqual(data["shimdata"]["total_iterations"], 1)

    def test_run_iteration_returns_true_when_finished(self):
        """run_iteration() returns True when max_iterations already reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(dimensions=(4,), periodic=(False,), lattice_data_root=Path(tmpdir))
            exp = Experiment(chain, _make_sync_sampler(), results_root=tmpdir,
                             anneal_time=1.0, max_iterations=0)
            chain._load_embeddings = mock.MagicMock()
            finished = exp.run_iteration([{"energy_scale": 1.0, "anneal_time": 1.0}])

            self.assertTrue(finished)
            self.assertEqual(list(exp.data_path.glob("iter*.pkl.lzma")), [])

    def test_flux_bias_shim_basic_update(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        chain.embedding_list = np.array([[0, 1, 2, 3]])
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler, flux_bias_shim_step=0.001)

        sc = SamplerCall(run_index=0)
        sc.shimdata = {"flux_biases": np.zeros(128), "total_iterations": 0}
        results = {"QubitMagnetization": np.array([0.1, -0.1, 0.2, -0.2])}
        exp._update_flux_bias_shim(sc, results)
        self.assertFalse(np.all(sc.shimdata["flux_biases"] == 0))

    def test_coupler_shim_basic_update(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        chain.embedding_list = np.array([[0, 1, 2, 3]])
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler, coupler_shim_step=0.01, energy_scale=1.0)

        sc = SamplerCall(run_index=0)
        bqm = chain.make_nominal_bqm()
        sc.nominal_bqms = [bqm]
        sc.shimdata = {
            "total_iterations": 0,
            "relative_coupler_strength": np.ones((1, chain.num_edges)),
        }
        results = {"CouplerFrustration": np.random.rand(1, chain.num_edges)}
        exp._update_coupler_shim(sc, results)
        self.assertEqual(sc.shimdata["relative_coupler_strength"].shape, (1, chain.num_edges))

    def test_parse_results_with_spin_reversal(self):
        chain = Chain(dimensions=(4,), periodic=(False,))
        chain.embedding_list = np.array([[0, 1, 2, 3]])
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler)
        exp.run_index = 0
        bqm = chain.make_nominal_bqm()
        samples = np.ones((10, 4))
        ss = dimod.SampleSet.from_samples_bqm(samples, bqm)

        sc = SamplerCall(run_index=0)
        sc.nominal_bqms = [bqm]
        sc.spin_reversal_transform = {0: True, 1: False, 2: True, 3: False}
        results = exp.parse_results(sc, ss)
        self.assertIn("QubitMagnetization", results)

    def test_get_shimdata_not_initialized(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler()
        exp = Experiment(chain, sampler)
        exp.already_initialized = False
        shimdata = exp._get_shimdata()
        self.assertEqual(shimdata["total_iterations"], 0)

class TestFastAnnealExperiment(unittest.TestCase):
    def test_default_params(self):
        chain = Chain(dimensions=(4,), periodic=(True,))
        sampler = _make_mock_sampler()
        exp = FastAnnealExperiment(chain, sampler)
        self.assertTrue(exp.param.get("fast_anneal"))
        self.assertEqual(exp.param["num_reads"], 100)

    def test_observables(self):
        obs_names = {type(o).__name__ for o in FastAnnealExperiment.observables_to_collect}
        self.assertIn("QubitMagnetization", obs_names)
        self.assertIn("SampleEnergy", obs_names)
        self.assertIn("ReferenceEnergy", obs_names)


if __name__ == "__main__":
    unittest.main()
