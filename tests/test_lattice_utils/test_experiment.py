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

from dwave.experimental.lattice_utils import experiment, lattice
from tests.test_lattice_utils._helpers import _make_mock_sampler


class TestSamplerCall(unittest.TestCase):
    def test_defaults(self):
        sc = experiment.SamplerCall(run_index=0)
        self.assertEqual(sc.run_index, 0)
        self.assertIsNone(sc.bqm)
        self.assertEqual(sc.shimdata, {})
        self.assertEqual(sc.logical_bqms, [])
        self.assertEqual(sc.sampler_params, {})

    def test_with_values(self):
        bqm = dimod.BQM(vartype="SPIN")
        sc = experiment.SamplerCall(
            run_index=5,
            embedded_bqm=bqm,
            shimdata={"total_iterations": 1},
            logical_bqms=[bqm],
            sampler_params={"num_reads": 100},
        )
        self.assertEqual(sc.run_index, 5)
        self.assertIs(sc.bqm, bqm)
        self.assertEqual(sc.shimdata["total_iterations"], 1)


class TestExperimentInit(unittest.TestCase):
    def test_default_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            self.assertEqual(exp.param["signed_energy_scale"], 1.0)
            self.assertEqual(exp.param["num_reads"], 100)
            self.assertIs(exp.inst, chain)


class TestApplyParam(unittest.TestCase):
    def test_data_path_with_schedule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.apply_param({"signed_energy_scale": 1.0, "anneal_schedule": [(0, 1), (5, 0.5)]})
            self.assertIn("asched", str(exp.data_path))

    def test_apply_param_unknown_sampler_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler(type_name="UnknownSampler")
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            with self.assertRaises(TypeError):
                exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})

    def test_apply_param_sets_run_index_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})
            self.assertEqual(exp.run_index, 0)

    def test_apply_param_resumes_from_existing_iterations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})
            for i in range(3):
                fn = exp.data_path / f"iter{i:05d}.pkl.lzma"
                fn.parent.mkdir(parents=True, exist_ok=True)
                with lzma.open(fn, "wb") as f:
                    pickle.dump({}, f)

            exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})
            self.assertEqual(exp.run_index, 3)


class TestShimdata(unittest.TestCase):
    def test_initial_shim_no_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.already_initialized = False
            shimdata = exp._make_initial_shim()
            self.assertEqual(shimdata["total_iterations"], 0)
            self.assertNotIn("flux_biases", shimdata)

    def test_initial_shim_with_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            chain.embedding_list = np.array([[0, 1, 2, 3]])
            sampler = _make_mock_sampler(num_qubits=128)
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            shimdata = exp._make_initial_shim()
            self.assertIn("flux_biases", shimdata)
            self.assertEqual(len(shimdata["flux_biases"]), 128)

    def test_initial_shim_with_preset_flux_biases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            chain.embedding_list = np.array([[0, 1, 2, 3]])
            sampler = _make_mock_sampler(num_qubits=128)
            fb = np.ones(128) * 0.01
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.param["flux_biases"] = fb
            shimdata = exp._make_initial_shim()
            np.testing.assert_array_almost_equal(shimdata["flux_biases"], fb)

    def test_load_shim_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.run_index = 1
            exp.data_path = Path(tmpdir)

            shimdata = {"total_iterations": 5, "flux_biases": np.zeros(10)}
            data = {"shimdata": shimdata}
            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            with lzma.open(fn, "wb") as f:
                pickle.dump(data, f)

            loaded = exp._load_shim()
            self.assertEqual(loaded["total_iterations"], 5)

    def test_load_shim_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.run_index = 1
            exp.data_path = Path(tmpdir)

            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            fn.touch()

            with self.assertRaises(FileNotFoundError):
                exp._load_shim()

    def test_load_shim_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.run_index = 1
            exp.data_path = Path(tmpdir)
            fn = Path(tmpdir) / "iter00000.pkl.lzma"
            fn.write_bytes(b"not a valid lzma file")

            with self.assertRaises(OSError):
                exp._load_shim()

    def test_load_shim_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.data_path = Path(tmpdir)
            exp.run_index = 1

            with self.assertRaises(FileNotFoundError):
                exp._load_shim()

    def test_get_shimdata_not_initialized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.already_initialized = False
            shimdata = exp._get_shimdata()
            self.assertEqual(shimdata["total_iterations"], 0)


class TestCouplerShim(unittest.TestCase):
    def test_coupler_shim_basic_update(self):
        """rcs += step_size * (frust - mean(frust)) within each orbit bin.

        signed_energy_scale=0.5 keeps the effective coupler |J*rcs*scale| below the
        truncation thresholds (>1 / <-2) so we can assert the raw update math.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(False,),
                data_root=tmpdir,
                orbit_type="global",  # all edges in one bin -> update is non-trivial
            )
            chain.embedding_list = np.array([[0, 1, 2, 3]])
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(
                inst=chain,
                sampler=sampler,
                config=experiment.ExperimentConfig(coupler_shim_step=0.01, signed_energy_scale=0.5),
            )

            sc = experiment.SamplerCall(run_index=0)
            sc.logical_bqms = [chain.make_bqm()]
            sc.shimdata = {
                "total_iterations": 0,
                "relative_coupler_strength": np.ones((1, chain.num_edges)),
            }
            # mean(frust) = 0.5, so delta = 0.01 * [-0.2, 0.0, 0.2] = [-0.002, 0.0, 0.002]
            # Post-update mean(|rcs|) = 1.0 exactly -> renormalization is a no-op.
            # Q = rcs * J(=1) * scale(=0.5) stays in [0.499, 0.501] -> no truncation.
            results = {"CouplerFrustration": np.array([[0.3, 0.5, 0.7]])}

            exp._update_coupler_shim(sc, results)

            np.testing.assert_array_almost_equal(
                sc.shimdata["relative_coupler_strength"],
                np.array([[0.998, 1.0, 1.002]]),
            )

    def test_coupler_shim_singleton_orbits_is_noop(self):
        """With singleton orbits, mean equals the single value, so rcs is unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(
                dimensions=(4,),
                periodic=(False,),
                data_root=tmpdir,
                orbit_type="singleton",
            )
            chain.embedding_list = np.array([[0, 1, 2, 3]])
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(
                inst=chain,
                sampler=sampler,
                config=experiment.ExperimentConfig(coupler_shim_step=0.01),
            )

            sc = experiment.SamplerCall(run_index=0)
            sc.logical_bqms = [chain.make_bqm()]
            rcs_before = np.ones((1, chain.num_edges))
            sc.shimdata = {
                "total_iterations": 0,
                "relative_coupler_strength": rcs_before.copy(),
            }
            results = {"CouplerFrustration": np.array([[0.3, 0.5, 0.7]])}

            exp._update_coupler_shim(sc, results)

            np.testing.assert_array_equal(sc.shimdata["relative_coupler_strength"], rcs_before)


class TestSaveLoadResults(unittest.TestCase):
    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
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
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.data_path = Path(tmpdir)
            with self.assertRaises(ValueError):
                exp._save_results({}, run_index=0, filename="test.pkl.lzma")

    def test_save_custom_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.data_path = Path(tmpdir)
            data = {"x": 1}
            exp._save_results(data, filename="custom.pkl.lzma")
            self.assertTrue((Path(tmpdir) / "custom.pkl.lzma").exists())

    def test_load_results_ignore_shim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})
            fn = exp.data_path / "iter00000.pkl.lzma"
            fn.parent.mkdir(parents=True, exist_ok=True)
            with lzma.open(fn, "wb") as f:
                pickle.dump({"value": 0, "shimdata": {}}, f)

            results = exp.load_results(num_iterations=1, ignore_shim=True)
            self.assertNotIn("shimdata", results[0])

    def test_load_results_starting_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})
            for i in range(10):
                fn = exp.data_path / f"iter{i:05d}.pkl.lzma"
                fn.parent.mkdir(parents=True, exist_ok=True)
                with lzma.open(fn, "wb") as f:
                    pickle.dump({"value": i, "shimdata": {}}, f)

            results = exp.load_results(num_iterations=3, start_iteration=2)
            self.assertEqual(len(results), 3)

    def test_load_results_corrupted_lzma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            exp.apply_param({"signed_energy_scale": 1.0, "anneal_time": 1.0})
            fn = exp.data_path / "iter00000.pkl.lzma"
            fn.parent.mkdir(parents=True, exist_ok=True)
            fn.write_bytes(b"corrupted data")

            with self.assertRaises(lzma.LZMAError):
                exp.load_results(num_iterations=1)

    def test_generate_data_type_conversions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(inst=chain, sampler=sampler)
            sc = experiment.SamplerCall(run_index=0)
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


class TestMakeBqm(unittest.TestCase):
    def test_make_bqm_no_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(
                inst=chain,
                sampler=sampler,
                config=experiment.ExperimentConfig(signed_energy_scale=0.5),
            )
            sc = experiment.SamplerCall(run_index=0)
            sc.logical_bqms = [chain.make_bqm()]
            sc.shimdata = {"total_iterations": 0}
            bqm = exp._make_bqm(sc)
            for u, v in chain.edge_list:
                self.assertAlmostEqual(bqm.quadratic[(u, v)], 0.5)

    def test_make_bqm_with_embeddings(self):
        """Physical biases = logical_bias * relative_coupler_strength * signed_energy_scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            chain.embedding_list = np.array([[10, 11, 12, 13]])  # offset to detect mapping
            sampler = _make_mock_sampler()
            exp = experiment.Experiment(
                inst=chain,
                sampler=sampler,
                config=experiment.ExperimentConfig(signed_energy_scale=0.5),
            )
            sc = experiment.SamplerCall(run_index=0)
            sc.logical_bqms = [chain.make_bqm()]
            sc.shimdata = {
                "total_iterations": 0,
                "relative_coupler_strength": np.full((1, chain.num_edges), 2.0),
            }

            bqm = exp._make_bqm(sc)

            # Physical variables come from the embedding, not the logical indices.
            self.assertEqual(set(bqm.variables), {10, 11, 12, 13})
            for u_log, v_log in chain.edge_list:
                self.assertAlmostEqual(bqm.quadratic[(10 + u_log, 10 + v_log)], 1.0)

            self.assertEqual(len(bqm.quadratic), chain.num_edges)


class TestRunIteration(unittest.TestCase):
    def test_run_iteration_basic(self):
        """run_iteration() exercises the full pipeline: build call, sample, parse, shim, save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            exp = experiment.Experiment(
                inst=chain, sampler=_make_mock_sampler(sync_response=True), max_iterations=1
            )
            chain._load_embeddings = mock.MagicMock()
            finished = exp.run_iteration([{"signed_energy_scale": 1.0, "anneal_time": 1.0}])

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
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            exp = experiment.Experiment(
                inst=chain,
                sampler=_make_mock_sampler(sync_response=True),
                config=experiment.ExperimentConfig(),
                max_iterations=0,
            )
            chain._load_embeddings = mock.MagicMock()
            finished = exp.run_iteration([{"signed_energy_scale": 1.0, "anneal_time": 1.0}])

            self.assertTrue(finished)
            self.assertEqual(list(exp.data_path.glob("iter*.pkl.lzma")), [])


class TestFastAnnealExperiment(unittest.TestCase):
    def test_default_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(True,), data_root=tmpdir)
            sampler = _make_mock_sampler()
            config = experiment.FastAnnealExperimentConfig()
            exp = experiment.Experiment(inst=chain, sampler=sampler, config=config)
            self.assertTrue(exp.param.get("fast_anneal"))
            self.assertEqual(exp.param["num_reads"], 100)


if __name__ == "__main__":
    unittest.main()
