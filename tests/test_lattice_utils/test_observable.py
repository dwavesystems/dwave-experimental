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

import tempfile
import unittest
from pathlib import Path

import dimod
import numpy as np

from dwave.experimental.lattice_utils import lattice, observable
from tests.test_lattice_utils._helpers import _make_mock_experiment, _make_triangular


class TestQubitMagnetization(unittest.TestCase):
    def test_qubit_magnetization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            samples = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]])
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp = _make_mock_experiment(chain)
            result = observable.QubitMagnetization().evaluate(exp, bqm, ss)
            np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])


class TestCouplerCorrelation(unittest.TestCase):
    def test_coupler_correlation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            exp = _make_mock_experiment(chain)
            alt = np.tile([1, -1, 1, -1], (4, 1))
            ss_alt = dimod.SampleSet.from_samples_bqm(alt, bqm)
            np.testing.assert_array_equal(
                observable.CouplerCorrelation().evaluate(exp, bqm, ss_alt),
                -np.ones(chain.num_edges),
            )


class TestCouplerFrustration(unittest.TestCase):
    def test_coupler_frustration(self):
        """All aligned (corr=1) -> frustration = 1.0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            samples = np.ones((4, 4))
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp = _make_mock_experiment(chain)
            np.testing.assert_array_almost_equal(
                observable.CouplerFrustration().evaluate(exp, bqm, ss), np.ones(chain.num_edges)
            )


class TestSampleEnergy(unittest.TestCase):
    def test_sample_energy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            # All-ones: energy = sum of J for 3 edges = 3
            samples = np.ones((1, 4))
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp_pos = _make_mock_experiment(chain, signed_energy_scale=1.0)
            np.testing.assert_array_almost_equal(
                observable.SampleEnergy().evaluate(exp_pos, bqm, ss), [3]
            )


class TestBitpackedSpins(unittest.TestCase):
    def test_bitpacked_spins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            samples = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp = _make_mock_experiment(chain)
            packed, shape = observable.BitpackedSpins().evaluate(exp, bqm, ss)
            self.assertEqual(shape, (2, 4))
            # Unpack and verify round-trip
            unpacked = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape)
            np.testing.assert_array_equal(unpacked, np.equal(samples, 1))


class TestReferenceEnergy(unittest.TestCase):
    def test_reference_energy_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ref.txt"
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            sample = np.array([1, -1, 1, -1])
            obs = observable.ReferenceEnergy()
            obs.save(path, -3, sample, "SA")

            exp = _make_mock_experiment(chain)
            energy, loaded_sample, method = obs.load(exp, bqm, path)
            self.assertEqual(energy, -3)
            self.assertEqual(method, "SA")
            np.testing.assert_array_equal(loaded_sample, sample)

    def test_reference_energy_evaluate_generates_and_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            obs = observable.ReferenceEnergy()

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
            chain = lattice.Chain(dimensions=(4,), periodic=(False,), data_root=tmpdir)
            bqm = chain.make_bqm()
            obs = observable.ReferenceEnergy()
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


class TestKinks(unittest.TestCase):
    def test_all_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(6,), periodic=(True,), data_root=tmpdir)
            bqm = chain.make_bqm()
            samples = np.ones((10, 6))
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp = _make_mock_experiment(chain)
            result = observable.KinkKinkCorrelator().evaluate(exp, bqm, ss)
            np.testing.assert_array_equal(result, np.zeros(6))

    def test_mixed_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = lattice.Chain(dimensions=(6,), periodic=(True,), data_root=tmpdir)
            bqm = chain.make_bqm()
            # [1,1,-1,-1,1,1]: kink at sites 2,4 (domain walls)
            samples = np.tile([1, 1, -1, -1, 1, 1], (20, 1))
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp = _make_mock_experiment(chain)
            result = observable.KinkKinkCorrelator().evaluate(exp, bqm, ss)
            expected = np.array([0.0, -0.25, 0.125, -0.25, 0.125, -0.25])
            np.testing.assert_array_almost_equal(result, expected)


class TestTriangularOP(unittest.TestCase):
    def test_uniform_state_vanishes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tri = _make_triangular(tmpdir, 3, 3, periodic=(True, False))
            bqm = tri.make_bqm()
            samples = np.ones((5, 9))
            ss = dimod.SampleSet.from_samples_bqm(samples, bqm)
            exp = _make_mock_experiment(tri)
            result = observable.TriangularOP().evaluate(exp, bqm, ss)
            np.testing.assert_array_almost_equal(np.abs(result), np.zeros(5), decimal=10)


if __name__ == "__main__":
    unittest.main()
