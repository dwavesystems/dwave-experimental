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

from dwave.system.testing import MockDWaveSampler

from dwave.experimental.shimming import shim_flux_biases


class FluxBiases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler = MockDWaveSampler()

    def test_validation(self):
        with self.subTest("invalid h"):
            with self.assertRaises(ValueError):
                shim_flux_biases({0: 1}, {}, {0: (0,)}, self.sampler)

    def test_sampler_called(self):
        with unittest.mock.patch.object(self.sampler, "sample_ising") as m:
            fb = shim_flux_biases({0: 0}, {}, {0: (0,)}, self.sampler)
            m.assert_called()

        self.assertIsInstance(fb, list)
        self.assertEqual(len(fb), self.sampler.properties['num_qubits'])
