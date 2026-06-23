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

import numpy as np

from dwave.experimental.lattice_utils.utils import (
    bootstrap,
    confidence_interval,
    generate_bootstrap_indices,
)


class TestUtils(unittest.TestCase):
    def test_bootstrap_all_nan_skipnan(self):
        rng = np.random.default_rng(seed=0)
        result = bootstrap(np.array([np.nan, np.nan]), rng, repetitions=5, skipnan=True)
        self.assertEqual(len(result), 5)
        for val in result:
            self.assertTrue(np.isnan(val))

    def test_bootstrap_skipnan_false(self):
        rng = np.random.default_rng(seed=0)
        result = bootstrap(np.array([1.0, 2.0, np.nan]), rng, repetitions=5, skipnan=False)
        self.assertEqual(len(result), 5)

    def test_bootstrap_custom_function(self):
        rng = np.random.default_rng(seed=0)
        result = bootstrap(np.arange(20), rng, repetitions=10, bootstrap_function=np.mean)
        self.assertEqual(len(result), 10)

    def test_generate_bootstrap_indices_correct_count(self):
        rng = np.random.default_rng(seed=0)
        indices = list(generate_bootstrap_indices(10, 5, rng))
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


if __name__ == "__main__":
    unittest.main()
