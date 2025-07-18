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

import re
import unittest
import unittest.mock

from dwave.system import DWaveSampler
from dwave.experimental.fast_reverse_anneal import (
    get_parameters, get_solver_name, SOLVER_FILTER,
)


class FRA(unittest.TestCase):

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_sampler_params(self):
        x_target_c_range = [0, 1]
        x_nominal_pause_time_values = [0, 1, 2]
        info = ['fastReverseAnnealTargetCRange', x_target_c_range,
                'fastReverseAnnealNominalPauseTimeValues', x_nominal_pause_time_values]

        with unittest.mock.MagicMock() as sampler:
            sampler.solver.edges = [(0,1)]
            sampler.solver.sample_qubo.return_value.result.return_value = \
                dict(x_get_fast_reverse_anneal_exp_feature_info=info)

            p = get_parameters(sampler)

            self.assertIn('x_target_c', p)
            self.assertEqual(p['x_target_c']['limits']['range'], x_target_c_range)

            self.assertIn('x_nominal_pause_time', p)
            self.assertEqual(p['x_nominal_pause_time']['limits']['set'], x_nominal_pause_time_values)

    @unittest.mock.patch('dwave.experimental.fast_reverse_anneal.api.Client')
    def test_default_solver_name(self, client):
        class Solver:
            name = "mock-solver"

        client.from_config.return_value.__enter__.return_value.get_solver.return_value = Solver()

        solver_name = get_solver_name()

        self.assertEqual(solver_name, Solver.name)


class LiveSmokeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.sampler = DWaveSampler(solver=SOLVER_FILTER)
        except:
            raise unittest.SkipTest('Fast reverse annealing solver not available.')

    @classmethod
    def tearDownClass(cls):
        cls.sampler.close()

    def tearDown(self):
        # make sure solver name is not cached, so the next test is not affected
        get_solver_name.cache_clear()

    def test_get_parameters(self):
        params = get_parameters(self.sampler)
        self.assertIn('x_target_c', params)
        self.assertIn('x_nominal_pause_time', params)

    def test_solver_selection(self):
        name = get_solver_name()
        self.assertIsNotNone(re.match(SOLVER_FILTER['name__regex'], name))
