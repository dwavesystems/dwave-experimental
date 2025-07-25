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

from dwave.experimental.fast_reverse_anneal import get_parameters


class FRA(unittest.TestCase):

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
