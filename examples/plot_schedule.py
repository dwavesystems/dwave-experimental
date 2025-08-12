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

"""Plot a family of fast reverse anneal schedules (one curve per allowed
``nominal_pause_time`` value), for a specific ``target_c``.
"""

from pprint import pprint
import numpy
from dwave.experimental import fast_reverse_anneal as fra


solver_name = fra.get_solver_name()
print("Solver:", solver_name)

params = fra.get_parameters(solver_name)
print("Parameters:")
pprint(params)

schedules = fra.load_schedules(solver_name)
print("Schedules:")
pprint(schedules)


t = numpy.arange(0.95, 1.03, 1e-4)
nominal_pause_times = params['x_nominal_pause_time']['limits']['set']

target_c = 0.5
fig = None
for nominal_pause_time in nominal_pause_times:
    fig = fra.plot_schedule(
        t, target_c=target_c, nominal_pause_time=nominal_pause_time,
        schedules=schedules, figure=fig)

filename = f'schedules for {target_c=}.png'
fig.savefig(filename)
print(f'Figure saved to: {filename}')
