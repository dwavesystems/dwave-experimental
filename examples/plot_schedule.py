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

import argparse
from typing import Union

from pprint import pprint
import numpy
import matplotlib.pyplot as plt
from dwave.experimental import fast_reverse_anneal as fra


def main(
    solver: Union[None, dict, str],
    x_target_c: float
):
    """Plot a family of fast reverse anneal schedules for a solver

    One curve is plotted for each calibrated ``nominal_pause_time``, for a specific ``target_c``.

    Args:
        solver: Name of the solver, or dictionary of characteristics.
        x_target_c: 
            The lowest value of the normalized control bias, c(s), attained during the fast 
            reverse anneal. This parameter sets the reversal distance of the reverse anneal.
    """
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

    target_c = x_target_c
    fig = None
    for nominal_pause_time in nominal_pause_times:
        fig = fra.plot_schedule(
            t, target_c=target_c, nominal_pause_time=nominal_pause_time,
            schedules=schedules, figure=fig)

    filename = f'schedules for {target_c=}.png'
    fig.savefig(filename)
    print(f'Figure saved to: {filename}')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="An example for plotting a family of fast reverse anneal schedules"
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver, by default=None",
        default=None,
    )
    parser.add_argument(
        "--x_target_c",
        type=float,
        help="Reverse anneal target point x_target_c, by default 0.25",
        default=0.25,
    )
    args = parser.parse_args()

    main(
        solver=args.solver_name,
        x_target_c=args.x_target_c
    )
