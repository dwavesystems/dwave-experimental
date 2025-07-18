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

from __future__ import annotations

import json
from importlib.resources import files
from typing import Optional

import numpy
import numpy.typing
import matplotlib.pyplot

from .api import get_solver_name

__all__ = ['load_schedules', 'linex', 'c_vs_t', 'plot_schedule']


def load_schedules(solver_name: Optional[str] = None) -> dict[float, dict[str, float]]:
    """Return per-solver approximation parameters for a family of fast reverse
    annealing schedules.

    Args:
        solver_name:
            Name of a QPU solver that supports fast reverse annealing.
            If unspecified, the default solver is used.
    """
    if solver_name is None:
        solver_name = get_solver_name()

    fra = files('dwave.experimental.fast_reverse_anneal')
    schedules = json.loads(fra.joinpath('data/schedules.json').read_bytes())
    if solver_name not in schedules:
        raise ValueError(f"Schedule parameters not found for {solver_name!r}")

    family = schedules[solver_name]['params']
    # reformat for easier access
    return {s['nominal_pause_time']: s for s in family}


def linex(t: numpy.typing.ArrayLike,
          c0: float,
          c2: float,
          t_min: float,
          a: float,
          ) -> numpy.typing.ArrayLike:
    """Linear-exponential (linex) function used to approximate a
    fast-reverse-annealing schedule.
    """
    return c0 + 2*c2/a**2*(numpy.exp(a*(t - t_min)) - a*(t - t_min) - 1)


def c_vs_t(t: numpy.typing.ArrayLike,
           target_c: float,
           nominal_pause_time: float = 0.0,
           upper_bound: float = 1.0,
           schedules: Optional[dict[str, float]] = None,
           ) -> numpy.typing.ArrayLike:
    """Time-dependence of the normalized control bias c(s) in linear-exponential
    fast-reverse-anneal waveforms
    """
    if schedules is None:
        schedules = load_schedules()

    schedule = schedules[nominal_pause_time]
    c2, a, t_min = schedule["c2"], schedule["a"], schedule["t_min"]

    # subtracting -0.25*nominal_pause_time helps in reducing the s=1 padding when upper_bound=1.0.
    return numpy.minimum(linex(1.025 - t - 0.25*nominal_pause_time, target_c, c2, t_min, a), upper_bound)


def plot_schedule(t: numpy.typing.ArrayLike,
                  target_c: float,
                  nominal_pause_time: float,
                  schedules: Optional[dict[str, float]] = None,
                  figure: Optional[matplotlib.pyplot.Figure] = None,
                  ) -> matplotlib.pyplot.Figure:
    """Plot the approximate fast reverse schedule for a given ``target_c`` and
    ``nominal_pause_time``, using time grid ``t``, optionally adding to figure
    ``fig``.
    """

    if figure is None:
        figure = matplotlib.pyplot.figure()
    ax = figure.gca()

    c = c_vs_t(t, target_c=target_c, nominal_pause_time=nominal_pause_time, schedules=schedules)

    ax.plot(t, c, label=nominal_pause_time)
    ax.set_xlabel("t [$\\mu s$]")
    ax.set_ylabel("c(s)")
    ax.set_title(f"Predicted fast-reverse-anneal waveforms, target_c = {target_c:.2f}")
    ax.legend(title="Nominal pause duration [$\\mu s$]")

    return figure
