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
import re
from importlib.resources import files
from typing import Optional

import numpy
import numpy.typing
import matplotlib.pyplot

from .api import get_solver_name

__all__ = ['load_schedules', 'linex', 'c_vs_t', 'plot_schedule']


def _get_schedules_data() -> dict[str, dict]:
    fra = files('dwave.experimental.fast_reverse_anneal')
    return json.loads(fra.joinpath('data/schedules.json').read_bytes())


def load_schedules(solver_name: Optional[str] = None) -> dict[float, dict[str, float]]:
    """Return per-solver approximation parameters for a family of fast reverse
    annealing schedules.

    Args:
        solver_name:
            Name of a QPU solver that supports fast reverse annealing.
            If unspecified, the default solver is used.

    Returns:
        A dict mapping allowed ``nominal_pause_time`` values to a schedule
        approximation curve (linear-exponential) parameters dict. For example:

        {0.0: {'a': -51.04360118925347,
               'c2': 9821.41471886313,
               'nominal_pause_time': 0.0,
               't_min': 1.0234109310649393},
         ...}

         See :meth:`.linex` for parameters description.

    Note:
        When ``solver_name`` is not specified, a call to SAPI has to be made to
        determine the default (fast reverse anneal) solver.

    """
    if solver_name is None:
        solver_name = get_solver_name()

    schedules = _get_schedules_data()

    def load_params(solver_name, schedules):
        if solver_name in schedules:
            return schedules[solver_name]['params']

        # try regex search before failing
        for pattern, schedule in schedules.items():
            if re.fullmatch(pattern, solver_name):
                return schedule['params']

        raise ValueError(f"Schedule parameters not found for {solver_name!r}")

    params = load_params(solver_name, schedules)

    # reformat for easier access
    return {s['nominal_pause_time']: s for s in params}


def linex(
    t: numpy.typing.ArrayLike,
    *,
    c0: float,
    c2: float,
    a: float,
    t_min: float,
) -> numpy.typing.ArrayLike:
    r"""Linear-exponential (linex) function used to approximate a
    fast-reverse-annealing schedule.

    Fast-reverse-annealing schedules can be approximated with the following
    linear exponential function:

    .. math::
        f(t) = c_0 + \frac{2 c_2}{a^2} \left(e^{a(t - t_{\min})} - a(t - t_{\min}) - 1\right)

    Args:
        t: Discrete time (in microseconds), given as a scalar or an array.
        c0: Ordinate offset coefficient.
        c2: Quadratic ordinate coefficient.
        a: Asymmetry parameter.
        t_min: Time offset parameter.

    Returns:
        The linear-exponential function evaluated at ``t``.
    """
    return c0 + 2*c2/a**2*(numpy.exp(a*(t - t_min)) - a*(t - t_min) - 1)


def c_vs_t(
    t: numpy.typing.ArrayLike,
    *,
    target_c: float,
    nominal_pause_time: float = 0.0,
    upper_bound: float = 1.0,
    schedules: Optional[dict[str, float]] = None,
) -> numpy.typing.ArrayLike:
    """Time-dependence of the normalized control bias c(s) in linear-exponential
    fast-reverse-anneal waveforms.

    Args:
        t:
            Discrete time (in microseconds), given as a scalar or an array.
        target_c:
            The lowest value of the normalized control bias, `c(s)`, reached
            during a fast reverse annealing.
        nominal_pause_time:
            Pause duration, in microseconds, for the fast-reverse-annealing schedule.
        upper_bound:
            Waveform's upper bound.
        schedules:
            Schedule family parameters, as returned by :meth:`.load_schedules`.

    Returns:
        Schedule waveform approximation evaluated at ``t``.

    """
    if schedules is None:
        schedules = load_schedules()

    schedule = schedules[nominal_pause_time]
    c2, a, t_min = schedule["c2"], schedule["a"], schedule["t_min"]

    return numpy.minimum(linex(t, c0=target_c, c2=c2, a=a, t_min=t_min), upper_bound)


def plot_schedule(
    t: numpy.typing.ArrayLike,
    *,
    target_c: float,
    nominal_pause_time: float = 0.0,
    schedules: Optional[dict[str, float]] = None,
    figure: Optional[matplotlib.pyplot.Figure] = None,
) -> matplotlib.pyplot.Figure:
    """Plot the approximate fast reverse schedule for a given ``target_c`` and
    ``nominal_pause_time``, using time grid ``t``, optionally adding to figure
    ``fig``.

    Example::
        import numpy
        import matplotlib.pyplot as plt
        from dwave.experimental.fast_reverse_anneal import plot_schedule

        t = numpy.arange(1.0, 1.04, 1e-4)
        fig = plot_schedule(t, target_c=0.0)
        plt.show()

    See also: ``examples/plot_schedule.py``.
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
