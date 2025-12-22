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
    r"""Return fast-reverse-annealing schedule-approximation parameters.

    The approximation parameters are for all allowed values of pause duration,
    and used in the following formula,

    .. |fra_approximation_formula| replace:: :math:`f(t) = c_0 + \frac{2 c_2}{a^2}
        \left(e^{a(t - t_{\min})} - a(t - t_{\min}) - 1\right)`

    .. |fra_approximation_params| replace:: where :math:`t` is discrete time, in
        microseconds, :math:`c_0` is an ordinate offset coefficient, :math:`c_2`
        is a quadratic ordinate coefficient, :math:`a` is an asymmetry
        parameter, and :math:`t_{\text{min}}` is a time offset parameter.

    |fra_approximation_formula|\ ,

    |fra_approximation_params|

    Args:
        solver_name:
            Name of a QPU solver that supports fast
            reverse annealing. If unspecified, a call to SAPI is made to
            determine the default solver (a QPU that supports fast reverse
            annealing) .

    Returns:
        A dict mapping supported values of the
        :ref:`parameter_qpu_nominal_pause_time` parameter to dicts of parameters
        that can be used to approximate the (linear-exponential) annealing
        schedule of the QPU. For example::

            {0.0: {'a': -51.04360118925347,
                'c2': 9821.41471886313,
                'nominal_pause_time': 0.0,
                't_min': 1.0234109310649393},
            ...}

    Examples:
        Obtain the schedule-approximation parameters for the default solver that
        supports fast reverse annealing.

        >>> from dwave.experimental import fast_reverse_anneal as fra
        ...
        >>> param_002 = fra.schedule.load_schedules()[0.02]
        >>> list(param_002.keys())
        ['nominal_pause_time', 'a', 'c2', 't_min']
        >>> tmin_002 = param_002['t_min']

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
    r"""Approximate the fast-reverse-annealing schedule at a given time.

    Uses a linear-exponential ("linex") function to approximate a
    fast-reverse-annealing schedule with the following linear-exponential
    function,

    |fra_approximation_formula|\ ,

    |fra_approximation_params|

    Args:
        t: Discrete time, in microseconds, as a scalar or an array.
        c0: :math:`c_0` is an ordinate offset coefficient.
        c2: :math:`c_2` is a quadratic ordinate coefficient.
        a: :math:`a` is an asymmetry parameter.
        t_min: :math:`t_{\text{min}}` is a time offset parameter.

    Returns:
        The linear-exponential function evaluated at ``t``.

    Examples:
        See source code of the :func:`c_vs_t` function for a usage example.
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
    """Calculate the approximate normalized control bias.

    Approximates the time-dependent normalized control bias :math:`c(s)` using a
    linear-exponential function, :func:`.linex`, for simulating
    fast-reverse-anneal waveforms.

    Args:
        t:
            Discrete time, in microseconds, as a scalar or an array.
        target_c:
            The lowest value of the normalized control bias, :math:`c(s)`,
            reached during a fast reverse annealing.
        nominal_pause_time:
            Pause duration, in microseconds, for the fast-reverse-annealing
            schedule.
        upper_bound:
            Waveform's upper bound.
        schedules:
            Schedule family parameters, as returned by the
            :func:`.load_schedules` function.

    Returns:
        Schedule waveform approximation evaluated at ``t``.

    Examples:
        Obtain an estimated normalized control bias :math:`c(s)`, at time 0.022,
        for a waveform that reaches :math:`c(s)=0` at its lowest point and
        nominally pauses there for 0.02 microsecond.

        >>> from dwave.experimental import fast_reverse_anneal as fra
        ...
        >>> c = c_vs_t(0.022,
        ...     target_c=0.0,
        ...     nominal_pause_time=0.02,
        ...     schedules=fra.schedule.load_schedules())

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
    """Plot the approximate fast-reverse waveform.

    Creates a plot of the approximate fast-reverse waveform for a given
    ``target_c`` and ``nominal_pause_time``, using time grid ``t``, optionally
    adding to an existing figure ``figure``.

    Example:

        >>> import numpy
        >>> import matplotlib.pyplot as plt                     # doctest: +SKIP
        >>> from dwave.experimental.fast_reverse_anneal import plot_schedule
        ...
        >>> t = numpy.arange(1.0, 1.04, 1e-4)
        >>> fig = plot_schedule(t, target_c=0.0)                # doctest: +SKIP
        >>> plt.show()                                          # doctest: +SKIP

    See also: `examples directory <https://github.com/dwavesystems/dwave-experimental/tree/main/examples>`_.
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
