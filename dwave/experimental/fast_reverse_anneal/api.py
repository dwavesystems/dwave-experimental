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

from functools import cache
from typing import Any, Optional, Union

from dwave.cloud import Client, Solver
from dwave.system import DWaveSampler

__all__ = ['SOLVER_FILTER', 'get_solver_name', 'get_parameters']


SOLVER_FILTER = dict(name__regex=r'Advantage2_prototype2.*|Advantage2_research1\..*')
"""Filter for an available solver that supports advanced annealing features.

Feature-based solver selection returns the first available solver that supports
features such as fast reverse annealing.

.. note:: currently SAPI does not support filtering for solvers with
    :ref:`experimental research <qpu_experimental_research>` features, so a
    simple pattern matching is used.

Example::
    >>> from dwave.system import DWaveSampler
    >>> from dwave.experimental import fast_reverse_anneal as fra
    ...
    >>> with DWaveSampler(solver=fra.SOLVER_FILTER) as sampler: # doctest: +SKIP
            sampler.sample(...)
"""


@cache
def get_solver_name() -> str:
    """Return the name of a solver that supports advanced annealing features.

    The result is memoized, so the API is queried only on first call.

    Examples:

        >>> from dwave.experimental.fast_reverse_anneal import get_solver_name
        ...
        >>> print(get_solver_name())                # doctest: +SKIP
        Advantage2_research1.4
    """
    with Client.from_config() as client:
        solver = client.get_solver(**SOLVER_FILTER)
        return solver.name


def get_parameters(sampler: Optional[Union[DWaveSampler, Solver, str]] = None,
                   ) -> dict[str, Any]:
    """Return available fast-reverse-annealing parameters and their information.

    Args:
        sampler:
            A :class:`~dwave.system.samplers.DWaveSampler` sampler that supports
            the fast-reverse-anneal (FRA) protocol. Alternatively, you can
            specify a :class:`~dwave.cloud.solver.StructuredSolver` solver or a
            solver name. If unspecified, :data:`.SOLVER_FILTER` is used to fetch
            an FRA-enabled solver.

    Returns:
        Each parameter available is described with its data type, value limits,
        an is-required flag, a default value (if it's optional), and a short
        description.

    Examples:
        Use an instantiated :class:`~dwave.system.samplers.DWaveSampler`
        sampler:

        >>> from dwave.system import DWaveSampler
        >>> from dwave.experimental import fast_reverse_anneal as fra
        ...
        >>> with DWaveSampler() as sampler:             # doctest: +SKIP
        ...    param_info = fra.get_parameters(sampler)
    """

    # inelegant, but convenient extensions
    if sampler is None or isinstance(sampler, str):
        if isinstance(sampler, str):
            filter = dict(name=sampler)
        else:
            filter = SOLVER_FILTER

        with Client.from_config() as client:
            solver = client.get_solver(**filter)
            return get_parameters(solver)

    if hasattr(sampler, 'solver'):
        solver: Solver = sampler.solver
    else:
        solver: Solver = sampler

    # get FRA param ranges
    computation = solver.sample_qubo(
        {next(iter(solver.edges)): 0},
        x_get_fast_reverse_anneal_exp_feature_info=True)

    result = computation.result()
    try:
        raw = result['x_get_fast_reverse_anneal_exp_feature_info']
    except KeyError:
        raise ValueError(f'Selected sampler ({solver.name}) does not support fast reverse anneal')

    info = dict(zip(raw[::2], raw[1::2]))

    # until parameter description is available via SAPI, we hard-code it here
    return {
        "x_target_c": {
            "type": "float",
            "required": True,
            "limits": {
                "range": info["fastReverseAnnealTargetCRange"],
            },
            "description": (
                "The lowest value of the normalized control bias, `c(s)`, "
                "reached during a fast reverse annealing."
            ),
        },
        "x_nominal_pause_time": {
            "type": "float",
            "required": False,
            "default": 0.0,
            "limits": {
                "set": info["fastReverseAnnealNominalPauseTimeValues"],
            },
            "description": (
                "Sets the pause duration, in microseconds, "
                "for fast-reverse-annealing schedules."
            ),
        },
    }
