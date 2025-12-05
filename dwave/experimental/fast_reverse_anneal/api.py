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
"""Feature-based solver selection filter to return the first available solver
that supports fast reverse annealing.

Note: currently SAPI doesn't provide a nice way to filter solvers with prototype
features (like fast reverse anneal), so we need to defer to a simple pattern
matching.

Example::
    from dwave.system import DWaveSampler
    from dwave.experimental import fast_reverse_anneal as fra

    with DWaveSampler(solver=fra.SOLVER_FILTER) as sampler:
        sampler.sample(...)
"""


@cache
def get_solver_name() -> str:
    """Return the name of a solver that supports fast reverse anneal.

    Note: the result is memoized, so the API is queried only on first call.
    """
    with Client.from_config() as client:
        solver = client.get_solver(**SOLVER_FILTER)
        return solver.name


def get_parameters(sampler: Optional[Union[DWaveSampler, Solver, str]] = None,
                   ) -> dict[str, Any]:
    """For a given sampler (or solver), return the available fast annealing
    parameters and their expanded info.

    Args:
        sampler:
            A :class:`~dwave.system.DWaveSampler` sampler that supports the fast
            reverse anneal (FRA) protocol. Alternatively, a :class:`dwave.cloud.Solver`
            solver can be provided, or a solver name. If unspecified,
            :attr:`.SOLVER_FILTER` is used to fetch a FRA-enabled solver.

    Returns:
        Each parameter available is described with: a data type, value limits,
        an is-required flag, a default value if it's optional, and a short text
        description.

    Examples:
        Use an instantiated :class:`~dwave.system.DWaveSampler` sampler:

        .. code:: python
            from dwave.system import DWaveSampler
            from dwave.experimental import fast_reverse_anneal as fra

            with DWaveSampler() as sampler:
                param_info = fra.get_parameters(sampler)
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
