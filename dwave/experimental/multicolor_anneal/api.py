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

from typing import Any

from dwave.cloud import Client, Solver
from dwave.system import DWaveSampler

from dwave.experimental.fast_reverse_anneal.api import SOLVER_FILTER, get_solver_name

__all__ = ['SOLVER_FILTER', 'get_solver_name', 'get_properties']


def get_properties(sampler: DWaveSampler | Solver | str | None = None
                   ) -> list[dict[str, Any]]:
    """For a given sampler/solver, return the multicolor anneal properties
    for each of the available annealing lines.

    Args:
        sampler:
            A :class:`~dwave.system.DWaveSampler` sampler that supports the multicolor
            anneal (MCA) protocol. Alternatively, a :class:`dwave.cloud.Solver`
            solver can be provided, or a solver name string. If unspecified,
            :attr:`.SOLVER_FILTER` is used to fetch an MCA-enabled solver.

    Returns:
        Annealing line properties for all available anneal lines, formatted
        as list of dicts in ascending order of anneal-line index.

    Examples:
        Retrieve MCA annealing lines' properties for a default solver, and
        print the number of anneal lines and first qubits on anneal line 0.

        >>> from dwave.experimental import multicolor_anneal as mca
        >>> annealing_lines = mca.get_properties()
        >>> len(annealing_lines)            # doctest: +SKIP
        6
        >>> annealing_lines[0]['qubits']    # doctest: +SKIP
        [2, 6, 9, 14, 17, 18, ...]
    """

    # inelegant, but convenient extensions
    if sampler is None or isinstance(sampler, str):
        if isinstance(sampler, str):
            filter = dict(name=sampler)
        else:
            filter = SOLVER_FILTER

        with Client.from_config() as client:
            solver = client.get_solver(**filter)
            return get_properties(solver)

    if hasattr(sampler, 'solver'):
        solver: Solver = sampler.solver
    else:
        solver: Solver = sampler

    # get MCA annealing lines and properties
    computation = solver.sample_qubo(
        {next(iter(solver.edges)): 0},
        x_get_multicolor_annealing_exp_feature_info=True)

    result = computation.result()
    try:
        return result['x_get_multicolor_annealing_exp_feature_info']
    except KeyError:
        raise ValueError(f'Selected sampler ({solver.name}) does not support multicolor annealing')
