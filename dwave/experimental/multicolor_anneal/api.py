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
    """Return multicolor-annealing properties for each annealing line.

    Args:
        sampler:
            A :class:`~dwave.system.samplers.DWaveSampler` sampler that supports
            the multicolor annealing (MCA) protocol. Alternatively, you can
            specify a :class:`~dwave.cloud.solver.StructuredSolver` solver or a
            solver name. If unspecified, :data:`.SOLVER_FILTER` is used to fetch
            an MCA-enabled solver.

    Returns:
        Annealing-line properties for all available annealing lines, formatted
        as list of dicts in ascending order of annealing-line index.

    Examples:
        Retrieve MCA properties for the annealing lines of a default solver, and
        print the number of lines and first qubits on line 0.

        >>> from dwave.experimental import multicolor_anneal as mca
        ...
        >>> annealing_lines = mca.get_properties()      # doctest: +SKIP
        >>> len(annealing_lines)                        # doctest: +SKIP
        6
        >>> annealing_lines[0]['qubits']                # doctest: +SKIP
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
