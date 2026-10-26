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

from dwave.cloud import Client, Solver
from dwave.system import DWaveSampler

from dwave.experimental.fast_reverse_anneal.api import SOLVER_FILTER, get_solver_name
from dwave.experimental.multicolor_anneal.utils import ExpFeatureInfo

__all__ = ['SOLVER_FILTER', 'get_solver_name', 'get_properties']


def get_properties(sampler: DWaveSampler | Solver | str | None = None
                   ) -> ExpFeatureInfo:
    """Return multicolor-annealing properties

    Args:
        sampler:
            A :class:`~dwave.system.samplers.DWaveSampler` sampler that supports
            the multicolor annealing (MCA) protocol. Alternatively, you can
            specify a :class:`~dwave.cloud.solver.StructuredSolver` solver or a
            solver name. If unspecified, :data:`.SOLVER_FILTER` is used to fetch
            an MCA-enabled solver.

    Returns:
        An :class:`~dwave.experimental.multicolor_anneal.utils.ExpFeatureInfo`
        named tuple with two fields:
        1. ``polarizing_line_info``: a dictionary of annealing-line independent
        properties.
        2. ``annealing_line_info``: annealing-line specific properties for all
        available annealing lines, formatted as a list of dicts in ascending
        order of annealing-line index.

    Examples:
        Retrieve MCA properties for the annealing lines of a default solver, and
        print the number of lines and first qubits on line 0.

        >>> from dwave.experimental import multicolor_anneal as mca
        ...
        >>> exp_feature_info = mca.get_properties()      # doctest: +SKIP
        >>> len(exp_feature_info[1])                        # doctest: +SKIP
        6
        >>> exp_feature_info[1][0]['qubits']                # doctest: +SKIP
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
        return ExpFeatureInfo(result['x_get_multicolor_annealing_exp_feature_info'])
    except KeyError:
        raise ValueError(f'Selected sampler ({solver.name}) does not support multicolor annealing')
