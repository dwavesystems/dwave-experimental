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

from typing import Any, Optional

from dwave.cloud import Client

__all__ = ['get_solver_name', 'get_parameters']


def get_solver_name() -> str:
    """Return the first available solver with fast reverse annealing enabled."""

    # TODO: use feature-based solver selection to get a FRA solver
    # NOTE: until we can use FBSS for FRA, we use a hard-coded name (prefix)
    filter = dict(name__regex=r'Advantage2_prototype2.*|Advantage2_research1\..*')

    with Client.from_config() as client:
        solver = client.get_solver(**filter)
        return solver.name


def get_parameters(solver_name: str) -> dict[str, Any]:
    """Retrieve available fast annealing parameters and their expanded info.

    For each parameter available, we return its data type, allowed value limits,
    if it's required, a default value if it's not required, and a short text
    description.
    """

    with Client.from_config() as client:
        solver = client.get_solver(name=solver_name)

        # get FRA param ranges
        computation = solver.sample_qubo(
            {next(iter(solver.edges)): 0},
            x_get_fast_reverse_anneal_exp_feature_info=True)

        raw = computation['x_get_fast_reverse_anneal_exp_feature_info']
        info = dict(zip(raw[::2], raw[1::2]))

    # until parameter description is available via SAPI, we hard-code it here
    return {
        "x_target_c": {
            "type": "float",
            "required": True,
            "limits": {
                "range": info["fastReverseAnnealTargetCRange"],
            },
            "description": "The lowest value of the normalized control bias, `c(s)`, during a fast reverse annealing.",
        },
        "x_nominal_pause_time": {
            "type": "float",
            "required": False,
            "default": 0.0,
            "limits": {
                "set": info["fastReverseAnnealNominalPauseTimeValues"],
            },
            "description": "Sets the pause duration for fast-reverse-annealing schedules.",
        },
    }
