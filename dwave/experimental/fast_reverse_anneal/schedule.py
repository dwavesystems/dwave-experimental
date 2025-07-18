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

from .api import get_solver_name

__all__ = ['load_schedules']


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
