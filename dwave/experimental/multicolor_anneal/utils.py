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

from typing import Union
from dwave_networkx import zephyr_coordinates


def qubit_to_Advantage2_annealing_line(
    n: Union[int, tuple], shape: tuple, coordinates=False
) -> int:
    """Return the annealing line associated to an Advantage2 qubit

    Advantage2 processors can allow for multicolor annealing based in
    some cases on a 6-line control schme. Compatibility with this
    scheme should be confirmed using a solver API or release notes.
    Based on the Zephyr coordinate system (u,w,k,j,z), a qubit
    can be uniquely assigned a color. u denotes qubit orientation
    j and z control aligned-displacement on the processor. See also
    dwave_networkx.zephyr_graph and dwave_networkx.zephyr_coordinates

    Args:
        n: qubit label
        shape: Advantage2 processor shape, accessible as a solver
            property properties['topology']['shape']
        coordinates: label format, if False the labeling is an integer
            if True the Zephyr coordinate labeling scheme is used.

    Returns:
        Integer annealing line assignment for Advantage2 processors
        using 6-annealing line control.

    Examples:
        Retrieve MCA annealing lines' properties for a default solver, and
        if a 6 color scheme is used confirm the programmatic mapping is
        in agreement with the multicolor annealing properties on all qubits
        and lines

        >>> from dwave.system import DWaveSampler
        >>> from dwave.experimental import multicolor_anneal, qubit_to_Advantage2_annealing_line

        >>> qpu = DWaveSampler()             # doctest: +SKIP
        >>> annealing_lines = multicolor_anneal.get_properties(qpu)            # doctest: +SKIP
        >>> if len(annealing_lines) == 6:            # doctest: +SKIP
        >>>     assert(all(qubit_to_Advantage2_annealing_line(n)==al_idx for al_idx, al in enumerate(annealing_lines) for n in al['qubits']))            # doctest: +SKIP
    """

    if coordinates:
        u, w, k, j, z = n
    else:
        u, w, k, j, z = zephyr_coordinates(*shape).linear_to_zephyr(n)

    return 3 * u + (1 - 2 * z - j) % 3
