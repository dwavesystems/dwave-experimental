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

import networkx as nx

from dwave_networkx import zephyr_coordinates

__all__ = ["qubit_to_Advantage2_annealing_line", "make_tds_graph"]


def qubit_to_Advantage2_annealing_line(n: int | tuple, shape: tuple) -> int:
    """Return the annealing line associated to an Advantage2 qubit

    Advantage2 processors can allow for multicolor annealing based in
    some cases on a 6-line control scheme. Compatibility with this
    scheme should be confirmed using a solver API or release notes.
    Based on the Zephyr coordinate system (u,w,k,j,z), a qubit
    can be uniquely assigned a color. u denotes qubit orientation
    j and z control aligned-displacement on the processor. See also
    dwave_networkx.zephyr_graph and dwave_networkx.zephyr_coordinates

    Args:
        n: qubit label, as an integer, or a Zephyr coordinate as a 5-tuple
        shape: Advantage2 processor shape, accessible as a solver
            property properties['topology']['shape']

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

    if isinstance(n, tuple):
        u, w, k, j, z = n
    else:
        u, w, k, j, z = zephyr_coordinates(*shape).linear_to_zephyr(n)

    return 3 * u + (1 - 2 * z - j) % 3


def make_tds_graph(
    target_graph: nx.Graph,
    detected_nodes: list[int] | None = None,
    sourced_nodes: list[int] | None = None,
) -> tuple[nx.Graph, dict]:
    """Decorate a target graph with detectors and sources.

    We add single node source and detector branches to nodes of a target
    graph.

    Args:
        target_graph: A networkx target graph
        detector_nodes: An iterable on the target nodes, if None
            all target nodes.
        source_nodes: An iterable on the target nodes, if None
            all target nodes.

    Returns:
        A copy of the graph where edges from target nodes (n) are
        added to ('target', n) and/or ('source', n) nodes.

    Raises:
        ValueError: If detected_nodes or sourced_nodes are not in the graph
            target_graph.
    """

    if detected_nodes is None:
        detected_nodes = target_graph.nodes()
    elif not set(detected_nodes).issubset(target_graph.nodes()):
        raise ValueError("detected_nodes are not compatible with the target graph")

    if sourced_nodes is None:
        sourced_nodes = target_graph.nodes()
    elif not set(sourced_nodes).issubset(target_graph.nodes()):
        raise ValueError("sourced_nodes are not compatible with the target graph")

    node_to_tds = (
        {n: "target" for n in target_graph.nodes()}
        | {("source", n): "source" for n in sourced_nodes}
        | {("detector", n): "detector" for n in detected_nodes}
    )

    tds_graph = target_graph.copy()
    tds_graph.add_edges_from((n, ("detector", n)) for n in detected_nodes)
    tds_graph.add_edges_from((n, ("source", n)) for n in sourced_nodes)

    return tds_graph, node_to_tds
