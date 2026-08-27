# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Hashable, Literal

from minorminer.subgraph import find_subgraph
import networkx as nx
from dwave.graphs import chimera_two_color, pegasus_four_color, zephyr_four_color

from dwave.experimental.embedding_methods.quotient_embedding_search import (
    _normalize_coordinate,
)

__all__ = ["find_labeled_subgraph"]


def node_labels_by_orientation(
    graph: nx.Graph, as_str: bool = True
) -> dict[Hashable, str] | dict[Hashable, int]:
    """Generate node labels from graph orientation classes.

    For supported D-Wave graph families, this function labels nodes by their
    physical qubit orientation in processor implementations (vertical or horizontal).

    For non-D-Wave graph families, a greedy coloring is used as a fallback. In
    that case, the graph must be bipartite so that the resulting coloring has
    exactly two color classes, which serve as orientation labels.

    Args:
        graph: Input graph whose nodes will be labeled by orientation.
        as_str: If ``True``, convert orientation labels to strings before returning. If
            ``False``, preserve the integer labels. Defaults to ``True``.

    Returns:
        A dictionary mapping graph nodes to orientation labels.

    Raises:
        ValueError: If greedy coloring is used and produces
            more than two colors.
    """
    match graph.graph.get("family", None):
        case "pegasus" | "zephyr":
            col = {n: n[0] for n in graph.nodes()}
        case "chimera":
            col = {n: n[2] for n in graph.nodes()}
        case _:
            col = nx.greedy_color(graph)
            if len(set(col.values())) != 2:
                raise ValueError(
                    "Orientation labeling requires a bipartite graph, but greedy "
                    "coloring produced more than 2 colors"
                )
    if as_str:
        return {k: str(v) for k, v in col.items()}
    else:
        return col


def node_labels_by_coloring(
    graph: nx.Graph, as_str: bool = True
) -> dict[Hashable, str] | dict[Hashable, int]:
    """Generate node labels from a family-specific graph coloring.

    For supported D-Wave graph families, canonical 2-coloring for Chimera and 4-coloring
    for Pegasus and Zephyr are used. There can be more than 1 valid
    coloring not related by isomorphism, so failure to find a coloring is
    not sufficient to rule out any 2 (or 4) colored subgraph isomorphism.

    For graphs without recognized D-Wave family metadata, a greedy coloring is used as a
    generic fallback.

    Args:
        graph: Input graph to color. The family graph metadata is used to select
            a family-specific coloring method where available.
        as_str: If ``True``, convert color labels to strings before returning. If
            ``False``, preserve the integer color labels. Defaults to ``True``.

    Returns:
        A dictionary mapping graph nodes to color labels.
    """
    if "family" in graph.graph and graph.graph["family"] in (
        "chimera",
        "pegasus",
        "zephyr",
    ):
        graph, to_source = _normalize_coordinate(
            graph, graph.graph["rows"], graph.graph["tile"]
        )
        if graph.graph["family"] == "chimera":
            col = {n: chimera_two_color(n) for n in graph.nodes()}
        elif graph.graph["family"] == "pegasus":
            col = {n: pegasus_four_color(n) for n in graph.nodes()}
        elif graph.graph["family"] == "zephyr":
            col = {n: zephyr_four_color(n) for n in graph.nodes()}
        col = {to_source(n): color for n, color in col.items()}
    else:
        col = nx.greedy_color(graph)

    if as_str:
        return {k: str(v) for k, v in col.items()}

    return col


def node_labels_by_quotient(
    graph: nx.Graph, expand_boundary_search: bool = True, as_str: bool = True
) -> dict[Hashable, str] | dict[Hashable, tuple]:
    """Generate quotient graph labels for nodes based on graph family and structure.

    This function assigns quotient labels to nodes. See
    :func:`~dwave.experimental.embedding_methods.quotient_embedding_search.greedy_quotient_sublattice_mapping`
    for a description of the quotient graph.

    For Zephyr graphs with ``expand_boundary_search=True``, boundary quotient
    nodes are collapsed with adjacent internal nodes, since
    those assignments also allow for embeddings.

    Args:
        graph: A Chimera, Pegasus or Zephyr NetworkX graph.
        expand_boundary_search: If ``True`` and the graph family is ``"zephyr"``,
            boundary quotient nodes are remapped to adjacent (interior perpendicular
            block offset) nodes. Defaults to ``True``.
        as_str: If ``True``, labels are converted to strings. If ``False``, labels
            remain as tuples. Defaults to ``True``.

    Returns:
        A dictionary mapping node coordinates to quotient labels.

    Raises:
        ValueError: If graph family is not found in metadata or is not 'zephyr',
            'pegasus', or 'chimera'.
    """
    if "family" in graph.graph and graph.graph["family"] in (
        "chimera",
        "pegasus",
        "zephyr",
    ):
        graph, to_source = _normalize_coordinate(
            graph, graph.graph["rows"], graph.graph["tile"]
        )
        if graph.graph["family"] == "chimera":
            col = {to_source(n): n[:3] for n in graph.nodes()}
        elif graph.graph["family"] == "pegasus":
            col = {to_source(n): n[:2] + (n[2] // 2,) + n[3:] for n in graph.nodes()}
        elif graph.graph["family"] == "zephyr":
            if expand_boundary_search:
                m = graph.graph["rows"]

                wmap = {0: 1, 2 * m: 2 * m - 1}
                col = {
                    to_source(n): n[:1] + (wmap.get(n[1], n[1]),) + n[3:]
                    for n in graph.nodes()
                }
            else:
                col = {to_source(n): n[:2] + n[3:] for n in graph.nodes()}
    else:
        raise ValueError("Unrecognized graph family")

    if as_str:
        # Whitespace-free: find_subgraph's underlying vertex-label parser rejects labels
        # containing spaces, which the default tuple repr would otherwise include.
        return {k: str(v).replace(" ", "") for k, v in col.items()}
    else:
        return col


def find_labeled_subgraph(
    source: nx.Graph,
    target: nx.Graph,
    labeling_method: Literal["orientation", "quotient", "coloring"] = "orientation",
    node_labels: tuple[dict, dict] | None = None,
    **kwargs,
) -> dict[Hashable, Hashable]:
    """Find a subgraph of target isomorphic to source that preserves node colors.

    This is a helper function that calls :code:``find_subgraph`` with ``node_labels``.
    Node labeling can significantly accelerate the search when well chosen. The
    supported labeling methods are intended for bipartite graphs and D-Wave source
    graphs (Zephyr, Pegasus, and Chimera) embedded onto D-Wave target graphs. However,
    in general it may be necessary to consider uncolored search or several
    application-specific node labelings to find embeddings.

    Args:
        source: Source graph.
        target: Target graph.
        labeling_method: Method to use for coloring the graphs. Options are:
            - 'orientation': It is assumed variables in source must
                map to variables of the same orientation in target.
                If the graph orientation is unclear (the graph is not
                Chimera, Pegasus or Zephyr) the orientation is assigned by
                a greedy min-coloring and an error is thrown if there are
                not precisely two colors.
            - 'quotient': Quotient labeling is used. The graphs must
                be of type Chimera, Pegasus or Zephyr. Qubits are labeled
                by orientation and horizontal displacement. Note that this is
                only a good choice if the grid parameter (number of rows
                and columns) of source and target are matched.
            - 'coloring': According to canonical 2-colorings of Chimera,
                or the 4-coloring of Pegasus/Zephyr. Note that these colorings
                are not unique. If the graph orientation
                is unclear, the coloring is assigned by a greedy min-coloring.
                A greedy coloring is not guaranteed to be unique or optimal.
                For high performance applications consider direct
                specification of the coloring via the node_labels argument.
        node_labels: A tuple of dicts mapping nodes in source and target graphs
            to labels.
        **kwargs: Additional keyword arguments to pass to find_subgraph.
            Use of a timeout > 0 is recommended.
    """
    if node_labels is None:
        # use provided coloring
        match labeling_method:
            case "orientation":
                node_labels = tuple(
                    node_labels_by_orientation(G) for G in (source, target)
                )
            case "quotient":
                if (
                    "family" not in target.graph
                    or target.graph["family"] != source.graph["family"]
                ):
                    raise ValueError(
                        "Source and target graph families should match for quotient "
                        "coloring"
                    )
                if target.graph["family"] not in ("chimera", "pegasus", "zephyr"):
                    raise ValueError(
                        "Quotient coloring is only implemented for Chimera, Pegasus "
                        "and Zephyr graph families"
                    )
                node_labels = tuple(
                    node_labels_by_quotient(G) for G in (source, target)
                )
            case "coloring":
                node_labels = tuple(
                    node_labels_by_coloring(G) for G in (source, target)
                )
            case _:
                raise ValueError(f"Unknown coloring method {labeling_method}")

    return find_subgraph(source, target, node_labels=node_labels, **kwargs)
