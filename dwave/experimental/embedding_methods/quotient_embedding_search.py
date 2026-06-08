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

import itertools
import warnings
from collections import namedtuple
from typing import Callable, Hashable, Literal, get_args

import networkx as nx
import numpy as np
from dwave.embedding import verify_embedding
from dwave_networkx import (
    zephyr_coordinates,
    zephyr_graph,
    pegasus_coordinates,
    pegasus_graph,
    chimera_coordinates,
    chimera_graph,
)

__all__ = ["quotient_search"]

YieldType = Literal["node", "edge", "rail-edge"]
SearchStrategy = Literal["by_quotient_rail", "by_quotient_node", "by_rail_then_node"]
GraphFamily = Literal["zephyr", "pegasus", "chimera"]
EmbeddingMapping = dict[Hashable, tuple[Hashable, ...]]

QuotientSearchMetadata = namedtuple(
    "QuotientSearchMetadata",
    ["max_num_yielded", "starting_num_yielded", "final_num_yielded"],
)


def _expected_coordinate_tuple_len(family: str) -> int:
    """Return expected coordinate tuple length for a D-Wave graph family."""
    return 5 if family == "zephyr" else 4


def _validate_graph_inputs(source: nx.Graph, target: nx.Graph) -> None:
    """Validate that source and target are supported D-Wave NetworkX graphs.

    Both source and target graphs must be networkx graph instances with a ``'family'`` metadata
    key set to one of ``'zephyr'``, ``'pegasus'``, or ``'chimera'``. Each graph must also contain
    ``'rows'``, ``'tile'`` and ``'labels'`` metadata keys.

    Args:
        source: Source D-Wave graph.
        target: Target D-Wave graph.

    Raises:
        TypeError: If inputs are not NetworkX graphs.
        ValueError: If either graph is not a supported family graph or is missing 'rows'/'tile'
            metadata, or if the source and target graphs are not of the same family.
    """
    if not isinstance(source, nx.Graph) or not isinstance(target, nx.Graph):
        raise TypeError("source and target must both be networkx.Graph instances")

    valid_families = set(get_args(GraphFamily))
    source_family = source.graph.get("family")
    target_family = target.graph.get("family")

    if source_family not in valid_families:
        raise ValueError(
            "source graph should be a supported family graph "
            f"{sorted(valid_families)}"
        )
    if target_family != source_family:
        raise ValueError(
            "target graph should be the same family as the source graph "
            f"(source: {source_family}, target: {target_family})"
        )

    for graph_name, graph in zip(("source", "target"), (source, target)):
        for key in ("rows", "tile", "labels"):
            if key not in graph.graph:
                raise ValueError(
                    f"{graph_name} graph is missing required '{key}' metadata"
                )


def _extract_graph_properties(
    source: nx.Graph, target: nx.Graph
) -> tuple[int, int, int]:
    """Extract and validate graph properties, returning ``(rows, tile count, and target
    tile count)``.

    Each graph must contain required metadata fields: 'rows' (number of rows) and 'tile'
    (tile count). All metadata values must be positive integers. The source and target graphs must
    have matching row counts. The target tile count must be greater than or equal to the source tile
    count to accommodate the embedding.

    Args:
        source: Source graph.
        target: Target graph.

    Returns:
        Source rows, source tile count and target tile count.

    Raises:
        TypeError: If metadata values are not integers.
        KeyError: If graph shape metadata is missing from either graph.
        ValueError: If graph shape metadata is inconsistent between the
            source and target or the target is smaller than the source.
    """
    m = source.graph["rows"]
    if source.graph["family"] == "pegasus":
        tp = 1
        t = 2
    else:
        tp = source.graph["tile"]
        t = target.graph["tile"]

    for v, name in zip((m, tp, t), ("rows", "source tile", "target tile")):
        if not isinstance(v, int):
            raise TypeError(f"graph '{name}' metadata must be an integer")
        if v <= 0:
            raise ValueError(f"graph '{name}' metadata must be positive")
    if not (m == target.graph["rows"] == target.graph["columns"] == source.graph["columns"]):
        raise ValueError("source and target must have matched square grid parameters")
    if t < tp:
        raise ValueError("target tile count must be >= source tile count")

    return m, tp, t


def _validate_search_parameters(
    search_strategy: str,
    yield_type: str,
    embedding: EmbeddingMapping | None = None,
    *,
    source_family: str,
    target_family: str,
    source_labels: str,
    target_labels: str,
) -> None:
    """Validate high-level search parameters.

    ``search_strategy`` must be one of ``'by_quotient_rail'``, ``'by_quotient_node'``, or
    ``'by_rail_then_node'``; ``yield_type`` must be one of ``'node'``, ``'edge'``, or
    ``'rail-edge'``; and ``embedding`` must be ``None`` or a ``dict`` representing a
    one-to-one chain mapping where each source key is a coordinate tuple and each value is a
    singleton target-node chain.

    Args:
        search_strategy: Search mode.
        yield_type: Optimization objective.
        embedding: Optional initial one-to-one chain mapping in the input graph node label format.
            If None, no validation of the embedding is performed.
        source_family: Source graph family metadata value.
        target_family: Target graph family metadata value.
        source_labels: Source graph labels metadata value.
        target_labels: Target graph labels metadata value.

    Raises:
        ValueError: If ``search_strategy`` or ``yield_type`` is invalid, if ``embedding``
            contains duplicate target nodes (i.e. is not one-to-one), if embedding chains are not
            singleton tuples, if source keys are not coordinate tuples of the expected family
            length, or if coordinate-valued target nodes do not match family conventions
            (Zephyr=5, Pegasus/Chimera=4).
        TypeError: If ``embedding`` is provided but is not a dictionary.
    """
    valid_ksearch = get_args(SearchStrategy)
    valid_yield_type = get_args(YieldType)

    if search_strategy not in valid_ksearch:
        raise ValueError(
            f"search_strategy must be one of {sorted(valid_ksearch)}. Got "
            f"'{search_strategy}'"
        )
    if yield_type not in valid_yield_type:
        raise ValueError(
            f"yield_type must be one of {sorted(valid_yield_type)}. Got '{yield_type}'"
        )
    if embedding is not None:
        if not isinstance(embedding, dict):
            raise TypeError(
                f"embedding must be a dictionary when provided. Got {type(embedding)}"
            )
        source_coord_len = _expected_coordinate_tuple_len(source_family)
        target_coord_len = _expected_coordinate_tuple_len(target_family)

        # Validate chain format: values are singleton tuples.
        for key, value in embedding.items():
            if not isinstance(value, tuple) or len(value) != 1:
                raise ValueError(
                    f"embedding values must be singleton tuples representing node chains. "
                    f"Got value {value} of type {type(value)}"
                    + (f" with length {len(value)}" if isinstance(value, tuple) else "")
                    + f" for key {key}"
                )

            if not isinstance(key, tuple) or len(key) != source_coord_len:
                raise ValueError(
                    f"source coordinate keys must be {source_coord_len}-tuples for "
                    f"family '{source_family}'. Got key {key}"
                )

            if target_labels == "coordinate":
                target_node = value[0]
                if (
                    not isinstance(target_node, tuple)
                    or len(target_node) != target_coord_len
                ):
                    raise ValueError(
                        f"target coordinate nodes must be {target_coord_len}-tuples for "
                        f"family '{target_family}'. Got target node {target_node} for "
                        f"source key {key}"
                    )
            if source_labels == "coordinate":
                if not isinstance(key, tuple) or len(key) != source_coord_len:
                    raise ValueError(
                        f"source coordinate keys must be {source_coord_len}-tuples for "
                        f"family '{source_family}'. Got key {key}"
                    )
        # Check one-to-one constraint: flatten all chains and ensure no duplicates
        all_target_nodes = []
        for chain in embedding.values():
            all_target_nodes.extend(chain)
        if len(all_target_nodes) != len(set(all_target_nodes)):
            raise ValueError(
                "embedding must be a one-to-one mapping: duplicate target nodes detected across "
                "chains. "
            )


def _normalize_coordinate(
    graph: nx.Graph,
    m: int,
    t: int,
    add_singleton_nodes: bool = False,
) -> tuple[nx.Graph, Callable[[tuple], Hashable]]:
    """Normalise the source graph to coordinate labels.

    This function maps graphs to the family-appropriate coordinate system.

    Args:
        graph: D-Wave NetworkX compatible graph, either linear or coordinate labelled.
        m: Number of rows (must be consistent with ``graph``).
        t: Source tile count (must be consistent with ``graph``).
        add_singleton_nodes: If ``True``, add any missing nodes in the coordinate-labelled
            source graph as singleton nodes.
    Returns:
        coordinate-labelled (tuple) source graph and a callable that maps coordinate nodes
        back to the original source labelling space

    Raises:
        ValueError: If source labels are unsupported.
    """
    if graph.graph["family"] == "zephyr":
        graph_generator = zephyr_graph
        shape = (m, t)
        coords = zephyr_coordinates(*shape)
        to_linear = coords.zephyr_to_linear
        to_tuple = coords.linear_to_zephyr
    elif graph.graph["family"] == "pegasus":
        graph_generator = pegasus_graph
        shape = (m,)
        coords = pegasus_coordinates(*shape)
        to_linear = coords.pegasus_to_linear
        to_tuple = coords.linear_to_pegasus
    elif graph.graph["family"] == "chimera":
        shape = (m, m, t)
        graph_generator = chimera_graph
        coords = chimera_coordinates(*shape)
        to_linear = coords.chimera_to_linear
        to_tuple = coords.linear_to_chimera

    # As necessary convert edge_list to coordinates and define inversion
    if graph.graph["labels"] == "int":
        edge_list = [(to_tuple(n1), to_tuple(n2)) for n1, n2 in graph.edges()]
        node_list = [to_tuple(n) for n in graph.nodes()]
    elif graph.graph["labels"] == "coordinate":
        edge_list = graph.edges()
        node_list = graph.nodes()

        def to_linear(n: tuple) -> tuple:
            return n

    else:
        raise ValueError("source graph has unknown labelling scheme")
    generator_args = dict(coordinates=True, node_list=node_list, edge_list=edge_list)
    if add_singleton_nodes:
        if graph.graph["family"] == "pegasus":
            # For Pegasus, we need to add singleton nodes with odd k indices to get the full single-rail graph
            generator_args["node_list"] = [
                (u, w // 6, (2 * w) % 12 + k, z)
                for u in range(2)
                for w in range(6 * m)
                for k in range(t)
                for z in range(m - 1)
            ]
            generator_args["fabric_only"] = False
        else:
            # Default works
            generator_args["node_list"] = None

    _source = graph_generator(*shape, **generator_args)
    if graph.graph["family"] == "pegasus" and t == 1:
        # Pegasus quotient search only works for single-rail source graphs, which are defined by having only even k indices.
        # If any odd k nodes are present, raise an error.
        if any(n[2] % 2 != 0 for n in _source.nodes()):
            raise ValueError(
                "Pegasus quotient search requires that the source graph only contains nodes with even k indices, "
                "which defines a pegasus subgraph with single rails as opposed to pairs of rails."
            )
    return _source, to_linear


def _boundary_proposals(
    u: int,
    w: int,
    tp: int,
    t: int,
    embedding: dict[tuple[int, int, int, int, int], tuple[int, int, int, int, int]],
    j: int = 0,
    z: int = 0,
) -> set[tuple[int, int, int, int, int]]:
    r"""Generate candidate targets for boundary expansion.

    This routine applies only to Zephyr quotient search.

    For a fixed quotient index ``(u, w, j, z)``, this function proposes all target ``k`` locations
    in that rail, then removes the entries already occupied by the currently mapped source
    :math:`k \in \{0, \dots, tp-1\}`.

    Args:
        u: Zephyr orientation.
        w: Zephyr column index.
        tp: Source tile count.
        embedding: Current one-to-one proposal mapping.
        j: Intra-cell orientation index. Default is 0.
        z: Row index. Default is 0.

    Returns:
        Available target coordinate nodes, each represented as the 5-tuple
        ``(u, w, k, j, z)``, with fixed ``(u, w, j, z)``.
    """
    all_target_coordinates = {(u, w, k, j, z) for k in range(t)}
    used_coordinates = {
        embedding[(u, w, k, j, z)] for k in range(tp) if (u, w, k, j, z) in embedding
    }
    return all_target_coordinates.difference(used_coordinates)


def _node_search(
    source: nx.Graph,
    target: nx.Graph,
    embedding: dict[tuple, tuple],
    *,
    expand_boundary_search: bool = True,
    ksymmetric: bool = False,
    yield_type: YieldType = "edge",
) -> dict[tuple, tuple]:
    r"""Greedy node-level quotient search

    subgraph isomorphisms (1:1 embeddings) are searched subject to the restriction that nodes in the
    source are mapped to nodes in the target aligned subject to the constraint of matched quotient
    graph structure.

    The following case describes specifically zephyr graphs, but the general approach applies to
    chimera and pegasus graphs with the appropriate quotient graph structure and coordinate conventions.
    
    The source and target are viewed in quotient blocks indexed by :math:`(u, w, j, z)`, each
    containing :math:`tp` source nodes. For each block, we propose target nodes with the same
    :math:`(u, w, j, z)` and varying target :math:`k`, optionally augmented with boundary proposals.

    The scoring objective is:

    .. math::

        \operatorname{score}(p) =
        \begin{cases}
        \sum\limits_{n \in B} \mathbf{1}[p_n \in V(T)] & \text{node yield}\\
        \sum\limits_{(n,m) \in E(S_B, S_\text{fixed})}
        \mathbf{1}[(p_n, \phi(m)) \in E(T)] & \text{edge yield}
        \end{cases}

    For a fixed quotient index :math:`q = (u, w, j, z)`, define the source block :math:`B_q` as

    .. math::

        B_q = \{(u, w, k, j, z) : k \in \{0, \dots, tp-1\}\}.

    A proposal :math:`p` is an assignment on that block, :math:`p: B_q \to V(T)`, and can be
    viewed as a length-``tp`` vector :math:`(p_0, \dots, p_{tp-1})` where :math:`p_k` is the
    proposed target node for source node :math:`(u, w, k, j, z)`.

    Here :math:`T` is the target graph, :math:`V(T)` is its node set, and :math:`E(T)` is its edge
    set. Let :math:`S` be the source graph and define the already-fixed outside set

    .. math::

        F_q = \{m \in V(S) \setminus B_q : m \in \operatorname{dom}(\phi)\},

    where :math:`\phi` is the current embedding. Then

    .. math::

        E(S_B, S_\text{fixed})
        := \{(n,m) \in E(S) : n \in B_q,\ m \in F_q\},

    i.e., the source edges that cross from the current block to already-fixed source nodes outside
    the block.

    In other words, node yield counts how many proposed nodes :math:`p_n` are present in
    :math:`V(T)`; while edge yield counts how many source edges crossing from the current block to
    already-fixed nodes are preserved as target edges :math:`(p_n, \phi(m)) \in E(T)`.

    Yield types in this node-level search are interpreted as follows: ``"node"`` maximises target
    node presence for each proposed block; ``"edge"`` maximises preserved cross-block
    source-to-fixed edge connectivity; and ``"rail-edge"`` follows the same node-level scoring as
    ``"edge"`` in this function (the distinction between ``"edge"`` and ``"rail-edge"`` is made
    in rail-level search).

    Args:
        source: Coordinate-labeled graph. 
        target: Coordinate-labeled graph. The family should be matched to the source family.
        embedding: Current mapping, updated in-place.
        expand_boundary_search: If ``True``, augment boundary columns using the adjacent
            internal column. Defaults to ``True``. This argument is only applicable to Zephyr
            graph applications, and is ignored for Pegasus and Chimera graph families.
        ksymmetric: If ``True``, assume the order of source ``k`` indices is interchangeable
            for scoring and use top-``tp`` selection. Defaults to ``False``.
        yield_type: ``"node"``, ``"edge"``, or ``"rail-edge"``. Defaults to ``"edge"``.

    Returns:
        Updated embedding.

    Raises:
        ValueError: If graph geometry metadata is inconsistent.
            This includes source/target family mismatch and unsupported geometry assumptions.
    """
    expand_boundary_search = source.graph["family"] == "zephyr" and expand_boundary_search
    m = source.graph["rows"]
    if source.graph["family"] == "pegasus":
        tp = 1  # Only non-trivial case
        t = 2  # Only case.
    else:
        tp = source.graph["tile"]
        t = target.graph["tile"]
    if source.graph["family"] != target.graph["family"]:
        raise ValueError(
            "source and target families should be matched for implemented searches"
        )
    if not (m == target.graph["rows"] == target.graph["columns"] == source.graph["columns"]):
        raise ValueError("source and target must have matched square grid parameters")

    if expand_boundary_search:
        # Visit interior columns first so boundary expansion can reuse already-assigned assignments:
        quotient_node_iterator = itertools.product(
            range(2),
            list(range(1, 2 * m)) + [0, 2 * m],
            range(2),
            range(m),
        )
        ksymmetric_original = ksymmetric
        def _quotient_to_var(nq, k):
            return nq[:2] + (k,) + nq[2:]
    else:
        if source.graph['family'] == 'zephyr':
            quotient_node_iterator = itertools.product(
                range(2), range(2 * m + 1), range(2), range(m)
            )
            def _quotient_to_var(nq, k):
                return nq[:2] + (k,) + nq[2:]
        elif source.graph['family'] == 'pegasus':
            quotient_node_iterator = itertools.product(
                range(2), range(6*m), range(m-1)
            )
            def _quotient_to_var(nq, k):
                u, w, z = nq
                return (u, w//6, 2*(w % 6) + k, z)
        elif source.graph['family'] == 'chimera':
            quotient_node_iterator = itertools.product(
                range(2), range(m), range(m)
            ) # Orientation, orthogonal displacement, parallel displacement
            def _quotient_to_var(nq, k):
                u, w, z = nq
                return (w * u + z * (1 - u), w * (1 - u) + z * u, u, k)
        else:
            raise ValueError('Unknown family')

    for nq in quotient_node_iterator:
        # Base proposals preserve (u, w, j, z) and search only over target k-indices:
        proposals = [_quotient_to_var(nq, k) for k in range(t)]

        if expand_boundary_search:
            u, w, j, z = nq
            if w == 0:
                ksymmetric = False
                # borrow candidates from adjacent internal column
                proposals += list(_boundary_proposals(u, 1, tp, t, embedding, j, z))
            elif w == 2 * m:
                ksymmetric = False
                proposals += list(
                    _boundary_proposals(u, 2 * m - 1, tp, t, embedding, j, z)
                )
            else:
                ksymmetric = ksymmetric_original

        if ksymmetric or yield_type != "edge":
            if yield_type == "node":
                # symmetry doesn't matter: just count how many proposed nodes are present in the
                # target:
                counts = [int(target.has_node(n_t)) for n_t in proposals]
            else:
                # Count preserved edges from already-mapped neighboring source nodes into each
                # proposed target node.
                source_neighbours = source.neighbors(_quotient_to_var(nq, 0))
                counts = [
                    sum(
                        int(target.has_edge(embedding[n_s], n_t))
                        for n_s in source_neighbours
                        if n_s in embedding
                    )
                    for n_t in proposals
                ]
            # performance: this is faster than selected = proposals[np.argsort()]...
            top_indices = np.argpartition(np.asarray(counts), -tp)[-tp:]
            selected = [proposals[idx] for idx in top_indices]
        else:
            # Nodes with different k indices in the source block are not interchangeable, so we
            # evaluate all permutations of the proposals:
            permutation_scores = {
                proposal_perm: sum(
                    int(target.has_edge(embedding[n], proposal_perm[k]))
                    for k in range(tp)
                    for n in source.neighbors(_quotient_to_var(nq, k))
                    if n in embedding
                )
                for proposal_perm in itertools.permutations(proposals, tp)
            }
            selected_key = max(permutation_scores, key=lambda k: permutation_scores[k])
            selected = list(selected_key)

        embedding.update(
            {_quotient_to_var(nq, k): proposal for k, proposal in zip(range(tp), selected)}
        )

    return embedding

def _rail_nodes(m, family):
    if family == 'chimera':
        def to_nodes(u, w, k):
            for z in range(m):
                yield (w * u + z * (1 - u), w * (1 - u) + z * u, u, k)
    elif family == 'zephyr':
        def to_nodes(u, w, k):
            for j in range(2):
                for z in range(m):
                    yield (u, w, k, j, z)
    elif family == 'pegasus':
        def to_nodes(u, w, k):
            for z in range(m - 1):
                yield (u, w // 6, 2 * (w % 6) + k, z)
    else:
        raise ValueError(f'Unknown rails for {family}')
    return to_nodes


def _rail_search(
    source: nx.Graph,
    target: nx.Graph,
    embedding: dict[tuple, tuple],
    *,
    expand_boundary_search: bool = True,
    ksymmetric: bool = False,
    yield_type: YieldType = "edge",
) -> dict[tuple, tuple]:
    r"""Greedy rail-level quotient search

    Implementation status: rail-level search supports Zephyr, Chimera, and Pegasus coordinate
    families. 

    Rails are connected components that consist of connected node sequences of the same orientation:
    vertical (u=0) or horizontal (u=1) qubits. Removing internal edges (those that connect
    qubits of differing orientation, rails are disconnected graph components.

    Rails are indexed by :math:`(u, w, k)`, where ``u`` denotes ``orientation`` and ``w`` denotes
    ``orthogonal_displacement``. For Zephyr these match the standard coordinate system.
    Zephyr rails contain all qubits (u, w, k, *, *).
    Chimera rails contain (*, w, u, k) for u=0, (w, *, u, k) for u=1.
    Pegasus rails contain (u, w//6, (2 w)%12 + k, *)

    Rails that differ only in the k index are related by automorphism in the fully-yielded graph.
    This code greedily searches for a mapping (1:1 embedding) of rails on the source graph to rails on the larger graph
    in order to maximize some objective outcome like edge-yield.

    The following description applies to Zephyr, but generalizes qualitatively to Chimera and Pegasus
    graphs:
    .. math::

        \mathcal{R}^{S}_{u,w} := \{(u, w, k_s) : k_s \in \{0, \dots, t_p-1\}\}.

    The search chooses :math:`t_p` target rails for each family :math:`\mathcal{R}^{S}_{u,w}`
    from candidate rails optionally augmented at boundaries (:math:`w=0` and :math:`w=2m`) using
    adjacent interior columns.

    Let the target rail indexed by :math:`(u, w_t, k_t)` be

    .. math::

        R^{T}_{u,w_t,k_t} :=
        \{(u, w_t, k_t, j, z) : j \in \{0,1\},\ z \in \{0,\dots,m-1\}\}.

    We can define its objective for ``yield_type='edge'`` as the number of edges preserved within
    that rail, i.e., the number of edges in the target subgraph induced by the proposed rail, or
    equivalently the number of edges in the source rail (which is fixed) that are preserved by the
    proposal:

    .. math::

        Q(u,w_t,k_t) := |E(T[R^{T}_{u,w_t,k_t}])|,

    or, for ``yield_type='node'``, the number of present target nodes in that rail. Here :math:`T`
    is the target graph and :math:`E(T[R])` is the edge set of the target subgraph induced by node
    set :math:`R`.
    For ``yield_type='edge'``, each proposal also gets an external connectivity term counting
    preserved edges from already-embedded neighbouring source nodes into the proposed target rail.

    .. math::

        \operatorname{score}(u,w_t,k_t)
        = Q(u,w_t,k_t)
        + \sum \mathbf{1}[\text{external source edge maps to a target edge}].

    Depending on ``ksymmetric``, the algorithm either selects the top :math:`t_p` rail proposals by
    score (treating source :math:`k` order as interchangeable), or evaluates permutations assigning
    proposal rails to source indices :math:`k_s \in \{0,\dots,t_p-1\}`.

    Yield types in this rail-level search are interpreted as follows: ``"node"`` scores each
    proposal rail by the number of present target nodes in that rail. ``"edge"`` prefers rails
    that both have many internal rail edges and connect well to already-embedded neighbouring
    rails. ``"rail-edge"`` focuses first on how good the rail itself is, measured by the number of
    target edges inside that rail; when permutations are evaluated, it also includes the same
    already-embedded neighbour consistency term as ``"edge"``.

    Example: suppose two candidate target rails have the same internal rail structure, but one of
    them has more edges to neighbouring rails that are already fixed in the embedding. Then
    ``"edge"`` prefers that better-connected rail, while ``"rail-edge"`` treats the two rails as
    equivalent in the top-rail selection path because it only compares their internal rail
    structure there.

    Selected rails are then expanded back to node assignments for all :math:`(j,z)` in
    each source rail.

    Args:
        source: Coordinate-labeled source graph.
        target: Coordinate-labeled target graph.
        embedding: Current mapping, updated in-place.
        expand_boundary_search: If ``True``, include adjacent-column rail proposals when
            :math:`w` is at a boundary. Defaults to ``True``. Boundary expansion is only 
            relevant for Zephyr, it is ignored for other graph families.
        ksymmetric: If ``True``, treat source :math:`k` order as interchangeable when scoring
            rails. Defaults to ``False``.
        yield_type: ``"node"``, ``"edge"``, or ``"rail-edge"``. Defaults to ``"edge"``.

    Returns:
        Updated embedding.

    Raises:
        ValueError: If duplicate target assignments are produced.
        ValueError: If graph family is unknown.
    """

    expand_boundary_search = (
        expand_boundary_search and source.graph["family"] == "zephyr"
    )
    m = source.graph["rows"]
    if source.graph["family"] == "pegasus":
        # Only non-trivial case: contraction of odd-couplers.
        u_index = 0
        tp = 1
        t = 2
        num_orthogonal_displacements = 6 * m
    elif source.graph["family"] == "zephyr":
        u_index = 0
        tp = source.graph["tile"]
        t = target.graph["tile"]
        num_orthogonal_displacements = 2 * m + 1    
    elif source.graph["family"] == "chimera":
        u_index = 2
        tp = source.graph["tile"]
        t = target.graph["tile"]
        num_orthogonal_displacements = m
    else:
        raise ValueError("unknown graph family")
    rail_nodes = _rail_nodes(m, source.graph["family"])    
    uw_iterator = list(itertools.product(range(2), range(num_orthogonal_displacements)))

    if yield_type == "node":
        rail_score = {
            (u, w, k): sum(target.has_node(node) for node in rail_nodes(u, w, k))
            for u, w in uw_iterator
            for k in range(t)
        }
    else:
        # Precompute per-rail edge number for fast proposal scoring.
        rail_score = {
            (u, w, k): target.subgraph(
                {node for node in rail_nodes(u, w, k)}
            ).number_of_edges()
            for u, w in uw_iterator
            for k in range(t)
        }

    # when optimising for edges, we consider all edges that do not share the same orientation
    source_external_edges = (
        source.edge_subgraph(
            {e for e in source.edges() if e[0][u_index] != e[1][u_index]}
        ).copy()
        if "edge" in yield_type
        else None
    )
    if source_external_edges:
        source_external_edges.add_nodes_from(source.nodes())  # Pathological edge case
    if expand_boundary_search:
        uw_iterator = list(
            itertools.product(range(2), list(range(1, 2 * m)) + [0, 2 * m])
        )
        ksymmetric_original = ksymmetric

    for u, w in uw_iterator:
        # rail proposals preserve orientation in the target graph and only move in (w, k) quotient
        # graph.
        proposals = [(w, k) for k in range(t)]

        if expand_boundary_search:
            if w == 0:
                # b[1:3] is taken because those are the w and k indices
                proposals += [
                    b[1:3] for b in _boundary_proposals(u, 1, tp, t, embedding)
                ]
                ksymmetric = False
            elif w == 2 * m:
                proposals += [
                    b[1:3] for b in _boundary_proposals(u, 2 * m - 1, tp, t, embedding)
                ]
                ksymmetric = False
            else:
                ksymmetric = ksymmetric_original

        if ksymmetric or yield_type == "node":
            if yield_type in ("node", "rail-edge"):
                counts = [rail_score[(u, w_t, k_t)] for w_t, k_t in proposals]
            else:
                # the other only possibility is that yield_type == "edge". The following check is
                # just to avoid linter complaint about source_external_edges being possibly None.
                if source_external_edges is None:
                    raise ValueError("internal error: missing external edge subgraph")
                counts = [
                    rail_score[(u, w_t, k_t)]
                    + sum(
                        int(target.has_edge(embedding[neigh_r], n))
                        for n, n_s in zip(rail_nodes(u, w_t, k_t), rail_nodes(u, w, 0))
                        # neigh_r will be nodes in the source graph with a different orientation
                        # to the current rail, that are neighbours of nodes in the current rail.
                        # Note that we pick k=0 because ksymmetric means that all k indices in the
                        # source rail are interchangeable, so we can just look at one of them.
                        for neigh_r in source_external_edges.neighbors(n_s)
                        if neigh_r in embedding
                    )
                    for w_t, k_t in proposals
                ]

            p_indices = np.argpartition(np.asarray(counts), -tp)[-tp:]
            # Apply chosen rails to all nodes in the quotient rail block.
            embedding.update(
                {
                    n1: n2
                    for k in range(tp)
                    for n1, n2 in zip(
                        rail_nodes(u, w, k), rail_nodes(u, *proposals[p_indices[k]])
                    )
                }
            )
        else:
            # this path is activated when ksymmetric is False and yield_type is either "edge" or
            # "rail-edge".
            if source_external_edges is None:
                raise ValueError("internal error: missing external edge subgraph")

            permutation_scores = {
                proposal_perm: sum(
                    rail_score[(u,) + proposal] for proposal in proposal_perm
                )
                + sum(
                    int(target.has_edge(embedding[n_neigh], n))
                    for k_s, proposal in enumerate(proposal_perm)
                    for n, n_s in zip(rail_nodes(u, *proposal), rail_nodes(u, w, k_s))
                    for n_neigh in source_external_edges.neighbors(n_s)
                    if n_neigh in embedding
                )
                for proposal_perm in itertools.permutations(proposals, tp)
            }
            selected = max(permutation_scores, key=lambda k: permutation_scores[k])
            embedding.update(
                {
                    n_s: n_t
                    for k in range(tp)
                    for n_s, n_t in zip(
                        rail_nodes(u, w, k), rail_nodes(u, *selected[k])
                    )
                }
            )

        if len(set(embedding.values())) != len(embedding):
            raise ValueError("Duplicate target coordinates detected in embedding")

    return embedding


def quotient_search(
    source: nx.Graph,
    target: nx.Graph,
    *,
    search_strategy: SearchStrategy = "by_quotient_rail",
    embedding: EmbeddingMapping | None = None,
    expand_boundary_search: bool = True,
    ksymmetric: bool = False,
    yield_type: YieldType = "edge",
) -> tuple[EmbeddingMapping, QuotientSearchMetadata]:
    r"""Compute a high-yield quotient embedding for supported D-Wave graph families.

    This routine starts from a source graph with ``m`` rows and ``tp`` tiles,
    and maps it into a target graph with the same ``m`` rows and ``t >= tp``
    tiles. It is designed for defective targets where a direct identity map may lose
    nodes or edges. Since a greedy method is used for embedding search, it is possible it fails to
    find a 1:1 embedding where one is viable. A complete method such as
    :code:``minorminer.subgraph.find_subgraph`` may be more appropriate in a scenario such as this,
    especially with customization of parameters to the target families. Similarly, when defect rates
    are high direct use of :code:``minorminer.find_embedding`` may be a more efficient strategy.

    The quotient construction and greedy update rules in this implementation are based on Zephyr,
    Pegasus or Chimera coordinate blocks presented in the D-Wave networkx format.

    The search is organized around the **quotient graph** of the topology, formed by
    contracting fine-grained coordinate indices so that each equivalence class maps to a single
    quotient node. Two coarsenings are used:

    - **Quotient node** block :math:`(u, w, j, z)`: groups the ``tp`` source nodes that share
      orientation ``u``, column ``w``, intra-cell index ``j``, and row ``z`` but differ in
      tile index :math:`k \in \{0, \dots, tp-1\}`.
    - **Quotient rail** block :math:`(u, w)`: groups all :math:`2 m \cdot tp` nodes that share
      orientation ``u`` and column ``w`` (i.e. a whole Zephyr rail family) before any
      :math:`(k, j, z)` variation.

    The function can be used in (1) node-level mode (``search_strategy='by_quotient_node'``), where
    each quotient node block :math:`(u,w,j,z)` is optimized by choosing target candidates with the
    same :math:`(u,w,j,z)` and selecting the highest-yield proposals; (2) rail-level mode
    (``search_strategy='by_quotient_rail'``): optimise each quotient rail block :math:`(u,w,:)` by
    selecting rails :math:`(u,w_t,k_t)` that maximise yield.; and (3) hybrid mode
    (``search_strategy='by_rail_then_node'``): rail search followed by node refinement.

    Family-specific support status:
        - Zephyr and Chimera: support all three search strategies.
        - Pegasus: supports all three strategies in the implemented quotient setting
            (``tp=1``, ``t=2``).

    When ``expand_boundary_search=True``, boundary columns ``w=0`` and ``w=2m`` are augmented using
    proposals drawn from adjacent internal columns. Whenever this behaviour is activated, nodes from
    the internal columns are assigned first, so that the unassigned nodes in the internal columns
    adjacent to the boundaries can be considered as proposals when optimising the boundary columns.

    Yield types control what the greedy search tries to preserve. ``"node"`` tries to place as
    many source nodes as possible onto target nodes that actually exist. ``"edge"`` tries to
    preserve source edges throughout the search. ``"rail-edge"`` is a mixed strategy: during rail
    search it first prefers rails that are internally well-formed, and if a node-refinement phase
    runs afterward it switches to ordinary edge-preservation scoring. The final yield for both
    ``"edge"`` and ``"rail-edge"`` is reported as a number of preserved source edges.

    Multi-family API note: this function accepts Zephyr/Pegasus/Chimera graph metadata and
    coordinate-form embedding validation at the API boundary. Strategy coverage is implemented for
    Zephyr, Chimera, and Pegasus (with the Pegasus quotient-search constraints above).

    Args:
        source: Source graph (linear or coordinate labels) from a supported D-Wave topology
            family: ``'zephyr'``, ``'pegasus'``, or ``'chimera'``.
        target: Target graph (linear or coordinate labels) from a supported D-Wave topology
            family: ``'zephyr'``, ``'pegasus'``, or ``'chimera'``.
        search_strategy: Search strategy. One of ``'by_quotient_rail'``,
            ``'by_quotient_node'``, or ``'by_rail_then_node'``. See full docstrings for a
            description of these. Defaults to ``'by_quotient_rail'``.
        embedding: Optional initial one-to-one chain mapping. If omitted,
            the identity on source coordinate indices is used (wrapped in singleton chains).
            Defaults to ``None``. This must be a chain mapping where each source node maps to
            a singleton tuple target chain, e.g. ``{source_node: (target_node,)}``.
        expand_boundary_search: Enable additional boundary proposals. Defaults to ``True``.
        ksymmetric: Assume source ``k`` ordering can be treated symmetrically during greedy
            selection when valid. Defaults to ``False``.
        yield_type: Optimization objective: ``'node'``, ``'edge'``, or ``'rail-edge'``.
            See full docstrings for a description of these. Defaults to ``'edge'``.

    Returns:
        A pruned one-to-one chain embedding of the form ``source_node -> (target_node,)`` (singleton
        chains) that contains only mappings whose target node exists in the target, and a
        :class:`QuotientSearchMetadata` namedtuple with fields ``max_num_yielded``,
        ``starting_num_yielded``, and ``final_num_yielded``.

    .. note::
        If you want to embed a Zephyr graph with parameter ``mp`` < ``m``, where ``m`` is the row
        count of the target, you can use
        ``minorminer.utils.parallel_embeddings.find_sublattice_embeddings`` to locate a compatible
        ``mp``-row sublattice first, then pass that induced subgraph as the target.

        .. code-block:: python

            import networkx as nx
            import dwave_networkx as dnx
            from minorminer.utils.parallel_embeddings import find_sublattice_embeddings

            # Build an mp-row Zephyr tile and locate it in the original target.
            tile = dnx.zephyr_graph(mp, target.graph["tile"], coordinates=True)
            tile_embs = find_sublattice_embeddings(
                S=tile,
                T=target,
                max_num_emb=1,
                one_to_iterable=False,
            )

            if tile_embs:
                tile_to_target = tile_embs[0]  # pick the first one
                mp_nodes = set(tile_to_target.values())
                target_mp = target.subgraph(mp_nodes).copy()

                # Relabel to canonical mp coordinates expected by source/target metadata.
                target_to_tile = {tgt: tile_n for tile_n, tgt in tile_to_target.items()}
                target_mp = nx.relabel_nodes(target_mp, target_to_tile, copy=True)
                target_mp.graph.update(family="zephyr", rows=mp, tile=target.graph["tile"],
                                       labels="coordinate")

                emb_mp, metadata = search_strategy(source, target_mp)

                # Map the final embedding back to the original target labels.
                emb_in_original_target = {
                    s: tuple(tile_to_target[v] for v in chain)
                    for s, chain in emb_mp.items()
                }

        If you want to refine a non-full-yield result with an external solver, run
        :func:`search_strategy` first and only call the refinement routine when
        ``metadata.final_num_yielded < metadata.max_num_yielded``.

        .. code-block:: python

            emb, metadata = quotient_search(source, target, yield_type="edge")
            if metadata.final_num_yielded < metadata.max_num_yielded:
                import minorminer

                initial_chains = {s: chain for s, chain in emb.items() if chain[0] in target}
                refined = minorminer.find_embedding(
                    S=source,
                    T=target,
                    initial_chains=initial_chains,
                    timeout=5,  # or whatever you want
                )
    """

    _validate_graph_inputs(source, target)
    m, tp, t = _extract_graph_properties(source, target)
    _validate_search_parameters(
        search_strategy,
        yield_type,
        embedding,
        source_family=source.graph["family"],
        target_family=target.graph["family"],
        source_labels=source.graph["labels"],
        target_labels=target.graph["labels"],
    )

    # Make sure source and target are in coordinate form (tuples)
    _source, to_source = _normalize_coordinate(source, m, tp, add_singleton_nodes=True)
    _target, to_target = _normalize_coordinate(target, m, t)

    if embedding is None:
        # Start with the identity mapping
        working_embedding = {n: n for n in _source.nodes()}
    else:
        # Convert chain format to internal single-node format
        working_embedding = {k: v[0] for k, v in embedding.items()}

    if yield_type == "node":
        max_num_yielded = source.number_of_nodes()
        num_yielded = sum(
            _target.has_node(working_embedding[n])
            for n in _source.nodes()
            if n in working_embedding
        )
    else:
        max_num_yielded = source.number_of_edges()
        num_yielded = sum(
            _target.has_edge(working_embedding[n1], working_embedding[n2])
            for n1, n2 in _source.edges()
            if n1 in working_embedding and n2 in working_embedding
        )

    full_yield = max_num_yielded == num_yielded
    starting_yield = num_yielded

    if not full_yield:
        supplement = search_strategy == "by_rail_then_node"

        if search_strategy == "by_quotient_rail" or supplement:
            working_embedding = _rail_search(
                source=_source,
                target=_target,
                embedding=working_embedding,
                # if search_strategy is by_rail_then_node, we expand boundary search only in the
                # node search, and disable it in the rail search:
                expand_boundary_search=((not supplement) and expand_boundary_search),
                ksymmetric=ksymmetric,
                yield_type=yield_type,
            )
            if supplement:
                working_embedding = _node_search(
                    source=_source,
                    target=_target,
                    embedding=working_embedding,
                    expand_boundary_search=expand_boundary_search,
                    ksymmetric=False,
                    yield_type=yield_type,
                )
        elif search_strategy == "by_quotient_node":
            working_embedding = _node_search(
                source=_source,
                target=_target,
                embedding=working_embedding,
                expand_boundary_search=expand_boundary_search,
                ksymmetric=ksymmetric,
                yield_type=yield_type,
            )

        if yield_type == "node":
            num_yielded = sum(
                _target.has_node(working_embedding[n])
                for n in _source.nodes()
                if n in working_embedding
            )
        else:
            num_yielded = sum(
                _target.has_edge(working_embedding[n1], working_embedding[n2])
                for n1, n2 in _source.edges()
                if n1 in working_embedding and n2 in working_embedding
            )
        full_yield = max_num_yielded == num_yielded

        if num_yielded < starting_yield:
            raise ValueError("Greedy quotient search reduced the objective value")

    # If there are unfeasible mappings to target nodes, the final working_embedding might contain
    # entries that map to non-existent target nodes. We prune those out before returning the final
    # embedding:
    pruned_embedding = {
        to_source(k): to_target(v)
        for k, v in working_embedding.items()
        if v in _target.nodes()
    }

    # Convert to chain format for return value
    pruned_embedding = {k: (v,) for k, v in pruned_embedding.items()}

    if full_yield and yield_type != "node":
        verify_embedding(emb=pruned_embedding, source=source, target=target)

    metadata = QuotientSearchMetadata(
        max_num_yielded=max_num_yielded,
        starting_num_yielded=starting_yield,
        final_num_yielded=num_yielded,
    )
    return pruned_embedding, metadata
