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

from collections.abc import Hashable

import dimod
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice.automorphism import schreier_rep

__all__ = [
    'reindex',
    'make_signed_bqm',
    'get_bqm_orbits',
    'get_unsigned_bqm_orbits',
    'get_orbits',
]

def reindex(mapping: dict[Hashable, int]) -> dict[Hashable, int]:
    """Reindex dictionary values to consecutive integers starting at zero.
    
    Args:
        mapping: Dictionary whose values represent indices or labels.
    """
    value_mapping = {v: i for i, v in enumerate(dict.fromkeys(mapping.values()))}
    return {k: value_mapping[v] for k, v in mapping.items()}


def make_signed_bqm(bqm: dimod.BQM) -> dimod.BQM:
    """Construct a signed expansion of a BQM.
    
    Takes a bqm and duplicates every spin s into two copies corresponding to
    s and -s.
    Each field h gets mapped to two opposing fields:
     h(s1) = -h(s2)
    each coupler gets mapped to four couplers:
     J(s1,s2) = J(-s1,-s2) = -J(s1,-s2) = -J(-s1,s2)

    Args:
        bqm: Input binary quadratic model.

    Returns:
        A new BQM with duplicated variables representing both signs of each spin.
     """
    # Nodes and edges added in a seemingly ugly way in order to get the order right.
    ret = dimod.BinaryQuadraticModel(vartype="SPIN")
    for var in bqm.variables:
        ret.add_variable(f"p{var}", bqm.linear[var])
    for var in bqm.variables:
        ret.add_variable(f"m{var}", -bqm.linear[var])

    for u, v in bqm.quadratic:
        ret.add_quadratic(f"p{u}", f"p{v}", bqm.quadratic[(u, v)])
    for u, v in bqm.quadratic:
        ret.add_quadratic(f"m{u}", f"m{v}", bqm.quadratic[(u, v)])
    for u, v in bqm.quadratic:
        ret.add_quadratic(f"p{u}", f"m{v}", -bqm.quadratic[(u, v)])
    for u, v in bqm.quadratic:
        ret.add_quadratic(f"m{u}", f"p{v}", -bqm.quadratic[(u, v)])

    return ret


def get_bqm_orbits(
    bqm: dimod.BQM,
) -> tuple[dict[Hashable, int], dict[tuple[Hashable, Hashable], int]]:
    """Take a bqm, perhaps a "signed" bqm from make_signed_bqm, and convert it
    into a vertex-colored graph as needed.

    Since the automorphism module only takes edge colorings, the couplings
    (J terms) need to be specified using auxiliary vertices.  Thus for every
    edge (u,v) of the BQM graph, we add a new vertex w(u,v) and give it the color
    corresponding to J(u,v) in the BQM.

    To avoid ambiguity, we add a pendant (degree 1) vertex corresponding to each
    original vertex.

    Args:
        bqm: Input binary quadratic model.

    Returns:
        A tuple ``(qubit_orbits, coupler_orbits)`` where ``qubit_orbits`` maps
        each node to an integer orbit label and ``coupler_orbits`` maps each
        edge to an integer orbit label. 
    """
    # The function first adds auxiliary elements to a BQM
    graph = nx.Graph()

    for v in bqm.variables:
        graph.add_node(f"hnode_{v}")
        graph.add_node(v)
        graph.add_edge(v, f"hnode_{v}")

    for u, v in bqm.quadratic:
        graph.add_edge(u, v)
        graph.add_node(f"Jnode_{u}_{v}")
        graph.add_edge(u, f"Jnode_{u}_{v}")
        graph.add_edge(v, f"Jnode_{u}_{v}")

    node_labels = list(graph.nodes)
    num_nodes = graph.number_of_nodes()
    node_to_idx = {node: i for i, node in enumerate(graph.nodes())}

    mapping_h = {h: [] for h in set(bqm.linear.values())}
    mapping_mp = {h: [] for h in set(bqm.linear.values())}
    mapping_J = {J: [] for J in set(bqm.quadratic.values())}

    for p, q in bqm.linear.items():
        mapping_h[q].append(f"hnode_{p}")
        mapping_mp[q].append(p)

    for p, q in bqm.quadratic.items():
        mapping_J[q].append(f"Jnode_{p[0]}_{p[1]}")

    # Make color classes
    coloring = []
    for nodes_h in mapping_h.values():
        coloring.append({node_to_idx[v] for v in nodes_h})
    for nodes_J in mapping_J.values():
        coloring.append({node_to_idx[e] for e in nodes_J})
    for nodes_mp in mapping_mp.values():
        coloring.append({node_to_idx[v] for v in nodes_mp})

    graph_coloring = {}
    node_colors = np.zeros(num_nodes)
    for i, color in enumerate(coloring):
        node_colors[list(color)] = i
        for node in color:
            graph_coloring[node_labels[node]] = i

    result = schreier_rep(graph, graph_coloring=graph_coloring)

    vertex_orbits = result.vertex_orbits_original_labels
    vertex_orbit_array = np.zeros(num_nodes, dtype=int)
    for i in range(len(vertex_orbits)):
        vertex_orbit_array[[node_to_idx[x] for x in vertex_orbits[i]]] = i
    qubit_orbits = {
        spin: vertex_orbit_array[node_to_idx[f"hnode_{spin}"]] for spin in bqm.variables
    }

    edge_orbits = result.edge_orbits_original_labels
    edge_orbit_array = np.zeros(num_nodes, dtype=int)
    for i in range(len(edge_orbits)):
        edge_orbit_array[[(node_to_idx[x], node_to_idx[y]) for x, y in edge_orbits[i]]] = i
    coupler_orbits = {
        (u, v): edge_orbit_array[node_to_idx[f"Jnode_{u}_{v}"]] for u, v in bqm.quadratic
    }

    return reindex(qubit_orbits), reindex(coupler_orbits)


def get_unsigned_bqm_orbits(
    signed_qubit_orbits: dict[Hashable, int],
    signed_coupler_orbits: dict[tuple[Hashable, Hashable], int],
    bqm: dimod.BQM,
) -> tuple[dict[Hashable, int], dict[tuple[Hashable, Hashable], int]]:
    """Convert orbits for a signed BQM into orbits for the corresponding unsigned BQM.

    Assumes that orbits are given for a signed BQM, and turns them into signed
    orbits for an unsigned BQM. We also need to keep track of self-symmetric pairs
    of spins.

    Args:
        signed_qubit_orbits: Mapping from signed variable labels to orbit indices.
        signed_coupler_orbits: Mapping from signed coupler pairs to orbit indices.
        bqm: Original unsigned BQM.

    Returns: 
        A tuple ``(qubit_orbits, coupler_orbits)`` where ``qubit_orbits`` maps
        each original variable to its orbit index and ``coupler_orbits`` maps
        each coupling to its orbit index. 
    """
    # Combine coupler orbits so that O(p1p2)=O(m1m2) and O(p1m2)=O(m1p2)
    for u, v in bqm.quadratic:
        signed_coupler_orbits[(f"p{u}", f"p{v}")] = min(
            signed_coupler_orbits[(f"p{u}", f"p{v}")],
            signed_coupler_orbits[(f"m{u}", f"m{v}")],
        )
        signed_coupler_orbits[(f"m{u}", f"m{v}")] = signed_coupler_orbits[(f"p{u}", f"p{v}")]

        signed_coupler_orbits[(f"m{v}", f"p{u}")] = min(
            signed_coupler_orbits[(f"m{v}", f"p{u}")],
            signed_coupler_orbits[(f"m{u}", f"p{v}")],
        )
        signed_coupler_orbits[(f"m{u}", f"p{v}")] = signed_coupler_orbits[(f"m{v}", f"p{u}")]

    qubit_orbits = {}
    for v in bqm.linear:
        qubit_orbits[v] = signed_qubit_orbits[(f"p{v}")]

    coupler_orbits = {}
    for u, v in bqm.quadratic:
        coupler_orbits[(u, v)] = signed_coupler_orbits[(f"p{u}", f"p{v}")]

    return reindex(qubit_orbits), reindex(coupler_orbits)


def get_orbits(bqm: dimod.BQM, edge_list: list[int, int]) -> tuple[NDArray, NDArray]:
    """Provide a bqm and receive a set of usable orbits derived from the signed BQM.

    Args:
        bqm: Ising model to analyze
        edge_list
    
    Returns:
        A tuple ``(qubit_orbits_array, coupler_orbits_array)`` where
        ``qubit_orbits_array`` is a 1-D array of length ``num_spins`` mapping
        each variable index to an orbit index, and ``coupler_orbits_array`` is a
        1-D array of length ``len(edge_list)`` mappig each entry of ``edge_list``
        to an orbit index.
    """
    signed_bqm = make_signed_bqm(bqm)
    signed_qubit_orbits, signed_coupler_orbits = get_bqm_orbits(signed_bqm)
    qubit_orbits, coupler_orbits = get_unsigned_bqm_orbits(
        signed_qubit_orbits,
        signed_coupler_orbits,
        bqm,
    )

    qubit_orbits_array = np.array([qubit_orbits[q] for q in range(len(qubit_orbits))]).astype(int)
    coupler_orbit_dict = {tuple(sorted(list(key))): val for key, val in coupler_orbits.items()}
    coupler_orbits_array = np.array(
        [coupler_orbit_dict[tuple(sorted(list(c)))] for c in edge_list]
    ).astype(int)

    return qubit_orbits_array, coupler_orbits_array
