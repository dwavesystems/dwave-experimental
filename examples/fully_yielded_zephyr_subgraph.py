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

import argparse

import matplotlib.pyplot as plt

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from minorminer import find_embedding
from minorminer.utils.parallel_embeddings import find_sublattice_embeddings

from dwave.experimental.embedding_methods import quotient_search


def main(family="zephyr", seed=None, m_t=12, m_s=5, t=4, t_s=2, node_yield=0.97):
    rng = np.random.default_rng(seed)

    print(
        f"This example demonstrates how to use quotient_search to find a full-yield embedding of "
        f"a smaller {family} graph into a larger, defective {family} graph. Since quotient_search "
        " finds embeddings for source and target graphs with the same number of rows, this example "
        "shows how to use find_sublattice_embeddings to search by horizontal "
        "or vertical displacement of a smaller graph, and how to use"
        "how to use quotient_search for a subgraph by exploting rail"
        "automorphisms. "
        "Note that sublattice search and quotient search are heuristic"
        "approaches for which failure is not a guarantee of non-existence,"
        " except in special circumstances."
    )
    if family == "zephyr":
        graph_generator = dnx.zephyr_graph
        shape_s = (m_s, t)
        shape_t = (m_t, t)
        shape_step6 = (m_s, t_s)
    elif family == "chimera":
        graph_generator = dnx.chimera_graph
        shape_s = (m_s, m_s, t)
        shape_t = (m_t, m_t, t)
        shape_step6 = (m_s, m_s, t_s)
    elif family == "pegasus":
        graph_generator = dnx.pegasus_graph
        shape_s = (m_s,)
        shape_t = (m_t,)
        shape_step6 = (m_s,)
    else:
        raise ValueError("Unknown family")
    tile = graph_generator(*shape_s, coordinates=True)
    target = graph_generator(*shape_t, coordinates=True)

    print(
        f"Step 1: Build two Zephyr graphs.\nThe smaller graph is the m={m_s}, t={t} tile we want to recover "
        f"({tile.number_of_nodes()} nodes, {tile.number_of_edges()} edges), and the larger graph is the"
        f" m={m_t}, t={t} target that will later be damaged "
        f"({target.number_of_nodes()} nodes, {target.number_of_edges()} edges)."
    )

    #  first, identify one complete m=m_s, t=4 sublattice in the pristine target.
    reference_embeddings = find_sublattice_embeddings(
        S=tile,
        T=target,
        max_num_emb=1,
        one_to_iterable=False,
        seed=seed,
    )
    print(
        "Step 2: In the defect-free target, search for one complete copy of the smaller graph. "
        f"The search found {len(reference_embeddings)} candidate sublattices. We will protect the "
        "first one so we know the damaged target still contains a valid solution."
    )

    # now, remove a fraction of random nodes from outside the sublattice that was found before
    protected_nodes = set(reference_embeddings[0].values())
    num_remove = int(
        (1 - node_yield) * (target.number_of_nodes() - len(protected_nodes))
    )
    removable_nodes = [n for n in target.nodes() if n not in protected_nodes]
    removed_idx = rng.choice(len(removable_nodes), size=num_remove, replace=False)
    removed_nodes = [removable_nodes[i] for i in removed_idx]
    target.remove_nodes_from(removed_nodes)
    # target.add_nodes_from(removed_nodes) # For sake of plotting we add back singleton nodes (no edges)
    print(
        "Step 3: Created a defective target by randomly removing qubits outside the protected "
        f"sublattice. We keep {len(protected_nodes)} nodes untouched, remove {len(removed_nodes)} "
        f"nodes, and end up with a damaged target containing {target.number_of_nodes()} nodes and "
        f"{target.number_of_edges()} edges."
    )

    # this finishes up creating our "defective" target graph, which, by construction, still contains at
    # least one complete m=m_s, t=4 sublattice, but is now missing a fraction of the nodes outside that
    # sublattice.

    # our example actually starts here. we start from this defective target graph, so we need to
    # discover a complete m=m_s, t=4 sublattice in the defective target.
    tile_embeddings = find_sublattice_embeddings(
        S=tile,
        T=target,
        max_num_emb=1,
        one_to_iterable=False,
        seed=seed,
    )
    tile_embedding = tile_embeddings[0]  # pick the first embedding.

    G = target.copy()
    G.add_nodes_from(
        removed_nodes
    )  # For sake of plotting we add back singleton nodes (no edges)
    node_color = ["r" if G.degree(n) == 0 else "lightgray" for n in G.nodes()]
    plt.figure("Successful search by displacement")
    dnx.draw_parallel_embeddings(
        G=G,
        embeddings=tile_embeddings,
        S=tile,
        one_to_iterable=False,
        shuffle_colormap=False,
        node_color=node_color,
    )

    print(
        f"Step 4: Starting only from the defective target, search again for a complete m={m_s}, t={t} "
        f"sublattice. The algorithm found {len(tile_embeddings)} valid sublattice(s); this example "
        "continues with the first one."
    )

    # Remove nodes also from the protected sublattice at
    # the same rate as elsewhere in the processor:
    num_remove = int((1 - node_yield) * len(protected_nodes))
    removed_nodes2 = [
        tuple(n)
        for n in rng.choice(list(protected_nodes), size=num_remove, replace=False)
    ]
    target.remove_nodes_from(removed_nodes2)

    # Relabel to canonical m_s coordinates before quotient_search.
    sublattice_nodes = set(tile_embedding.values())
    target_sub = target.subgraph(sublattice_nodes).copy()
    inv_map = {
        target_node: tile_node for tile_node, target_node in tile_embedding.items()
    }
    target_sub = nx.relabel_nodes(target_sub, inv_map, copy=True)
    target_sub.graph.update(family=family, rows=m_s, tile=t, labels="coordinate")

    print(
        f"Step 5: Now apply the node_yield parameter to remove a fraction of nodes also on the "
        f"sublattice, all sublattices are statistically similar, but "
        f"the example continues with the same sublattice as before."
        f"Relabel the recovered sublattice using shape {shape_s} coordinates. "
        f"The relabeled subgraph has {target_sub.number_of_nodes()} "
        f"nodes and {target_sub.number_of_edges()} edges."
    )

    # embed source zephyr(mp=m_s, tp=t_s) into the found complete m=m_s, t=4 sublattice.
    source = graph_generator(*shape_step6, coordinates=True)
    if source.graph["family"] == "pegasus":
        # Graphs with reduced rails are not created by the same
        # generators.
        source.remove_nodes_from(
            [n for n in source.nodes() if n[2] % 2 != 0]
        )  # remove odd k nodes to get single-rail pegasus graph

    print(
        "Step 6: With any defects present on a sublattice, the graph of shape "
        f"{shape_s} will not embed. "
        f"However a graph of shape {shape_step6} compatible with the quotient graph, "
        f"of {source.number_of_nodes()} nodes and {source.number_of_edges()} edges may be feasible."
    )

    emb, metadata = quotient_search(source, target_sub, yield_type="edge")

    print(
        "Step 7: Run quotient_search on the canonical sublattice. It successfully placed "
        f"{metadata.final_num_yielded} of {metadata.max_num_yielded} source edges."
    )

    # If not full-yield, refine with minorminer.find_embedding.
    best_embedding = emb
    if metadata.final_num_yielded < metadata.max_num_yielded:
        print(
            "Step 8: The quotient search did not reach full yield, so we pass its chains to minorminer "
            "as an initial guess and try to refine the embedding."
        )
        refined = find_embedding(
            S=source,
            T=target_sub,
            initial_chains=emb,
            timeout=50,
        )
        if refined:
            best_embedding = refined
            print(
                "The refinement returned an embedding, improving the result to "
                f"{len(best_embedding)} mapped source nodes."
            )
        else:
            print(
                "The refinement step did not return a better embedding, "
                "so the script keeps the quotient-search result."
            )
        fig_name = (
            "Successful graph-minor search by quotient search + minorminer refinement"
        )
    else:
        print(
            "Step 8: The quotient search already achieved full yield, so no refinement step is needed."
        )
        node_color = [
            "r" if target_sub.degree(n) == 0 else "lightgray"
            for n in target_sub.nodes()
        ]
        fig_name = "Successful subgraph automorphism search by quotient search"
    plt.figure(fig_name)
    plt.title(fig_name)
    G = target_sub.copy()
    G.add_nodes_from(
        inv_map[n] for n in removed_nodes2
    )  # For sake of plotting we add back singleton nodes (no edges)
    node_color = ["r" if G.degree(n) == 0 else "lightgray" for n in G.nodes()]
    dnx.draw_parallel_embeddings(
        G=G,
        embeddings=[emb],
        S=source,
        one_to_iterable=True,
        shuffle_colormap=False,
        node_color=node_color,
    )

    # map back to original target labels, which can be used as the effective embedding for the source
    # into the original target.
    embedding_in_original_target = {
        s: tuple(tile_embedding[v] for v in chain)
        for s, chain in best_embedding.items()
    }
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find a full-yield subgraph embedding on a defective target graph."
    )
    parser.add_argument(
        "--family",
        choices=("zephyr", "chimera", "pegasus"),
        default="zephyr",
        help="Graph family to run the example with.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible runs.",
    )
    args = parser.parse_args()

    main(family=args.family, seed=args.seed)
