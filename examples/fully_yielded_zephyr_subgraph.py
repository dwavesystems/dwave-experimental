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

import dwave_networkx as dnx
import networkx as nx
import numpy as np
from minorminer import find_embedding
from minorminer.utils.parallel_embeddings import find_sublattice_embeddings

from dwave.experimental.embedding_methods import zephyr_quotient_search

seed = 12345
rng = np.random.default_rng(seed)

print(
    "This example demonstrates how to use zephyr_quotient_search to find a full-yield embedding of "
    "a smaller Zephyr graph into a larger, defective Zephyr graph. Since zephyr_quotient_search "
    " finds embeddings for source and target graphs with the same number of rows, this example "
    "shows how to use find_sublattice_embeddings to first identify a complete sublattice in the "
    "defective target that matches the smaller source graph's parameters, and then run "
    "zephyr_quotient_search on that sublattice. "
)

tile = dnx.zephyr_graph(6, 4, coordinates=True)
target = dnx.zephyr_graph(12, 4, coordinates=True)
print(
    "Step 1: Build two Zephyr graphs.\nThe smaller graph is the m=6, t=4 tile we want to recover "
    f"({tile.number_of_nodes()} nodes, {tile.number_of_edges()} edges), and the larger graph is the"
    " m=12, t=4 target that will later be damaged "
    f"({target.number_of_nodes()} nodes, {target.number_of_edges()} edges)."
)

#  first, identify one complete m=6, t=4 sublattice in the pristine target.
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


# now, remove 10% random nodes from outside the sublattice that was found before
protected_nodes = set(reference_embeddings[0].values())
num_remove = int(0.1 * target.number_of_nodes())
removable_nodes = [n for n in target.nodes() if n not in protected_nodes]
removed_idx = rng.choice(len(removable_nodes), size=num_remove, replace=False)
removed_nodes = [removable_nodes[i] for i in removed_idx]
target.remove_nodes_from(removed_nodes)

print(
    "Step 3: Created a defective target by randomly removing qubits outside the protected "
    f"sublattice. We keep {len(protected_nodes)} nodes untouched, remove {len(removed_nodes)} "
    f"nodes, and end up with a damaged target containing {target.number_of_nodes()} nodes and "
    f"{target.number_of_edges()} edges."
)

# this finishes up creating our "defective" target graph, which, by construction, still contains at
# least one complete m=6, t=4 sublattice, but is now missing 10% of the nodes outside that
# sublattice.

# our example actually starts here. we start from this defective target graph, so we need to
# discover a complete m=6, t=4 sublattice in the defective target.
tile_embeddings = find_sublattice_embeddings(
    S=tile,
    T=target,
    max_num_emb=1,
    one_to_iterable=False,
    seed=seed,
)
tile_embedding = tile_embeddings[0]  # pick the first embedding.

print(
    "Step 4: Starting only from the defective target, search again for a complete m=6, t=4 "
    f"sublattice. The algorithm found {len(tile_embeddings)} valid sublattice(s); this example "
    "continues with the first one."
)

# Relabel to canonical m=6 coordinates before zephyr_quotient_search.
sublattice_nodes = set(tile_embedding.values())
target_sub = target.subgraph(sublattice_nodes).copy()
inv_map = {target_node: tile_node for tile_node, target_node in tile_embedding.items()}
target_sub = nx.relabel_nodes(target_sub, inv_map, copy=True)
target_sub.graph.update(family="zephyr", rows=6, tile=4, labels="coordinate")

print(
    "Step 5: Relabel the recovered sublattice into canonical m=6 coordinates so quotient search can"
    f" work on a standard Zephyr graph. The relabeled subgraph has {target_sub.number_of_nodes()} "
    f"nodes and {target_sub.number_of_edges()} edges."
)

# embed source zephyr(mp=6, tp=2) into the found complete m=6, t=4 sublattice.
source = dnx.zephyr_graph(6, 2, coordinates=True)
print(
    "Step 6: Build the source graph we actually want to place into that recovered sublattice. "
    f"Here the source is a Zephyr m=6, t=2 graph with {source.number_of_nodes()} nodes and "
    f"{source.number_of_edges()} edges."
)

emb, metadata = zephyr_quotient_search(source, target_sub, yield_type="edge")

print(
    "Step 7: Run zephyr_quotient_search on the canonical sublattice. It successfully placed "
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
else:
    print(
        "Step 8: The quotient search already achieved full yield, so no refinement step is needed."
    )

# map back to original target labels, which can be used as the effective embedding for the source
# into the original target.
embedding_in_original_target = {
    s: tuple(tile_embedding[v] for v in chain)
    for s, chain in best_embedding.items()
}
