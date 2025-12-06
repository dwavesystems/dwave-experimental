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
"""
An example to show embedding for multicolor annealing.
"""

import argparse
from typing import Union, Optional

import matplotlib.pyplot as plt
import networkx as nx

from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from minorminer.subgraph import find_subgraph
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    get_solver_name,
    SOLVER_FILTER,
    qubit_to_Advantage2_annealing_line,
    make_tds_graph,
)
from dwave_networkx import (
    zephyr_graph,
    draw_zephyr,
    draw_parallel_embeddings,
    chimera_graph,
    zephyr_coordinates,
)
from hybrid.decomposers import _chimeralike_to_zephyr


def main(
    use_client: bool = False,
    solver: Union[None, dict, str] = None,
    detector_line: int = 0,
    source_line: int = 3,
):
    """Demonstrate 3-examples of multicolor annealing.

    In the first example we demonstrate embedding of a single
    qubit problem in many places on the processor, such that
    every qubit is connected to a source and detector qubit on the
    specified lines.

    Args:
        use_client: Whether or not to use a specific solver. If
            True, the solver field is used to establish a client
            connection, and embedding proceeds with respect to the
            associated solver graph. If False the code runs locally
            using a defect-free Advantage2 processor graph at a
            Zephyr[6] scale, using the standard 6-anneal line scheme.
        solver: Name of the solver, or dictionary of characteristics.
        detector_line: The integer index of the detector line.
        source_line: The integer index of the source line.

    Raises:
        ValueError: If the number of lines is less than 3, or
        ValueError: If the detector_line or source_line is not in
            the range [0, num_lines)
    """

    # when available, use feature-based search to default the solver.
    if use_client:
        qpu = DWaveSampler(solver=solver)
        annealing_lines = get_properties(qpu)
        line_assignments = {
            n: al_idx for al_idx, al in enumerate(annealing_lines) for n in al["qubits"]
        }
        num_lines = len(annealing_lines)
    else:
        shape = [6, 4]
        qpu = MockDWaveSampler(topology_type="zephyr", topology_shape=[6, 4])
        line_assignments = {
            n: qubit_to_Advantage2_annealing_line(n, shape) for n in qpu.nodelist
        }
        num_lines = 6
    # Plot the colored graph:
    T = qpu.to_networkx_graph()

    cmap = plt.colormaps.get_cmap("plasma")
    colors = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    def target_assignments(n: int):
        line = line_assignments[n]
        if line == detector_line:
            return "detector"
        elif line == source_line:
            return "source"
        else:
            return "target"

    Tnode_to_tds = {n: target_assignments(n) for n in qpu.nodelist}
    node_color = [colors[line_assignments[n]] for n in T.nodes()]
    plt.figure("annealing lines")
    draw_zephyr(T, node_color=node_color, node_size=5, edge_color="grey")

    print(
        "Embed many single-target variable problems, with a source and detector"
        "associated to every qubit."
    )
    target_graph = nx.Graph()
    target_graph.add_node(0)
    S, Snode_to_tds = make_tds_graph(target_graph)
    subgraph_kwargs = dict(node_labels=(Snode_to_tds, Tnode_to_tds), as_embedding=True)

    embs = find_multiple_embeddings(
        S, T, max_num_emb=None, embedder_kwargs=subgraph_kwargs, one_to_iterable=True
    )
    used_nodes = {v[0] for emb in embs for v in emb.values()}
    node_color = [
        colors[line_assignments[n]] if n in used_nodes else "grey" for n in T.nodes()
    ]
    plt.figure("Parallel Embedding")
    draw_parallel_embeddings(T, embeddings=embs, S=S, node_color=node_color)

    print(
        "Embed a loop of length 64, with a source and detector"
        "associated to every qubit Embedding for a ring of length L."
    )
    L = 64
    target_graph = nx.from_edgelist((i, (i + 1) % L) for i in range(L))
    S, Snode_to_tds = make_tds_graph(target_graph)
    subgraph_kwargs = dict(node_labels=(Snode_to_tds, Tnode_to_tds), as_embedding=True)
    emb = find_subgraph(S, T, **subgraph_kwargs)
    used_nodes = {v[0] for v in emb.values()}
    node_color = [
        colors[line_assignments[n]] if n in used_nodes else "grey" for n in T.nodes()
    ]
    plt.figure("Loop embedding")
    draw_parallel_embeddings(T, embeddings=[emb], S=S, node_color=node_color)

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A target-detector-source embedding example"
    )
    parser.add_argument(
        "--use_client", action="store_true", help="Add this flag to use the client"
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver when use_client is True, by default an experimental system supporting fast reverse anneal",
        default=SOLVER_FILTER,
    )
    args = parser.parse_args()

    main(
        use_client=args.use_client,
        solver=args.solver_name,
    )
