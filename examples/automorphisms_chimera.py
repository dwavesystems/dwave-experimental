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
An example to demonstrate calculating the Schreier-Sims representation of a 
chimera graph with one unit cell, as well as the number of automorphisms and
vertex orbits.
"""

from typing import Optional
import argparse

import dwave_networkx as dnx

from dwave.experimental.automorphism import schreier_rep, array_to_cycle


def main(
    chimera_unit_cells: int,
    num_samples: Optional[int],
    verbose: bool,
):

    graph = dnx.chimera_graph(chimera_unit_cells)
    result = schreier_rep(graph, num_samples=num_samples)

    #print the elements of the group vector in cycle notation
    if verbose == True:
        print('Schreier-Sims representation (not including identity):')
        for i in sorted(result.u_map.keys()):
            print(f'U_{i}:')
            for h in result.u_vector[result.u_map[i]]:
                print(array_to_cycle(h))
        print('\nnumber of automorphisms: ', result.num_automorphisms)
        print('vertex orbits:')

        print(result.vertex_orbits)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="An automorphism calculation of the chimera graph example"
    )
    parser.add_argument(
        "--num_unit_cells",
        type=int,
        help="Number of unit cells of the chimera graph, by default 1",
        default=1,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of coset representatives to sample for automorphism composition using the random-Schreier method. Alternatively, the default of None corresponds to using all coset representives",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        help="Prints the left transversals of the Schreier-Sims representation of a graph, the number of automorphisms, and the vertex orbits",
        default=True,
    )
    args = parser.parse_args()

    main(
        chimera_unit_cells=args.num_unit_cells,
        num_samples=args.num_samples,
        verbose=args.verbose
    )
