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
cycle graph with 8 vertices, as well as the number of automorphisms,
vertex orbits, and a random uniform sampling of 3 automorphisms.
"""

import networkx as nx

from dwave.experimental.automorphism import array_to_cycle, sample_automorphisms, schreier_rep


G = nx.cycle_graph(8)

result = schreier_rep(G)

#print the elements of the group vector in cycle notation
print('Schreier-Sims representation (not including identity):')
for i in sorted(result.u_map.keys()):
    print(f'U_{i}:')
    for h in result.u_vector[result.u_map[i]]:
        print(array_to_cycle(h))
print('\nnumber of automorphisms: ', result.num_automorphisms)
print('vertex orbits:')

print(result.vertex_orbits)

print('Randomly sampling 3 automorphisms: ')
num_samples = 3
for sampled_automorphism in sample_automorphisms(result.u_vector, num_samples):
    print(array_to_cycle(sampled_automorphism))
