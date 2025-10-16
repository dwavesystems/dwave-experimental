import dwave_networkx as dnx
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dwave.experimental.automorphism import *
G = dnx.chimera_graph(1)

result = schreier_rep(G, num_samples=4)

#print the elements of the group vector in cycle notation
print('Schreier-Sims representation (not including identity):')
for i in sorted(result.u_map.keys()):
    print(f'U_{i}:')
    for h in result.u_vector[result.u_map[i]]:
        print(array_to_cycle(h))
print('\nnumber of automorphisms: ',result.num_automorphisms)
print('vertex orbits:')

print(result.vertex_orbits)