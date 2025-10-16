import dwave_networkx as dnx
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dwave.experimental.automorphism import *
G = nx.cycle_graph(8)

result = schreier_rep(G, num_samples=4)

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