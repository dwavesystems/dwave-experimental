import matplotlib.pyplot as plt
import numpy as np

import dimod
from dwave.experimental.shimming import shim_flux_biases
from dwave.system import DWaveSampler
from minorminer.subgraph import find_subgraph
from dwave.experimental.shimming import shim_flux_biases
from hybrid.decomposers import make_origin_embeddings
from dwave.embedding import embed_bqm

print("Creates a large cubic spin glass, shim to restore symmetry in fast annealing.")

qpu = DWaveSampler()

# Find a set of chains sufficient to embed a cubic lattice at full yield,
# adapt (by vacancies) to tolerate missing qubits in the target QPU.
embedding = make_origin_embeddings(qpu_sampler=qpu, lattice_type="cubic")[0]


def ordered_neighbors(v1, v2):
    "Return true if variables are ordered, and geometric distance 1"
    return sum(a != b for a, b in zip(v1, v2)) == 1 and sum(v2) - sum(v1) == 1


edge_list = [
    (v1, v2) for v1 in embedding for v2 in embedding if ordered_neighbors(v1, v2)
]

# Define a spin glass with random couplings, using extended J-range (-2) chains
source_bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
    h={},
    J={e: -(2 * np.random.random() - 1) for e in edge_list},
)

bqm = embed_bqm(source_bqm, embedding, qpu.adjacency, chain_strength=2)

# Set up solver parameters for a fast reverse anneal experiment
learning_schedule = (
    4e-5 * 3 / np.arange(3, 22)
)  # Not optimized, but likely sufficient for a signal in Advantage/Advantage2 generation processors.
sampling_params = dict(
    num_reads=1024,
    fast_anneal=True,
    auto_scale=False,
    annealing_time=qpu.properties["fast_anneal_time_range"][0],
)

# Find flux biases that restore average magnetization, ideally this cancels
# the impact of low-frequency environment fluxes coupling into the qubit body
flux_biases, fb_history, mag_history = shim_flux_biases(
    bqm=bqm,
    sampler=qpu,
    sampling_params=sampling_params,
    learning_schedule=learning_schedule,
)

mag_array = np.array(list(mag_history.values()))
flux_array = np.array(list(fb_history.values()))

# Note that sampling error may dominate a signal with num_reads=1024.
plt.figure("all_mags")
plt.plot(
    np.sort(mag_array[:, 0]),
    np.arange(mag_array.shape[0]) / mag_array.shape[0],
    label="Before",
)
plt.plot(
    np.sort(mag_array[:, -1]),
    np.arange(mag_array.shape[0]) / mag_array.shape[0],
    label="After",
)
plt.legend()
plt.xlabel("Magnetization")
plt.ylabel("Cumulative Distribution Functino")
plt.savefig("DwaveExperimentalMag.png")

plt.figure("all_fluxes")
plt.plot(flux_array.transpose())
plt.xlabel("Shim iteration")
plt.ylabel("Flux bias ($\\Phi_0$)")
# plt.legend(fb_history.keys(), title="Qubit index")
plt.savefig("DwaveExperimentalFlux.png")
plt.show()
