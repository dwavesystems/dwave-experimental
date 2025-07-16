import matplotlib.pyplot as plt
import numpy as np

import dimod
from dwave.experimental.shimming import shim_flux_biases
from dwave.system.testing import MockDWaveSampler
from dwave.system import DWaveSampler
from minorminer.subgraph import find_subgraph
from dwave.experimental.shimming import shim_flux_biases

print(
    "Creates a small ferromagnetic problem and shims such that the expected magnetization is 0 averaging over all up and all down initial conditions"
)

# Change to 'Advantage2_research1.1' upon release, when available use
# feature-based search to default the solver.
qpu = DWaveSampler(solver="Advantage2_prototype2_x_internal")

# Embedding for a ring of length L
loop_length = 4
edge_list = [(i, (i + 1) % loop_length) for i in range(loop_length)]  # A loop
embedding = find_subgraph(edge_list, qpu.edgelist)
if len(embedding) != loop_length:
    raise RunTimeError("A {loop_length} loop cannot be embedded on the solver")

# Define a ferromagnetic Ising model over programmable qubits and couplers
bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
    h={q: 0 for q in embedding.values()},
    J={(embedding[v1], embedding[v2]): -1 for v1, v2 in edge_list},
)

# Set up solver parameters for a fast reverse anneal experiment, in a regime
# with weak correlations (small target_c sufficient).
learning_schedule = 1e-5 * 3 / np.arange(3, 13)
initial_state = 3 * np.ones(qpu.properties["num_qubits"])  # 3 denotes inactive
for q in embedding.values():
    initial_state[q] = 1
sampling_params = dict(
    num_reads=1024,
    reinitialize_state=True,
    x_target_c=0.25,
    x_nominal_pause_time=0.0,
    anneal_schedule=[[0, 1], [1, 1]],
    auto_scale=False,
    initial_state=initial_state,
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

mag_array = np.reshape(mag_array, (mag_array.shape[0], mag_array.shape[1] // 2, 2))

plt.figure("all_mags")
for experiment_sign in range(2):
    plt.plot(
        mag_array[:, :, experiment_sign].transpose(),
        label=f"Initial state all {-1 + 2*experiment_sign}",
    )
plt.plot(
    np.mean(mag_array, axis=2).transpose(), color="black", label="Experiment average"
)
plt.xlabel("Shim iteration")
plt.ylabel("Magnetization")
plt.legend()
plt.savefig("DwaveExperimentalMag.png")

plt.figure("all_fluxes")
plt.plot(flux_array.transpose())
plt.xlabel("Shim iteration")
plt.ylabel("Flux bias ($\\Phi_0$)")
plt.legend(fb_history.keys(), title="Qubit index")
plt.savefig("DwaveExperimentalFlux.png")
plt.show()
