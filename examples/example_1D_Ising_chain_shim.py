# Copyright 2026 D-Wave
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

"""Shimming example for 1D Ising chain."""

from pathlib import Path
import os

from dwave.system import DWaveSampler
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from dwave.experimental.lattice_utils import lattice, experiment, observable
from dwave.experimental.lattice_utils.utils import bootstrap, confidence_interval

# Set up the parameters

# Get a QPU sampler with Zephyr topology.
sampler = DWaveSampler(solver=dict(topology__type="zephyr"))

NUM_SPINS = 256
SIGNED_ENERGY_SCALE = -1.6

# We will simulate four orders of magnitude in anneal time.
# File format rounds to the nearest picosecond, so we will do so explicitly here.
ANNEAL_TIMES = np.round(np.geomspace(0.01, 100, 3), 6)

errorbar_style = {"marker": "", "linestyle": "", "capsize": 2}
point_style = {"linestyle": "", "markersize": 5}
cm = plt.get_cmap("tab10")

# Create a folder to save figures in if it doesn't already exist.
Path("figures").mkdir(exist_ok=True)

data_root = Path(__file__).resolve().parents[1]

# Make a lattice instance for a periodic 256-spin chain, so we can embed it.
inst = lattice.Chain(
    dimensions=(NUM_SPINS,),
    data_root=data_root,
    periodic=(True,),
    orbit_type="standard",
)

# Find parallel embeddings of the lattice heuristically. The embed_lattice
# function is heuristic and is run here with a default timeout (10s) and no
# tuning of any parameters. Larger and more complex lattices can take longer
# to embed.
inst.embed_lattice(sampler)

# Time to make an experiment.  The aim in this example is to demonstrate a coupler
# and flux-bias shim on a chain, at fixed energy scale and varying anneal time.
# Since the susceptibility to changes in the shim changes as a function of anneal
# time, we will run each anneal time using separate shim steps.
# We will also set the orbit_type to 'standard', allowing the use of graph
# automorphisms to determine symmetries in the system that can be exploited by
# shimming. In this case, all couplers are equivalent (they go in the same orbit)
# so the coupler shim will compel them all to have the same spin-spin correlation
# for a given parameterization.
flux_bias_shim_step = {
    0.01: 5e-6,
    1.0: 2e-6,
    100: 0.5e-6,
}
coupler_shim_step = {
    0.01: 0.05,
    1.0: 0.2,
    100: 1.0,
}
for anneal_time in ANNEAL_TIMES:

    config = experiment.FastAnnealExperimentConfig(
        signed_energy_scale=SIGNED_ENERGY_SCALE,
        coupler_shim_step=coupler_shim_step[anneal_time],
        flux_bias_shim_step=flux_bias_shim_step[anneal_time],
    )
    exp = experiment.Experiment(inst=inst, sampler=sampler, max_iterations=100, config=config)

    # Make parameter list. We will only vary anneal time.
    for _ in range(120):
        done = exp.run_iteration([{"anneal_time": anneal_time}], progress=True)
        if done:
            break


# We will make a dict for the data we want to analyze.  For each anneal time
# we will load all iterations into a corresponding dict entry.
# Disjoint embeddings are given along a separate axis in the results, except
# flux biases and anneal offsets, which are given as a single
# array since they are indexed by physical qubit.

mag = {}  # average qubit magnetization
frust = {}  # average coupler frustration (kink density)
cshim = {}  # coupler shim
fbshim = {}  # flux bias shim
for anneal_time in ANNEAL_TIMES:
    exp.apply_param({"anneal_time": anneal_time})
    res = exp.load_results()
    mag[anneal_time] = np.array([it["QubitMagnetization"] for it in res])
    frust[anneal_time] = np.array([it["CouplerFrustration"] for it in res])
    cshim[anneal_time] = np.array(
        [it["shimdata"]["relative_coupler_strength"].ravel() for it in res]
    )
    fbshim[anneal_time] = np.array([it["shimdata"]["flux_biases"] for it in res])

title = (
    f"1D chain shim, "
    f"{'x'.join([str(dim) for dim in inst.dimensions])}, "
    f"J={exp.param['signed_energy_scale']}, "
    f"{sampler.solver.name}"
)
fig, axes = plt.subplots(3, 6, figsize=(16, 8), sharex="col", sharey="col")
fig.suptitle(title, fontsize=16)

for iat, anneal_time in enumerate(ANNEAL_TIMES):

    # Plot std of qubit magnetization
    ax = axes[iat, 0]
    ax.plot(mag[anneal_time].std(axis=(1, 2)), label=r"Mag std")
    ax.set_ylabel("Magnetization std")

    # Plot histograms of first and last iterations
    ax = axes[iat, 1]
    ax.hist(mag[anneal_time][:5].ravel(), label="First 5 iterations", alpha=0.5)
    ax.hist(mag[anneal_time][-5:].ravel(), label="Last 5 iterations", alpha=0.5)
    ax.set_yticks([])
    ax.set_ylabel("Frequency")

    # Plot std of coupler frustration
    ax = axes[iat, 2]
    ax.plot(frust[anneal_time].std(axis=(1, 2)), label=r"Frust std")
    ax.set_ylabel("Frustration std")

    # Plot histograms of first and last iterations
    ax = axes[iat, 3]
    ax.hist(frust[anneal_time][:5].ravel(), label="First iteration", alpha=0.5)
    ax.hist(frust[anneal_time][-5:].ravel(), label="Last iteration", alpha=0.5)
    ax.set_yticks([])
    ax.set_ylabel("Frequency")

    # Plot flux biases
    ax = axes[iat, 4]
    ax.plot(fbshim[anneal_time], alpha=0.2)
    ax.set_ylabel("Flux bias")

    # Plot coupler shim
    ax = axes[iat, 5]
    ax.plot(cshim[anneal_time], alpha=0.2)
    ax.set_ylabel("Rel. cplr. strength")


axes[iat, 0].set_xlabel("Iteration")
axes[iat, 1].set_xlabel("Magnetization")
axes[iat, 1].legend()
axes[iat, 2].set_xlabel("Iteration")
axes[iat, 3].set_xlabel("Frustration")
axes[iat, 3].legend()
axes[iat, 4].set_xlabel("Iteration")
axes[iat, 5].set_xlabel("Iteration")

axes[0, 0].set_title("Magnetization std")
axes[0, 1].set_title("Magnetization")
axes[0, 2].set_title("Frustration std")
axes[0, 3].set_title("Frustration")
axes[0, 4].set_title("Flux offset shim")
axes[0, 5].set_title("Coupler shim")

fig.tight_layout()

filename = title
for bad_symbol in "/: ;,":
    filename = filename.replace(bad_symbol, "_")
fig.savefig(Path(os.getcwd()) / "figures" / f"{filename}.png")
plt.show()
