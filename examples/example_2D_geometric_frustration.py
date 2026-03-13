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

"""Now we're doing to do a similar example but on triangular lattices, which are
embedded using two qubits per chain."""

from pathlib import Path
import os

from dwave.system import DWaveSampler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from dwave.experimental.lattice_utils import lattice, experiment, observable
from dwave.experimental.lattice_utils.utils import bootstrap, confidence_interval

# Set up a dict for collating statistics
m_dict = {}
psi_dict = {}

# Just an Advantage2 prototype.
sampler = DWaveSampler(solver="Advantage2_system3.1")

ANNEAL_TIMES = np.round(0.005 * np.logspace(0, 2, 17), 6)

errorbar_style = {"marker": '', "linestyle": '', "capsize": 2}
point_style = {"marker": 'o', "linestyle": ''}
# Create a folder to save figures in if it doesn't already exist
Path("figures").mkdir(exist_ok=True)

inst = lattice.DimerizedTriangular(
    dimensions=(9, 12),
    periodic=(True, False),
    sampler=sampler,
    orbit_type="explicit",
    halve_boundary_couplers=True,
    chain_strength=2,
)
inst.embed_lattice(
    sampler,
    max_number_of_embeddings=1,
    timeout=10000,
    remove_external_edges=True,
    remove_odd_edges=True,
    draw_reduced_graph=True,
)
# Now must make the orbits: chain and no-chain.
coupler_orbit = np.array(
    [inst.make_nominal_bqm().quadratic[edge] == -2 for edge in inst.edge_list],
    dtype=int,
)
qubit_orbit = np.ones(inst.num_spins, dtype=int)
inst.initialize_orbits(qubit_orbits=qubit_orbit, coupler_orbits=coupler_orbit)

exp = experiment.FastAnnealExperiment(
    inst=inst,
    sampler=sampler,
    num_reads=100,
    readout_thermalization=100,
    max_iterations=210,
    results_root=Path("./results"),
    automorph_embeddings=False,
    energy_scale=0.8,
    coupler_shim_step=0.1,
    flux_bias_shim_step=5e-6,
)

exp.observables_to_collect = [
    observable.QubitMagnetization(),
    observable.CouplerCorrelation(),
    observable.CouplerFrustration(),
    observable.SampleEnergy(),
    observable.TriangularOP(),
    observable.ReferenceEnergy(),
    observable.BitpackedSpins(),
]

# Make parameter list
parameter_list = [{"anneal_time": rate} for rate in ANNEAL_TIMES]

for _ in range(1000):
    done = exp.run_iteration(parameter_list)
    if done:
        break

# We will make some lists for the data we want to analyze, and for each iteration
# of the experiment we will load theresults and append the observable to the list.
frust = []  # average coupler frustration (kink density)
cshim = []  # coupler shim
fbshim = []  # flux bias shim
opmag = []
psi = []
ene = []

for param in parameter_list:
    exp.apply_param(param)
    res = exp.load_results(num_iterations=1000)
    frust.append(np.array([np.mean(i["CouplerFrustration"]) for i in res]))
    cshim.append(np.asarray([i["shimdata"]["relative_coupler_strength"].ravel() for i in res]))
    fbshim.append(np.asarray([i["shimdata"]["flux_biases"].ravel() for i in res]))
    opmag.append(np.array([np.mean(np.abs(i["TriangularOP"])) for i in res]))
    ene.append(np.array([np.mean(i["SampleEnergy"]) for i in res]))
    psi.append(np.asarray([i["TriangularOP"] for i in res]))

title=f"DimerizedTriangular, {'x'.join([str(dim) for dim in inst.dimensions])}, " \
    f"J={exp.param["energy_scale"]}, {sampler.solver.name}"
fig, axes = plt.subplots(3, 3, figsize=(16, 10))
fig.suptitle(title, fontsize=16)
rng = np.random.default_rng(0)
x = np.linspace(0, 2*np.pi, 400)
plt.tight_layout()
plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.9, left=0.07, bottom=0.07)

ax = axes[0, 0]
ax.loglog()

M = np.asarray(opmag)
bs = np.asarray([bootstrap(m, bootstrap_function=np.nanmedian, seed=None) for m in M])
ci = np.asarray([confidence_interval(i) for i in bs])

errorbar_handle = ax.errorbar(ANNEAL_TIMES, ci[:, 0], yerr=[ci[:, 1], ci[:, 2]], **errorbar_style)
ax.plot(
    ANNEAL_TIMES,
    ci[:, 0],
    color=errorbar_handle[0]._color,
    markerfacecolor=np.array(to_rgb(errorbar_handle[0]._color)) / 2 + 0.5,
    **point_style,
)

ax.set_title("<m>")
ax.set_ylabel("<m>")
ax.set_xlabel("$t_a$ (μs)")
ax.set_xlim([0.002, 9e-1])
ax.grid(which="both", alpha=0.3)

ax = axes[0, 1]
ax.loglog()
y = np.sqrt(np.asarray([np.mean(_**2) for _ in fbshim]))
ax.plot(ANNEAL_TIMES, y, marker="o", linestyle="-")
ax.set_title("RMS flux bias shim")
ax.set_xlabel("$t_a$ (μs)")
ax.set_ylabel("RMS flux bias")
ax.grid(which="both", alpha=0.3)

ax = axes[0, 2]
ax.loglog()
y = np.sqrt(np.asarray([np.mean((_ - 1) ** 2) for _ in cshim]))
ax.plot(ANNEAL_TIMES, y, marker="o", linestyle="-")
ax.set_title("RMS coupler shim")
ax.set_xlabel("$t_a$ (μs)")
ax.set_ylabel("RMS coupler shim")
ax.grid(which="both", alpha=0.3)

ax = axes[1, 0]
ax.plot(fbshim[0])
ax.set_title(f"Flux bias shim, t_a={ANNEAL_TIMES[0]:.3f}μs")
ax.set_xlabel("Iteration")
ax.grid(which="both", alpha=0.3)

ax = axes[1, 1]
ax.plot(fbshim[1])
ax.set_title(f"Flux bias shim, t_a={ANNEAL_TIMES[1]:.3f}μs")
ax.set_xlabel("Iteration")
ax.grid(which="both", alpha=0.3)

ax = axes[1, 2]
ax.plot(fbshim[6])
ax.set_title(f"Flux bias shim, t_a={ANNEAL_TIMES[-1]:.3f}μs")
ax.set_xlabel("Iteration")
ax.grid(which="both", alpha=0.3)

ax = axes[2, 0]
ax.plot(cshim[0])
ax.set_title(f"Coupler shim, t_a={ANNEAL_TIMES[0]:.3f}μs")
ax.set_xlabel("Iteration")
ax.grid(which="both", alpha=0.3)

ax = axes[2, 1]
ax.plot(cshim[1])
ax.set_title(f"Coupler shim, t_a={ANNEAL_TIMES[1]:.3f}μs")
ax.set_xlabel("Iteration")
ax.grid(which="both", alpha=0.3)

ax = axes[2, 2]
ax.plot(cshim[6])
ax.set_title(f"Coupler shim, t_a={ANNEAL_TIMES[-1]:.3f}μs")
ax.set_xlabel("Iteration")
ax.grid(which="both", alpha=0.3)

filename = title
for bad_symbol in "/: ;,":
    filename = filename.replace(bad_symbol, "_")
fig.savefig(Path(os.getcwd()) / 'figures' / f"{filename}.png")
plt.show()

# Put kink density in a dict so we can plot them all together.
m_dict[sampler.solver.name] = np.asarray(opmag)
psi_dict[sampler.solver.name] = np.asarray(psi)

# Now plot the order parameters together, for a nice comparison.
fig2, ax2 = plt.subplots(2, 1, figsize=(8, 12))
title=f'Triangular, global orbit, J={exp.param["energy_scale"]}'
fig2.suptitle(title, fontsize=16)

M = m_dict[sampler.solver.name]

bs = np.asarray([bootstrap(m, bootstrap_function=np.nanmedian, seed=None) for m in M[:, :5]])
ci = np.asarray([confidence_interval(i) for i in bs])
errorbar_handle = ax2[0].errorbar(
    ANNEAL_TIMES,
    ci[:, 0],
    yerr=[ci[:, 1], ci[:, 2]],
    **errorbar_style,
)

facecolor = np.array(to_rgb(errorbar_handle[0]._color)) / 2 + 0.5
ax2[0].plot(
    ANNEAL_TIMES,
    ci[:, 0],
    color=errorbar_handle[0]._color,
    markerfacecolor=facecolor,
    label="first 5 iterations of shim",
    **point_style,
)
bs = np.asarray([bootstrap(m, bootstrap_function=np.nanmedian, seed=None) for m in M[:, -5:]])
ci = np.asarray([confidence_interval(i) for i in bs])
errorbar_handle = ax2[0].errorbar(
    ANNEAL_TIMES,
    ci[:, 0],
    yerr=[ci[:, 1], ci[:, 2]],
    **errorbar_style,
)
facecolor = np.array(to_rgb(errorbar_handle[0]._color)) / 2 + 0.5
ax2[0].plot(
    ANNEAL_TIMES,
    ci[:, 0],
    color=errorbar_handle[0]._color,
    markerfacecolor=facecolor,
    label="last 5 iterations of shim",
    **point_style,
)

ax2[0].loglog()
ax2[0].grid(which="both", alpha=0.3)
ax2[0].set_title(f"<m>: {sampler.solver.name}")
ax2[0].set_ylabel("<m>")
ax2[0].set_xlabel("$t_a$ (μs)")
ax2[0].set_ylim([0.15, 1.2])
ax2[0].legend()

# And heatmaps of psi.
index = len(M) - 1
M = psi_dict[sampler.solver.name][index][-10:].ravel()

x = np.real(M)
y = np.imag(M)

NUM_BINS = 41
extent = (-2, 2, -1.95, 1.95)

hb = ax2[1].hexbin(x, y, gridsize=NUM_BINS, cmap="inferno", extent=extent)
ax2[1].set_title(f"ψ, t_a={ANNEAL_TIMES[index]:.3f}μs")
cb = fig2.colorbar(hb, ax=ax2[1])
cb.set_label("count")
ax2[1].plot([-1 / np.sqrt(3), 1 / np.sqrt(3)], [-1, 1], color=(0, 0, 0, 0.1), linestyle="-")
ax2[1].plot([-1 / np.sqrt(3), 1 / np.sqrt(3)], [1, -1], color=(0, 0, 0, 0.1), linestyle="-")
ax2[1].plot([-2 / np.sqrt(3), 2 / np.sqrt(3)], [0, 0], color=(0, 0, 0, 0.1), linestyle="-")
ax2[1].axis([-1.4, 1.4, -1.4, 1.4])
ax2[1].set_aspect("equal", "box")

filename = title
for bad_symbol in "/: ;,":
    filename = filename.replace(bad_symbol, "_")
fig2.savefig(Path(os.getcwd()) / 'figures' / f"{filename}.png")

plt.show()
