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

"""Example for running an ensemble of 3D spin glasses, embedded using two qubits
per cubic site.

This example uses several more advanced features.  First, we use the SpinGlassMixin
class to apply a random distribution of coupling values to each edge of the cubic
lattice. Second, we use a different kind of experiment:
SpinGlassExperiment, which, if multiple embeddings of the lattice are
available, embeds a different spin glass realization (random seed) in each
embedding. Third, it uses the EmbeddedLattice class, which allows a structured
embedding with an explicit set of couplings present between any two
chains (and within any given chain).
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dwave.system import DWaveSampler
from matplotlib.colors import to_rgb
from scipy.stats import bootstrap

from dwave.experimental.lattice_utils import experiment, lattice, observable

# Get a QPU sampler with Zephyr topology.
sampler = DWaveSampler(solver={"topology__type": 'zephyr'})

# We will simulate four orders of magnitude in anneal time.
# File format rounds to the nearest picosecond, so we will do so explicitly here.
# We add a long anneal time to help find ground-state energies.
ANNEAL_TIMES = np.round(np.geomspace(0.005, 0.03, 9), 6)
ANNEAL_TIMES = np.append(ANNEAL_TIMES, 100)

errorbar_style = {"marker": "", "linestyle": "", "capsize": 2}
point_style = {"linestyle": "", "markersize": 5, "marker": "o"}
cm = plt.get_cmap("tab10")

# Create a folder to save figures in if it doesn't already exist.
Path("figures").mkdir(exist_ok=True)
data_root = Path(__file__).resolve().parents[1]

# Construct 3D dimer lattices of sizes 3x3x3 and 4x4x4 and find embeddings. These
# lattices have twice as many physical spins as logical sites, with periodic
# couplings in the z dimension.

exps = {}
for L in [3, 4]:
    for precision in [1, 256]:
        # Find at least two parallel embeddings, allowing a 100-second search timeout.
        # SpinGlassExperiment assigns a different disorder seed to each embedding.

        # We explore bimodal (precision=1) and uniformly distributed discrete
        # couplings (precision=256). Coupling precision only changes the weights,
        # not the geometry, so embeddings found for one precision can be reused
        # for the other.

        cubic_spin_glass = lattice.DimerizedCubicSpinGlass(
            dimensions=(L, L, L),
            periodic=(False, False, True),
            precision=precision,
            data_root=data_root,
        )
        cubic_spin_glass.embed_lattice(
            sampler,
            timeout=100,
            min_number_of_embeddings=2,
            max_number_of_embeddings=40,
        )

        # Apply flux-bias shimming to the dimerized cubic spin glasses at a fixed
        # energy scale. Each anneal time has a separate shim history because the
        # appropriate flux-bias correction may depend on anneal time.
        config = experiment.FastAnnealExperimentConfig(
            num_random_instances=40,
            flux_bias_shim_step=1e-6,
        )
        exps[L, precision] = experiment.SpinGlassExperiment(
            lattice=cubic_spin_glass,
            sampler=sampler,
            max_iterations=40,
            config=config,
        )
        exps[L, precision].observables_to_collect.append(observable.EdwardsAndersonOverlap())

for exp in exps.values():
    print(
        f"Running experiment with L={exp.lattice.dimensions[0]:d}, "
        f"precision={exp.lattice.logical_lattice.precision:d}."
    )
    # Run parameter list. We will only vary anneal time.
    for _ in range(100):
        done = exp.run_iteration([{"anneal_time": anneal_time} for anneal_time in ANNEAL_TIMES])
        if done:
            break


# We will make a dict for the data we want to analyze.  For each anneal time
# we will load all iterations into a corresponding dict entry.
# Disjoint embeddings are given along a separate axis in the results, except
# flux biases and anneal offsets, which are given as a single
# array since they are indexed by physical qubit.

resene = {}  # residual energy
ea = {}

for precision in [1, 256]:
    for L in [3, 4]:
        exp = exps[L, precision]
        for anneal_time in ANNEAL_TIMES:
            exp.apply_param({"anneal_time": anneal_time})
            res = exp.load_results(num_iterations=40)

            # To compute the residual energy, compute the sample energy and
            # subtract the reference energy.
            # For this example we simply assume that the reference energy is the
            # ground-state energy.
            resene[precision, L, anneal_time] = np.array([it["SampleEnergy"] for it in res])
            resene[precision, L, anneal_time] -= np.tile(
                np.atleast_3d(np.array([it["ReferenceEnergy"] for it in res])),
                (1, 1, resene[precision, L, anneal_time].shape[2]),
            )

            ea[precision, L, anneal_time] = np.array([it["EdwardsAndersonOverlap"] for it in res])

title = f"3D dimer spin glass, J={exp.param['signed_energy_scale']}, {sampler.solver.name}"
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex="col", sharey="col")
fig.suptitle(title, fontsize=16)

for iprec, precision in enumerate([1, 256]):
    # Plot <q^2> with bootstrap confidence intervals

    for iL, L in enumerate([3, 4]):
        ax = axes[iprec, 0]
        ax.loglog()
        # Compute mean and bootstrap CI for each anneal time
        cis = [
            bootstrap(
                (ea[precision, L, anneal_time][:, :, 1].flatten(),), np.mean, confidence_level=0.95
            ).confidence_interval
            for anneal_time in ANNEAL_TIMES[:-1]
        ]
        y = np.array(
            [ea[precision, L, anneal_time][:, :, 1].mean() for anneal_time in ANNEAL_TIMES[:-1]]
        )
        y_lower = y - np.array([ci.low for ci in cis])
        y_upper = np.array([ci.high for ci in cis]) - y
        yerr = np.array([y_lower, y_upper])
        color = cm(iL)

        ax.errorbar(ANNEAL_TIMES[:-1], y, yerr=yerr, **errorbar_style, color=color)
        ax.plot(
            ANNEAL_TIMES[:-1],
            y,
            color=color,
            markerfacecolor=np.array(to_rgb(color)) / 2 + 0.5,
            label=f"L={L:d}, prec={precision:d}",
            **point_style,
        )
        ax.set_xlabel("Anneal Time (µs)")
        ax.set_ylabel(r"$\langle q^2 \rangle$")
        ax.set_title(r"Edwards-Anderson Overlap $\langle q^2 \rangle$")
        ax.legend()

        # Plot residual energy in the second column in a similar manner.
        ax = axes[iprec, 1]
        ax.loglog()

        residual_energy = [
            resene[precision, L, anneal_time].mean(axis=2) / (L * L * L)
            for anneal_time in ANNEAL_TIMES[:-1]
        ]
        residual_energy_cis = [
            bootstrap((res_en.flatten(),), np.mean, confidence_level=0.95).confidence_interval
            for res_en in residual_energy
        ]
        y = np.array([re.mean() for re in residual_energy])
        y_lower = y - np.array([ci.low for ci in residual_energy_cis])
        y_upper = np.array([ci.high for ci in residual_energy_cis]) - y
        yerr = np.array([y_lower, y_upper])

        ax.errorbar(ANNEAL_TIMES[:-1], y, yerr=yerr, **errorbar_style, color=color)
        ax.plot(
            ANNEAL_TIMES[:-1],
            y,
            color=color,
            markerfacecolor=np.array(to_rgb(color)) / 2 + 0.5,
            label=f"L={L:d}, prec={precision:d}",
            **point_style,
        )
        ax.set_xlabel("Anneal Time (µs)")
        ax.set_ylabel("Residual Energy Per Logical Spin")
        ax.set_title("Residual Energy Per Logical Spin")
        ax.legend()

        # Now we are ready to do the Binder cumulant.
        ax = axes[iprec, 2]
        ax.loglog()

        def binder_cumulant(_q2, _q4, axis=-1):
            _meanq2 = np.array(_q2).mean(axis=axis)
            _meanq4 = np.array(_q4).mean(axis=axis)
            return 1.5 - 0.5 * _meanq4 / (_meanq2**2)

        cis = {}
        U_mean = {}
        for anneal_time in ANNEAL_TIMES[:-1]:
            q_squared = ea[precision, L, anneal_time][:, :, 1].ravel()
            q_fourth = ea[precision, L, anneal_time][:, :, 2].ravel()
            U_mean[anneal_time] = binder_cumulant(q_squared, q_fourth)

            cis[anneal_time] = bootstrap(
                (q_squared, q_fourth), binder_cumulant, confidence_level=0.95, paired=True
            ).confidence_interval

        y = np.array([U_mean[anneal_time] for anneal_time in ANNEAL_TIMES[:-1]])
        y_lower = y - np.array([ci.low for ci in cis.values()])
        y_upper = np.array([ci.high for ci in cis.values()]) - y
        yerr = np.array([y_lower, y_upper])

        ax.errorbar(ANNEAL_TIMES[:-1], y, yerr=yerr, **errorbar_style, color=color)
        ax.plot(
            ANNEAL_TIMES[:-1],
            y,
            color=color,
            markerfacecolor=np.array(to_rgb(color)) / 2 + 0.5,
            label=f"L={L:d}, prec={precision:d}",
            **point_style,
        )
        ax.set_xlabel("Anneal Time (µs)")
        ax.set_ylabel("$U$")
        ax.set_title("Binder Cumulant $U$")
        ax.legend()

        # Plot Binder cumulant against rescaled anneal time t * L^-2.85, approximating a collapse.
        ax = axes[iprec, 3]
        ax.loglog()

        ax.errorbar(
            np.array(ANNEAL_TIMES[:-1]) * L**-2.85,
            y,
            yerr=yerr,
            **errorbar_style,
            color=color,
        )
        ax.plot(
            np.array(ANNEAL_TIMES[:-1]) * L**-2.85,
            y,
            color=color,
            markerfacecolor=np.array(to_rgb(color)) / 2 + 0.5,
            label=f"L={L:d}, prec={precision:d}",
            **point_style,
        )
        ax.set_xlabel("Rescaled Anneal Time $t_aL^{-2.85}$")
        ax.set_ylabel("$U$")
        ax.set_title("Binder Cumulant Collapse")
        ax.legend()

fig.tight_layout()

filename = re.sub(r"[^A-Za-z0-9._-]", "_", title)
fig.savefig(Path(os.getcwd()) / "figures" / f"{filename}.png")
plt.show()
