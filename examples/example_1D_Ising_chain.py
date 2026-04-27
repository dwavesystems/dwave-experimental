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

"""Example for 1D Ising chain."""

from pathlib import Path
import os

from dwave.system import DWaveSampler
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from dwave.experimental.lattice_utils import lattice, experiment, observable
from dwave.experimental.lattice_utils.utils import bootstrap, confidence_interval

# Set up a dict for collating kink densities
kd_dict = {}
kkc_dict = {}

# Set up the parameters

# Two samplers: an Advantage2 prototype and an Advantage system.
samplers = [
    DWaveSampler(solver="Advantage2_system3.1"),
    DWaveSampler(solver="Advantage_system4.1"),
]

NUM_SPINS = 256

# Two energy scales: one strong coupling and one weak coupling.
ENERGY_SCALES = (-1.8, 0.1)

# Minimum anneal time is 5ns. We will simulate four orders of magnitude in anneal time.
# File format rounds to the nearest picosecond, so we will do so explicitly here.
ANNEAL_TIMES = np.round(np.geomspace(0.005, 50, 21), 6)

errorbar_style = {"marker": '', "linestyle": '', "capsize": 2}
point_style = {"marker": 'o', "linestyle": ''}
# Create a folder to save figures in if it doesn't already exist
Path("figures").mkdir(exist_ok=True)

for sampler in samplers:

    # Make a lattice instance for a periodic 256-spin chain, so we can embed it.
    inst = lattice.Chain(
        dimensions=(NUM_SPINS,),
        periodic=(True,),
        sampler=sampler,
        orbit_type="standard",
    )

    # Find parallel embeddings of the lattice heuristically. The embed_lattice
    # function is heuristic and is run here with a default timeout (10s) and no
    # tuning of any parameters. Larger and more complex lattices can take longer
    # to embed.
    inst.embed_lattice(sampler)

    # Time to make an experiment. We will also set the orbit_type to 'standard',
    # which will allow the use of graph automorphisms to determine symmetries in
    # the system that can be exploited by shimming. In this case, all couplers
    # are equivalent (they go in the same orbit) so the coupler shim will compel
    # them all to have the same spin-spin correlation for a given parameterization.

    # Here we will do some shimming: flux bias shim and coupler shim.  We will
    # run two energy scales: a very strong one (negative, ferromagnetic) and a
    # very weak one (positive, antiferromagnetic).  Positive and negative energy
    # scales are equivalent by gauge transformation, but we run the strong coupling
    # on the FM side because the maximum FM magnitude (-2) is larger than the
    # maximum AFM magnitude (+1).
    for energy_scale in ENERGY_SCALES:
        exp = experiment.FastAnnealExperiment(
            inst=inst,
            sampler=sampler,
            loop_data_files=30,
            max_iterations=5,
            energy_scale=energy_scale,
            coupler_shim_step=0.05,
            flux_bias_shim_step=1e-6,
        )
        # Every experiment has an attribute (a set) of observables to compute and
        # save while the experiment runs.Here we can add non-default observables.
        # In this case we will add the kink-kink correlator (CITE).  The observable
        # object is designed to provide a standard interface for adding whatever
        # experiment-specific observables you might require.
        exp.observables_to_collect.add(observable.KinkKinkCorrelator())

        # Make parameter list. We will only vary anneal time.
        parameter_list = [{"anneal_time": time} for time in ANNEAL_TIMES]

        for _ in range(20):
            done = exp.run_iteration(parameter_list)
            if done:
                break

        # Now we will run some analysis.  Let's first just plot kink density as
        # a function of annealing time. Kink density is the same as the average
        # "FrustrationProbability" observable for a given coupler, which is already
        # gathered by default since it is required for the coupler shim.

        # We will make some lists for the data we want to analyze, and for each
        # iteration of the experiment we will load the results and append the
        # observable to the list.

        frust = []  # average coupler frustration (kink density)
        cshim = []  # coupler shim
        fbshim = []  # flux bias shim
        kkc = []  # kink-kink correlator
        for param in parameter_list:
            exp.apply_param(param)
            res = exp.load_results(num_iterations=1000)

            frust.append(np.array([np.mean(it["CouplerFrustration"]) for it in res]))
            cshim.append(
                np.asarray([it["shimdata"]["relative_coupler_strength"].ravel() for it in res])
            )
            fbshim.append(np.asarray([it["shimdata"]["flux_biases"] for it in res]))
            kkc.append(
                np.reshape(np.asarray([it["KinkKinkCorrelator"] for it in res]), (-1, NUM_SPINS))
            )

        title = f"1D chain, {'x'.join([str(dim) for dim in inst.dimensions])}, J={exp.param["energy_scale"]}, {sampler.solver.name}"
        fig, axes = plt.subplots(3, 3, figsize=(16, 10))
        fig.suptitle(title, fontsize=16)
        rng = np.random.default_rng(0)
        x = np.linspace(0, 2*np.pi, 400)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.9)

        ax = axes[0, 0]
        ax.loglog()
        x_theory = ANNEAL_TIMES[0]
        y_theory = np.mean(frust[0])
        theoryfit = np.polyfit(
            np.log([x_theory, x_theory * 2]), np.log([y_theory, y_theory * (2**-0.5)]), 1
        )
        ax.plot(
            ANNEAL_TIMES,
            np.exp(np.polyval(theoryfit, np.log(ANNEAL_TIMES))),
            linestyle="-",
            color=[0.8, 0.8, 0.8],
            label="theory",
        )

        #x = ANNEAL_TIMES
        M = np.asarray(frust)
        bs = np.asarray([bootstrap(_, bootstrap_function=np.nanmedian, seed=None) for _ in M])
        ci = np.asarray([confidence_interval(_) for _ in bs])

        errorbar_handle = ax.errorbar(
            ANNEAL_TIMES,
            ci[:, 0],
            yerr=[ci[:, 1], ci[:, 2]],
            **errorbar_style,
        )
        ax.plot(
            ANNEAL_TIMES,
            ci[:, 0],
            color=errorbar_handle[0]._color,
            markerfacecolor=np.array(to_rgb(errorbar_handle[0]._color)) / 2 + 0.5,
            **point_style,
        )

        ax.set_title("Kink density (with ~t_a^{-1/2} guideline)")
        ax.set_ylabel("kink density")
        ax.set_xlabel("$t_a$ (μs)")
        ax.set_ylim([5e-4, 5e-1])
        ax.set_xlim([0.002, 9e1])
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
        kd_dict[sampler.solver.name, energy_scale] = np.asarray(frust)
        kkc_dict[sampler.solver.name, energy_scale] = np.asarray(kkc)

# Now plot the kink densities together, for a nice comparison.
fig2, ax2 = plt.subplots(1, 2, figsize=(10, 8))
title = f"1D chain kink density, {'x'.join([str(dim) for dim in inst.dimensions])}"
fig2.suptitle(title, fontsize=16)

for isampler, sampler in enumerate(samplers):

    for energy_scale in [-1.8, 0.1]:
        M = kd_dict[sampler.solver.name, energy_scale]

        # Kink density plot
        theoryx = ANNEAL_TIMES[0]
        theoryy = np.mean(M[0])
        theoryfit = np.polyfit(
            np.log([theoryx, theoryx * 2]), np.log([theoryy, theoryy * (2**-0.5)]), 1
        )
        ax2[isampler].plot(
            ANNEAL_TIMES,
            np.exp(np.polyval(theoryfit, np.log(ANNEAL_TIMES))),
            linestyle="-",
            color=[0.8, 0.8, 0.8],
            label="theory",
        )
        ax2[isampler].set_title(f"Kink density: {sampler.solver.name}")
        ax2[isampler].loglog()
        ax2[isampler].grid(which="both", alpha=0.3)
        ax2[isampler].set_ylabel("Kink density")
        ax2[isampler].set_xlabel("$t_a$ (μs)")
        ax2[isampler].set_ylim([5e-4, 5e-1])
        ax2[isampler].set_xlim([0.002, 9e1])

        x = ANNEAL_TIMES
        M = kd_dict[sampler.solver.name, energy_scale]
        bs = np.asarray([bootstrap(m, bootstrap_function=np.nanmedian, seed=None) for m in M])
        ci = np.asarray([confidence_interval(b) for b in bs])

        errorbar_handle = ax2[isampler].errorbar(
            ANNEAL_TIMES,
            ci[:, 0],
            yerr=[ci[:, 1], ci[:, 2]],
            **errorbar_style
        )
        ax2[isampler].plot(
            x,
            ci[:, 0],
            color=errorbar_handle[0]._color,
            markerfacecolor=np.array(to_rgb(errorbar_handle[0]._color)) / 2 + 0.5,
            **point_style,
        )

filename = title
for bad_symbol in "/: ;,":
    filename = filename.replace(bad_symbol, "_")
fig2.savefig(Path(os.getcwd()) / 'figures' / f"{filename}.png")
plt.show()

# Now we will analyze the kink-kink correlator for the fastest anneals (5ns)
fig3, ax3 = plt.subplots(1, 2, figsize=(10, 8))
dims = 'x'.join(map(str, inst.dimensions))
time_ns = ANNEAL_TIMES[0] * 1000
title=f"1D chain kink-kink correlator, {dims}, {time_ns:.1f} ns"
fig3.suptitle(title, fontsize=16)

for isampler, sampler in enumerate(samplers):

    for energy_scale in [-1.8, 0.1]:
        magnetization = kkc_dict[sampler.solver.name, energy_scale][0].T
        kd = np.mean(kd_dict[sampler.solver.name, energy_scale][0])

        # Kink density plot
        ax3[isampler].grid(which="both", alpha=0.3)
        ax3[isampler].set_title(f"Kink-kink correlator: {sampler.solver.name}")
        ax3[isampler].set_ylabel("Kink-kink correlator")
        ax3[isampler].set_xlabel("Normalized distance")
        ax3[isampler].set_ylim([-0.15, 0.15])
        ax3[isampler].set_xlim([0.01, 1.5])

        x = np.arange(NUM_SPINS) * kd
        M = magnetization

        bs = np.asarray([bootstrap(m, bootstrap_function=np.nanmedian, seed=None) for m in M])
        ci = np.asarray([confidence_interval(i) for i in bs])

        errorbar_handle = ax3[isampler].errorbar(
            x,
            ci[:, 0],
            yerr=[ci[:, 1], ci[:, 2]],
            **errorbar_style
        )
        ax3[isampler].plot(
            x,
            ci[:, 0],
            marker='o',
            linestyle='',
            color=errorbar_handle[0]._color,
            markerfacecolor=np.array(to_rgb(errorbar_handle[0]._color)) / 2 + 0.5
        )

filename = title
for bad_symbol in "/: ;,":
    filename = filename.replace(bad_symbol, "_")
fig3.savefig(Path(os.getcwd()) / 'figures' / f"{filename}.png")
plt.show()
