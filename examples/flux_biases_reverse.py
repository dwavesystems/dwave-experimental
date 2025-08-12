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
An example to demonstrate simple calibration refinement of flux_biases for
fast reverse anneal applied to a 1D coupled ring. 
"""

import argparse
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import dimod
from dwave.system import DWaveSampler
from minorminer.subgraph import find_subgraph

from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alpha_phi


def main(
    solver: Union[None, dict, str],
    loop_length: int,
    num_iters: int,
    coupling_strength: float,
    x_target_c: float,
):
    """Refine the calibration of a ferromagnetic loop.

    See also the calibration refinement tutorial  <https://doi.org/10.3389/fcomp.2023.1238988>

    Args:
        solver: name of the solver, or dictionary of characteristics.
        num_iters: number of gradient descent steps.
        coupling_strength: coupling strength on the loop.
        x_target_c: schedule target point for reverse anneal.
    """

    # when available, use feature-based search to default the solver.
    qpu = DWaveSampler(
        solver=dict(name__regex=r"Advantage2_prototype2.*|Advantage2_research1\..*")
    )

    # Embedding for a ring of length L
    edge_list = [(i, (i + 1) % loop_length) for i in range(loop_length)]  # A loop
    embedding = find_subgraph(edge_list, qpu.edgelist)
    if len(embedding) != loop_length:
        raise RuntimeError("A {loop_length} loop cannot be embedded on the solver")

    # Define a ferromagnetic Ising model over programmable qubits and couplers
    bqm = dimod.BQM.from_ising(
        h={q: 0 for q in embedding.values()},
        J={(embedding[v1], embedding[v2]): coupling_strength for v1, v2 in edge_list},
    )

    # Set up solver parameters for a fast reverse anneal experiment.
    # A regime with weak correlations allows greater efficiency, and might be
    # smoothly extraplated to more strongly correlated regimes (small target_c
    # allows for sufficiently weak dependence on the initial condition).
    # The calibration refinement is expected to be a smooth function for the
    # programmed Hamiltonian and sampling parameters.
    sampling_params = dict(
        num_reads=1024,
        reinitialize_state=True,
        x_target_c=x_target_c,
        x_nominal_pause_time=0.0,
        anneal_schedule=[[0, 1], [1, 1]],
        auto_scale=False,
        initial_state={q: 1 for q in embedding.values()},
    )

    # A geometric decay is sufficient for a bulk low-frequency correction.
    learning_schedule = qubit_freezeout_alpha_phi() / np.arange(1, num_iters + 1)

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
    for k in range(mag_array.shape[0]):
        for experiment_sign, color in zip(range(2), "br"):
            plt.plot(
                mag_array[k, :, experiment_sign].transpose(),
                label=f"Initial state all {-1 + 2*experiment_sign}" if k == 0 else None,
                color=color,
            )
        plt.plot(
            np.mean(mag_array[k, :, :], axis=1).transpose(),
            color="black",
            label="Experiment average" if k == 0 else None,
        )
    plt.xlabel("Number of gradient descent steps")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.savefig("DwaveExperimentalMagReverse.png")

    plt.figure("all_fluxes")
    plt.plot(flux_array.transpose())
    plt.xlabel("Number of gradient descent steps")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    plt.legend(fb_history.keys(), title="Qubit index")
    plt.savefig("DwaveExperimentalFluxReverse.png")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A fast reverse anneal calibration refinement example"
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver, by default an experimental system supporting fast reverse anneal",
        default=dict(name__regex=r"Advantage2_prototype2.*|Advantage2_research1\..*"),
    )
    parser.add_argument(
        "--loop_length", type=int, help="Length of the loop, by default 4", default=4
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        help="Number of gradient descent steps, by default 10. A geometrically decaying learning rate is used 1/num_steps",
        default=10,
    )
    parser.add_argument(
        "--x_target_c",
        type=float,
        help="Reverse anneal point x_target_c, should be early enough for magnetization not to be polarized by the initial condition, by default 0.25",
        default=0.25,
    )
    parser.add_argument(
        "--coupling_strength",
        type=float,
        help="Coupling strength on the ring, by default -1 (ferromagnetic)",
        default=-1,
    )
    args = parser.parse_args()

    main(
        solver=args.solver_name,
        loop_length=args.loop_length,
        num_iters=args.num_iters,
        coupling_strength=args.coupling_strength,
        x_target_c=args.x_target_c,
    )
