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
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np

import dimod
from dwave.system import DWaveSampler
from minorminer.subgraph import find_subgraph

from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alpha_phi
from dwave.experimental.fast_reverse_anneal import SOLVER_FILTER


def main(
    solver: Union[None, dict, str],
    loop_length: int,
    num_steps: int,
    coupling_strength: float,
    x_target_c: float,
    x_target_c_updates: Optional[list] = None,
    x_nominal_pause_time: float,
    use_hypergradient: bool = True,
    beta_hypergradient: float = 0.4,
):
    """Refine the calibration of a ferromagnetic loop.

    See also the calibration refinement tutorial  <https://doi.org/10.3389/fcomp.2023.1238988>

    Args:
        solver: name of the solver, or dictionary of characteristics.
        num_steps: number of gradient descent steps.
        coupling_strength: coupling strength on the loop.
        x_target_c: schedule target point for reverse anneal.
        x_target_c_updates: a list of x_target_c to average over.
        use_hypergradient: use the adaptive learning rate. If False,
            a fixed geometric decay is used.
        x_nominal_pause_time: pause time at target point for reverse anneal.
        use_hypergradient: use the adaptive learning rate. If set to False,
            a fixed geometric decay is used.
        beta_hypergradient: adaptive learning rate parameter
    """

    # when available, use feature-based search to default the solver.
    qpu = DWaveSampler(solver=solver)

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
        x_nominal_pause_time=x_nominal_pause_time,
        anneal_schedule=[[0, 1], [1, 1]],
        auto_scale=False,
        initial_state={q: 1 for q in embedding.values()},
    )
    if x_target_c_updates is not None:
        sampling_params_updates = [
            {"x_target_c": x_target_c} for x_target_c in x_target_c_updates
        ]
        symmetrize_experiments = False
    else:
        sampling_params_updates = None
        symmetrize_experiments = True

    alpha = 0.1*qubit_freezeout_alpha_phi()
    if use_hypergradient:
        # A geometric decay is sufficient for a bulk low-frequency correction.
        learning_schedule = None
    else:
        learning_schedule = alpha / np.arange(1, num_steps + 1)
    
    # Find flux biases that restore average magnetization, ideally this cancels
    # the impact of low-frequency environment fluxes coupling into the qubit body
    flux_biases, fb_history, mag_history = shim_flux_biases(
        bqm=bqm,
        sampler=qpu,
        sampling_params=sampling_params,
        learning_schedule=learning_schedule,
        sampling_params_updates=sampling_params_updates,
        symmetrize_experiments=symmetrize_experiments,
        use_hypergradient=use_hypergradient,
        beta_hypergradient=beta_hypergradient,
        num_steps=num_steps,
        alpha=alpha,
    )

    mag_array = np.array(list(mag_history.values()))
    flux_array = np.array(list(fb_history.values()))
    if sampling_params_updates is None:
        batch_size = 2
    else:
        batch_size = len(sampling_params_updates)
    mag_array = np.reshape(
        mag_array, (mag_array.shape[0], mag_array.shape[1] // batch_size, batch_size)
    )

    plt.figure("all_mags")
    for k in range(mag_array.shape[0]):
        if sampling_params_updates is None:
            for experiment_sign, color in zip(range(2), "br"):
                plt.plot(
                    mag_array[k, :, experiment_sign].transpose(),
                    label=(
                        f"Initial state all {-1 + 2*experiment_sign}"
                        if k == 0
                        else None
                    ),
                    color=color,
                )
            plt.plot(
                np.mean(mag_array[k, :, :], axis=1).transpose(),
                color="black",
                label="Experiment average" if k == 0 else None,
            )
            plt.legend()
        else:
            plt.plot(
                np.mean(mag_array[k, :, :], axis=1).transpose(),
                label="Experiment average" if k == 0 else None,
            )
            plt.legend(fb_history.keys(), title="Qubit index")
    plt.xlabel("Number of gradient descent steps")
    plt.ylabel("Magnetization")
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
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--loop_length", type=int, help="Length of the loop, by default 4", default=4
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        help="Number of gradient descent steps, by default 10. A geometrically decaying learning rate is used 1/num_steps",
        default=10,
    )
    parser.add_argument(
        "--x_target_c",
        type=float,
        help="Reverse anneal point x_target_c, should be early enough for magnetization not to be polarized by the initial condition, by default 0.22",
        default=0.22,
    )
    parser.add_argument(
        "--x_target_c_average",
        type=bool,
        help="Use an average over x_target_c in 0.2 to 0.22 for a fixed initial condition",
        default=False,
    )
    parser.add_argument(
        "--x_nominal_pause_time",
        type=float,
        help="Reverse anneal dwell time.",
        default=0.0,
    )
    parser.add_argument(
        "--coupling_strength",
        type=float,
        help="Coupling strength on the ring, by default -1 (ferromagnetic)",
        default=-1,
    )
    parser.add_argument(
        "--use_hypergradient",
        type=bool,
        help="Enables hypergradient descent optimization instead of a fixed geometric decay",
        default=True,
    )
    parser.add_argument(
        "--beta_hypergradient",
        type=float,
        help="Specifies the adaptive learning rate hyperparameter beta",
        default=0.4,
    )
    args = parser.parse_args()
    if args.x_target_c_average:
        # The target_c regime is experimentally dependent, it must
        # be early enough and broad enough for the sign on the
        # initial_condition to contributed negligibly.
        x_target_c_updates = np.arange(0.2, 0.22, 0.001)
    else:
        x_target_c_updates = None

    main(
        solver=args.solver_name,
        loop_length=args.loop_length,
        num_steps=args.num_steps,
        coupling_strength=args.coupling_strength,
        x_target_c=args.x_target_c,
        x_target_c_updates=x_target_c_updates,
        x_nominal_pause_time=args.x_nominal_pause_time,
        use_hypergradient=args.use_hypergradient,
        beta_hypergradient=args.beta_hypergradient,
    )
