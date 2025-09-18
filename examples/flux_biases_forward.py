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
fast (forward) anneal applied to a full scale simple-cubic spin-glass problem
with 2-qubit embedding at maximum (extended range) chain strength. 
"""


import argparse
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

import dimod
from hybrid.decomposers import make_origin_embeddings
from dwave.embedding import embed_bqm
from dwave.system import DWaveSampler

from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alpha_phi


def main(
    solver: Union[None, dict, str],
    num_iters: int,
    coupling_strength: float = -1,
    annealing_time: Union[None, float] = None,
    use_hypergradient: bool = False,
    beta_hypergradient: float = 0.4,
):
    """Refine the calibration of a large spin glass.

    See also the calibration refinement tutorial  <https://doi.org/10.3389/fcomp.2023.1238988>

    Args:
        solver: name of the solver, or dictionary of characteristics.
        loop_length: length of the loop.
        num_iters: number of gradient descent steps.
        coupling_strength: coupling strength on the cubic lattice.
        annealing_time: annealing_time in microseconds
        use_hypergradient: use the adaptive learning rate
        beta_hypergradient: adaptive learning rate parameter
    """
    qpu = DWaveSampler(solver=solver)
    if annealing_time is None:
        annealing_time = qpu.properties["fast_anneal_time_range"][0]
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
    source_bqm = dimod.BQM.from_ising(
        h={},
        J={e: coupling_strength * (2 * np.random.random() - 1) for e in edge_list},
    )
    bqm = embed_bqm(source_bqm, embedding, qpu.adjacency, chain_strength=2)

    # Set up solver parameters for a forward anneal experiment.
    # A regime with weak correlations allows greater efficiency, and might be
    # smoothly extraplated to more strongly correlated regimes.
    # The calibration refinement is expected to be a smooth function for the
    # programmed Hamiltonian and sampling parameters.
    sampling_params = dict(
        num_reads=1024,
        fast_anneal=True,
        auto_scale=False,
        annealing_time=annealing_time,
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
        use_hypergradient=use_hypergradient,
        beta_hypergradient=beta_hypergradient,
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
    plt.savefig("DwaveExperimentalMagForward.png")

    plt.figure("all_fluxes")
    plt.plot(flux_array.transpose())
    plt.xlabel("Shim iteration")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    plt.savefig("DwaveExperimentalFluxForward.png")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A fast anneal calibration refinement example"
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver, by default=None",
        default=None,
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        help="Number of gradient descent steps, by default 10. A geometrically decaying learning rate is used 1/num_steps.",
        default=10,
    )
    parser.add_argument(
        "--coupling_strength",
        type=float,
        help="Coupling strength on the cubic lattice, couplers are random +/-coupling_strength, but default the maximum programmable scale (1).",
        default=-1,
    )
    parser.add_argument(
        "--annealing_time",
        type=float,
        help="annealing_time in microseconds. Smallest valuable supported by default.",
        default=None,
    )
    parser.add_argument(
        "--use_hypergradient",
        type=bool,
        help="Enables hypergradient descent optimization instead of the default learning schedule.",
        default=True,
    )
    parser.add_argument(
        "--beta_hypergradient",
        type=float,
        help="Specifies a custom multiplicative hyperparameter beta",
        default=0.4,
    )
    args = parser.parse_args()

    main(
        solver=args.solver_name,
        num_iters=args.num_iters,
        coupling_strength=args.coupling_strength,
        annealing_time=args.annealing_time,
        use_hypergradient=args.use_hypergradient,
        beta_hypergradient=args.beta_hypergradient,      
    )
