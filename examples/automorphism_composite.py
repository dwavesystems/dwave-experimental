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
An example to show noise mitigation by use of automorphic averaging
"""

import os
import argparse

import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler  # For code testing
from dwave.preprocessing import SpinReversalTransformComposite
from minorminer.subgraph import find_subgraph

from dwave.experimental.automorphism.automorphism_composite import AutomorphismComposite


def main(
    solver=None,
    use_cache: bool = True,
    L: int = 4,
    subgraph_timeout: int = 60,
    srts: bool = True,
):
    """An example to demonstrate noise mitigation by averaging on automorphic embeddings

    Examine forward-anneal statistics for a fully-frustrated square-lattice Ising model on a Torus.
    The expected energy per edge <J_ij s_i s_j> is identical up to symmetry breaking
    effects (noise) resulting from calibration and bath non-idealities. Averaging automorphisms
    enhances the signal to noise ratio, resulting in improved statistics.

    Args:

    """
    qpu = DWaveSampler(solver=solver, profile='defaults')

    J = {
        ((x, y), ((x + uv[0]) % L, (y + uv[1]) % L)): -1 + 2 * (x % 2 and uv[0] == 0)
        for x in range(L)
        for y in range(L)
        for uv in [(0, 1), (1, 0)]
    }
    emb = find_subgraph(
        J.keys(), qpu.edgelist, as_embedding=True, timeout=subgraph_timeout
    )
    if not emb:
        raise ValueError(
            "For the selected solver, an embedding was not found in 60 seconds. Modifying the solver_name."
            "size of lattice L or timeout in subgraph search should resolve this issue"
        )

    sampler0 = FixedEmbeddingComposite(qpu, emb)
    samplerA = AutomorphismComposite(
        FixedEmbeddingComposite(qpu, emb), G=nx.from_edgelist(J.keys())
    )

    samplers = {"Fixed Embedding": sampler0, "Automorphism average": samplerA}
    if srts:
        samplerSRT = FixedEmbeddingComposite(SpinReversalTransformComposite(qpu), emb)
        samplerASRT = AutomorphismComposite(samplerSRT, G=nx.from_edgelist(J.keys()))
        samplers.update(
            {"SRT average": samplerSRT, "SRT and Auto. average": samplerASRT}
        )
    num_progs = 10
    fn_cache = f"cache/{solver}_automorphism_composite.pkl"
    if use_cache and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            signed_correlations = pickle.load(f)
    else:
        signed_correlations = {sn: [] for sn in samplers.keys()}

    print('Employing different averaging techniques, we can decrease the '
          'observed variance in statistics, which are expected to be equal '
          'in the absence of symmetry breaking effects in environmental noise '
          'device control and calibration.')

    
    print('Identical signed edge expectations are anticipated based on the; '
          'programmed model, a fully frustrated square lattice on a torus. '
          'Specifically <sign(Jij) s_i s_j> = -0.5 is anticipated if samples '
          'are restricted to the ground state space.')
    for sn, sampler in samplers.items():
        for num_prog in range(num_progs):
            if len(signed_correlations[sn]) < num_prog:
                ss = sampler.sample_ising(
                    {}, J, num_reads=1000
                )  # Could be asyncrhonized
                samps = ss.record.sample
                no = ss.record.num_occurrences
                var_to_idx = {v: idx for idx, v in enumerate(ss.variables)}
                signed_correlations[sn].append(
                    [
                        Jij
                        * np.sum(no * samps[:, var_to_idx[i]] * samps[:, var_to_idx[j]])
                        / np.sum(no)
                        for (i, j), Jij in J.items()
                    ]
                )

                if use_cache:
                    os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
                    with open(fn_cache, "wb") as f:
                        pickle.dump(signed_correlations, f)
        plt.plot(
            np.sort(np.mean(signed_correlations[sn][:num_prog], axis=0)),
            np.arange(len(J)) / len(J),
            label=sn,
        )
        plt.xlabel(r"Edge-wise expected energy, J_{ij} <s_i s_j>")
        plt.ylabel("Cumulative Distribution Function")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    print(
        "Averaging over Automorphisms, and/or Spin Reversal Transforms,"
        " can mitigate for symmetry breaking noise in statistics."
    )

    parser = argparse.ArgumentParser(description="An AutomorphismComposte example")
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Add this flag to save experimental data, and reload when available at matched parameters",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver, the default DWaveSampler when not specified",
        default=None,
    )
    parser.add_argument(
        "--srts",
        action="store_true",
        help="Add this flag to invoke additional noise mitigation with the SpinReversalTransformComposite",
    )

    args = parser.parse_args()

    main(
        use_cache=args.use_cache,
        solver=args.solver_name,
        srts=args.srts,
    )
