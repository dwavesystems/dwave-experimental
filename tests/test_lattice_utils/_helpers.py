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

"""Shared test fixtures for the ``lattice_utils`` test package."""

import dimod

from dwave.experimental.lattice_utils import experiment, lattice

__all__ = ["_make_triangular", "_make_experiment", "_make_embedded_chain"]


def _make_triangular(
    data_root,
    ly=3,
    lx=3,
    periodic=(True, False),
    orbit_type="singleton",
    halve_boundary_couplers=False,
):
    return lattice.Triangular(
        dimensions=(ly, lx),
        periodic=periodic,
        data_root=data_root,
        orbit_type=orbit_type,
        halve_boundary_couplers=halve_boundary_couplers,
    )


def _make_experiment(lattice, signed_energy_scale=1.0, num_random_instances=1, sampler=None):
    config = experiment.ExperimentConfig(
        signed_energy_scale=signed_energy_scale,
        num_random_instances=num_random_instances,
    )
    if sampler is None:
        sampler = dimod.ExactSolver()
    exp = experiment.Experiment(lattice=lattice, sampler=sampler, config=config)
    return exp


def _make_embedded_chain(chain_nodes, data_root):
    return lattice.EmbeddedLattice(
        logical_lattice=lattice.Chain(
            dimensions=(len(chain_nodes),),
            periodic=(False,),
            data_root=data_root,
        ),
        chain_nodes=chain_nodes,
        dimensions=(sum(len(chain) for chain in chain_nodes.values()),),
        periodic=(False,),
    )
