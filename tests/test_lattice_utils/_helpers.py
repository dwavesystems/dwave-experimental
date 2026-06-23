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

"""Shared test fixtures for the ``lattice_utils`` test package."""

from unittest import mock

import dimod
import numpy as np

from dwave.experimental.lattice_utils import lattice

__all__ = [
    "_make_triangular",
    "_make_mock_sampler",
    "_make_mock_experiment",
    "_make_embedded_chain",
]


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


def _make_mock_sampler(
    num_qubits=128,
    nodelist=None,
    solver_name="TestSolver",
    type_name="DWaveSampler",
    *,
    sync_response=False,
):
    """Build a mock sampler resembling DWaveSampler.

    Production code detects the sampler via ``type(sampler).__name__`` and reads
    ``nodelist``, ``properties["num_qubits"]``, and ``solver.name``. When
    ``sync_response`` is True, ``sampler.sample(...)`` returns a mock response
    mimicking the DWaveSampler async interface (``.done()`` -> True,
    ``.samples()`` -> all-ones ndarray) used by ``run_iteration`` tests.
    """
    sampler = mock.MagicMock(spec=dimod.Sampler)
    type(sampler).__name__ = type_name
    if nodelist is None:
        nodelist = list(range(num_qubits))
    sampler.nodelist = nodelist
    sampler.properties = {"num_qubits": num_qubits}
    sampler.solver = mock.MagicMock()
    sampler.solver.name = solver_name
    if sync_response:
        response = mock.MagicMock()
        response.done.return_value = True
        response.samples.return_value = np.ones((10, num_qubits), dtype=float)
        sampler.sample.return_value = response
    return sampler


def _make_mock_experiment(inst, signed_energy_scale=1.0, run_index=0, num_random_instances=1):
    """Return a lightweight mock Experiment with .inst and .param."""
    exp = mock.MagicMock()
    exp.inst = inst
    exp.param = {
        "signed_energy_scale": signed_energy_scale,
        "num_random_instances": num_random_instances,
    }
    exp.run_index = run_index
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
