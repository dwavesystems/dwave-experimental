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

from dwave.experimental.lattice_utils.experiment import Experiment
from dwave.experimental.lattice_utils.observable import (
    QubitMagnetization,
    CouplerCorrelation,
    CouplerFrustration,
    SampleEnergy,
    BitpackedSpins,
    ReferenceEnergy
)

__all__ = ['FastAnnealExperiment']

class FastAnnealExperiment(Experiment):
    # Set default parameters
    default_parameters = {
        "energy_scale": 1.0,
        "automorph_embeddings": False,
        "spin_reversal_transform": False,
        "spin_reversal_transform_seed": None,
        "num_reads": 100,
        "num_random_instances": None,
        "readout_thermalization": 100,
        "fast_anneal": True,
        "anneal_time": 1.0,
        "flux_bias_shim_step": 0.0,
        "coupler_shim_step": 0.0,
        "coupler_damp": 0.0,
        "anneal_offset_shim_step": 0.0,
        "anneal_offset_damp": 0.0,
        "individual_qubit_anneal_offsets": None,
        "target_magnetization": 0.0,
        "logical_software": False,
    }
    observables_to_collect = {
        QubitMagnetization(),
        CouplerCorrelation(),
        CouplerFrustration(),
        SampleEnergy(),
        BitpackedSpins(),
        ReferenceEnergy(),
    }
