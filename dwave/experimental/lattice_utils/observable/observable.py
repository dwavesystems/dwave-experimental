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

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import dimod

from dwave.experimental.lattice_utils.lattice import Lattice

__all__ = [
    'Observable',
    'QubitMagnetization',
    'CouplerCorrelation',
    'CouplerFrustration',
    'SampleEnergy',
    'BitpackedSpins',
    'ReferenceEnergy',
]

class Observable(ABC):
    """The observable class does not take any parameters.  Its primary
    functionality is through the required 'evaluate' method, which requires
    parameters 'experiment' and 'bqm' defining the context, and 'sample_set'
    which provides the samples on which we compute the observable.  Output is a
    numpy array of arbitrary type (usually float).
    """
    def __init__(self):
        self.name: str = type(self).__name__

    @abstractmethod
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        raise NotImplementedError


class QubitMagnetization(Observable):
    """Compute the mean magnetization of each qubit."""
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        sample_array = dimod.as_samples(sample_set)[0].astype(float)
        return np.mean(sample_array, axis=0)


class CouplerCorrelation(Observable):
    """Compute pairwise spin correlations for each coupler."""
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        sample_array = dimod.as_samples(sample_set)[0].astype(float)
        if len(experiment.inst.edge_list) == 0:
            return []
        row, col = np.asarray(experiment.inst.edge_list).T

        # Surprisingly, it's faster to multiply the whole matrix.
        spin_product = np.matmul(sample_array.T, sample_array)[row, col] / len(sample_array)
        return spin_product


class CouplerFrustration(Observable):
    """Compute the mean coupler frustration for each edge."""
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        sample_array = dimod.as_samples(sample_set)[0].astype(float)
        if len(experiment.inst.edge_list) == 0:
            return []
        row, col = np.asarray(experiment.inst.edge_list).T

        # Surprisingly, it's faster to multiply the whole matrix.
        spin_product = np.matmul(sample_array.T, sample_array)[row, col] / len(sample_array)
        coupler_signs = (
            np.sign([bqm.quadratic[edge] for edge in experiment.inst.edge_list])
            * np.sign(experiment.param["energy_scale"])
        )

        return spin_product * coupler_signs / 2 + 1 / 2


class SampleEnergy(Observable):
    """Compute sample energies with respect to the nominal BQM.

    Energies exclude the magnitude of ``energy_scale`` but include its sign.
    """
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        return sample_set.data_vectors["energy"] * np.sign(experiment.param["energy_scale"])


class BitpackedSpins(Observable):
    """Return bitpacked spins and a tuple of the array size."""
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> tuple[NDArray, tuple[int, int]]:
        sample_array = dimod.as_samples(sample_set)[0]

        # Bitpack solutions
        results_bool = np.equal(sample_array, 1)
        results_bitpacked = np.packbits(results_bool)
        results_shape = sample_array.shape

        return results_bitpacked, results_shape


class ReferenceEnergy(Observable):
    """Return a cached reference energy, computing it and saving it if needed."""
    def evaluate(self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
        path: str | Path | None = None,
        inst: Lattice | None = None,
    ) -> float:

        if path is None:
            path = get_reference_energy_path(experiment, bqm=bqm)

        if path.exists():
            energy, sample, method_string = self.load(experiment, bqm, path)
            return energy

        # And if we can't load, we generate a reference sample.
        if experiment is not None:
            energy, sample, method_string = experiment.inst._optimize(bqm)
        else:
            energy, sample, method_string = inst._optimize(bqm)

        self.save(path, energy, sample, method_string)

        return energy

    def load(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        path: str | Path | None = None,
    ) -> tuple[float, NDArray, str]:
        """Load and get the full data tuple, not just the energy."""
        if path is None:
            path = get_reference_energy_path(experiment, bqm=bqm)
        with open(path, "r") as f:
            method_string = f.readline().strip()
            energy = float(f.readline().strip())

        sample = np.loadtxt(path, skiprows=2)

        return energy, sample, method_string

    def save(self, path: str | Path, energy: float, sample: NDArray, method_string: str) -> None:
        """Save the reference energy to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, sample, fmt="%d", header=f"{method_string}\n{energy}", comments="")

    def update(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample,
        path: str | Path | None = None,
    ) -> None:
        """Update the cached reference energy if the provided sample improves it.

        Use this when you get an energy that is lower than the reference energy.
        We want to keep the old method string unless it is specified.
        """
        reference_energy, _, reference_method_string = self.load(experiment, bqm, path)

        new_energy = bqm.energy(sample)

        if new_energy < reference_energy:
            if path is None:
                path = get_reference_energy_path(experiment, bqm=bqm)
            self.save(path, new_energy, sample, reference_method_string)
            print(f"Updated energy from {reference_energy} to {new_energy}.")
        else:
            raise ValueError


def get_reference_energy_path(
    experiment: Experiment | None = None,
    root: str | Path | None = None,
    bqm: dimod.BQM | None = None,
    dummy_experiment_data_dict: dict[str, Any] | None = None,
) -> Path:
    """Return the path to the reference energy file for the given experiment and BQM.
    
    This needs to be fixed if you have something not in the instance
    pathstring that needs to be taken into account, for example if the ground-state
    energies depend on the chip.
    """
    if bqm is None:
        raise NotImplementedError  # defunct.

    # Allow for generation of dummy experiment data without all the overhead,
    # for running without an actual experiment.
    if experiment is None:
        experiment_data_dict = dummy_experiment_data_dict
    else:
        experiment_data_dict = {
            "run_index": experiment.run_index,
            "num_random_instances": experiment.param["num_random_instances"],
            "inst": experiment.inst,
        }

    if root is None:
        root = experiment_data_dict["inst"].lattice_data_root
    else:
        root = Path(root)

    path = root / "reference_energies" / experiment_data_dict["inst"]._get_instance_pathstring()

    # Use hash.  BQM is not hashable so use the experiment.inst data to generate a tuple.
    bqm_as_tuple = tuple(bqm.linear[v] for v in sorted(bqm.variables)) + tuple(
        bqm.quadratic[e] for e in experiment_data_dict["inst"].edge_list
    )
    bqm_hash = hash(bqm_as_tuple)
    path = path / str(bqm_hash)

    return path.with_suffix('.txt')
