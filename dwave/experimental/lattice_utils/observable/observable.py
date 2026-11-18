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

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import dimod
import numpy as np
from numpy.typing import NDArray

from dwave.experimental.lattice_utils.lattice.lattice import Lattice

if TYPE_CHECKING:
    from dwave.experimental.lattice_utils.experiment.experiment import Experiment

__all__ = [
    'Observable',
    'QubitMagnetization',
    'CouplerCorrelation',
    'CouplerFrustration',
    'SampleEnergy',
    'BitpackedSpins',
    'ReferenceEnergy',
]

ObservableResult: TypeAlias = NDArray | float | int | tuple[NDArray, tuple[int, int]]


class Observable(ABC):
    """Abstract base class for observables in lattice experiments.

    Each observable should inherit from this class and implement the 'evaluate'
    method, which computes the observable from a given sample set.
    """

    def __init__(self):
        self.name: str = type(self).__name__

    @abstractmethod
    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> ObservableResult:
        """Compute the observable from the provided sample set.

        Args:
            experiment: Experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: The samples used to compute the observable.
        """


class QubitMagnetization(Observable):
    """Compute the mean magnetization of each qubit."""

    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        """Return per-qubit mean spin values over the provided samples.

        Args:
            experiment: Experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: Samples used to compute the magnetization.

        Returns:
            A numpy array containing the mean magnetization for each qubit.
        """
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
        """Return per-coupler pairwise spin correlations over the provided samples.

        Args:
            experiment: Experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: Samples used to compute the coupler correlations.

        Returns:
            A numpy array containing the pairwise spin correlations for each coupler.
        """
        sample_array = dimod.as_samples(sample_set)[0].astype(float)
        if not experiment.lattice.edge_list:
            return np.empty(0, dtype=float)

        row, col = np.asarray(experiment.lattice.edge_list).T
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
        """Return the mean coupler frustration over the provided samples.

        Args:
            experiment: Experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: Samples used to compute the mean coupler frustration.

        Returns:
            A numpy array containing the mean coupler frustration for each edge.
        """
        sample_array = dimod.as_samples(sample_set)[0].astype(float)
        if not experiment.lattice.edge_list:
            return np.empty(0, dtype=float)

        row, col = np.asarray(experiment.lattice.edge_list).T
        spin_product = np.matmul(sample_array.T, sample_array)[row, col] / len(sample_array)
        coupler_signs = np.sign(
            [bqm.quadratic[edge] for edge in experiment.lattice.edge_list]
        ) * np.sign(experiment.param["signed_energy_scale"])

        return spin_product * coupler_signs / 2 + 1 / 2


class SampleEnergy(Observable):
    """Compute sample energies with respect to the nominal BQM.

    Energies exclude the magnitude of ``signed_energy_scale`` but include its sign.
    """

    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> NDArray:
        """Return signed sample energies from the sample set.

        Args:
            experiment: Experiment context providing ``signed_energy_scale``.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: Samples containing energy data.

        Returns:
            A numpy array containing the sample energies multiplied by the sign
            of ``signed_energy_scale``.
        """
        return sample_set.data_vectors["energy"] * np.sign(experiment.param["signed_energy_scale"])


class BitpackedSpins(Observable):
    """Compute bitpacked spins."""

    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
    ) -> tuple[NDArray, tuple[int, int]]:
        """Return bitpacked spin samples and their original array shape.

        Args:
            experiment: Experiment object containing the context for this observable.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: Samples containing the spin values to unpack.

        Returns:
            A tuple containing the bitpacked spin array and the original array
            shape.
        """
        sample_array = dimod.as_samples(sample_set)[0]

        # Bitpack solutions
        results_bool = np.equal(sample_array, 1)
        results_bitpacked = np.packbits(results_bool)
        results_shape = sample_array.shape

        return results_bitpacked, results_shape


class ReferenceEnergy(Observable):
    """Return a cached reference energy, computing it and saving it if needed."""

    def evaluate(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample_set: dimod.SampleSet,
        path: str | Path | None = None,
        lattice: Lattice | None = None,
    ) -> float:
        """Get the reference energy for the given BQM, computing and caching it
        if needed.

        Args:
            experiment: The experiment for which to get the reference energy. Used
                to determine the path for caching and loading the reference energy.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample_set: The sample set is not used in this observable, but is
                included in the signature for consistency with other observables.
            path: Optional path to load/save the reference energy. If not provided,
                a default path will be generated based on the experiment and BQM.

        Returns:
            The reference energy for the given BQM.
        """
        if path is not None:
            path = Path(path)
        else:
            path = get_reference_energy_path(bqm, experiment)

        if path.exists():
            energy, sample, method_string = self.load(experiment, bqm, path)
            return energy

        # And if we can't load, we generate a reference sample.
        if experiment is not None:
            energy, sample, method_string = experiment.lattice.optimize(bqm)
        elif lattice is not None:
            energy, sample, method_string = lattice.optimize(bqm)
        else:
            raise ValueError(
                "Must provide either an experiment or a lattice to compute reference energy."
            )

        self.save(path, energy, sample, method_string)

        return energy

    def load(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        path: str | Path | None = None,
    ) -> tuple[float, NDArray, str]:
        """Load and get the full data tuple, not just the energy.

        Args:
            experiment: The experiment for which to load the reference energy.
            bqm: The binary quadratic model corresponding to the problem instance.
            path: Optional path to load the reference energy. If not provided,
                a default path will be generated based on the experiment and BQM.

        Returns:
            A tuple containing the reference energy, the corresponding sample
            as a NumPy array, and a string indicating the optimization method used.
        """
        if path is not None:
            path = Path(path)
        else:
            path = get_reference_energy_path(bqm, experiment)

        with open(path, "r") as f:
            method_string = f.readline().strip()
            energy = float(f.readline().strip())

        sample = np.loadtxt(path, skiprows=2)

        return energy, sample, method_string

    def save(self, path: str | Path, energy: float, sample: NDArray, method_string: str) -> None:
        """Save the reference energy to disk.

        Args:
            path: Path to save the reference energy file.
            energy: The reference energy to save.
            sample: The corresponding sample to save.
            method_string: A string indicating the optimization method used to
                obtain the reference energy.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, sample, fmt="%d", header=f"{method_string}\n{energy}", comments="")

    def update(
        self,
        experiment: Experiment,
        bqm: dimod.BQM,
        sample: NDArray,
        path: str | Path | None = None,
    ) -> None:
        """Update the cached reference energy if the provided sample improves it.

        Use this when you get an energy that is lower than the reference energy.
        We want to keep the old method string unless it is specified.

        Args:
            experiment: The experiment for which to update the reference energy.
            bqm: The binary quadratic model corresponding to the problem instance.
            sample: The new sample that may improve the reference energy.
            path: Optional path to load/save the reference energy. If not provided,
                a default path will be generated based on the experiment and BQM.
        """
        if path is not None:
            path = Path(path)

        reference_energy, _, reference_method_string = self.load(experiment, bqm, path)
        new_energy = bqm.energy(sample)

        if new_energy >= reference_energy:
            raise ValueError("New energy is not better than reference energy, not updating.")

        if path is None:
            path = get_reference_energy_path(bqm, experiment)
        self.save(path, new_energy, sample, reference_method_string)


def get_reference_energy_path(
    bqm: dimod.BQM,
    experiment: Experiment | None = None,
    root: str | Path | None = None,
    dummy_experiment_data_dict: dict[str, Any] | None = None,
) -> Path:
    """Return the path to the reference energy file for the given experiment and BQM.

    This should be revised if relevant factors are not captured in the instance
    pathstring, for example when ground-state energies depend on the specific chip.

    Args:
        bqm: The binary quadratic model for which to get the reference energy path.
        experiment: The experiment for which to get the reference energy path.
        root: Optional root directory to use instead of the experiment's data root.
        dummy_experiment_data_dict: A dictionary containing the keys ``run_index``,
            ``num_random_instances``, and ``lattice`` to use when no experiment is
            provided. This allows for generation of dummy experiment data without
            all the overhead, for running without an actual experiment.

    Returns:
        The path to the reference energy file.
    """
    if experiment is None:
        if dummy_experiment_data_dict is None:
            raise ValueError("Provide either 'experiment' or 'dummy_experiment_data_dict'.")
        experiment_data_dict = dummy_experiment_data_dict
    else:
        experiment_data_dict = {
            "run_index": experiment.run_index,
            "num_random_instances": experiment.param["num_random_instances"],
            "lattice": experiment.lattice,
        }

    if root is None:
        root = experiment_data_dict["lattice"].data_root
    else:
        root = Path(root)

    path = (
        root
        / "lattice_data"
        / "reference_energies"
        / experiment_data_dict["lattice"]._get_instance_pathstring()
    )

    # Use hash. BQM is not hashable so use the experiment.lattice data to generate a tuple.
    bqm_as_tuple = tuple(bqm.linear[v] for v in sorted(bqm.variables)) + tuple(
        bqm.quadratic[e] for e in experiment_data_dict["lattice"].edge_list
    )
    bqm_hash = hash(bqm_as_tuple)
    path = path / str(bqm_hash)

    return path.with_suffix('.txt')
