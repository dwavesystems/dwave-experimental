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

import copy
import lzma
import os
import pickle
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dimod
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if TYPE_CHECKING:
    from tqdm.std import tqdm as tqdm_bar

from dwave.experimental.lattice_utils.experiment.samplercall import SamplerCall
from dwave.experimental.lattice_utils.lattice.lattice import Lattice
from dwave.experimental.lattice_utils.observable import (
    BitpackedSpins,
    CouplerCorrelation,
    CouplerFrustration,
    QubitMagnetization,
    ReferenceEnergy,
    SampleEnergy,
)

__all__ = ['Experiment', 'ExperimentConfig']

DW_TEAL = "#17bebb"
DW_BLUE = "#2a7de1"
DW_ORANGE = "#f37820"


@dataclass
class ExperimentConfig:
    """Container for the parameters that define an experiment."""

    signed_energy_scale: float = 1.0
    num_reads: int = 100
    anneal_time: float = 1.0
    num_random_instances: int | None = 1
    readout_thermalization: int = 100
    flux_bias_shim_step: float = 0.0
    coupler_shim_step: float = 0.0
    anneal_offset_shim_step: float = 0.0
    target_magnetization: float = 0.0


class Experiment:
    """Base class for running experiments on lattice instances.

    Includes common functionality for managing parameters, running iterations,
    parsing results, and saving data.

    Args:
        lattice: The lattice instance to run the experiment on.
        sampler: The dimod sampler to use for sampling.
        max_iterations: The maximum number of iterations to run the experiment for.
        config: An ExperimentConfig object containing experiment parameters.
    """

    def __init__(
        self,
        *,
        lattice: Lattice,
        sampler: dimod.Sampler,
        reference_energy_sampler: dimod.Sampler | None = None,
        reference_energy_sampler_kwargs: dict[str, Any] | None = None,
        max_iterations: int | None = None,
        config: ExperimentConfig | None = None,
    ):
        if config is None:
            config = ExperimentConfig()

        self.lattice = lattice
        self.sampler = sampler
        self.reference_energy_sampler = reference_energy_sampler
        self.reference_energy_sampler_kwargs = reference_energy_sampler_kwargs
        self.param = vars(config).copy()
        self.experiment_results_root = lattice.data_root / "results"
        self.data_path = None
        self.run_index = 0
        self.config = config
        self.max_iterations = max_iterations
        self.already_initialized: bool = False
        self.observables_to_collect = [
            QubitMagnetization(),
            CouplerCorrelation(),
            CouplerFrustration(),
            SampleEnergy(),
            BitpackedSpins(),
            ReferenceEnergy(),
        ]

    def load_results(
        self,
        num_iterations: int = 100,
        start_iteration: int | None = None,
        result_fields: list[str] | None = None,
        ignore_shim: bool = False,
    ) -> list[dict[str, Any]]:
        """Load results from the highest-numbered iterations of the experiment.

        .. warning::
            Results are stored as LZMA-compressed **pickle** files
            (``iter*.pkl.lzma``) and reloaded here with :func:`pickle.load`,
            which is not secure against maliciously constructed data:
            unpickling can execute arbitrary code. Only load result
            directories that you created yourself or obtained from a fully
            trusted source.

        Args:
            num_iterations: Maximum number of iterations to load.
            start_iteration: If provided, load results starting from this
                iteration index. Otherwise the most recent ``num_iterations``
                results are loaded.
            result_fields: Subset of fields to extract from each result file. If
                ``None``, all fields present in the first result file are used.
            ignore_shim: If true, the ``shim_data`` field is removed from the
                returned results.

        Returns:
            A list of dictionaries containing the results for each iteration.
        """
        fnlist = self._get_sorted_results_file_list()
        if start_iteration is not None:
            fnlist = fnlist[max(start_iteration, 0) : max(start_iteration + num_iterations, 0)]
        else:
            fnlist = fnlist[-num_iterations:]

        results = []
        for filename in fnlist:

            try:
                with lzma.open(filename, "rb") as f:
                    data = pickle.load(f)
            except lzma.LZMAError as e:
                raise lzma.LZMAError(f"Failing to load {filename}", e)

            fields = list(data.keys()) if result_fields is None else result_fields
            if ignore_shim:
                fields = [f for f in fields if f != "shim_data"]

            results.append({k: data[k] for k in fields})

        return results

    def apply_param(self, param: dict[str, float]) -> None:
        """Apply a parameter configuration to the experiment.

        Parameters are formatted to ensure filename consistency, which can be
        important for loading data.

        Args:
            param: Dictionary of parameter values to apply to the experiment.
        """
        param = self._format_parameter_list([param])[0]

        # anneal_time and anneal_schedule are mutually exclusive
        if "anneal_schedule" in param:
            self.param.pop("anneal_time", None)
        elif "anneal_time" in param:
            self.param.pop("anneal_schedule", None)

        self.param.update(param)

        self.data_path = self.experiment_results_root / self._get_relative_data_path()
        self.already_initialized = self._prepare_run_index()

    def run_iteration(
        self,
        parameter_list: list,
        progress: bool = False,
        scaling_factor: float = 1.0,
    ) -> bool:
        """Run one experiment iteration for each parameter set in ``parameter_list``.

        For each parametrization, this method applies the parameters, builds the
        sampler call, submits the sampling job, waits for completion, parses the
        returned results, updates the shim, and saves the results.

        Args:
            parameter_list: List of parameter dictionaries to run.
            progress: If true, displays a progress bar for waiting on results.
            scaling_factor: A multiplicative factor to apply to the BQM before sampling.

        Returns:
            A boolean value corresponding to whether or not the experiment is
            finished.
        """
        if progress and tqdm is None:
            raise ImportError("Progress reporting requires the optional 'tqdm' dependency.")

        embeddings = getattr(self.lattice, "embedding_list", None)
        if embeddings is None or len(embeddings) == 0:
            raise RuntimeError("Call lattice.embed_lattice(sampler) before running.")

        if progress:
            tqdm.write(
                f"\n{type(self.lattice).__name__}={self.lattice.dimensions}, "
                f"J={self.param['signed_energy_scale']}, "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"({self.lattice._get_instance_pathstring()}/{self._get_solver_pathstring()})"
            )

        parameter_list = self._format_parameter_list(parameter_list)
        response_dict = {}
        call_dict = {}

        create_bar = (
            self._make_progress_bar(
                total=len(parameter_list),
                desc="Creating sampler calls",
                colour=DW_BLUE,
            )
            if progress
            else None
        )

        for index, param in enumerate(parameter_list):
            self.apply_param(param)
            call_dict[index] = self._build_sampler_call()
            if call_dict[index] is None:
                call_dict.pop(index)
            else:
                response_dict[index] = self.sampler.sample(
                    call_dict[index].bqm * scaling_factor,
                    **call_dict[index].sampler_params,
                )
            if create_bar is not None:
                create_bar.update()

        if create_bar is not None:
            create_bar.close()

        if not call_dict:
            if progress:
                tqdm.write(
                    f"***\n***\nFINISHED for all {len(parameter_list)} parameterizations.\n***\n***"
                )
            return True

        wait_bar = (
            self._make_progress_bar(
                total=len(call_dict),
                desc=" Awaiting/parsing data",
                colour=DW_TEAL,
            )
            if progress
            else None
        )

        # Get and manage all the results
        while response_dict:
            made_progress = False

            for index, val in response_dict.items():

                if val.done():
                    self.apply_param(parameter_list[index])
                    results = self.parse_results(call_dict[index], val)
                    self._update_shim(call_dict[index], results)
                    savedata = self._generate_data_to_save(call_dict[index], results)
                    self._save_results(savedata)
                    if wait_bar is not None:
                        wait_bar.update()
                    del response_dict[index]
                    made_progress = True
                    break

            if not made_progress:
                time.sleep(0.1)  # Waiting for results to come in

        if wait_bar is not None:
            wait_bar.close()

        if progress:
            self._print_iteration_status(call_dict, len(parameter_list))

        return False

    def parse_results(self, call: SamplerCall, response: dimod.SampleSet) -> dict[str, Any]:
        """Parse a sampler response into per-embedding observable results.

        Args:
            call: Sampler call metadata, including the logical BQMs
            response: Raw sample set returned by the sampler.

        Returns:
            Dictionary mapping observable names to their evaluated results across
            embeddings.
        """
        embedding_list = self.lattice.embedding_list
        myarr = response.samples(sorted_by=None)
        sample_arrays = [myarr[:, emb].copy() for emb in embedding_list]

        sample_set = {}
        for iemb, sample_array in enumerate(sample_arrays):
            sample_set[iemb] = dimod.SampleSet.from_samples_bqm(
                sample_array, call.logical_bqms[iemb]
            )

        results = {}
        for observable in self.observables_to_collect:
            results[observable.name] = []
            for iemb, sample_array in enumerate(sample_arrays):
                bqm = call.logical_bqms[iemb]
                obs_result = observable.evaluate(self, bqm, sample_set[iemb])
                results[observable.name].append(obs_result)

            if isinstance(results[observable.name][0], np.ndarray):
                results[observable.name] = np.asarray(results[observable.name])

        return results

    def _make_progress_bar(
        self,
        *,
        total: int,
        desc: str,
        colour: str,
        bar_format: str | None = None,
        initial: int | float = 0,
    ) -> tqdm_bar:
        """Create a tqdm progress bar with consistent formatting.

        The optional ``tqdm`` dependency is required to call this method. If
        ``tqdm`` is not installed, an ImportError will be raised.
        """
        if tqdm is None:
            raise ImportError("Progress reporting requires the optional 'tqdm' dependency.")

        if bar_format is None:
            bar_width = min(100, max(total, 20))
            bar_format = f"{{desc}}: |{{bar:{bar_width}}}{{r_bar}}{{bar:-{bar_width}b}}"

        return tqdm(
            total=total,
            initial=initial,
            desc=desc,
            bar_format=bar_format,
            colour=colour,
        )

    def _print_iteration_status(
        self,
        call_dict: dict[int, SamplerCall],
        num_params: int,
    ) -> None:
        """Print a summary of the iteration status, including progress and iteration ranges.

        The optional ``tqdm`` dependency is required to call this method. If
        ``tqdm`` is not installed, an ImportError will be raised.
        """
        if tqdm is None:
            raise ImportError("Progress reporting requires the optional 'tqdm' dependency.")

        iteration_range = (
            f"Iteration range "
            f"{min(call.shim_data['total_iterations'] for call in call_dict.values())}-"
            f"{max(call.shim_data['total_iterations'] for call in call_dict.values())} "
        )
        if self.max_iterations is None:
            tqdm.write("        Total progress: " + iteration_range)
            return

        total = num_params * self.max_iterations
        progress_value = (
            sum(call.shim_data["total_iterations"] for call in call_dict.values())
            + (num_params - len(call_dict)) * self.max_iterations
        )

        progress_string = (
            f"{progress_value / total * 100:.1f}%  "
            f"Iteration range "
            f"{min(call.shim_data['total_iterations'] for call in call_dict.values())}-"
            f"{max(call.shim_data['total_iterations'] for call in call_dict.values())} "
            f"of {self.max_iterations} "
            f"({num_params - len(call_dict)} of {num_params} parameters finished)"
        )

        bar_width = min(100, max(num_params, 20))
        bar_format = f"{{desc}}: |{{bar:{bar_width}}}| {progress_string}"

        total_bar = self._make_progress_bar(
            total=total,
            desc="        Total progress",
            bar_format=bar_format,
            colour=DW_ORANGE,
            initial=progress_value,
        )
        total_bar.close()

    def _save_results(
        self,
        data_dict: dict[str, Any],
        run_index: int | None = None,
        filename: str | None = None,
    ) -> None:
        """Save results to disk using LZMA-compressed pickle."""
        if filename is None:
            if run_index is None:
                run_index = self.run_index
            filename = f"iter{run_index:05d}.pkl.lzma"
        else:
            if run_index is not None:
                raise ValueError("Cannot specify both filename and run_index.")

        # Write to a temp directory first to reduce disk write errors from killed jobs.
        with tempfile.TemporaryDirectory(dir=self.data_path) as tmp:
            temp_filename = Path(tmp) / filename
            with lzma.open(temp_filename, "wb") as f:
                pickle.dump(data_dict, f)
            os.rename(temp_filename, self.data_path / filename)

    def _get_sorted_results_file_list(self) -> list[str]:
        """Return result filenames sorted lexicographically."""
        return [str(fn) for fn in sorted(self.data_path.glob("iter*.pkl.lzma"))]

    def _get_next_run_index(self) -> tuple[int, bool]:
        """Get the next run index based on the existing files in the data path."""
        if not self.data_path.exists():
            return 0, False

        fnlist = list(self.data_path.glob("iter*.pkl.lzma"))
        if not fnlist:
            return 0, False

        latest_file_iter = max(int(fn.stem.split(".")[0][4:]) for fn in fnlist)
        return latest_file_iter + 1, True

    def _prepare_run_index(self) -> bool:
        """Prepare the run index for the next iteration, creating the data path if needed."""
        if self.data_path is None:
            raise RuntimeError("No parameterization selected. Call apply_param() first.")

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.run_index, already_initialized = self._get_next_run_index()
        return already_initialized

    def _get_solver_pathstring(self) -> str:
        """Construct a pathstring for the solver.

        Structured to support additional sampler types in the future.
        """
        pathstring = None
        rules = [
            (lambda s: s == "DWaveSampler", "qpu"),
            (lambda s: s == "MockDWaveSampler", "MockDWaveSampler"),
        ]
        for check, label in rules:
            if check(type(self.sampler).__name__):
                pathstring = label
        if pathstring is None:
            raise TypeError("Sampler type not compatible with known possibilities")

        if pathstring in ["qpu"]:
            pathstring += f"/{self.sampler.solver.name}"

        return pathstring

    def _get_parameter_pathstring(self) -> str:
        """Construct a pathstring for the experimental parameters.

        Assumes a forward anneal. Annealing time format is in microseconds (up
        to 999.9999us), with six decimal places (picosecond resolution).
        """
        signed_energy_scale = self.param["signed_energy_scale"]

        if "anneal_time" in self.param:
            parts = [
                f"energyscale{signed_energy_scale:0.3}",
                f'atime{self.param["anneal_time"]:010.6f}us',
            ]
        elif "anneal_schedule" in self.param:
            parts = [
                f"energyscale{signed_energy_scale:0.3}",
                f'asched{self.param["anneal_schedule"]}',
            ]
        else:
            raise ValueError(
                "Parameter list must contain either 'anneal_time' or 'anneal_schedule'."
            )

        parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in parts]
        return "/".join(parts)

    def _get_relative_data_path(self) -> str:
        """Make a subdirectory name for a sampler call's data."""
        return "/".join(
            [
                self.lattice._get_instance_pathstring(),
                self._get_solver_pathstring(),
                self._get_parameter_pathstring(),
            ]
        )

    def _make_logical_bqms(self) -> list[dimod.BQM]:
        """Make logical BQMs for the experiment."""
        logical_bqm = self.lattice.make_bqm()

        if not hasattr(self.lattice, "embedding_list"):
            return [logical_bqm]

        return [logical_bqm] * len(self.lattice.embedding_list)

    def _build_sampler_call(self) -> None | SamplerCall:
        """Build the sampler call using attributes of the experiment and instance.

        Returns a SamplerCall.
        """
        sampler_call = SamplerCall(run_index=self.run_index)
        sampler_call.logical_bqms = self._make_logical_bqms()
        sampler_call.shim_data = self._get_shim_data()

        # Here we can find out that we're finished.
        if (
            self.max_iterations is not None
            and sampler_call.shim_data["total_iterations"] >= self.max_iterations
        ):
            return None

        sampler_call.bqm = self._make_bqm(sampler_call)
        sampler_call.sampler_params = self._make_sampler_params(shim_data=sampler_call.shim_data)

        return sampler_call

    def _format_parameter_list(
        self,
        parameter_list: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        """Deduplicate and format the parameter list for filename consistency.

        Some parameters can cause bugs if they are not appropriately formatted,
        rounded, etc. in accordance with filenames.
        """
        ret_unique = []
        ret = copy.deepcopy(parameter_list)
        for entry in ret:
            if "anneal_time" in entry:
                entry["anneal_time"] = np.round(entry["anneal_time"], 6)
            if "anneal_schedule" in entry:
                entry["anneal_schedule"] = [tuple(np.round(p, 6)) for p in entry["anneal_schedule"]]

            if entry not in ret_unique:
                ret_unique.append(entry)

        return ret_unique

    def _generate_data_to_save(
        self,
        sampler_call: SamplerCall,
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct a single dictionary containing results and shim data for saving."""
        savedata = {}
        for key in results:
            if isinstance(results[key], np.ndarray):
                if results[key].dtype == "complex128":
                    savedata[key] = results[key].astype(np.complex64)
                elif results[key].dtype == "float64":
                    savedata[key] = results[key].astype(np.float32)
                else:
                    savedata[key] = results[key]
            else:
                savedata[key] = results[key].copy()

        savedata["shim_data"] = {}
        for key in sampler_call.shim_data:
            if isinstance(sampler_call.shim_data[key], np.ndarray):
                savedata["shim_data"][key] = sampler_call.shim_data[key].astype(np.float32)
            elif isinstance(sampler_call.shim_data[key], int):
                savedata["shim_data"][key] = sampler_call.shim_data[key]
            else:
                savedata["shim_data"][key] = sampler_call.shim_data[key].copy()

        return savedata

    def _make_sampler_params(self, **kwargs) -> dict[str, Any]:
        """Construct a dictionary containing sampler parameters."""
        ret = {
            "answer_mode": "raw",
            "auto_scale": False,
            "flux_drift_compensation": False,
            "readout_thermalization": int(self.param["readout_thermalization"]),
            "num_reads": self.param["num_reads"],
            "label": os.path.join(self._get_relative_data_path(), f"iter{self.run_index:05d}"),
        }

        if "shim_data" in kwargs:
            if "flux_biases" in kwargs["shim_data"]:
                ret["flux_biases"] = list(kwargs["shim_data"]["flux_biases"])
            if "anneal_offsets" in kwargs["shim_data"]:
                ret["anneal_offsets"] = list(kwargs["shim_data"]["anneal_offsets"])

        if self.param.get("fast_anneal", False):
            ret["fast_anneal"] = True

        if "anneal_schedule" in self.param:
            ret["anneal_schedule"] = self.param["anneal_schedule"]
        elif "anneal_time" in self.param:
            ret["annealing_time"] = self.param["anneal_time"]

        return ret

    def _get_shim_data(self) -> dict[str, Any]:
        """Load shim data if possible, otherwise make an initial shim."""
        if self.already_initialized:
            return self._load_shim()
        return self._make_initial_shim()

    def _make_initial_shim(self) -> dict[str, Any]:
        """Create the initial shim and dictate what shim will be saved and modified."""
        shim_data = {"total_iterations": 0}
        if hasattr(self.lattice, "embedding_list"):
            num_embeddings = len(self.lattice.embedding_list)
            shim_data["flux_biases"] = np.zeros(self.sampler.properties["num_qubits"])
            shim_data["anneal_offsets"] = np.zeros(self.sampler.properties["num_qubits"])
            shim_data["relative_coupler_strength"] = np.ones(
                (num_embeddings, self.lattice.num_edges)
            )

        if self.param.get("flux_biases", None) is not None:
            shim_data["flux_biases"] = self.param.get("flux_biases")

        return shim_data

    def _get_latest_iteration_filename(self) -> Path:
        """Return the filename of the most recently completed iteration."""
        return self.data_path / f"iter{self.run_index - 1:05d}.pkl.lzma"

    def _load_shim(self) -> dict[str, Any]:
        """Load shim data from the most recently completed iteration."""
        filename = self._get_latest_iteration_filename()

        if os.path.getsize(filename) == 0:
            os.remove(filename)
            raise FileNotFoundError(f"{filename} does not exist")

        try:
            with lzma.open(filename, "rb") as f:
                data = pickle.load(f)
                shim_data = data["shim_data"]
            return shim_data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{filename} does not exist") from e
        except Exception as e:
            raise OSError("Failed to open file") from e

    def _update_shim(self, sampler_call: SamplerCall, results: dict[str, Any]) -> None:
        """Update shim parameters according to shim data and parameters."""
        if (
            "flux_biases" in sampler_call.shim_data
            and self.param.get("flux_bias_shim_step", 0) != 0
        ):
            self._update_flux_bias_shim(sampler_call, results)
        if (
            "relative_coupler_strength" in sampler_call.shim_data
            and self.param.get("coupler_shim_step", 0) != 0
        ):
            self._update_coupler_shim(sampler_call, results)

        sampler_call.shim_data["total_iterations"] += 1

    def _update_flux_bias_shim(self, sampler_call: SamplerCall, results: dict[str, Any]) -> None:
        """Update flux-bias shim values based on qubit magnetization."""
        target_magnetization = self.param["target_magnetization"]
        qubit_magnetization = results["QubitMagnetization"]
        flux_biases = sampler_call.shim_data["flux_biases"]
        shim_step = self.param["flux_bias_shim_step"]

        steps = shim_step * (qubit_magnetization.ravel() - target_magnetization)
        flux_biases[self.lattice.embedding_list.ravel()] -= steps
        mean_magnetization = np.mean(qubit_magnetization)

        if target_magnetization > 0:
            if mean_magnetization < target_magnetization - 0.001:
                flux_biases *= 1.01
            elif mean_magnetization > target_magnetization + 0.001:
                flux_biases /= 1.01

        elif target_magnetization < 0:
            if mean_magnetization > target_magnetization + 0.001:
                flux_biases *= 1.01
            elif mean_magnetization < target_magnetization - 0.001:
                flux_biases /= 1.01

    def _update_coupler_shim(
        self,
        sampler_call: SamplerCall,
        results: dict[str, Any],
        step_size: float | None = None,
    ) -> None:
        """Update relative coupler strength based on measured frustration."""
        orbits = self.lattice.coupler_orbits
        signed_energy_scale = self.param["signed_energy_scale"]
        relative_coupler_strength = sampler_call.shim_data["relative_coupler_strength"]

        # Allow for zero step size, which will just truncate the shim.
        if step_size is None:
            step_size = self.param["coupler_shim_step"]
        if step_size == 0:
            return

        # Get the set over which we normalize.
        normalization_basis = np.ones_like(orbits, dtype=bool)

        # Assume we have multiple embeddings of the same BQM.
        bqms = sampler_call.logical_bqms
        if len(bqms) > 1 and any(bqm != bqms[0] for bqm in bqms[1:]):
            raise NotImplementedError("Case for distinct embedded BQMs not implemented yet.")

        bqm = bqms[0]
        logical_values = np.array([bqm.quadratic[edge] for edge in self.lattice.edge_list])
        coupler_signs = np.sign(logical_values)
        for orbit_bin in range(max(orbits) + 1):
            bin_edges = np.argwhere(orbits == orbit_bin).ravel()
            if step_size != 0:
                frust = results["CouplerFrustration"][:, bin_edges]
                meanfrust = np.mean(frust)
                relative_coupler_strength[:, bin_edges] += step_size * (frust - meanfrust)

            # Damp the couplers (push toward default value)
            if "coupler_damp" in self.param and self.param["coupler_damp"] > 0:
                excess = relative_coupler_strength[:, bin_edges] - np.mean(
                    relative_coupler_strength[:, bin_edges]
                )
                relative_coupler_strength[:, bin_edges] -= (
                    np.multiply(coupler_signs[bin_edges], excess) * self.param["coupler_damp"]
                )

            # New truncation method... previous is buggy when we mix signs of logical values.
            # Let's try being more explicit.
            for iemb in range(len(relative_coupler_strength)):
                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * logical_values[bin_edges]
                    * signed_energy_scale
                    > 1
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    0.99999 / logical_values[bin_edges[violators]] / signed_energy_scale
                )

                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * logical_values[bin_edges]
                    * signed_energy_scale
                    < -2
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    -1.99999 / logical_values[bin_edges[violators]] / signed_energy_scale
                )

        # Renormalize each orbit after truncation
        for orbit_bin in range(np.max(orbits) + 1):
            bin_edges = orbits == orbit_bin
            mean_relative = np.mean(
                np.abs(relative_coupler_strength[:, bin_edges * normalization_basis])
            )
            relative_coupler_strength[:, bin_edges] /= mean_relative

        # And truncate again
        for orbit_bin in range(np.max(orbits) + 1):
            bin_edges = np.argwhere(orbits == orbit_bin).ravel()

            # New truncation method... previous is buggy when we mix signs of logical values.
            # Let's try being more explicit.
            for iemb in range(len(relative_coupler_strength)):
                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * logical_values[bin_edges]
                    * signed_energy_scale
                    > 1
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    0.99999 / logical_values[bin_edges[violators]] / signed_energy_scale
                )

                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * logical_values[bin_edges]
                    * signed_energy_scale
                    < -2
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    -1.99999 / logical_values[bin_edges[violators]] / signed_energy_scale
                )

        Q = logical_values * relative_coupler_strength * signed_energy_scale
        Q_max = np.max(Q)
        Q_min = np.min(Q)
        if Q_max > 1 or Q_min < -2:
            raise ValueError(
                "Effective coupler strengths violate hardware bounds: "
                f"min={Q_min:.6f}, max={Q_max:.6f}"
            )

    def _make_bqm(self, sampler_call: SamplerCall) -> dimod.BQM:
        """Construct a BQM for the current sampler call."""
        signed_energy_scale = self.param["signed_energy_scale"]
        relative_coupler_strength = sampler_call.shim_data["relative_coupler_strength"]
        bqm = dimod.BQM(vartype="SPIN")

        for iemb, emb in enumerate(self.lattice.embedding_list):
            logical_bqm = sampler_call.logical_bqms[iemb].copy()

            for v in range(self.lattice.num_spins):
                if logical_bqm.degree(v) > 0:
                    bqm.add_linear(emb[v], logical_bqm.linear[v])

            for iedge, edge in enumerate(self.lattice.edge_list):
                bias = (
                    logical_bqm.quadratic[tuple(edge)]
                    * relative_coupler_strength[iemb, iedge]
                    * signed_energy_scale
                )
                bqm.add_quadratic(emb[edge[0]], emb[edge[1]], bias)

        return bqm
