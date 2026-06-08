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

import tempfile
import lzma
import os
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass

import dimod
import numpy as np
from tqdm.auto import tqdm

from dwave.experimental.lattice_utils.lattice import Lattice
from dwave.experimental.lattice_utils.observable import (
    QubitMagnetization,
    CouplerCorrelation,
    CouplerFrustration,
    SampleEnergy,
    BitpackedSpins,
    ReferenceEnergy,
)
from dwave.experimental.lattice_utils.experiment.samplercall import SamplerCall

__all__ = ['Experiment', 'ExperimentConfig']

DW_TEAL = "#17bebb"
DW_BLUE = "#2a7de1"
DW_ORANGE = "#f37820"


@dataclass
class ExperimentConfig:
    """Container for the parameters that define an experiment."""

    energy_scale: float = 1.0
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
        inst: The lattice instance to run the experiment on.
        sampler: The dimod sampler to use for sampling.
        max_iterations: The maximum number of iterations to run the experiment for.
        config: An ExperimentConfig object containing experiment parameters.
    """

    def __init__(
        self,
        *,
        inst: Lattice,
        sampler: dimod.Sampler,
        max_iterations: int | None = None,
        config: ExperimentConfig,
    ):
        self.inst = inst
        self.sampler = sampler
        self.param = dict(vars(config))
        self.experiment_results_root = inst.data_root / "results"
        self.data_path = None
        self.run_index = 0
        self.config = config
        self.max_iterations = max_iterations
        self.already_initialized: bool = False
        self.observables_to_collect = {
            QubitMagnetization(),
            CouplerCorrelation(),
            CouplerFrustration(),
            SampleEnergy(),
            BitpackedSpins(),
            ReferenceEnergy(),
        }

    def load_results(
        self,
        num_iterations: int = 100,
        start_iteration: int | None = None,
        result_fields: list[str] | None = None,
        quiet: bool = True,
        ignore_shim: bool = False,
    ) -> list[dict[str, Any]]:
        """Load results from the highest-numbered iterations of the experiment.

        Args:
            num_iterations: Maximum number of iterations to load.
            start_iteration: If provided, load results starting from this
                iteration index. Otherwise the most recent ``num_iterations``
                results are loaded.
            result_fields: Subset of fields to extract from each result file. If
                ``None``, all fields present in the first result file are used.
            quiet: If false, prints a message when each result file is loaded.
            ignore_shim: If true, the ``shimdata`` field is removed from the
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

            if not quiet:
                print(f"Loaded {filename} at {datetime.now()}")
            if result_fields is None:
                result_fields = list(data.keys())
                if ignore_shim:
                    result_fields.remove("shimdata")

            results.append({k: data[k] for k in result_fields})

        return results

    def apply_param(self, param: dict[str, float]) -> None:
        """Apply a parameter configuration to the experiment.

        Parameters are formatted to ensure filename consistency, which can be
        important for loading data.

        Args:
            param: Dictionary of parameter values to apply to the experiment.
        """
        param = self._format_parameter_list([param])[0]
        for param_name, param_val in param.items():
            self.param[param_name] = param_val

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
        try:
            self.inst._load_embeddings(self.sampler)
        except FileNotFoundError as e:
            raise FileNotFoundError("No Embedding Found: ", e) from e

        tqdm.write(
            f"\n{type(self.inst).__name__}={self.inst.dimensions}, "
            f"J={self.param['energy_scale']}, "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"({self.inst._get_instance_pathstring()}/{self._get_solver_pathstring()})"
        )

        parameter_list = self._format_parameter_list(parameter_list)
        response_dict = {}
        call_dict = {}

        create_bar = self._make_progress_bar(
            total=len(parameter_list),
            desc="Creating sampler calls",
            colour=DW_BLUE,
            enabled=progress,
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

        if len(call_dict) == 0:
            if progress:
                tqdm.write(
                    f"***\n***\nFINISHED for all {len(parameter_list)} parameterizations.\n***\n***"
                )
            return True

        wait_bar = self._make_progress_bar(
            total=len(call_dict),
            desc=" Awaiting/parsing data",
            colour=DW_TEAL,
            enabled=progress,
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
                    self._save_results(savedata, quiet=True)
                    if wait_bar is not None:
                        wait_bar.update()
                    del response_dict[index]
                    made_progress = True
                    break

            if not made_progress:
                time.sleep(0.1)  # Waiting for results to come in

        if wait_bar is not None:
            wait_bar.close()

        self._print_iteration_status(call_dict, len(parameter_list), enabled=progress)
        return False

    def parse_results(self, call: SamplerCall, response: dimod.SampleSet) -> dict[str, Any]:
        """Parse a sampler response into per-embedding observable results.

        Args:
            call: Sampler call metadata, inluding the nominal BQMs
            response: Raw sample set returned by the sampler.

        Returns:
            Dictionary mapping observable names to their evaluated results across
            embeddings.
        """
        if hasattr(self.inst, "embedding_list"):
            embedding_list = self.inst.embedding_list
            myarr = response.samples(sorted_by=None)
            sample_arrays = [myarr[:, emb].copy() for emb in embedding_list]
        else:
            sample_arrays = [response.samples(sorted_by=None)[:, np.arange(self.inst.num_spins)]]

        sample_set = {}
        for iemb, sample_array in enumerate(sample_arrays):
            sample_set[iemb] = dimod.SampleSet.from_samples_bqm(
                sample_array, call.nominal_bqms[iemb]
            )

        results = {}
        for observable in set(self.observables_to_collect):
            results[observable.name] = []
            for iemb, sample_array in enumerate(sample_arrays):
                bqm = call.nominal_bqms[iemb]
                obs_result = observable.evaluate(self, bqm, sample_set[iemb])
                results[observable.name].append(obs_result)

            if type(results[observable.name][0]) == np.ndarray:
                results[observable.name] = np.asarray(results[observable.name])

        return results

    def _make_progress_bar(
        self,
        *,
        total: int,
        desc: str,
        colour: str,
        enabled: bool,
        bar_format: str | None = None,
        initial: int | float = 0,
    ) -> tqdm | None:
        """Create a tqdm progress bar with consistent formatting."""
        if not enabled:
            return None

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
        enabled: bool,
    ) -> None:
        """Print a summary of the iteration status, including progress and iteration ranges."""
        if not enabled:
            return
        iteration_range = (
            f"Iteration range "
            f"{min(call.shimdata['total_iterations'] for call in call_dict.values())}-"
            f"{max(call.shimdata['total_iterations'] for call in call_dict.values())} "
        )
        if self.max_iterations is None:
            tqdm.write("        Total progress: " + iteration_range)
            return

        total = num_params * self.max_iterations
        progress_value = (
            sum(call.shimdata["total_iterations"] for call in call_dict.values())
            + (num_params - len(call_dict)) * self.max_iterations
        )

        progress_string = (
            f"{progress_value / total * 100:.1f}%  "
            f"Iteration range "
            f"{min(call.shimdata['total_iterations'] for call in call_dict.values())}-"
            f"{max(call.shimdata['total_iterations'] for call in call_dict.values())} "
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
            enabled=enabled,
            initial=progress_value,
        )
        total_bar.close()

    def _save_results(
        self,
        data_dict: dict[str, Any],
        run_index: int | None = None,
        quiet: bool = False,
        filename: str | None = None,
    ) -> None:
        """Save results to disk using LZMA-compressed pickle."""
        if filename is None:
            if run_index is None:
                run_index = self.run_index
            filename = f"iter{run_index:05d}.pkl.lzma"
        else:
            if run_index is not None:
                raise ValueError

        # Write to a temp directory first to reduce disk write errors from killed jobs.
        with tempfile.TemporaryDirectory(dir=self.data_path) as tmp:
            temp_filename = Path(tmp) / filename
            with lzma.open(temp_filename, "wb") as f:
                pickle.dump(data_dict, f)
            os.rename(temp_filename, self.data_path / filename)

        if not quiet:
            print(f"Saved {filename} at {datetime.now()}")

    def _get_sorted_results_file_list(self) -> list[str]:
        """Return result filenames sorted lexicographically."""
        fnlist = list(self.data_path.glob("iter*.pkl.lzma"))
        fnlist.sort()
        return [str(fn) for fn in fnlist]

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
        energy_scale = self.param["energy_scale"]

        if "anneal_time" in self.param:
            pathstring = f'energyscale{energy_scale:0.3}/atime{self.param["anneal_time"]:010.6f}us'
        elif "anneal_schedule" in self.param:
            pathstring = f'energyscale{energy_scale:0.3}/asched{self.param["anneal_schedule"]}'
        else:
            raise ValueError

        # Strip spaces and replace other unswanted symbols with underscores.
        pathstring = pathstring.replace(" ", "_")
        for bad_symbol in ":;,":
            pathstring = pathstring.replace(bad_symbol, "")

        return pathstring

    def _get_relative_data_path(self) -> str:
        """Make a subdirectory name for a sampler call's data."""
        return "/".join(
            [
                self.inst._get_instance_pathstring(),
                self._get_solver_pathstring(),
                self._get_parameter_pathstring(),
            ]
        )

    def _make_nominal_bqms(self) -> list[dimod.BQM]:
        """Make nominal BQMs (one per embedding) for the experiment."""
        nominal_bqm = self.inst.make_nominal_bqm()

        if not hasattr(self.inst, "embedding_list"):
            return [nominal_bqm]

        return [nominal_bqm] * len(self.inst.embedding_list)

    def _build_sampler_call(self) -> None | SamplerCall:
        """Build the sampler call using attributes of the experiment and instance.

        Returns a SamplerCall.
        """
        sampler_call = SamplerCall(run_index=self.run_index)
        sampler_call.nominal_bqms = self._make_nominal_bqms()
        sampler_call.shimdata = self._get_shimdata()

        # Here we can find out that we're finished.
        if (
            self.max_iterations is not None
            and sampler_call.shimdata["total_iterations"] >= self.max_iterations
        ):
            return None

        sampler_call.bqm = self._make_bqm(sampler_call)
        sampler_call.sampler_params = self._make_sampler_params(shimdata=sampler_call.shimdata)

        return sampler_call

    def _format_parameter_list(
        self,
        parameter_list: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        """Deduplicate and format the parameter list for filename consistency.

        Some parameters can cause bugs if they are not appropriately formatted,
        rounded, etc. in accordance with filenames.
        """
        ret = parameter_list.copy()
        for entry in ret:
            if "anneal_time" in entry:
                entry["anneal_time"] = np.round(entry["anneal_time"], 6)
            if "anneal_schedule" in entry:
                entry["anneal_schedule"] = [tuple(np.round(p, 6)) for p in entry["anneal_schedule"]]

        # We want the elements to be unique, of course.
        ret_unique = []
        for entry in ret:
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
            if type(results[key]) == np.ndarray:
                if results[key].dtype == "complex128":
                    savedata[key] = results[key].astype(np.complex64)
                elif results[key].dtype == "float64":
                    savedata[key] = results[key].astype(np.float32)
                else:
                    savedata[key] = results[key]
            else:
                savedata[key] = results[key].copy()

        savedata["shimdata"] = {}
        for key in sampler_call.shimdata:
            if type(sampler_call.shimdata[key]) == np.ndarray:
                savedata["shimdata"][key] = sampler_call.shimdata[key].astype(np.float32)
            elif type(sampler_call.shimdata[key]) == int:
                savedata["shimdata"][key] = sampler_call.shimdata[key]
            else:
                savedata["shimdata"][key] = sampler_call.shimdata[key].copy()

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

        if "shimdata" in kwargs:
            if "flux_biases" in kwargs["shimdata"]:
                ret["flux_biases"] = list(kwargs["shimdata"]["flux_biases"])
            if "anneal_offsets" in kwargs["shimdata"]:
                ret["anneal_offsets"] = list(kwargs["shimdata"]["anneal_offsets"])

        if self.param.get("fast_anneal", False):
            ret["fast_anneal"] = True

        ret["annealing_time"] = self.param["anneal_time"]

        return ret

    def _get_shimdata(self) -> dict[str, Any]:
        """Load shim data if possible, otherwise make an initial shim."""
        if self.already_initialized:
            return self._load_shim()
        return self._make_initial_shim()

    def _make_initial_shim(self) -> dict[str, Any]:
        """Create the initial shim and dictate what shim will be saved and modified."""
        shimdata = {"total_iterations": 0}
        if hasattr(self.inst, "embedding_list"):
            num_embeddings = len(self.inst.embedding_list)
            shimdata["flux_biases"] = np.zeros(self.sampler.properties["num_qubits"])
            shimdata["anneal_offsets"] = np.zeros(self.sampler.properties["num_qubits"])
            shimdata["relative_coupler_strength"] = np.ones((num_embeddings, self.inst.num_edges))

        if self.param.get("flux_biases", None) is not None:
            shimdata["flux_biases"] = self.param.get("flux_biases")

        return shimdata

    def _get_latest_iteration_filename(self) -> Path:
        """Return the filename of the most recently completed iteration."""
        return self.data_path / f"iter{self.run_index - 1:05d}.pkl.lzma"

    def _load_shim(self):
        """Load shim data from the most recently completed iteration."""
        filename = self._get_latest_iteration_filename()

        if os.path.getsize(filename) == 0:
            os.remove(filename)
            raise FileNotFoundError(f"{filename} does not exist")

        try:
            with lzma.open(filename, "rb") as f:
                data = pickle.load(f)
                shimdata = data["shimdata"]
            return shimdata
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{filename} does not exist") from e
        except Exception as e:
            raise OSError("Failed to open file") from e

    def _update_shim(self, sampler_call: SamplerCall, results: dict[str, Any]):
        """Update shim parameters according to shim data and parameters."""
        if "flux_biases" in sampler_call.shimdata and self.param.get("flux_bias_shim_step", 0) != 0:
            self._update_flux_bias_shim(sampler_call, results)
        if (
            "relative_coupler_strength" in sampler_call.shimdata
            and self.param.get("coupler_shim_step", 0) != 0
        ):
            self._update_coupler_shim(sampler_call, results)

        sampler_call.shimdata["total_iterations"] += 1

    def _update_flux_bias_shim(self, sampler_call: SamplerCall, results: dict[str, Any]):
        """Update flux-bias shim values based on qubit magnetization."""
        target_magnetization = self.param["target_magnetization"]
        qubit_magnetization = results["QubitMagnetization"]
        flux_biases = sampler_call.shimdata["flux_biases"]
        shim_step = self.param["flux_bias_shim_step"]

        steps = shim_step * (qubit_magnetization.ravel() - target_magnetization)
        flux_biases[self.inst.embedding_list.ravel()] -= steps
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
        orbits = self.inst.coupler_orbits
        energy_scale = self.param["energy_scale"]
        relative_coupler_strength = sampler_call.shimdata["relative_coupler_strength"]

        # Allow for zero step size, which will just truncate the shim.
        if step_size is None:
            step_size = self.param["coupler_shim_step"]
        if step_size == 0:
            return

        # Get the set over which we normalize.
        normalization_basis = np.ones_like(orbits, dtype=bool)

        # Assume we have multiple embeddings of the same BQM.
        bqms = sampler_call.nominal_bqms
        if len(bqms) > 1 and any(bqm != bqms[0] for bqm in bqms[1:]):
            raise NotImplementedError("Case for distinct embedded BQMs not implemented yet.")

        bqm = bqms[0]
        nominal_values = np.array([bqm.quadratic[edge] for edge in self.inst.edge_list])
        coupler_signs = np.sign(nominal_values)
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

            # New truncation method... previous is buggy when we mix signs of nominal values.
            # Let's try being more explicit.
            for iemb in range(len(relative_coupler_strength)):
                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * nominal_values[bin_edges]
                    * energy_scale
                    > 1
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    0.99999 / nominal_values[bin_edges[violators]] / energy_scale
                )

                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * nominal_values[bin_edges]
                    * energy_scale
                    < -2
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    -1.99999 / nominal_values[bin_edges[violators]] / energy_scale
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

            # New truncation method... previous is buggy when we mix signs of nominal values.
            # Let's try being more explicit.
            for iemb in range(len(relative_coupler_strength)):
                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * nominal_values[bin_edges]
                    * energy_scale
                    > 1
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    0.99999 / nominal_values[bin_edges[violators]] / energy_scale
                )

                violators = (
                    relative_coupler_strength[iemb, bin_edges]
                    * nominal_values[bin_edges]
                    * energy_scale
                    < -2
                )
                relative_coupler_strength[iemb, bin_edges[violators]] = (
                    -1.99999 / nominal_values[bin_edges[violators]] / energy_scale
                )

        Q = nominal_values * relative_coupler_strength * energy_scale
        Q_max = np.max(Q)
        Q_min = np.min(Q)
        if Q_max > 1 or Q_min < -2:
            raise ValueError(
                "Effective coupler strengths violate hardware bounds: "
                f"min={Q_min:.6f}, max={Q_max:.6f}"
            )

    def _make_bqm(self, sampler_call: SamplerCall) -> dimod.BQM:
        """Construct a BQM for the current sampler call."""
        energy_scale = self.param["energy_scale"]
        bqm = dimod.BQM(vartype="SPIN")
        if not hasattr(self.inst, "embedding_list"):
            nominal_bqm = sampler_call.nominal_bqms[0]

            for v in range(self.inst.num_spins):
                # Make sure variables appear in the correct order when dealing with software solvers
                bqm.add_variable(v)
                if v in nominal_bqm.variables:
                    bqm.add_linear(v, nominal_bqm.linear[v])

            for iedge, edge in enumerate(self.inst.edge_list):
                bqm.add_quadratic(edge[0], edge[1], nominal_bqm.quadratic[*edge] * energy_scale)

            return bqm

        relative_coupler_strength = sampler_call.shimdata["relative_coupler_strength"]
        for iemb, emb in enumerate(self.inst.embedding_list):
            nominal_bqm = sampler_call.nominal_bqms[iemb].copy()

            for v in range(self.inst.num_spins):
                # Don't touch degree-zero spins.  Relevant to partial yield.
                if nominal_bqm.degree(v) > 0:
                    bqm.add_linear(emb[v], nominal_bqm.linear[v])

            for iedge, edge in enumerate(self.inst.edge_list):
                bias = (
                    nominal_bqm.quadratic[*edge]
                    * relative_coupler_strength[iemb, iedge]
                    * energy_scale
                )
                bqm.add_quadratic(emb[edge[0]], emb[edge[1]], bias)

        return bqm
