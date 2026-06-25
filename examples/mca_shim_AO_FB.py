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
An example to show coarse-grained calibration refinement of flux_biases and anneal_offsets for multicolor annealing.
"""

import argparse
import hashlib
import json
import os
import re
from typing import Literal

import pickle
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import dimod
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    make_tds_graph,
    make_tds_x_schedules,
    make_tds_x_schedule_delays,
    qubit_to_Advantage2_annealing_line,
    SOLVER_FILTER,
)
from dwave.experimental.shimming import shim_flux_biases


def _figure_path(
    figures_dir: str, figure_label: str, cache_str: str | None = None
) -> str:
    """Create a stable file path for a matplotlib figure label."""
    safe_label = re.sub(r"[^0-9A-Za-z._-]+", "_", figure_label).strip("_")
    suffix = f"_{cache_str}" if cache_str else ""
    return os.path.join(figures_dir, f"{safe_label}{suffix}.png")


def _save_open_figures(figures_dir: str, cache_str: str | None = None) -> None:
    """Save currently open matplotlib figures to disk."""
    os.makedirs(figures_dir, exist_ok=True)
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        figure_label = fig.get_label() or f"Figure_{fig_num}"
        fig.savefig(_figure_path(figures_dir, figure_label, cache_str))




def _calc_anneal_offsets(
    frequencies: np.ndarray,
    psd: np.ndarray,
    expected_A: float,
    dAdc: float,
    dAfit: float = 0.5,
    fit_to_expected_A: bool = True,
):
    """Determine the anneal_offset necessary to synchronize frequency.

    After fully decoupling from the source, the signal is expected to be
    well described by a cosine subject to an exponentially decaying envelope,
    controlled by the T2 coherence time.
    The power spectral density is therefore a Lorentzian peaked at the
    oscillating frequency. The peak can be efficiently estimated with small
    bias as the expectation on a symmetric interval about the anticipated
    frequency. This is a simple heuristic estimator, reasonably robust to
    experimental non-idealities.

    root(A(c)^2 + [B(c) delta h]^2) is expected to describe the frequency,
    where delta h is non-zero owing to flux_biases. If flux biases are small
    we can correct the frequency accounting for a only using
    delta c = (A(c) - <A(c)>)/ [dA/dc], where dA/dc is known approximately
    from the schedule.

    Args:
        frequencies: Frequencies at which power is provided (GHz).
        psd: Power spectral density, the absolute discrete fourier transform
            value squared at each frequency.
        expected_A: Expected/desired peak position (GHz).
        dAdc: Approximate rate of change of A with c (anneal offset).
        dAfit: Fractional range around expected_A to use for estimation.
        fit_to_expected_A: If True, use expected_A as the reference; if False, use the mean.

    Returns:
        Estimated anneal offsets per qubit/embedding in units of normalized c.
    """

    # NB a symmetric window only works for frequencies in the range,
    # and some bias is introduced by use of a target.
    Amin = expected_A * dAfit
    Amax = expected_A * (1 + dAfit)

    Afilter = np.logical_and(frequencies < Amax, frequencies > Amin)
    mean_A_est = np.sum(
        psd[:, Afilter] * frequencies[Afilter][np.newaxis, :], axis=1
    ) / np.sum(psd[:, Afilter], axis=1)
    mu = np.mean(mean_A_est)
    print(
        "A: target, estimated_mean, and standard deviation",
        expected_A,
        np.mean(mean_A_est),
        np.std(mean_A_est),
    )
    if fit_to_expected_A:
        mu = expected_A
    print()
    dcs = (mean_A_est - mu) / dAdc
    return dcs


def make_y(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs=-1
) -> np.ndarray:
    """Make a noise-free model signal

    y(delays) = min(1, np.exp(-delays / T2)) * np.cos(2* np.pi * A * delays)

    Args:
        delays: time(s) of measurement
        A: frequency
        T2: exponential envelope time scale. Defaulted as
            from T_phi = 12ns and T1 = 32ns, typical of Advantage2 research.
    Returns:
        A model signal
    """
    return sign_Jts_fbs * (
        (delays < 0)
        + (delays > 0) * np.exp(-delays / T2) * np.cos(2 * np.pi * A * delays)
    )


def dy_dt0(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs=-1
) -> list[np.ndarray]:
    """Calculate derivative of signal model with respect to time delay.

    Computes the derivative of the model signal y(t) = exp(-t/T2) * cos(2*pi*A*t)
    with respect to the time delay parameter.

    Args:
        delays: Time delays at which to evaluate the derivative (microseconds).
        A: Frequency (GHz).
        T2: Exponential envelope time scale (microseconds).
        sign_Jts_fbs: Sign convention for the Josephson coupling term (default: -1).

    Returns:
        Array of derivatives evaluated at each delay.
    """
    y0 = np.clip(np.exp(-delays / T2), a_min=0, a_max=1)
    dy0_dt0 = -y0 / T2 * (delays > 0)  # Only contributes when not clipped
    y1 = np.cos(2 * np.pi * A * delays)
    dy1_dt0 = -2 * np.pi * A * np.sin(2 * np.pi * A * delays)

    return sign_Jts_fbs * (dy0_dt0 * y1 + y0 * dy1_dt0)


def dy_dA(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs=-1
) -> list[np.ndarray]:
    """Calculate derivative of signal model with respect to frequency.

    Computes the derivative of the model signal y(t) = exp(-t/T2) * cos(2*pi*A*t)
    with respect to the frequency parameter A.

    Args:
        delays: Time delays at which to evaluate the derivative (microseconds).
        A: Frequency (GHz).
        T2: Exponential envelope time scale (microseconds).
        sign_Jts_fbs: Sign convention for the Josephson coupling term (default: -1).

    Returns:
        Array of frequency derivatives evaluated at each delay.
    """
    y0 = np.clip(np.exp(-delays / T2), a_min=0, a_max=1)
    dy1_dA = -2 * np.pi * delays * np.sin(2 * np.pi * A * delays)

    return sign_Jts_fbs * (y0 * dy1_dA)


def artificial_data(
    delays: np.ndarray,
    A: float,
    T2: float = 0.0101,
    num_independent_samples: int = float("Inf"),
    prng: np.random.Generator | int | None = None,
):
    """Create an artificial data set

    with variance of (1 - y(t)^2) in the measured state. Given independent
    and identically distributed samples we can model noise as normally
    distributed.

    Args:
        delays: Time of measurement (microseconds).
        A: Frequency (GHz).
        T2: Exponential envelope time scale (microseconds).
        num_independent_samples: Number of samples to model.
        prng: Pseudo random number generator or seed.

    Returns:
        A model signal with sampling noise.
    """
    y = make_y(delays, A, T2)
    if num_independent_samples != float("Inf"):
        prng = np.random.default_rng(prng)
        return y + np.sqrt((1 - y**2) / num_independent_samples) * prng.normal(
            size=len(y)
        )
    else:
        return y


def run_parallel_experiment(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    qpu_parameters: dict,
    delays: np.ndarray | list,
    line_detector: int,
) -> np.ndarray:
    """Collect detector magnetization for a set of independent embeddings

    See documentation example, here we simply parallelize.

    Args:
        sampler: Parallel embedding composite sampler wrapping the QPU sampler.
        bqm: Binary Quadratic Model.
        qpu_parameters: Parameters passed to the QPU sampler.
        delays: Detector x_schedule_delays (microseconds).
        line_detector: Detector line index.

    Returns:
        Numpy array of detector magnetizations (delays x embeddings).
    """
    delay0 = qpu_parameters["x_schedule_delays"][line_detector]
    mean_Z_detector = []
    for delay in tqdm(delays):
        qpu_parameters["x_schedule_delays"][line_detector] = delay0 + delay
        # Return as a list of samplesets, instead of aggregated:
        samplesets, _ = sampler.sample_multiple(
            [bqm] * len(sampler.embeddings), **qpu_parameters
        )
        # Extract detector magnetization from each sampleset
        detector_samples = [
            dimod.keep_variables(sampleset, [("detector", 0)]).record.sample
            for sampleset in samplesets
        ]
        mean_Z_detector.append([np.mean(sample) for sample in detector_samples])
    qpu_parameters["x_schedule_delays"][line_detector] = delay0
    return np.array(mean_Z_detector)


def plot_shim(
    mag_history: dict,
    flux_history: dict,
    num_experiments: int = 1,
    label: str = "",
    max_qubit_labels: int = 10,
):
    """Plot the iterative flux_bias_shim process.

    Args:
        mag_history: the magnetizations estimated throughout the iterative
            process for every embedding.
        flux_history: the flux_biases assignments throughout the iterative
            process for every embedding.
        num_experiments: Number of programmings per flux iteration. Using 1
            by default it should be noted that 2 magnetizations may be
            be measured per step in flux_biases.
        label: a label for the plots, used in legends.
        max_qubit_labels: maximum number of qubit labels to include in legend,
            if larger, defaults to no labels.
    """
    mag_array = np.array(list(mag_history.values()))
    flux_array = np.array(list(flux_history.values()))

    mag_array = np.reshape(
        mag_array,
        (mag_array.shape[0], mag_array.shape[1] // num_experiments, num_experiments),
    )

    plt.figure("Magnetization_by_shim_iteration")
    plt.title(r"Magnetization by iteration, $\langle Z\rangle_{detector}$")
    for experiment_sign in range(num_experiments):
        if num_experiments > 1:
            plt.plot(
                mag_array[:, :, experiment_sign].transpose(),
                label=f"Initial state all {-1 + 2*experiment_sign}",
            )
        else:
            plt.plot(
                mag_array[:, :, experiment_sign].transpose(),
            )
    if num_experiments > 1:
        plt.plot(
            np.mean(mag_array, axis=2).transpose(),
            color="black",
            label="Experiment average",
        )
        plt.legend()
        plt.xlabel("Shim iteration")
    else:
        plt.xlabel("Programming")
        if mag_array.shape[0] <= max_qubit_labels:
            plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    plt.ylabel("Magnetization")

    plt.figure("Flux_bias_by_shim_iteration")
    plt.title("All detector flux biases")
    plt.plot(flux_array.transpose())
    plt.xlabel("Shim iteration")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    if mag_array.shape[0] <= max_qubit_labels:
        plt.legend(flux_history.keys(), title=f"{label} Qubit index")


def _plot_tds_schedules(
    x_polarizing_schedule: list[list[float]],
    x_anneal_schedules: list[list[list[float]]],
):
    """Plots the piecewise linear schedules used

    Args:
        x_polarizing_schedule: The polarization signal.
        x_anneal_schedules: The list of anneal schedules, one per line.
    """
    plt.figure("PWL multi-color annealing schedules")
    plt.title("PWL schedules")
    for line, schedule in enumerate(x_anneal_schedules):
        plt.plot(
            [x for x, _ in schedule], [y for _, y in schedule], label=f"Line {line}"
        )
    plt.plot(
        [x for x, _ in x_polarizing_schedule],
        [y for _, y in x_polarizing_schedule],
        label="Polarizing schedule",
        linestyle="dashed",
        color="black",
    )
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Schedule value")
    plt.legend()


def imshow_data(
    mean_Z_detector: np.ndarray,
    delays: np.ndarray,
    colormap_type: Literal["default", "divergent"],
    first: int = 0,
    last: int | None = None,
    context_str: str = "",
):
    """Display detector magnetization data as a heatmap.

    Creates an image plot of detector magnetization values organized by delay times,
    with optional divergent or default colormaps.

    Args:
        mean_Z_detector: 2D array of detector magnetizations (delays x embeddings).
        delays: Array of time delay values (nanoseconds after conversion).
        colormap_type: Type of colormap to use ("default" or "divergent").
        first: Starting index for the plotted range.
        last: Ending index for the plotted range. If None, uses full array length.
        context_str: Optional context string to append to figure title.
    """
    fig_title = f"Timeseries_{colormap_type}_colormap{context_str}"
    if colormap_type == "divergent":
        vmin, vmax, cmap = -1, 1, "RdBu"
    else:
        vmin, vmax, cmap = None, None, None
    plt.figure(fig_title)
    plt.title(f"Real space magnetizations: {context_str}")
    plt.imshow(mean_Z_detector, vmin=vmin, vmax=vmax, cmap=cmap)
    if last is None:
        last = mean_Z_detector.shape[0]
    yticks_dict = {
        first: f"{1000 * delays[first]:.3g}",
        last - 1: f"{1000 * delays[last-1]:.3g}",
    }
    yticks_dict.update(
        {
            0: str(1000 * delays[0]),
            mean_Z_detector.shape[0] - 1: str(1000 * delays[-1]),
        }
    )
    plt.yticks(
        list(yticks_dict.keys()),
        list(yticks_dict.values()),
    )
    plt.xlabel("Target-Detector-Source embedding")
    plt.ylabel("Delay, nanoseconds")


def _get_experiment_id(args, num_char: int = 8, verbose=True):
    """Generate a unique hash identifier for the current experiment parameters.

    Creates a reproducible hash of the experiment arguments (excluding save_figures)
    to enable consistent caching and figure naming.

    Args:
        args: Argument namespace containing experiment parameters.
        num_char: Number of characters to use from the hash (default: 8).
        verbose: If True, print experiment parameters and identifier.

    Returns:
        Hash string identifier for the experiment.
    """
    if verbose:
        print()
        print("Demo parameters:")
        print(vars(args))
    vars_args = vars(args).copy()
    vars_args.pop(
        "save_figures", None
    )  # save_figures is not relevant to the experiment data, so we exclude it from the hash
    args_string = json.dumps(vars_args, sort_keys=True)
    identifier = hashlib.sha256(args_string.encode("utf-8")).hexdigest()[:num_char]
    if verbose:
        print("Demo identifier (labels cached data and saved figures):", identifier)
    return identifier


def _fix_standard_c_range(anneal_schedules):
    """Clip anneal schedule values to the standard [0, 1] range.

    Modifies schedule arrays in-place to ensure all control parameter values
    are within the valid normalized range [0, 1].

    Args:
        anneal_schedules: List of anneal schedules to clip. Schedules are modified in-place.
    """
    for anneal_schedule in anneal_schedules:
        for tc in anneal_schedule:
            if tc[1] > 1.0:
                tc[1] = 1.0
            elif tc[1] < 0.0:
                tc[1] = 0.0


def _plot_time_series(
    embs,
    line_assignments,
    mean_Z_detector,
    delays,
    line_color,
    plotted_emb_idxs=None,
    label_emb_idxs=None,
    xlabel="Delay, nanoseconds",
    ylabel="Detector magnetizations",
):
    """Plot time series data for selected embeddings with line-based coloring.

    Creates a line plot of detector magnetization or other signals across delay times,
    with each embedding colored by its assigned annealing line.

    Args:
        embs: List of embeddings, each containing qubit assignments.
        line_assignments: Dict mapping qubits to annealing line indices.
        mean_Z_detector: 2D array of data (delays x embeddings).
        delays: Array of delay time values.
        line_color: List of colors indexed by annealing line.
        plotted_emb_idxs: Set of embedding indices to plot. If None, plots all.
        label_emb_idxs: Set of embedding indices to label in legend. If None, uses plotted_emb_idxs.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """

    if plotted_emb_idxs is None:
        plotted_emb_idxs = set(range(len(embs)))
    if label_emb_idxs is None:
        label_emb_idxs = plotted_emb_idxs

    for emb_idx in plotted_emb_idxs:
        q = embs[emb_idx][0][0]
        line_idx = line_assignments[q]
        if emb_idx in label_emb_idxs:
            if len(label_emb_idxs) == len(plotted_emb_idxs):
                label = f"line {line_idx}(qubit {q})"
            else:
                label = f"line {line_idx}"
        else:
            label = None
        plt.plot(
            delays, mean_Z_detector[:, emb_idx], color=line_color[line_idx], label=label
        )
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid()


def main(
    cache_str: str | None = None,
    solver: dict | str | None = None,
    line_detector: int = 0,
    line_source: int = 3,
    seed: int | None = None,
    max_num_embeddings: int | None = None,
    target_c: float | None = None,
    expected_A: float | None = 1.33,
    skip_flux_bias_refinement: bool = False,
    verify_anneal_offsets: bool = True,
    delay_min: float = 0.005,
    delay_max: float = 0.015,
    delay_min_fit: float | None = None,
    delay_max_fit: float | None = None,
    fn_schedule: str = "09-1317A-D_Advantage2_research1_4_annealing_schedule.xlsx",
    use_01_c_range: bool = False,
    symmetrize_c_bounds: bool = True,
    num_reads: int = 500,
    save_figures: bool = False,
):
    """Demonstrate t-d-s variability and mitigation strategies

    An ideal single-qubit target system might be prepared in
    a polarized state |1> whose evolution is subsequently
    described by H(c) = A(c) + B(c) h.
    Control limitations dictate that the A(s), B(s) and h realized
    by different qubits at a common c varies. An h error in the detector
    qubit can also contribute to errors in measurement.
    Methods are demonstrated for synchronization of frequency with
    use of anneal offsets incorporating a simple decoherence model, and
    shimming of a detector flux bias to restore symmetry.

    Higher accuracy shimming, and shimming of target flux_biases may also be
    desirable, but are beyond the scope of the example. Note that we can
    use simple statistic to determine flux_bias assignment on a detector
    relative to a target. E.g. a) when decoupled from the source and detector
    a 1 qubit model frequency omega=root(A(s)^2 + B(s)^2 h^2) is a convex
    monotonic function of the linear field, b) When decoupled from the source
    the response of the detector magnetization to a flux_bias perturbation is
    maximized.

    Args:
        cache_str:
            A unique experimental identifier. If not None, a directory
            cache/ is created which is populated with experimental data. The cache
            is checked for compatible experimental data before running an experiment,
            and if compatible data is present the data is reloaded rather than
            running new jobs through the client.
        solver:
            Name of the solver, or dictionary of characteristics.
        line_detector:
            The integer index of the detector line.
        line_source:
            The integer index of the source line.
        seed:
            Random seed used for embedding generation.
        max_num_embeddings:
            Maximum number of embeddings to find. If None, search for all available embeddings.
        target_c:
            Normalized control bias at which the target qubits are held.
            Either expected_A or target_c should be specified, not both.
            target_c is inferred from the schedule and expected_A by default.
        expected_A:
            The expected qubit frequency in GHz.
            Either expected_A or target_c should be specified, not both.
            When None expected_A is inferred from target_c and the schedule.
        skip_flux_bias_refinement:
            When set to True, flux-bias refinement is skipped. When set to False,
            detector flux biases are refined to achieve zero expected magnetization
            at long delay.
        verify_anneal_offsets:
            When set to True, data is collected and analyzed with anneal_offsets applied.
            When set to False, this verification stage is skipped.
            Anneal_offsets are modified so that the peak power-spectral density is peaked at a common
            value for all qubits. This peak value characterizes the frequency of the target
            qubit in simple, well-calibrated models.
        delay_min: The delay on the detector line for which data is collected.
        delay_max: The maximum delay on the detector line for which data is collected. Between
            delay_min and delay_max the spacing in time reflects the target frequency that we
            are seeking to resolve for anneal_offset refinement.
        delay_min_fit:
            A lower bound on the timeseries window used for inference of the target power spectral density.
            A value that is too small can bias the estimator by introduction of effects related
            to coupling to the source line.
        delay_max_fit:
            An upper bound on the timeseries window used for inference of the target power spectral density.
            Too large a value reduces the efficiency of the estimator, since delays much larger than the
            T1 coherence time are dominated by noise.
        fn_schedule: A schedule file that is used to estimate an appropriate sampling interval for delay
            time and an appropriate scale for anneal_offset synchronization. This should be matched to the
            solver.
        use_01_c_range:
            When set to True, restricts the schedule range to [0,1]. This lowers the detector and source quench
            rates, impacting fidelity and some other parameters.
        save_figures:
            When True, save generated figures to a ``figures`` folder.
        num_reads:
            The number of reads to perform for each measurement.

    Raises:
        ValueError: If the number of lines is less than 3, or
        if {detector_line, source_line} is not a size 2 subset of
        set(range(num_lines))
    """
    print()
    print(
        "A variety of plots are shown to demonstrate heuristic correction of "
        "flux_biases on detectors, and target qubit frequency "
        "desynchronization, from small amounts of data. "
    )
    if delay_max_fit is None:
        delay_max_fit = delay_max  # Can be automated for SNR in principle.
    elif delay_max_fit > delay_max:
        raise ValueError("Fit window exceeds data window")
    if delay_min_fit is None:
        delay_min_fit = delay_min  # Can be automated for SNR in principle.
    elif delay_min_fit < delay_min:
        raise ValueError("Fit window exceeds data window")
    if delay_min_fit > delay_max_fit:
        raise ValueError("Fit window is empty")
    # Schedule based approximations, expected_A and dA/dc are approximated.
    qpu_anneal_schedule = pd.read_excel(
        fn_schedule, sheet_name="Fast-Annealing Schedule"
    )
    plt.figure("Schedule")
    plt.title("Annealing Schedule")
    delta_vs_s = qpu_anneal_schedule[::-1]
    plt.plot(delta_vs_s["s"], delta_vs_s["A(s) (GHz)"], label="A(s)")
    plt.plot(delta_vs_s["s"], delta_vs_s["B(s) (GHz)"], label="B(s)")
    if (target_c is None) == (expected_A is None):
        raise ValueError("Exactly one of target_c or expected_A must be specified.")
    if target_c is None:
        target_c = np.interp(
            expected_A, delta_vs_s["A(s) (GHz)"], delta_vs_s["s"]
        )  # Expected normalized control bias at which to hold the qubits
    else:
        expected_A = np.interp(
            1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
        )  # Expected frequency of detector magnetization oscillations
    target_B = np.interp(1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["B(s) (GHz)"])
    stage_idx = 0
    print()
    print(f"Stage {stage_idx}: Plot the annealing schedule.")
    print(
        "Schedule expectations: ",
        "target_c",
        target_c,
        " A(target_c)",
        expected_A,
        "GHz B(target_c)",
        target_B,
        "GHz",
    )
    dc = 0.01
    expected_Aminus = np.interp(
        1 - (target_c - dc), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )
    expected_Aplus = np.interp(
        1 - (target_c + dc), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )
    dAdc = (expected_Aplus - expected_Aminus) / (2 * dc)
    plt.plot(
        [target_c, target_c],
        [0, np.max(delta_vs_s["A(s) (GHz)"])],
        label=f"c={target_c}",
    )
    plt.plot(
        [0, target_c - 0.01],
        [expected_Aminus, expected_Aminus],
        linestyle="dotted",
        color="black",
    )
    plt.plot(
        [target_c - 0.01, target_c - 0.01],
        [0, expected_Aminus],
        linestyle="dotted",
        color="black",
    )
    plt.plot(
        [0, target_c + 0.01],
        [expected_Aplus, expected_Aplus],
        linestyle="dotted",
        color="black",
    )
    plt.plot(
        [target_c + 0.01, target_c + 0.01],
        [0, expected_Aplus],
        linestyle="dotted",
        color="black",
    )
    plt.xlabel("Normalized control bias, c")
    plt.ylabel("Energy scale, GHz")
    plt.ylim([0, 2 * max(expected_A, target_B)])
    plt.xlim([0, 1])
    plt.legend()

    solver_props_cache = f"{solver}_properties.pkl"
    solver_efi_cache = f"{solver}_exp_feature_info.pkl"
    try:
        qpu = DWaveSampler(solver=solver)
        exp_feature_info = get_properties(qpu)
        with open(solver_props_cache, "wb") as f:
            pickle.dump(qpu.properties, f)
        with open(solver_efi_cache, "wb") as f:
            pickle.dump(exp_feature_info, f)
        online = True
    except Exception as error:
        print(
            "Connection to QPU error, if use_cache=True previously saved data will still be "
            "loaded and processed."
        )
        if not (
            os.path.isfile(solver_props_cache) and os.path.isfile(solver_efi_cache)
        ):
            raise FileNotFoundError(
                f"Fallback pickle cache files are missing: "
                f"{solver_props_cache}, {solver_efi_cache} and "
                f"{error}"
            )
        with open(solver_props_cache, "rb") as f:
            properties = pickle.load(f)
        qpu = MockDWaveSampler(
            properties=properties,
            nodelist=properties["qubits"],
            edgelist=properties["couplers"],
        )
        with open(solver_efi_cache, "rb") as f:
            exp_feature_info = pickle.load(f)
        online = False
    if len(exp_feature_info) != 2:
        raise ValueError('Legacy format')
    zephyr_shape = qpu.properties["topology"]["shape"]
    line_assignments = {
        n: al_idx for al_idx, al in enumerate(exp_feature_info[1]) for n in al["qubits"]
    }
    num_lines = len(exp_feature_info[1])
    target_lines = set(range(num_lines)) - {line_detector, line_source}
    cmap = plt.colormaps.get_cmap("plasma")
    line_color = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    detector_lines = (line_detector,)
    source_lines = (line_source,)
    x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=target_lines,
        target_c=target_c,
        detector_lines=detector_lines,
        source_lines=source_lines,
        use_01_c_range=use_01_c_range,
        symmetrize_c_bounds=symmetrize_c_bounds,
    )
    _plot_tds_schedules(
        x_polarizing_schedule,
        x_anneal_schedules,
    )
    x_schedule_delays = make_tds_x_schedule_delays(
        x_anneal_schedules=x_anneal_schedules,
        quenched_lines=detector_lines + source_lines,
        target_c=target_c,
        decimal_places=6
    )
    x_schedule_delays = [0.0]*num_lines
    dt = 1 / expected_A / 1000 / 4  # Appropriate scale for frequency resolution.
    delays = np.linspace(
        delay_min, delay_max, round((delay_max - delay_min) / dt) + 1, endpoint=True
    )
    dt = delays[1] - delays[0]  # Rounding

    dt_hd = 0.00001  # 0.01 nanoseconds
    high_density_delays = np.linspace(
        delay_min, delay_max, round((delay_max - delay_min) / dt_hd), endpoint=False
    )
    dt_hd = high_density_delays[1] - high_density_delays[0]  # Rounding
    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Demonstrate simple model data.")
    print("Model: y(t) = cos(2*pi*(A+dA)*(t+dt)) * exp(-gamma*(t+dt))")
    print("Includes sampling error, detector vs source delays, and perturbations.")
    delay_perturbation = 0.002 * np.random.random()
    for idx, A in enumerate([expected_Aminus, expected_A, expected_Aplus]):
        label = f"A={A:.3g}"
        signal = artificial_data(
            delays + delay_perturbation,
            A * 1000,
            num_independent_samples=num_reads,
        )
        high_density_signal = artificial_data(
            high_density_delays + delay_perturbation,
            A * 1000,
            num_independent_samples=float("Inf"),  # No noise
        )
        fig = plt.figure("artificial_timeseries")
        next_color = fig.gca()._get_lines.get_next_color()
        plt.title("y=cos(2pi A t)exp(-t/T)+sampling error")
        plt.plot(
            (delays + delay_perturbation) * 1000,
            signal,
            label=label,
            marker=".",
            linestyle=None,
            color=next_color,
        )
        plt.plot(
            (high_density_delays + delay_perturbation) * 1000,
            high_density_signal,
            linestyle="dotted",
            color=next_color,
        )
        plt.xlabel("Time, nanoseconds")
        plt.ylabel(r"Magnetization, $\langle y \rangle_{detector}$")
        plt.legend()

        plt.figure("artificial_psd")
        psd_title = "Approximate Lorentzian PSD ~ A/((f-A)^2 + A^2)"
        plt.title(psd_title)
        frequencies = np.arange(len(delays) // 2) / dt / len(delays) / 1000
        psd = np.abs(np.fft.fft(signal)) ** 2 / len(signal) ** 2
        frequencies_hd = (
            np.arange(len(high_density_delays) // 2)
            / dt_hd
            / len(high_density_delays)
            / 1000
        )
        psd_hd = (
            np.abs(np.fft.fft(high_density_signal)) ** 2 / len(high_density_signal) ** 2
        )
        plt.plot(
            frequencies,
            psd[: len(psd) // 2],
            label=label,
            marker=".",
            linestyle=None,
            color=next_color,
        )
        plt.plot(
            frequencies_hd,
            psd_hd[: len(psd_hd) // 2],
            linestyle="dotted",
            color=next_color,
        )
        plt.ylabel(r"Power Spectral Density, $|\langle {\hat y}\rangle(\omega)|^2$")
        plt.xlabel(r"Frequency ($\omega$), GHz")
        plt.xlim([0, frequencies[-1]])
        plt.legend()

    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next (experimental) stages.")
    plt.show()

    anneal_offsets = [0.0] * qpu.properties["num_qubits"]
    flux_biases = [0.0] * qpu.properties["num_qubits"]
    qpu_parameters = dict(
        num_reads=num_reads,
        answer_mode="raw",
        x_disable_filtering=True,
        x_schedule_delays=x_schedule_delays,
        x_anneal_schedules=x_anneal_schedules,
        x_polarizing_schedule=x_polarizing_schedule,
        flux_biases=flux_biases,
        anneal_offsets=anneal_offsets,
    )

    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Find T-D-S embeddings for parallel programming")
    print("(see mca_embedding.py example).")
    T = qpu.to_networkx_graph()

    def _target_assignments(n: int):
        line = line_assignments[n]
        if line == line_detector:
            return "detector"
        elif line == line_source:
            return "source"
        else:
            return "target"

    Tnode_to_tds = {n: _target_assignments(n) for n in qpu.nodelist}
    target_graph = nx.Graph()
    target_graph.add_node(0)
    S, Snode_to_tds = make_tds_graph(target_graph)
    subgraph_kwargs = dict(node_labels=(Snode_to_tds, Tnode_to_tds), as_embedding=True)
    fn_cache = f"cache/emb_{cache_str}.pkl"
    if cache_str:
        os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
    if cache_str and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            embs = pickle.load(f)
    else:
        embs = find_multiple_embeddings(
            S,
            T,
            max_num_emb=max_num_embeddings,
            embedder_kwargs=subgraph_kwargs,
            one_to_iterable=True,
            seed=seed,
        )
        with open(fn_cache, "wb") as f:
            pickle.dump(embs, f)
    # Reorder by target line for ease of analysis:
    embs_by_line = {i: [] for i in target_lines}
    for i, emb in enumerate(embs):
        q = emb[0][0]
        embs_by_line[line_assignments[q]].append(emb)
    plotted_emb_idxs = np.cumsum(
        [len(embs_by_line[i]) for i in target_lines if len(embs_by_line[i]) > 0]
    )
    plotted_emb_idxs -= plotted_emb_idxs[0]
    embs = [emb for i in target_lines for emb in embs_by_line[i]]

    sampler = ParallelEmbeddingComposite(qpu, embeddings=embs)

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes()}, {e: -1 for e in S.edges()}
    )
    if not skip_flux_bias_refinement:
        stage_idx += 1
        print()
        print(
            f"Stage {stage_idx}: Shim flux biases for zero detector magnetization in"
            " the limit of long delay (at equilibrium). "
            "This requires 10-20 programmings by default."
        )
        fn_cache = f"cache/FB_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                flux_biases, flux_history, mag_history = pickle.load(f)
        else:
            if not online:
                raise (RuntimeError, "QPU not available, and no cached data found.")
            # Require zero magnetization in the limit of long delay (where
            # source impact has decayed away.
            bqm_embedded = dimod.BinaryQuadraticModel("SPIN").from_ising(
                {emb[n][0]: h for emb in embs for n, h in bqm.linear.items()},
                {
                    tuple(emb[n][0] for n in e): J
                    for emb in embs
                    for e, J in bqm.quadratic.items()
                },
            )
            shimmed_variables = {
                n
                for n in bqm_embedded.variables
                if qubit_to_Advantage2_annealing_line(
                    n, zephyr_shape, num_lines=num_lines
                )
                == line_detector
            }
            delay0 = qpu_parameters["x_schedule_delays"][line_detector]
            qpu_parameters["x_schedule_delays"][
                line_detector
            ] = 0.1  # Sources should be depolarized during measurement (equilibrium)
            flux_biases, flux_history, mag_history = shim_flux_biases(
                bqm=bqm_embedded,
                sampler=qpu,
                sampling_params=qpu_parameters,
                shimmed_variables=shimmed_variables,
            )
            qpu_parameters["x_schedule_delays"][line_detector] = delay0

            if cache_str:
                os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
                with open(fn_cache, "wb") as f:
                    pickle.dump((flux_biases, flux_history, mag_history), f)
        plot_shim(
            mag_history,
            flux_history,
        )
        qpu_parameters["flux_biases"] = flux_biases
        print("flux_biases refinement complete.")
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next (experimental) stages.")
        plt.show()
    stage_idx += 1
    print()
    n_embs = len(embs)
    print(f"Stage {stage_idx}: Collect data for {n_embs} parallel embeddings.")
    print(
        "Apply delays on detector line vs source, sampling at twice the Nyquist frequency."
    )
    fn_cache = f"cache/AO_It0_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
        if not online:
            raise (RuntimeError, "QPU not available, and no cached data found.")

        mean_Z_detector = run_parallel_experiment(
            sampler, bqm, qpu_parameters, delays, line_detector
        )
        if cache_str:
            os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
            np.save(fn_cache, mean_Z_detector)

    first = np.argmax(delays >= delay_min_fit)
    last = np.argmax(delays >= delay_max_fit) + 1
    ld = last - first
    if ld < 1:
        raise ValueError("Fit window is empty: t-fit range too small for expected_A")

    frequencies = np.arange(ld) / dt / 1000 / ld  # GHz
    psd = np.array(
        [
            np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
            for i in range(len(embs))
        ]
    ) / (last - first)

    print(
        "Plotting real-space data and power spectral density (discrete Fourier transform)."
    )

    plt.figure("Timeseries")
    plt.title("Time series for several qubits using distinct target lines")
    _plot_time_series(
        embs,
        line_assignments,
        mean_Z_detector,
        delays * 1000,
        line_color,
        plotted_emb_idxs=plotted_emb_idxs,
        label_emb_idxs=plotted_emb_idxs,
    )
    for colormap_type in ["divergent", "default"]:
        imshow_data(
            mean_Z_detector=mean_Z_detector,
            delays=delays * 1000,
            colormap_type=colormap_type,
            first=first,
            last=last,
        )
    plt.figure("PSD")
    plt.title("Power associated with magnetization time series")
    _plot_time_series(
        embs,
        line_assignments,
        psd[:, : ld // 2].T,
        frequencies[: ld // 2],
        line_color,
        label_emb_idxs=plotted_emb_idxs,
        xlabel=r"Frequency ($\omega$), GHz",
        ylabel=r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$",
    )
    plt.plot(
        [expected_A, expected_A],
        [0, np.max(psd)],
        color="black",
        linestyle="dashed",
        label="Schedule prediction",
    )

    # Calculate anneal_offsets for synchronization
    anneal_offsets = y = _calc_anneal_offsets(
        frequencies, psd, expected_A, dAdc
    )  # Per embedding
    # anneal offsets can be realized line-wise by changing target_c,
    # or qubit-wise by modification of anneal_offset. We can
    # correct for the mean with a line_offset, and then qubit-wise
    # variation with the anneal offset.

    plt.figure("Proposed anneal_offsets")

    plt.plot(sorted(y), np.arange(len(y)) / len(y))
    plt.xlabel(
        f"Proposed anneal offset, RMS(A0)={np.sqrt(np.mean(np.array(y)**2)):.3g}"
    )
    plt.ylabel("Cumulative distribution function")
    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next (experimental) stages.")
    plt.show()
    if verify_anneal_offsets:
        stage_idx += 1
        print()
        print(
            f"Stage {stage_idx}: Rerun time series with estimated anneal offsets applied."
        )
        fn_cache = f"cache/AO_It1_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            mean_Z_detector = np.load(fn_cache)
        else:
            if not online:
                raise (RuntimeError, "QPU not available, and no cached data found.")

            for emb, ao in zip(embs, anneal_offsets):
                qpu_parameters["anneal_offsets"][
                    emb[0][0]
                ] -= ao  # Apply correction to target on each embedding
            mean_Z_detector = run_parallel_experiment(
                sampler, bqm, qpu_parameters, delays, line_detector
            )
            if cache_str:
                np.save(fn_cache, mean_Z_detector)
            if save_figures:
                _save_open_figures("figures/", cache_str)
            print("Close figures to end.")
            plt.show()
        psd = np.array(
            [
                np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
                for i in range(len(embs))
            ]
        ) / (last - first)

        print(
            "Plotting real-space data and power spectral density (discrete Fourier transform)."
        )

        plt.figure("Timeseries_after_anneal_offsets")
        plt.title("Time series after anneal offsets")
        for emb_idx in plotted_emb_idxs:
            q = embs[emb_idx][0][0]
            line_idx = line_assignments[q]
            plt.plot(
                delays * 1000,
                mean_Z_detector[:, emb_idx],
                color=line_color[line_idx],
                label=f"line {line_idx}; q={q}",
            )
        plt.ylabel("Detector magnetizations")
        plt.xlabel("Detector delay, ns")
        plt.legend()
        plt.grid()
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next (experimental) stages.")
        plt.show()
        for colormap_type in ["divergent", "default"]:
            imshow_data(
                mean_Z_detector=mean_Z_detector,
                delays=delays,
                colormap_type=colormap_type,
                first=first,
                last=last,
                context_str="after anneal offsets",
            )

        plt.figure("PSD_w_AO")
        plt.title(
            "Power associated with magnetization time series after anneal offsets"
        )
        for emb_idx, emb in enumerate(embs):
            q = emb[0][0]
            line_idx = line_assignments[q]
            if emb_idx in plotted_emb_idxs:
                label = f"target-qubit line={line_idx}"
            else:
                label = None
            plt.plot(
                frequencies[: ld // 2],
                psd[emb_idx, : ld // 2],
                color=line_color[line_idx],
                label=label,
            )
        plt.plot(
            [expected_A, expected_A],
            [0, np.max(psd)],
            color="black",
            linestyle="dashed",
            label="Schedule prediction",
        )
        plt.legend()
        plt.ylabel(r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
        plt.xlabel(r"Frequency ($\omega$), GHz")
        plt.grid(True)

        plt.figure("AnnealOffsets")
        anneal_offsets0 = anneal_offsets
        anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, expected_A, dAdc
        )  # Per embedding
        for emb_idx, emb in enumerate(embs):
            q = emb[0][0]
            line_idx = line_assignments[q]
            if emb_idx in plotted_emb_idxs:
                label = f"target-qubit line={line_idx}"
            else:
                label = None

            plt.plot(
                anneal_offsets0[emb_idx],
                anneal_offsets[emb_idx],
                color=line_color[line_idx],
                marker="x",
                label=label,
            )
        plt.xlabel(
            "Frequency discrepancy (proposed c-<c> change) before anneal_offset shim"
        )
        plt.ylabel(
            "Frequency discrepancy (proposed c-<c> change) after anneal_offset shim"
        )
        plt.grid(True)
        plt.legend()

    if save_figures:
        _save_open_figures("figures", cache_str)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A target-detector-source embedding example"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Save/reload experimental data for current parameters. Note: QPU is "
        "identified only by solver parameters; if graph_id changes, new "
        "embeddings may be required.",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="QPU solver name. Default: experimental system with fast reverse anneal.",
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--line_detector",
        type=int,
        help="Detector line",
        default=0,  # First vertical qubit line
    )
    parser.add_argument(
        "--line_source",
        type=int,
        help="Source line",
        default=3,  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for embedding generation.",
        default=None,
    )
    parser.add_argument(
        "--max_num_embeddings",
        type=int,
        help="Max embeddings to find (default: all available).",
        default=None,
    )
    parser.add_argument(
        "--expected_A",
        type=float,
        help="Expected qubit frequency (GHz). Schedule infers corresponding target_c.",
        default=1.33,
    )
    parser.add_argument(
        "--schedule_fn",
        type=str,
        help="Annealing schedule filename used to infer target_c and dA/dc.",
        default="09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    )
    parser.add_argument(
        "--delay_min",
        type=float,
        help="Initial delay time (us) for data collection",
        default=0.0,
    )
    parser.add_argument(
        "--delay_max",
        type=float,
        help="Final delay time (us) for data collection.",
        default=0.01,
    )
    parser.add_argument(
        "--delay_min_fit",
        type=float,
        help="Initial delay (us) for frequency estimation (default: matches delay_min). "
        "Choose smallest delay with non-polarized signal.",
        default=None,
    )
    parser.add_argument(
        "--delay_max_fit",
        type=float,
        help="Final delay (us) for frequency estimation (default: matches delay_max). "
        "Choose largest delay with low noise.",
        default=None,
    )
    parser.add_argument(
        "--skip_flux_bias_refinement",
        action="store_true",
        help="Skip flux-bias refinement (run with zero flux biases).",
    )
    parser.add_argument(
        "--skip_anneal_offset_verification",
        action="store_true",
        help="Skip the data analysis stage with anneal offsets applied.",
    )
    parser.add_argument(
        "--use_01_c_range",
        action="store_true",
        help="Restrict schedule to [0, 1] range. Lowers quench rates, affecting fidelity. "
        "TODO: add symmetrized or overshoot c-range options.",
    )
    parser.add_argument(
        "--no_symmetrize_c_bounds",
        "--no-symmetrize-c-bounds",
        dest="symmetrize_c_bounds",
        action="store_false",
        default=True,
        help="Disable symmetric c-bounds.",
    )
    parser.add_argument(
        "--save_figures",
        action="store_true",
        help="Save figures to figures/ folder with hash-based names.",
    )

    args = parser.parse_args()

    if args.use_cache:
        cache_str = _get_experiment_id(args, num_char=8)
    else:
        cache_str = None
    main(
        cache_str=cache_str,
        solver=args.solver_name,
        line_detector=args.line_detector,
        line_source=args.line_source,
        expected_A=args.expected_A,
        fn_schedule=args.schedule_fn,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        delay_min_fit=args.delay_min_fit,
        delay_max_fit=args.delay_max_fit,
        verify_anneal_offsets=not args.skip_anneal_offset_verification,
        skip_flux_bias_refinement=args.skip_flux_bias_refinement,
        use_01_c_range=args.use_01_c_range,
        symmetrize_c_bounds=args.symmetrize_c_bounds,
        save_figures=args.save_figures,
    )
