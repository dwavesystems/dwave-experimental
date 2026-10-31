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
An example to show coarse-grained calibration refinement for multi-color annealing.

Calibration can be refined relative to the baseline since certain static or low
frequency control errors are a function of the specific waveforms and programmed
Hamiltonian.
flux_biases, x_schedule_delays and anneal_offsets are refined sequentially by simple iterative methods
that typically succeed to improve calibration on a plurality of qubits.

This example builds many parallel target-detector-source (T-D-S) embeddings that
are refined in parallel.
Optionally, the embeddings can be made consistent with embedding of a loop,
in which case the final plot demonstrates an interference pattern arising from
pi/2-pulse initialization (up to limitations of decoherence and control error).
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
import re
import warnings
from collections.abc import Collection, Iterable, Sequence
from typing import Any, Literal

import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import networkx as nx
import numpy as np
from tqdm import tqdm

import dimod
from dimod.typing import Variable
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.graphs import draw_parallel_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    line_assignments_from_feature_info,
    make_tds_graph,
    make_tds_x_schedules,
    SOLVER_FILTER,
)
from dwave.experimental.shimming import shim_tds_flux_biases

# Re-exported so ``from mca_experiment_helpers.common import *`` makes every name
# used by the verbatim stage bodies available (including the underscore-prefixed
# helpers and the third-party modules/functions those bodies reference).
__all__ = [
    # third-party modules and helpers used by stage bodies
    "defaultdict",
    "hashlib",
    "json",
    "os",
    "re",
    "pickle",
    "plt",
    "SymLogNorm",
    "nx",
    "np",
    "tqdm",
    "dimod",
    "Variable",
    "DWaveSampler",
    "MockDWaveSampler",
    "ParallelEmbeddingComposite",
    "find_multiple_embeddings",
    "draw_parallel_embeddings",
    "get_properties",
    "line_assignments_from_feature_info",
    "make_tds_graph",
    "make_tds_x_schedules",
    "SOLVER_FILTER",
    "shim_tds_flux_biases",
    # module helpers
    "_figure_path",
    "_apply_tight_layout",
    "_save_open_figures",
    "_calc_anneal_offsets",
    "run_parallel_experiment",
    "plot_shim",
    "_plot_tds_schedules",
    "imshow_data",
    "_get_experiment_id",
    "_plot_time_series",
    "estimate_decoupling_timescale",
    "_to_independent_tds",
    "compute_delay_grid",
    "create_solver",
    "connect_qpu",
    "MCAContext",
]


def _figure_path(
    figures_dir: str, figure_label: str, cache_str: str | None = None
) -> str:
    """Create a stable file path for a matplotlib figure label."""
    safe_label = re.sub(r"[^0-9A-Za-z._-]+", "_", figure_label).strip("_")
    suffix = f"_{cache_str}" if cache_str else ""
    return os.path.join(figures_dir, f"{safe_label}{suffix}.png")


def _apply_tight_layout() -> None:
    """Apply tight_layout to all open figures so axis labels are not truncated."""
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        # Skip figures already using a layout engine (e.g. constrained) to avoid
        # the "figure layout has changed to tight" warning.
        if fig.get_layout_engine() is None:
            fig.tight_layout()


def _save_open_figures(figures_dir: str, cache_str: str | None = None) -> None:
    """Save currently open matplotlib figures to disk."""
    os.makedirs(figures_dir, exist_ok=True)
    _apply_tight_layout()
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        figure_label = fig.get_label() or f"Figure_{fig_num}"
        fig.savefig(_figure_path(figures_dir, figure_label, cache_str))


def _calc_anneal_offsets(
    frequencies: np.ndarray,
    psd: np.ndarray,
    target_A: float,
    dAdc: float,
    dAfit: float = 0.5,
    fit_to_target_A: bool = True,
) -> np.ndarray:
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
        target_A: Expected/desired peak position (GHz).
        dAdc: Approximate rate of change of A with c (anneal offset).
        dAfit: Fractional range around target_A to use for estimation.
        fit_to_target_A: If True, use target_A as the reference; if False, use the mean.

    Returns:
        Estimated anneal offsets per qubit/embedding in units of normalized c.
    """

    # NB a symmetric window only works for frequencies in the range,
    # and some bias is introduced as a function of the window.
    Amin = target_A * dAfit
    Amax = target_A * (1 + dAfit)

    Afilter = np.logical_and(frequencies < Amax, frequencies > Amin)
    mean_A_est = np.sum(
        psd[:, Afilter] * frequencies[Afilter][np.newaxis, :], axis=1
    ) / np.sum(psd[:, Afilter], axis=1)
    mu = np.mean(mean_A_est)
    print(
        f"A: target={target_A:.3g}, estimated_mean={np.mean(mean_A_est):.3g}, and standard deviation={np.std(mean_A_est):.3g}",
    )
    if fit_to_target_A:
        mu = target_A
    dcs = (mean_A_est - mu) / dAdc
    return dcs


def run_parallel_experiment(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    sampling_params: dict,
    detector_lines: Iterable[int],
    *,
    delays: np.ndarray | Sequence[float] = (0.0,),
    detected_vars: None | Sequence[Variable] = None,
) -> np.ndarray:
    """Collect detector magnetization for a set of independent embeddings

    Runs a Target-Detector-Source quench experiment on many parallel
    embeddings with the specified delays applied to detector lines.
    Sample averaged magnetization are calculated on detected qubits in
    each embedding and returned as a numpy array.

    Args:
        sampler: Parallel embedding composite sampler wrapping the QPU sampler.
        bqm: Binary Quadratic Model.
        sampling_params: Parameters passed to the QPU sampler.
        delays: Detector x_schedule_delays (microseconds).
        detector_lines: Iterable of detector line indices.
        detected_vars: Iterable of bqm variables to keep for
           detector magnetization calculation.

    Raises:
        ValueError if 'x_anneal_schedules' is not a key of
        `sampling_params`

    Returns:
        Numpy array of detector magnetizations (delays x embeddings).
    """
    if "x_anneal_schedules" not in sampling_params:
        raise ValueError("No multi-color anneal specified")

    if detected_vars is None:
        # Could check line assignment of qubits (should be consistent between
        # embeddings)
        detected_vars = [
            v for v in bqm.variables if type(v) == tuple and v[0] == "detector"
        ]

    reset_delay = "x_schedule_delays" in sampling_params
    x_schedule_delays = sampling_params.pop(
        "x_schedule_delays", [0.0] * len(sampling_params["x_anneal_schedules"])
    )
    baseline_delays = x_schedule_delays.copy()
    mean_Z_detector = []
    for delay in tqdm(delays, disable=len(delays) == 1):
        for line in detector_lines:
            x_schedule_delays[line] = baseline_delays[line] + delay
        # Return as a list of samplesets, instead of aggregated:
        samplesets, _ = sampler.sample_multiple(
            [bqm] * len(sampler.embeddings),
            x_schedule_delays=x_schedule_delays,
            **sampling_params,
        )
        # Extract detector magnetization from each sampleset
        detector_samples = [
            dimod.keep_variables(sampleset, detected_vars).record.sample
            for sampleset in samplesets
        ]
        if len(detected_vars) == 1:
            mean_Z_detector.append([np.mean(sample) for sample in detector_samples])
        else:
            mean_Z_detector.append(
                [np.mean(sample, axis=0) for sample in detector_samples]
            )
    if reset_delay:
        sampling_params["x_schedule_delays"] = baseline_delays
    return np.array(mean_Z_detector)


def plot_shim(
    mag_history: dict,
    flux_history: dict,
    num_experiments: int = 1,
    label: str = "",
    max_qubit_labels: int = 10,
    plt_show_block: None | bool = None,
    line_assignments: dict[int, int] | None = None,
) -> None:
    """Plot the iterative flux-bias calibration refinement process.

    Args:
        mag_history: the magnetizations estimated throughout the iterative
            process for every embedding. One per programming.
        flux_history: the flux_biases assignments throughout the iterative
            process for every embedding. One per flux iteration.
        num_experiments: Number of programmings per flux iteration. Using 1
            by default it should be noted that 2 magnetizations may be
            measured per step in flux_biases.
        label: a label for the plots, used in legends.
        max_qubit_labels: maximum number of qubit labels to include in legend,
            if larger, defaults to no labels. Ignored when ``line_assignments``
            is provided.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
        line_assignments: Optional mapping from qubit to annealing line index.
            When provided, qubit traces are drawn one line at a time, each line
            with a distinct color and a single legend entry. When None, every
            qubit is drawn with the default color cycle (current behavior).
    """
    line_to_color: dict = {}
    if line_assignments is None:
        # All values share a length, so a single cast is safe.
        line_to_color = {None: None}
        line_groups = [(None, list(mag_history.keys()), list(flux_history.keys()))]
    else:
        lines_sorted = sorted(
            {line_assignments[q] for q in set(mag_history) | set(flux_history)}
        )
        cmap = plt.colormaps.get_cmap("plasma")
        denom = max(len(lines_sorted) - 1, 1)
        line_to_color = {line: cmap(i / denom) for i, line in enumerate(lines_sorted)}
        # Lengths can differ between lines (e.g. td_shim_type="sequential"), so
        # each line is cast and reshaped independently.
        line_groups = [
            (
                line,
                [q for q in mag_history if line_assignments[q] == line],
                [q for q in flux_history if line_assignments[q] == line],
            )
            for line in lines_sorted
        ]

    plt.figure("All_Qubit_Magnetization_by_calibration_refinement_iteration")
    plt.title(r"Magnetization by iteration, $\langle Z\rangle_{detector}$")
    bulk_data = []
    for line, mag_keys, flux_keys in line_groups:
        if not mag_keys:
            continue
        color = line_to_color[line]
        mag_array = np.array([mag_history[q] for q in mag_keys])
        mag_array = np.reshape(
            mag_array,
            (
                mag_array.shape[0],
                mag_array.shape[1] // num_experiments,
                num_experiments,
            ),
        )
        y0 = 0
        for experiment_sign in range(num_experiments):
            y = mag_array[:, :, experiment_sign].transpose()
            if color is None:
                if num_experiments > 1:
                    plt.plot(y, label=f"Initial state all {-1 + 2*experiment_sign}")
                else:
                    plt.plot(y)
            else:
                traces = plt.plot(y, color=color)
                if experiment_sign == 0 and traces:
                    traces[0].set_label(f"{label} line {line}".strip())
            y0 = y0 + y / num_experiments
        if color is None and num_experiments > 1:
            plt.plot(
                np.mean(mag_array, axis=2).transpose(),
                color="black",
                label="Experiment average",
            )
        bulk_data.append((line, color, y0))

    if line_assignments is None:
        if num_experiments > 1:
            plt.legend()
            plt.xlabel("Calibration refinement iteration")
        else:
            plt.xlabel("Programming")
            if len(mag_history) <= max_qubit_labels:
                plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    else:
        plt.xlabel("Calibration refinement iteration")
        plt.legend(title=f"{label} Line".strip())
    plt.ylabel("Magnetization")

    plt.figure("Bulk_Magnetization_by_calibration_refinement_iteration")
    for line, color, y0 in bulk_data:
        n_qubits = y0.shape[1]
        rms = np.sqrt(np.mean(y0**2, axis=1))
        if n_qubits > 1:
            # Jackknife standard error: recompute RMS leaving out one qubit at a time.
            loo_ss = np.sum(y0**2, axis=1, keepdims=True) - y0**2
            rms_loo = np.sqrt(loo_ss / (n_qubits - 1))
            rms_se = np.sqrt(
                (n_qubits - 1)
                / n_qubits
                * np.sum((rms_loo - rms_loo.mean(axis=1, keepdims=True)) ** 2, axis=1)
            )
        else:
            rms_se = np.zeros_like(rms)
        if color is None:
            plt.errorbar(np.arange(len(rms)), rms, yerr=rms_se, capsize=3)
        else:
            plt.errorbar(
                np.arange(len(rms)),
                rms,
                yerr=rms_se,
                capsize=3,
                color=color,
                label=f"{label} line {line}".strip(),
            )
    if line_assignments is not None:
        plt.legend(title=f"{label} Line".strip())
    plt.xlabel("Calibration refinement iteration")
    plt.ylabel("Root Mean Square Magnetization")

    plt.figure("Flux_bias_by_calibration_refinement_iteration")
    plt.title("All refined flux_biases")
    for line, mag_keys, flux_keys in line_groups:
        if not flux_keys:
            continue
        color = line_to_color[line]
        flux_array = np.array([flux_history[q] for q in flux_keys])
        if color is None:
            plt.plot(flux_array.transpose())
        else:
            traces = plt.plot(flux_array.transpose(), color=color)
            if traces:
                traces[0].set_label(f"{label} line {line}".strip())
    plt.xlabel("Calibration refinement iteration")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    if line_assignments is not None:
        plt.legend(title=f"{label} Line".strip())
    elif len(mag_history) <= max_qubit_labels:
        plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    _apply_tight_layout()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def _plot_tds_schedules(
    x_polarizing_schedule: list[list[float]],
    x_anneal_schedules: list[list[list[float]]],
    detector_lines: Collection[int] | None = None,
    source_lines: Collection[int] | None = None,
    target_lines: Collection[int] | None = None,
    plt_show_block: None | bool = None,
) -> None:
    """Plots the piecewise linear schedules used

    Args:
        x_polarizing_schedule: The polarization signal.
        x_anneal_schedules: The list of anneal schedules, one per line.
        detector_lines: Indices of detector lines, used to annotate the legend.
        source_lines: Indices of source lines, used to annotate the legend.
        target_lines: Indices of target lines, used to annotate the legend.
            Lines in none of the above are labelled as unused. When none of the
            line collections are provided, lines are labelled by number only.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
    """
    detector_set = set(detector_lines or ())
    source_set = set(source_lines or ())
    target_set = set(target_lines or ())
    annotate_type = any(
        lines is not None for lines in (detector_lines, source_lines, target_lines)
    )

    def _line_type(line: int) -> str:
        if line in detector_set:
            return "detector"
        if line in source_set:
            return "source"
        if line in target_set:
            return "target"
        return "unused"

    plt.figure("PWL multi-color annealing schedules")
    plt.title("x anneal and polarizing schedules")
    for line, schedule in enumerate(x_anneal_schedules):
        label = f"Line {line}"
        if annotate_type:
            label += f" ({_line_type(line)})"
        plt.plot([x for x, _ in schedule], [y for _, y in schedule], label=label)
    plt.plot(
        [x for x, _ in x_polarizing_schedule],
        [y for _, y in x_polarizing_schedule],
        label="Polarizing schedule",
        linestyle="dashed",
        color="black",
    )
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Normalized CJJ control bias, c")
    plt.legend()
    _apply_tight_layout()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def imshow_data(
    mean_Z_detector: np.ndarray,
    delays: np.ndarray,
    colormap_type: Literal["default", "divergent"] = "divergent",
    linthresh: float = 1.0,
    first: int = 0,
    last: int | None = None,
    context_str: str = "",
    plt_show_block: None | bool = None,
    ax=None,
    vertical_lines: list[float] | None = None,
) -> None:
    """Display detector magnetization data as a heatmap.

    Creates an image plot of detector magnetization values organized by delay times,
    with optional divergent or default colormaps.

    Args:
        mean_Z_detector: 2D array of detector magnetizations (delays x embeddings).
        delays: Array of time delay values (microseconds); ytick labels are
            rendered in nanoseconds (values are multiplied by 1000).
        colormap_type: Type of colormap to use ("default" or "divergent").
        linthresh: Threshold for the symmetric logarithmic colormap
            (only relevant if colormap_type is "divergent").
        first: Index whose delay value is highlighted as an additional ytick
            label. Does not restrict the plotted range.
        last: Index one past the last delay value highlighted as an additional
            ytick label. If None, uses ``mean_Z_detector.shape[0]``. Does not
            restrict the plotted range.
        context_str: Optional context string to append to figure title.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
        ax: Optional matplotlib axes instance for plotting into an existing figure.
        vertical_lines: lines at which to draw vertical lines
    """
    if colormap_type == "divergent":
        fig_title = f"det_{context_str}"
        norm, cmap = SymLogNorm(linthresh=linthresh, vmin=-1, vmax=1), "RdBu"
    else:
        fig_title = f"det_spacetime_{colormap_type}_{context_str}"
        norm, cmap = None, None
    if ax is None:
        ax = plt.figure(fig_title).gca()
        ax.set_title(f"Real-space magnetization {context_str}".strip())
    # aspect="auto" lets axes fill their allotted cell so inter-subplot spacing survives resizing.
    ax.imshow(mean_Z_detector, norm=norm, cmap=cmap, aspect="auto")
    if last is None:
        last = mean_Z_detector.shape[0]
    yticks_dict = {
        first: f"{1000 * delays[first]:.3g}",
        last - 1: f"{1000 * delays[last-1]:.3g}",
    }
    yticks_dict.update(
        {
            0: f"{1000 * delays[0]:.3g}",
            mean_Z_detector.shape[0] - 1: f"{1000 * delays[-1]:.3g}",
        }
    )
    ax.set_yticks(
        list(yticks_dict.keys()),
        list(yticks_dict.values()),
    )
    ax.set_xlabel("Target-Detector-Source embedding")
    ax.set_ylabel("Delay, nanoseconds")
    if vertical_lines is not None:
        for x in vertical_lines:
            ax.axvline(x=x, color="black", linewidth=1.5)
    _apply_tight_layout()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def _get_experiment_id(
    args: argparse.Namespace, num_char: int = 8, verbose: bool = True
) -> str:
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
    vars_args = vars(args).copy()
    vars_args.pop(
        "save_figures", None
    )  # save_figures is not relevant to the experiment data, so we exclude it from the hash
    if vars_args.get("solver_name", None) == SOLVER_FILTER:
        vars_args["solver_name"] = "DefaultSolver"
        print(
            "NB: The default solver is used; the experiment identifier is computed "
            "as if solver_name='DefaultSolver', so it does not distinguish between "
            "different physical solvers resolved at runtime."
        )
    args_string = json.dumps(vars_args, sort_keys=True)
    identifier = hashlib.sha256(args_string.encode("utf-8")).hexdigest()[:num_char]
    if verbose:
        print("Demo parameters:")
        print(vars(args))
        print(
            "Demo identifier (labels cached data and saved figures, employ --use-cache for data reuse):",
            identifier,
        )
    return identifier


def _plot_time_series(
    embs: list,
    line_assignments: dict[int, int],
    mean_Z_detector: np.ndarray,
    delays: np.ndarray,
    line_color: list | None = None,
    plotted_emb_idxs: Collection[int] | None = None,
    label_emb_idxs: Collection[int] | None = None,
    xlabel: str = "Delay, nanoseconds",
    ylabel: str = "Detector magnetizations",
    plt_show_block: None | bool = None,
    ax=None,
) -> None:
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
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
        ax: Optional matplotlib axes instance for plotting into an existing figure.
    """

    if ax is None:
        ax = plt.gca()

    if plotted_emb_idxs is None:
        plotted_emb_idxs = set(range(len(embs)))
    if label_emb_idxs is None:
        label_emb_idxs = plotted_emb_idxs

    for emb_idx in plotted_emb_idxs:
        q = embs[emb_idx][0][0]
        line_idx = line_assignments[q]
        if emb_idx in label_emb_idxs:
            if len(label_emb_idxs) == len(plotted_emb_idxs):
                label = f"tar-line {line_idx}(qubit {q})"
            else:
                label = f"tar-line {line_idx}"
        else:
            label = None
        if line_color is not None:
            color = line_color[line_idx]
        else:
            color = None
        ax.plot(delays, mean_Z_detector[:, emb_idx], color=color, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def estimate_decoupling_timescale(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    sampling_params: dict,
    detector_lines: Iterable[int],
    detected_vars: Sequence[Variable] | None = None,
    preparation_orientation: float = 1.0,
    t_guess: float = 0.0,
    t_min: float | None = None,
    t_max: float | None = None,
    target_A: float = 2000,
    T2: float = 0.0101,
    threshold_cycle_av: float = 0.9,
    phase_accuracy: float = 1.0,
    verbose: bool = False,
    by_det_src_pair: bool = True,
    line_assignments: dict[int, int] | None = None,
    source_lines: Iterable[int] | None = None,
    interpolate_final_interval: bool = False,
) -> tuple[dict[int, float], list, dict[int, float]]:
    """Estimate bulk decoupling timescales

    Source and detector waveforms typically overlap, in order that the detection
    is not of a source-coupled system a delay is required.
    When the source is coupled the magnetization plateaus, after decoupling the
    magnetization follows a damped oscillation given pi/2-pulse initialization.
    Departure from the plateau to zero is rapid and persistent in a proxy signal
    with significantly reduced oscillation amplitude (provided target_A is not
    too inaccurate) y_proxy(t) = y(t) + y(t + 1/(2*target_A)).

    A bound on the decoupling timescale is first established to O(T2).
    The interval is then searched by bisection to obtain a value that characterizes
    decoupling in the bulk (mean) to accuracy O(1/target_A), variation between lines
    and qubits on a given line can be of a comparable scale. Series for separated
    lines or qubits can be evaluated by similar principles, and deviations reduced
    by application of anneal offsets on detectors and sources.

    Source-detectors are tightly synchronized by line in this demo as an initial stage.
    It would be more natural to coarsely synchronize, align target frequencies,
    and then refine the line-wise synchronization and finally use per-qubit
    anneal_offsets approximating some c(t) on source and detectors. A linear
    approximation assuming a common dc/dt and per-qubit-delays may be sufficient.

    Args:
        sampler: Parallel embedding composite sampler wrapping the QPU sampler.
        bqm: Binary Quadratic Model defining the target-detector-source system.
        sampling_params: Parameters passed to the QPU sampler.
        detector_lines: Iterable of detector line indices.
        detected_vars: Iterable of bqm variables to keep for detector
            magnetization calculation (default: (("detector", 0),)).
        preparation_orientation: Orientation of the preparation pulse (default: 1).
        t_guess: Initial delay guess used to seed the search (microseconds, default: 0.0).
        t_min: Known lower bound on the decoupling delay. If None, it is estimated.
        t_max: Known upper bound on the decoupling delay. If None, it is estimated.
        target_A: Target frequency for the decoupling sequence (MHz)
        T2: Decoherence time (microseconds)
        threshold_cycle_av: Threshold characterizing the decoupled regime (default: 0.9).
            The half-cycle average (t and t+1/(2*target_A)) magnetization is only larger than
            the threshold, in absolute value, in the source-coupled regime.
        verbose: Whether to print progress information (default: False).
        by_det_src_pair: When True (default), resolve a decoupling delay for each
            detector line independently, reusing sampled data across lines to
            the extent possible. When False, a single bulk delay is estimated
            from the median over all detector lines and shared across them.
        line_assignments: Mapping from qubit to line index. Required when
            by_det_src_pair is True to group embeddings by their detector line.
        source_lines: Iterable of source line indices, ordered so the first is
            the reference source line. When more than one source line is present
            (and by_det_src_pair is True), detector delays are resolved using only the
            reference-source embeddings, and each additional source line returns a
            decoupling delay relative to the reference source line.
        phase_accuracy: The desired accuracy for the decoupling timescale estimate. If
            only one source and detector line are present, an accuracy 1. is defaulted.
            With multiple source or detector lines, a high accuracy is appropriate
            to account for line-wise desynchronizations.
        interpolate_final_interval: When True, refine each returned delay past the
            bisection bracket by linearly interpolating the half-cycle-averaged
            proxy to its threshold crossing (used by the refinement stage).

    Returns:
        A tuple containing:
        - A dict of line index -> delay. Detector lines map to absolute decoupling
          delays (resolved against the reference source line when several source
          lines are present). With more than one source line, each source line also
          maps to a delay relative to the reference source line (reference -> 0.0).
          With by_det_src_pair=False a single bulk estimate is repeated for each detector line.
          Each delay is refined past the bisection bracket by linearly interpolating
          the half-cycle-averaged proxy to its threshold crossing.
        - A list of (delay, mag) pairs representing the decoupling timescale search data.
        - A dict of detected (detector) qubit -> decoupling time, estimated
          per qubit by linearly interpolating the two sampled points (one either side)
          nearest the half-cycle-averaged proxy threshold. Qubits without sampled points
          bracketing the threshold are omitted.
    """
    detector_lines = list(detector_lines)
    half_cycle = 1 / target_A / 2.0
    data = []
    cache = {}

    def _measure(delay):
        """Sample per-embedding detector magnetization, reusing cached delays."""
        if delay not in cache:
            mag = run_parallel_experiment(
                sampler=sampler,
                bqm=bqm,
                sampling_params=sampling_params,
                detector_lines=detector_lines,
                delays=[delay],
                detected_vars=detected_vars,
            )
            cache[delay] = mag[0, :]
            data.append((delay, cache[delay]))
        return cache[delay]

    def _search(vec, t_guess, t_min, t_max):
        """Bracket, then bisect, the decoupling delay for a magnetization selector."""
        if verbose and (t_min is None or t_max is None):
            print("Estimate bounds O(T1) on decoupling time scale (two programmings)")
        while t_min is None or t_max is None:
            v = vec(t_guess)
            if np.median(v) * preparation_orientation < threshold_cycle_av:
                t_max = t_guess
                if interpolate_final_interval:
                    # Cache the half-cycle companion so this delay can pair and
                    # seed the per-line/per-qubit interpolating estimators.
                    vec(t_guess - half_cycle)
                t_guess -= T2
            else:
                v_half = vec(t_guess - half_cycle)
                if (
                    preparation_orientation / 2 * np.median(v + v_half)
                    < threshold_cycle_av
                ):
                    t_max = t_guess - half_cycle
                    t_guess -= T2
                else:
                    t_min = t_guess
                    t_guess += T2
        if verbose:
            print(
                f"Estimate decoupling timescale to accuracy better than {phase_accuracy}/target_A, with ~log(T2*target_A) programmings"
            )
        while t_max - t_min > phase_accuracy / target_A:
            t_guess = (t_min + t_max) / 2
            v = vec(t_guess)
            if np.median(v) * preparation_orientation < threshold_cycle_av:
                t_max = t_guess
                if interpolate_final_interval:
                    # Cache the half-cycle companion so this delay can pair and
                    # seed the per-line/per-qubit interpolating estimators.
                    vec(t_guess - half_cycle)
            else:
                v_half = vec(t_guess - half_cycle)
                if (
                    preparation_orientation / 2 * np.median(v + v_half)
                    < threshold_cycle_av
                ):
                    t_max = t_guess - half_cycle
                else:
                    t_min = t_guess

        if interpolate_final_interval:
            # Refine within the final bracket: linearly interpolate the smooth
            # half-cycle-averaged proxy to the time where it crosses the threshold.

            def _proxy(delay):
                return (
                    preparation_orientation
                    / 2
                    * np.median(vec(delay) + vec(delay - half_cycle))
                )

            s_min = _proxy(t_min)
            s_max = _proxy(t_max)
            if s_min == s_max:
                return t_guess
            frac = (s_min - threshold_cycle_av) / (s_min - s_max)
            frac = min(max(frac, 0.0), 1.0)
            return t_min + frac * (t_max - t_min)
        else:
            return t_guess

    def _per_qubit_threshold_times():
        """Per-qubit threshold crossing time from a linear fit of the two
        sampled points (one either side) nearest the smooth half-cycle proxy
        threshold. Keyed by each embedding's detected (detector) qubit."""
        # Only delays with a half-cycle companion let the proxy be formed.
        paired_delays = sorted(t for t in cache if (t - half_cycle) in cache)
        threshold_times = {}
        for i, emb in enumerate(sampler.embeddings):
            qubit = emb[("detector", 0)][0]
            points = [
                (
                    t,
                    preparation_orientation
                    / 2
                    * (cache[t][i] + cache[t - half_cycle][i]),
                )
                for t in paired_delays
            ]
            above = [(t, p) for t, p in points if p >= threshold_cycle_av]
            below = [(t, p) for t, p in points if p < threshold_cycle_av]
            if not above or not below:
                continue
            # Nearest point to the threshold from each side brackets the crossing.
            ta, pa = min(above, key=lambda tp: tp[1])
            tb, pb = max(below, key=lambda tp: tp[1])
            if pa == pb:
                continue
            threshold_times[qubit] = ta + (tb - ta) * (pa - threshold_cycle_av) / (
                pa - pb
            )
        return threshold_times

    if not by_det_src_pair:
        # Non-default: a single bulk delay from the median over all detector lines.
        t_bulk = _search(_measure, t_guess, t_min, t_max)
        return (
            {line: t_bulk for line in detector_lines},
            data,
            _per_qubit_threshold_times(),
        )

    if line_assignments is None:
        raise ValueError("line_assignments is required when by_det_src_pair=True")
    # Column i of each measurement corresponds to sampler.embeddings[i]; map
    # each embedding to its detector and source line to resolve lines independently.
    emb_detector_lines = np.array(
        [line_assignments[emb[("detector", 0)][0]] for emb in sampler.embeddings]
    )
    emb_source_lines = np.array(
        [line_assignments[emb[("source", 0)][0]] for emb in sampler.embeddings]
    )
    source_lines = None if source_lines is None else list(source_lines)
    multi_source = source_lines is not None and len(source_lines) > 1
    reference_source = source_lines[0] if source_lines else None
    reference_detector = detector_lines[0]

    def _resolve(mask, context):
        if not mask.any():
            raise ValueError(f"No embedding available for {context}")
        if verbose:
            print(f"Resolving decoupling delay for {context}")
        return _search(lambda delay, m=mask: _measure(delay)[m], t_guess, t_min, t_max)

    t_decoupled = {}
    # Decoupling is relative to the source; resolve detector lines against the
    # reference source line so returned detector delays share a source baseline.
    for line in detector_lines:
        mask = emb_detector_lines == line
        if multi_source:
            mask = mask & (emb_source_lines == reference_source)
        t_decoupled[line] = _resolve(mask, f"detector line {line}")

    if multi_source:
        # Additional source lines return a delay relative to the reference source
        # line, measured on the reference detector line.
        t_reference = t_decoupled[reference_detector]
        t_decoupled[reference_source] = 0.0
        for source_line in source_lines[1:]:
            mask = (emb_detector_lines == reference_detector) & (
                emb_source_lines == source_line
            )
            t_decoupled[source_line] = t_reference - _resolve(
                mask,
                f"source line {source_line} on reference detector line {reference_detector}",
            )

    return t_decoupled, data, _per_qubit_threshold_times()


def _to_independent_tds(embs: list[dict]) -> list[dict]:
    """Convert coupled embeddings to decoupled embeddings.

    Assume the label scheme returned by :code:`tds_graph`:
        i: target qubit
        ("source", i): source qubit i
        ("detector", i): detector qubit i

    If a source/detector or target is absent for some i, the
    embedding value is assigned to None.
    """
    max_i = max(k if isinstance(k, int) else k[1] for emb in embs for k in emb)
    independent_tds = []
    for emb in embs:
        independent_tds += [
            {
                0: emb[i] if i in emb else None,
                ("source", 0): emb[("source", i)] if ("source", i) in emb else None,
                ("detector", 0): (
                    emb[("detector", i)] if ("detector", i) in emb else None
                ),
            }
            for i in range(max_i + 1)
        ]
    return independent_tds


def compute_delay_grid(
    ctx: "MCAContext",
    delay_range: float,
    rel_sampling_rate: float = 2,
    rel_start: float = 1.0,
) -> "MCAContext":
    """Build the regularly spaced delay grid used by the sampling stages.

    Args:
        ctx: Shared context providing ``target_A`` and ``dt_dc``.
        delay_range: Width of the delay window past the decoupling time
            (``fit_window`` for target-frequency fitting, ``loop_delay_range``
            for the loop experiment).
        rel_sampling_rate: Samples collected per Nyquist period (2 for the
            target-frequency stage, 4 for the finer-grained loop stage).
        rel_start: Offset of the window start from the decoupling time, as a
            multiple of ``dt_dc`` (1.0 for the target-frequency stage, 0.0 for
            the loop stage).

    Returns:
        The same ``ctx``, with ``delay_min``, ``delay_max``, ``delays`` and
        ``dt`` populated.
    """
    target_A = ctx.target_A
    dt_dc = ctx.dt_dc

    # Nyquist frequency in MHz, resolving up to 2*target_A.
    nyquist_frequency = target_A * 1000 * 2
    delay_min = (
        rel_start
        * dt_dc  # For reasonable confidence source is fully decoupled (c < 0).
    )
    delay_max = delay_min + delay_range
    # Collecting regularly spaced data on [delay_min, delay_min + delay_range],
    # post source decoupling, is not optimal for inference of calibration
    # errors, but (IMO) allows intuitive estimators and data series.
    delays = np.linspace(
        delay_min,
        delay_max,
        round((delay_max - delay_min) * nyquist_frequency * rel_sampling_rate) + 1,
    )

    ctx.delay_min = delay_min
    ctx.delay_max = delay_max
    ctx.delays = delays
    ctx.dt = delays[1] - delays[0]
    return ctx


class MCAContext:
    """Mutable state shared between the calibration-refinement stages.

    The stages (see the ``stage_*`` modules) read and write attributes on this
    object, allowing the monolithic workflow to be split into small, individually
    readable scripts while preserving the original sequential data flow.
    """

    #: Names of every attribute threaded between stages. Config values mirror the
    #: parameters of :func:`mca_experiment_helpers.runner.main`; the remainder are computed.
    _FIELDS = (
        # configuration (see runner.main for documentation)
        "cache_str",
        "solver_name",
        "detector_lines",
        "source_lines",
        "target_lines",
        "seed",
        "max_num_embeddings",
        "target_c",
        "target_A",
        "target_B",
        "dAdc",
        "td_shim_type",
        "t_decoupled",
        "by_det_src_pair",
        "refine_frequencies",
        "verify_anneal_offsets",
        "schedule_fn",
        "num_reads",
        "use_common_c_bounds",
        "use_overshoot",
        "save_figures",
        "J_td",
        "J_ts",
        "J_tt",
        "loop_length",
        "preparation_orientation",
        "colormap_type",
        "refine_delays",
        "dt_dc",
        "fit_window",
        "loop_delay_range",
        # runtime / computed state
        "stage_idx",
        "qpu",
        "online",
        "cached_chip_id",
        "exp_feature_info",
        "line_color",
        "x_anneal_schedules",
        "x_polarizing_schedule",
        "x_schedule_delays",
        "anneal_offsets",
        "flux_biases",
        "sampling_params",
        "T",
        "Tnode_to_tds",
        "S",
        "Snode_to_tds",
        "S_experiment",
        "embs_experiment",
        "embs",
        "embs_by_line",
        "sum_by_line",
        "cumsum_by_line",
        "line_exemplars",
        "n_embs",
        "sampler",
        "bqm",
        "delay_min",
        "delay_max",
        "delays",
        "dt",
        "mean_Z_detector",
        "first",
        "last",
        "frequencies",
        "psd",
        "anneal_offsets_target",
        "ax_second_iter_timeseries",
        "ax_second_iter_psd",
        "heatmap_after_axes",
        "t_decoupled_by_qubit",
        "t_mags",
    )

    def __init__(self, **config):
        for name in self._FIELDS:
            setattr(self, name, None)
        for name, value in config.items():
            if name not in self._FIELDS:
                raise TypeError(f"Unexpected configuration field: {name!r}")
            setattr(self, name, value)

    # The fields in ``_FIELDS`` are populated dynamically; these declarations let
    # static type checkers accept attribute access/assignment on the context.
    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


def create_solver(
    solver_name: dict | str | None, *, cache_empty: bool = False
) -> DWaveSampler:
    """Create the QPU sampler used by the experiment.

    This is the single place the solver is created, so every stage fails the
    same way: an error is raised only because the solver could not be created.

    Args:
        solver_name: Name of the solver, or dictionary of characteristics, as
            accepted by :class:`~dwave.system.DWaveSampler`.
        cache_empty: Set when cached data is enabled but cannot substitute for
            the solver. The error message then notes that retrieval of data is
            not possible.

    Returns:
        The connected :class:`~dwave.system.DWaveSampler`.

    Raises:
        RuntimeError: If the solver cannot be created or reached. The message
            names the underlying error.
    """
    try:
        qpu = DWaveSampler(solver=solver_name)
        # The client can serve solver data from its cache without a connection;
        # refreshing forces a round trip so success means the API is reachable.
        qpu.client.get_solvers(refresh=True)
    except Exception as error:
        note = (
            " and cache empty (retrieval of data not possible)"
            if cache_empty
            else ""
        )
        raise RuntimeError(
            f"Solver creation failed{note}: {type(error).__name__}: {error}"
        ) from error
    return qpu


def connect_qpu(ctx: MCAContext) -> MCAContext:
    """Create the solver on demand after a stage's cache lookup fails.

    Stages try cached data first. When that misses and the context is offline
    (``ctx.online`` is False, because the waveforms stage substituted a
    :class:`~dwave.system.testing.MockDWaveSampler` built from cached
    properties), call this to create the real solver. There is no fallback to
    cached data: an error is raised only if the solver cannot be created.

    On success ``ctx.qpu`` is replaced, ``ctx.sampler`` is rebuilt over the new
    solver from the existing embeddings (``ctx.embs``), and ``ctx.online`` is set
    to True. Stages must re-read ``ctx.qpu`` and ``ctx.sampler`` afterwards, as
    any local references still point at the offline mock. Does nothing when
    ``ctx.online`` is already True.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.

    Returns:
        The same ``ctx``.

    Raises:
        RuntimeError: If the solver cannot be created. A warning (not an error)
            is issued if its ``chip_id`` differs from the solver the cached
            properties, embeddings and schedules were built for.
    """
    if ctx.online:
        return ctx
    qpu = create_solver(ctx.solver_name, cache_empty=bool(ctx.cache_str))
    chip_id = qpu.properties.get("chip_id")
    if (
        ctx.cached_chip_id is not None
        and chip_id is not None
        and chip_id != ctx.cached_chip_id
    ):
        warnings.warn(
            f"Solver created on demand has chip_id {chip_id!r}, but the cached "
            f"properties, embeddings and schedules were built for {ctx.cached_chip_id!r}.",
            stacklevel=2,
        )
    print(f"Solver connected on cache miss: {qpu.solver.identity}")
    ctx.qpu = qpu
    if ctx.embs is not None:
        ctx.sampler = ParallelEmbeddingComposite(qpu, embeddings=ctx.embs)
    ctx.online = True
    return ctx
