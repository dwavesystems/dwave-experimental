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

This example builds many parallel target-detector-source embeddings, collects
detector magnetization time series, estimates per-embedding frequency spread,
and demonstrates two mitigation strategies:

1. detector-only or TDS flux-bias shimming
2. per-embedding anneal-offset refinement
"""

import argparse
import hashlib
import json
import os
import re
from typing import Collection, Iterable, Literal

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import dimod
from dimod.typing import Variable
from dwave.system import DWaveSampler
from dwave.system.testing import MockDWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    make_tds_graph,
    make_tds_x_schedules,
    #   make_tds_x_schedule_delays,
    SOLVER_FILTER,
)
from dwave.experimental.shimming import shim_flux_biases, shim_tds_flux_biases


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
        "A: target, estimated_mean, and standard deviation",
        target_A,
        np.mean(mean_A_est),
        np.std(mean_A_est),
    )
    if fit_to_target_A:
        mu = target_A
    print()
    dcs = (mean_A_est - mu) / dAdc
    return dcs


def make_yC(
    delays: np.ndarray,
    A: float,
    *,
    T2: float = 0.0101,
    phi_s: float = 0.0,
    phi_d: float = 0.0,
    sign_Jts_fbs: int = 1,
) -> np.ndarray:
    """Make a noise-free model signal for computational basis preparation and detection.

    Eq 6. from arXiv 2603.15534 at theta_d = theta_s = pi/2
    Note that this model assumes decoupling from the source, and ideally
    controlled flux biases (in particular note phi_q=0 on the target).
    The source polarization is assumed to be instantaneously removed at delay t=0,
    with measurement performed at delay t.

    y(t) = cos(2* pi * A * t + phi_s - phi_d) exp(-t/T2)

    Args:
        delays: Measurement times (microseconds)
        A: frequency
        T2: exponential envelope time scale. Defaulted as
            from T_phi = 12ns and T1 = 32ns, typical of Advantage2 research.
        phi_s: Bloch sphere rotation (azimuthal angle) for the source.
        phi_d: Bloch sphere rotation (azimuthal angle) for the detector.
        sign_Jts_fbs: Sign of the product of J_ts and fb_s (default: 1).
    Returns:
        A model signal
    """
    return (
        sign_Jts_fbs
        * np.exp(-delays / T2)
        * np.cos(2 * np.pi * A * delays + phi_s - phi_d)
    )


def make_yE(
    delays: np.ndarray,
    *,
    T1: float = 0.032,
    theta_s: float = np.pi / 2,
    sign_fbd: int = 1,
) -> np.ndarray:
    """Make a noise-free model signal for energy basis preparation and detection.

    Eq 6. from arXiv 2603.15534 at theta_d = 0, with theta_s = pi/2 by default.
    Note that this model assumes decoupling from the source, and ideally
    controlled flux biases (in particular note phi_q=0 on the target, and the
    flux bias on the detector must be small compared to the polarizing signal).
    The source polarization is assumed to be instantaneously removed.

    y(t) = 1 - exp(-t / T1) * (1 - cos(theta_s))

    Args:
        delays: Measurement times (microseconds)
        T1: Exponential envelope time scale (microseconds). Defaulted to a value
            typical of Advantage2 research systems.
        theta_s: Bloch sphere rotation (polar angle) for the source (default: pi/2).
    Returns:
        A model signal
    """
    return sign_fbd * (1 - np.exp(-delays / T1) * (1 - np.cos(theta_s)))


def make_y(
    delays,
    A: float,
    *,
    T1: float = 0.032,
    T2: float = 0.0101,
    sign_Jts_fbs: int = -1,
    sign_fbd: int = 1,
    theta_d: float = np.pi / 2,
    theta_s: float = np.pi / 2,
    phi_d: float = 0.0,
    phi_s: float = 0.0,
    t0: float = 0.0,
):
    """Make a noise-free model signal for arbitrary basis preparation and detection.

    The source polarization is fixed to sign(J_ts * fb_s):
    y(t) = sign(J) y(theta_d, theta_s, phi_d, phi_s) for t > t0, and
           sign(J) for t < t0   # Constrained by source polarization

    where y(theta_d, theta_s, phi_d, phi_s) matches Eq. 6 of arXiv:2603.15534.

    Assume that |Ip(s_tar) J_td| is large compared to fb_d. So that until the source
    is decoupled, the influence of fb_d can be neglected.

    Args:
        delays: Measurement times (microseconds)
        A: frequency
        T1: assumed coherence time (microseconds)
        T2: assumed coherence time (microseconds)
        sign_Jts_fbs: polarization of target when subject to strong source coupling.
        sign_fbd: polarization of decoupled detector (default: 1)
        theta_d: detection basis angle
        theta_s: source basis angle
        phi_d: detection phase
        phi_s: source phase
        t0: Source decoupling time (default: 0.0)
    Returns:
        A model signal assuming ideal detection and instantaneous source decoupling.

    """
    rd = delays - t0
    return sign_Jts_fbs * (rd <= 0) + (rd > 0) * (
        np.cos(theta_d) * make_yE(rd, T1=T1, theta_s=theta_s, sign_fbd=sign_fbd)
        + np.sin(theta_d)
        * np.sin(theta_s)
        * make_yC(rd, A=A, T2=T2, sign_Jts_fbs=sign_Jts_fbs, phi_d=phi_d, phi_s=phi_s)
    )


def dyC_dt0(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs: int = 1
) -> np.ndarray:
    """Calculate derivative of signal model with respect to time delay.

    Computes the derivative of the model signal produced by ``make_y`` with
    respect to the time delay. For ``t > 0`` the underlying signal is
    ``sign_Jts_fbs * exp(-t/T2) * cos(2*pi*A*t)``; for ``t < 0`` it is
    constant so the derivative is zero.

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


def dyC_dA(
    delays: np.ndarray, A: float, T2: float = 0.0101, sign_Jts_fbs: int = 1
) -> np.ndarray:
    """Calculate derivative of signal model with respect to frequency.

    EFFECTIVELY OBSOLETE FUNCTION

    Computes the derivative of the ``make_y`` signal
    ``sign_Jts_fbs * exp(-t/T2) * cos(2*pi*A*t)`` (for ``t > 0``) with respect
    to the frequency parameter ``A``.

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
    *,
    T1: float = 0.032,
    T2: float = 0.0101,
    sign_Jts_fbs: int = 1,
    theta_d: float = np.pi / 2,
    theta_s: float = np.pi / 2,
    phi_d: float = 0.0,
    phi_s: float = 0.0,
    t0: float = 0.0,
    num_independent_samples: float = float("Inf"),
    prng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Create an artificial data set

    y(t) = ideal signal produced by ``make_y``
    with variance of (1 - y(t)^2) in the measured state. Given independent
    and identically distributed samples we can model noise as normally
    distributed. The signal is clipped so that the value is always physical
    within the range [-1, 1] (only impacts small num_independent_samples).

    Args:
        delays: Time of measurement (microseconds).
        A: Frequency (GHz).
        T1: Assumed coherence time (microseconds).
        T2: Exponential envelope time scale (microseconds).
        sign_Jts_fbs: Polarization sign on the target (default: -1).
        theta_d: Detection basis angle (default: pi/2).
        theta_s: Source basis angle (default: pi/2).
        phi_d: Detection phase (default: 0.0).
        phi_s: Source phase (default: 0.0).
        t0: Reference time for the start of the measurement (default: 0.0).
        num_independent_samples: Number of samples to model.
        prng: Pseudo random number generator or seed.

    Returns:
        A model signal with sampling noise.
    """

    y = make_y(
        delays,
        A,
        T1=T1,
        T2=T2,
        sign_Jts_fbs=sign_Jts_fbs,
        theta_d=theta_d,
        theta_s=theta_s,
        phi_d=phi_d,
        phi_s=phi_s,
        t0=t0,
    )

    if num_independent_samples != float("Inf"):
        prng = np.random.default_rng(prng)
        return np.clip(
            y
            + np.sqrt((1 - y**2) / num_independent_samples) * prng.normal(size=len(y)),
            a_min=-1,
            a_max=1,
        )
    else:
        return y


def run_parallel_experiment(
    sampler: ParallelEmbeddingComposite,
    bqm: dimod.BinaryQuadraticModel,
    sampling_params: dict,
    delays: np.ndarray | list,
    detector_lines: Iterable[int],
    detected_vars: Iterable[Variable] = (("detector", 0),),
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
        mean_Z_detector.append([np.mean(sample) for sample in detector_samples])
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
) -> None:
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
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
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
    plt.title("All shimmed flux_biases")
    plt.plot(flux_array.transpose())
    plt.xlabel("Shim iteration")
    plt.ylabel("Flux bias ($\\Phi_0$)")
    if mag_array.shape[0] <= max_qubit_labels:
        plt.legend(flux_history.keys(), title=f"{label} Qubit index")
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def _plot_tds_schedules(
    x_polarizing_schedule: list[list[float]],
    x_anneal_schedules: list[list[list[float]]],
    plt_show_block: None | bool = None,
) -> None:
    """Plots the piecewise linear schedules used

    Args:
        x_polarizing_schedule: The polarization signal.
        x_anneal_schedules: The list of anneal schedules, one per line.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
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
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def imshow_data(
    mean_Z_detector: np.ndarray,
    delays: np.ndarray,
    colormap_type: Literal["default", "divergent"],
    first: int = 0,
    last: int | None = None,
    context_str: str = "",
    plt_show_block: None | bool = None,
) -> None:
    """Display detector magnetization data as a heatmap.

    Creates an image plot of detector magnetization values organized by delay times,
    with optional divergent or default colormaps.

    Args:
        mean_Z_detector: 2D array of detector magnetizations (delays x embeddings).
        delays: Array of time delay values (microseconds); ytick labels are
            rendered in nanoseconds (values are multiplied by 1000).
        colormap_type: Type of colormap to use ("default" or "divergent").
        first: Index whose delay value is highlighted as an additional ytick
            label. Does not restrict the plotted range.
        last: Index one past the last delay value highlighted as an additional
            ytick label. If None, uses ``mean_Z_detector.shape[0]``. Does not
            restrict the plotted range.
        context_str: Optional context string to append to figure title.
        plt_show_block: If not None (default), then execute
            :code:`plt.show(block=plt_show_block)` to display the figure.
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
    if verbose:
        print()
        print("Demo parameters:")
        print(vars(args))
    vars_args = vars(args).copy()
    vars_args.pop(
        "save_figures", None
    )  # save_figures is not relevant to the experiment data, so we exclude it from the hash
    if vars_args.get("solver_name", None) == SOLVER_FILTER:
        vars_args["solver_name"] = "DefaultSolver"
    args_string = json.dumps(vars_args, sort_keys=True)
    identifier = hashlib.sha256(args_string.encode("utf-8")).hexdigest()[:num_char]
    if verbose:
        print("Demo identifier (labels cached data and saved figures):", identifier)
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
        if line_color is not None:
            color = line_color[line_idx]
        else:
            color = None
        plt.plot(delays, mean_Z_detector[:, emb_idx], color=color, label=label)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.grid()
    if plt_show_block is not None:
        plt.show(block=plt_show_block)


def estimate_decoupling_timescale(
    sampler,
    bqm,
    sampling_params,
    detector_lines,
    detected_vars=(("detector", 0),),
    t_guess=0.0,
    t_min=None,
    t_max=None,
    target_A: float = 2000,
    T2: float = 0.0101,
    threshold_cycle_av=0.9,
    verbose: bool = True,
):
    """Estimate bulk decoupling delay based on target_A and T2.

    Uncertainty in the computational basis magnetization under pi/2 pulse (polarized)
    preparation is small compared to the to the maximum expected amplitude following
    decoupling for a duration t >= 1/target_A (time for one cycle) in typical experiments
    at baseline calibration, max(|y(t >= 1/omega)|) < exp(-1/(A*T2)). We can therefore
    perform a bisection search to determine a crude source-decoupling timescale.

    Args:
        target_A: Target amplitude for the decoupling sequence (MHz)
        T2: Decoherence time (milliseconds)
        threshold_cycle_av: Threshold characterizing the decoupled regime (default: 0.9).
            The half-cycle average (t and t+1/(2*target_A)) magnetization is only larger than
            the threshold, in absolute value, in the source-coupled regime.

    Returns:
        A tuple containing:
        - The final t_guess value representing the estimated decoupling timescale.
        - A list of (t_guess, mag) pairs representing the decoupling timescale search data.
    """
    threshold = np.exp(-1)  # Separates onset from no onset.
    if verbose:
        print("Estimate decoupling timescale to O(T1), most likely two programmings")
    # Estimate upper and lower bound to precision T1:
    data = []
    while t_min is None or t_max is None:
        mag = run_parallel_experiment(
            sampler=sampler,
            bqm=bqm,
            sampling_params=sampling_params,
            delays=[t_guess],
            detector_lines=detector_lines,
            detected_vars=detected_vars,
        )
        data.append((t_guess, mag[0, :]))
        if abs(np.median(mag)) < threshold:
            t_max = t_guess
            t_guess -= T2
        else:
            t_min = t_guess
            t_guess += T2
    # Bisection search to precision 1/target_A
    if verbose:
        print(
            "Estimate decoupling timescale to O(1/omega), most likely several programmings"
        )
    while t_max - t_min > 1 / target_A:
        t_guess = (t_min + t_max) / 2
        mag = run_parallel_experiment(
            sampler=sampler,
            bqm=bqm,
            sampling_params=sampling_params,
            delays=[t_guess],
            detector_lines=detector_lines,
            detected_vars=detected_vars,
        )
        data.append((t_guess, mag[0, :]))
        if abs(np.median(mag)) < threshold_cycle_av:
            t_max = t_guess
        else:
            # Check pinned at minus 1/2 cycle
            mag = run_parallel_experiment(
                sampler=sampler,
                bqm=bqm,
                sampling_params=sampling_params,
                delays=[t_guess - 1 / target_A / 2.0],
                detector_lines=detector_lines,
                detected_vars=detected_vars,
            )
            data.append((t_guess - 1 / target_A / 2.0, mag[0, :]))
            if abs(np.median(mag)) < threshold_cycle_av:
                t_max = t_guess - 1 / target_A / 2.0
            else:
                t_min = t_guess
    return t_guess, data


def main(
    cache_str: str | None = None,
    solver: dict | str | None = None,
    detector_lines: Iterable[int] = (0,),
    source_lines: Iterable[int] = (3,),
    seed: int | None = None,
    max_num_embeddings: int | None = None,
    target_c: float | None = None,
    target_A: float | None = 2.0,
    apply_flux_bias_shim: Literal["None", "Detector", "TDS"] = "Detector",
    source_decoupling_detection: bool = True,
    verify_anneal_offsets: bool = True,
    delay_min: float | None = None,
    delay_max: float | None = None,
    delay_min_fit: float | None = None,
    delay_max_fit: float | None = None,
    schedule_fn: str = "09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    use_01_c_range: bool = False,
    symmetrize_c_bounds: bool = True,
    num_reads: int = 500,
    use_common_bounds: bool = True,
    save_figures: bool = False,
    T2: float = 0.0101,
) -> None:
    """Demonstrate T-D-S variability and mitigation strategies.

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
    use simple statistics to determine flux-bias assignment on a detector
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
        detector_lines:
            An iterable of integer indices of the detector lines.
        source_lines:
            An iterable of integer indices of the source lines.
        seed:
            Random seed used for embedding generation.
        max_num_embeddings:
            Maximum number of embeddings to find. If None, search for all available embeddings.
        target_c:
            Normalized control bias at which the target qubits are held.
            Either target_A or target_c should be specified, not both.
            target_c is inferred from the schedule and target_A by default.
        target_A:
            The expected qubit frequency in GHz.
            Some methods such as TDS are known to be unstable at lower frequency.
            By default more data is used at higher frequency (density of sampling),
            is in proportion to the Nyquist frequency.
            When None target_A is inferred from target_c and the schedule.
            Either target_A or target_c should be specified, not both.
        apply_flux_bias_shim:
            When set to "None", flux_biases are not modified. When "Detector", flux_biases
            are modified on detector qubits to achieve zero expected magnetization at
            long delay. When "TDS", flux_biases are modified on TDS qubits.
        verify_anneal_offsets:
            When set to True, data is collected and analyzed with anneal_offsets applied.
            When set to False, this verification stage is skipped.
            Anneal_offsets are modified so that the peak power-spectral density is peaked at a common
            value for all qubits. This peak value characterizes the frequency of the target
            qubit in simple, well-calibrated models.
        delay_min: The delay on the detector line for which data is collected. If None
            determined by estimation of source decoupling.
        delay_max: The maximum delay on the detector line for which data is collected. By
            default delay_min + 2 * T2
        delay_min_fit:
            A lower bound on the timeseries window used for inference of the target power spectral density.
            A value that is too small can bias the estimator by introduction of effects related
            to coupling to the source line.
        delay_max_fit:
            An upper bound on the timeseries window used for inference of the target power spectral density.
            Too large a value reduces the efficiency of the estimator, since delays much larger than the
            T1 coherence time are dominated by noise.
        schedule_fn: A schedule file that is used to estimate an appropriate sampling interval for delay
            time and an appropriate scale for anneal_offset synchronization. This should be matched to the
            solver.
        use_01_c_range:
            When ``True``, restricts generated schedule c-values to [0, 1].
        symmetrize_c_bounds:
            If ``True``, use a schedule range symmetric in c about zero.
            `use_01_c_range` must be False.
        save_figures:
            When True, save generated figures to a ``figures`` folder.
        num_reads:
            The number of reads to perform for each measurement.

    Raises:
        ValueError: If the fit window (``delay_min_fit``, ``delay_max_fit``)
            is incompatible with the data window (``delay_min``, ``delay_max``)
            or empty; if neither or both of ``target_c`` and ``target_A`` are
            specified; if ``exp_feature_info`` has an unexpected (legacy)
            format; or if the fit window contains fewer than one sample.
        FileNotFoundError: If the QPU is offline and the fallback pickle
            caches for solver properties or experimental feature info are
            missing.
        RuntimeError: If the QPU is unavailable and no cached data is found
            for a stage that requires new sampling.
    """

    print()
    print(
        "A variety of plots are shown to demonstrate heuristic correction of "
        "flux_biases on detectors, and target qubit frequency "
        "desynchronization, from small amounts of data. "
    )
    # Schedule based approximations, target_A and dA/dc are approximated.
    print(f"Schedule file used: {schedule_fn}")
    qpu_anneal_schedule = pd.read_excel(
        schedule_fn, sheet_name="Fast-Annealing Schedule"
    )
    plt.figure("Schedule")
    plt.title("Annealing Schedule")
    delta_vs_s = qpu_anneal_schedule[::-1]
    plt.plot(delta_vs_s["s"], delta_vs_s["A(s) (GHz)"], label="A(s)")
    plt.plot(delta_vs_s["s"], delta_vs_s["B(s) (GHz)"], label="B(s)")
    if (target_c is None) == (target_A is None):
        raise ValueError("Exactly one of target_c or target_A must be specified.")
    if target_c is None:
        target_c = np.interp(
            target_A, delta_vs_s["A(s) (GHz)"], delta_vs_s["s"]
        )  # Expected normalized control bias at which to hold the qubits
    else:
        target_A = np.interp(
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
        target_A,
        "GHz B(target_c)",
        target_B,
        "GHz",
    )
    dc = 0.01
    target_Aminus = np.interp(
        1 - (target_c - dc), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )
    target_Aplus = np.interp(
        1 - (target_c + dc), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )
    dAdc = (target_Aplus - target_Aminus) / (2 * dc)
    plt.plot(
        [target_c, target_c],
        [0, np.max(delta_vs_s["A(s) (GHz)"])],
        label=f"c={target_c:.3g}",
    )
    plt.plot(
        [0, target_c - 0.01],
        [target_Aminus, target_Aminus],
        linestyle="dotted",
        color="black",
    )
    plt.plot(
        [target_c - 0.01, target_c - 0.01],
        [0, target_Aminus],
        linestyle="dotted",
        color="black",
    )
    plt.plot(
        [0, target_c + 0.01],
        [target_Aplus, target_Aplus],
        linestyle="dotted",
        color="black",
    )
    plt.plot(
        [target_c + 0.01, target_c + 0.01],
        [0, target_Aplus],
        linestyle="dotted",
        color="black",
    )
    plt.xlabel("Normalized control bias, c")
    plt.ylabel("Energy scale, GHz")
    plt.ylim([0, 2 * max(target_A, target_B)])
    plt.xlim([0, 1])
    plt.legend()

    if cache_str:
        qpu_fn = f"cache/qpu_{cache_str}.pkl"
    try:
        qpu = DWaveSampler(solver=solver)
        print(
            f"Solver connected to (check matches schedule file): {qpu.solver.identity}"
        )
        exp_feature_info = get_properties(qpu)
        if cache_str:
            with open(qpu_fn, "wb") as f:
                pickle.dump((qpu.properties, exp_feature_info), f)
        online = True
    except Exception as error:
        if not cache_str:
            raise (error)
        elif not os.path.isfile(qpu_fn):
            raise FileNotFoundError(
                f"use_cache=True, but cache files are missing and no client "
                f"is available: {error}"
            )
        else:
            with open(qpu_fn, "rb") as f:
                properties, exp_feature_info = pickle.load(f)
        qpu = MockDWaveSampler(
            properties=properties,
            nodelist=properties["qubits"],
            edgelist=properties["couplers"],
        )
        online = False
    if len(exp_feature_info) != 2:
        raise ValueError("Legacy format")
    line_assignments = {
        n: al_idx for al_idx, al in enumerate(exp_feature_info[1]) for n in al["qubits"]
    }
    num_lines = len(exp_feature_info[1])
    target_lines = set(range(num_lines)) - set(detector_lines) - set(source_lines)
    cmap = plt.colormaps.get_cmap("plasma")

    line_color = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=target_lines,
        target_c=target_c,
        detector_lines=detector_lines,
        source_lines=source_lines,
        use_01_c_range=use_01_c_range,
        use_common_bounds=use_common_bounds,
        symmetrize_c_bounds=symmetrize_c_bounds,
    )
    _plot_tds_schedules(
        x_polarizing_schedule,
        x_anneal_schedules,
    )
    # x_schedule_delays = make_tds_x_schedule_delays(
    #    x_anneal_schedules=x_anneal_schedules,
    #    quenched_lines=set(detector_lines) | set(source_lines),
    #    target_c=target_c,
    #    decimal_places=6,
    # )  # Quench rates implied by linear PWL are unreliable, especially with overshoot.
    x_schedule_delays = [0.0] * num_lines

    dt0 = 1 / target_A / 1000 / 4  # Appropriate scale for frequency resolution.
    delay_max_art = 2 * T2
    delay_min_art = 0.0
    delays = np.linspace(
        delay_min_art,
        delay_max_art,
        round((delay_max_art - delay_min_art) / dt0) + 1,
        endpoint=True,
    )
    dt = dt0  # Rounding

    dt_hd = 0.00001  # 0.01 nanoseconds
    high_density_delays = np.linspace(
        delay_min_art,
        delay_max_art,
        round((delay_max_art - delay_min_art) / dt_hd),
        endpoint=False,
    )
    dt_hd = high_density_delays[1] - high_density_delays[0]  # Rounding
    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Demonstrate simple model data.")
    print("Model: y(t) = cos(2*pi*(A+dA)*(t+dt)) * exp(-gamma*(t+dt))")
    print("Includes sampling error, detector vs source delays, and perturbations.")
    for A in [target_Aminus, target_A, target_Aplus]:
        delay_perturbation = 1 / (A * 1000) * np.random.random()  # up to 1 cycle.
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

        # Artificial pi/2 pulse preparation and energy basis measurement
        signal = artificial_data(
            delays,
            A * 1000,
            num_independent_samples=num_reads,
            theta_d=0.0,
            t0=delay_perturbation,
        )
        high_density_signal = artificial_data(
            high_density_delays + delay_perturbation,
            A * 1000,
            num_independent_samples=float("Inf"),  # No noise
            theta_d=0.0,
        )
        fig = plt.figure("Energy basis measurement")
        plt.title("y=1 - exp(-t/T2) + sampling error")
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
        plt.tight_layout()

    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next (experimental) stages.")
    plt.show()

    anneal_offsets = [0.0] * qpu.properties["num_qubits"]
    flux_biases = [0.0] * qpu.properties["num_qubits"]
    sampling_params = dict(
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
        if line in detector_lines:
            return "detector"
        elif line in source_lines:
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

    embs = [emb for i in target_lines for emb in embs_by_line[i]]
    n_embs = len(embs)

    sampler = ParallelEmbeddingComposite(qpu, embeddings=embs)

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes()}, {e: -1 for e in S.edges()}
    )

    if apply_flux_bias_shim != "None":
        x_polarizing_schedule = sampling_params.pop("x_polarizing_schedule")
        fn_cache = f"cache/FB_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                flux_biases, flux_history, mag_history = pickle.load(f)
        else:
            if not online:
                raise RuntimeError("QPU not available, and no cached data found.")
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
                if line_assignments[n] in detector_lines
            }

            if apply_flux_bias_shim == "TDS":
                print(
                    "Shim flux biases for zero magnetization on detector "
                    "and target qubits at equilibrium (with sources depolarized)"
                )
                print()

                flux_biases, flux_history, mag_history = shim_tds_flux_biases(
                    bqm=bqm_embedded,
                    sampler=qpu,
                    sampling_params=sampling_params,
                    target_lines=set(target_lines),
                    detector_lines=set(detector_lines),
                    line_assignments=line_assignments,
                    # exp_feature_line_info=exp_feature_info[1],  # Can replace explicit sampling_params
                    # target_c=target_c,  # Can replace explicit sampling_params
                )
            elif apply_flux_bias_shim == "Detector":
                print(
                    "Shim flux biases for zero detector magnetization in"
                    " the limit of long delay (at equilibrium)."
                )
                print()
                # Deep for float Sequence with ndarray or list of floats:
                x_schedule_delays = sampling_params["x_schedule_delays"].copy()
                for line in detector_lines:
                    sampling_params["x_schedule_delays"][
                        line
                    ] = 0.1  # Documented limit.
                # with open('polarized17.pkl', 'wb') as f:
                #     ising = bqm_embedded.to_ising()
                #     pickle.dump((sampling_params, ising, qpu.solver.identity, shimmed_variables), f)  # Reinstate for debugging
                flux_biases, flux_history, mag_history = shim_flux_biases(
                    bqm=bqm_embedded,
                    sampler=qpu,
                    sampling_params=sampling_params,
                    shimmed_variables=shimmed_variables,
                )
                sampling_params["x_schedule_delays"] = x_schedule_delays
            else:
                raise ValueError("Unknown method")
            polarization_candidates = [
                (i, flux_biases[i])
                for i in range(len(flux_biases))
                if abs(flux_biases[i]) > 1e-4
            ]
            if polarization_candidates:
                print(
                    "WARNING: Anomalously large flux biases could indicate "
                    "a calibration issue, check magnetization plots "
                    "for evidence of polarization and report bad qubits."
                )
                print(polarization_candidates)
            if cache_str:
                os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
                with open(fn_cache, "wb") as f:
                    pickle.dump((flux_biases, flux_history, mag_history), f)
        plot_shim(
            mag_history,
            flux_history,
        )
        sampling_params["flux_biases"] = flux_biases
        print("flux_biases refinement complete.")
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next (experimental) stages.")
        plt.show()
        sampling_params["x_polarizing_schedule"] = x_polarizing_schedule

    if source_decoupling_detection:
        stage_idx += 1
        print()
        print(f"Stage {stage_idx}: Detect delay for source decoupling.")
        print(
            "Whilst the source is coupled a polarized signal is detected."
            "Larmour precision proceeds from the point where decoupling occurs, "
            "determine this processor and PWL anneal_schedules specific value."
        )
        print(
            "Quality of outcome is contingent on decent calibration of flux_biases (theta_s~pi/2, theta_d~pi/2, target near-enough degeneracy at equilibrium)"
        )
        t_decoupled, t_mags = estimate_decoupling_timescale(
            sampler=sampler,
            bqm=bqm,
            sampling_params=sampling_params,
            detector_lines=detector_lines,
            target_A=target_A * 1000,
        )
        plt.figure("source_decoupling")
        x = [t for t, _ in t_mags]
        y = [mag for _, mag in t_mags]
        plt.plot(
            x,
            y,
            linestyle="None",
            marker=".",
        )
        plt.xlabel("Time")
        plt.ylabel("Magnitude")
        plt.title("Source Decoupling Detection")
        plt.axvline(
            t_decoupled, color="r", linestyle="--", label="Estimated Decoupling Time"
        )
        plt.legend()
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next (experimental) stages.")
    if delay_min is None:
        delay_min = t_decoupled + 1 / (target_A * 1000)  # Ignore first cycle.
    if delay_min_fit is None:
        delay_min_fit = delay_min
    if delay_max is None:
        delay_max = delay_min + 2 * T2
    if delay_max_fit is None:
        delay_max_fit = delay_max  # Can be automated for SNR in principle.
    if not (delay_min <= delay_min_fit < delay_max_fit <= delay_max):
        raise ValueError("The fit window is incompatible with the data window")

    delays = np.linspace(delay_min, delay_max, round((delay_max - delay_min) / dt0) + 1)
    dt = delays[1] - delays[0]

    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Collect data for {n_embs} parallel embeddings.")
    print(
        "Apply delays on detector line vs source, sampling at twice the Nyquist frequency."
    )
    fn_cache = f"cache/AO_It0_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
        if not online:
            raise RuntimeError("QPU not available, and no cached data found.")

        mean_Z_detector = run_parallel_experiment(
            sampler, bqm, sampling_params, delays, detector_lines
        )
        if cache_str:
            os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
            np.save(fn_cache, mean_Z_detector)

    first = np.argmax(delays >= delay_min_fit)
    last = np.argmax(delays >= delay_max_fit) + 1
    ld = last - first
    if ld < 1:
        raise ValueError("Fit window is empty: t-fit range too small for target_A")

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

    line_exemplars = {line_assignments[emb[0][0]]: idx for idx, emb in enumerate(embs)}

    plt.figure("Timeseries")
    plt.title("Time series for several qubits using distinct target lines")
    _plot_time_series(
        embs,
        line_assignments,
        mean_Z_detector,
        delays * 1000,
        line_color,
        plotted_emb_idxs=line_exemplars.values(),
        label_emb_idxs=line_exemplars.values(),
    )
    for colormap_type in ["divergent", "default"]:
        imshow_data(
            mean_Z_detector=mean_Z_detector,
            delays=delays,
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
        label_emb_idxs=line_exemplars.values(),
        xlabel=r"Frequency ($\omega$), GHz",
        ylabel=r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$",
    )
    plt.plot(
        [target_A, target_A],
        [0, np.max(psd)],
        color="black",
        linestyle="dashed",
        label="Schedule prediction",
    )

    # Calculate anneal_offsets for synchronization
    anneal_offsets = y = _calc_anneal_offsets(
        frequencies, psd, target_A, dAdc
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
                raise RuntimeError("QPU not available, and no cached data found.")

            for emb, ao in zip(embs, anneal_offsets):
                sampling_params["anneal_offsets"][
                    emb[0][0]
                ] -= ao  # Apply correction to target on each embedding
            mean_Z_detector = run_parallel_experiment(
                sampler, bqm, sampling_params, delays, detector_lines
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
        plt.title("Time series after anneal_offsets")
        _plot_time_series(
            embs,
            line_assignments,
            mean_Z_detector,
            delays * 1000,
            line_color,
            plotted_emb_idxs=line_exemplars.values(),
            label_emb_idxs=line_exemplars.values(),
        )
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
        _plot_time_series(
            embs,
            line_assignments,
            psd[:, : ld // 2].T,
            frequencies[: ld // 2],
            line_color,
            label_emb_idxs=line_exemplars.values(),
            xlabel=r"Frequency ($\omega$), GHz",
            ylabel=r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$",
        )
        plt.plot(
            [target_A, target_A],
            [0, np.max(psd)],
            color="black",
            linestyle="dashed",
            label="Schedule prediction",
        )
        plt.legend()

        plt.figure("AnnealOffsets")
        anneal_offsets0 = anneal_offsets
        anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding
        legend_idxs = set(line_exemplars.values())
        for emb_idx, emb in enumerate(embs):
            q = emb[0][0]
            line_target = line_assignments[q]
            if emb_idx in legend_idxs:
                label = f"target-qubit line={line_target}"
            else:
                label = None
            plt.plot(
                anneal_offsets0[emb_idx],
                anneal_offsets[emb_idx],
                color=line_color[line_target],
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
        description=(
            "Target-detector-source embedding demo with optional flux-bias "
            "shimming and anneal-offset refinement."
        )
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help=(
            "Cache and reload experiment artifacts keyed by CLI parameters. "
            "If solver graph_id changes, cached embeddings may be invalid."
        ),
    )
    parser.add_argument(
        "--solver-name",
        dest="solver_name",
        type=str,
        help="QPU solver name. Default research system with fast reverse anneal.",
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--detector-lines",
        dest="detector_lines",
        type=int,
        nargs="+",
        help="Detector lines (one or more integer indices).",
        default=[0],  # First vertical qubit line
    )
    parser.add_argument(
        "--source-lines",
        dest="source_lines",
        type=int,
        nargs="+",
        help="Source lines (one or more integer indices).",
        default=[3],  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for embedding generation.",
        default=None,
    )
    parser.add_argument(
        "--max-num-embeddings",
        dest="max_num_embeddings",
        type=int,
        help="Max embeddings to find (default: all available).",
        default=None,
    )
    parser.add_argument(
        "--target-A",
        dest="target_A",
        type=float,
        help="Expected qubit frequency (GHz). Schedule infers corresponding target_c.",
        default=2.0,
    )
    parser.add_argument(
        "--apply-flux-bias-shim",
        dest="apply_flux_bias_shim",
        type=str,
        choices=["None", "Detector", "TDS"],
        default="TDS",
        help="Flux-bias shimming mode: 'None' disables shimming, "
        "'Detector' shims detector qubits to zero measured magnetization, "
        "and 'TDS' alternates detector/target roles for TDS shimming "
        " (this can cause divergences, particularly for small frequencies and "
        "when target_c is desynchronized.",
    )
    parser.add_argument(
        "--delay-min",
        dest="delay_min",
        type=float,
        help="Initial delay time (us) for data collection",
        default=None,
    )
    parser.add_argument(
        "--delay-max",
        dest="delay_max",
        type=float,
        help="Final delay time (us) for data collection",
        default=None,  # Oscillations not completely decayed
    )
    parser.add_argument(
        "--delay-min-fit",
        dest="delay_min_fit",
        type=float,
        help="Initial delay (us) for frequency estimation (default: matches delay_min). "
        "Choose smallest delay with non-polarized signal.",
        default=None,
    )
    parser.add_argument(
        "--delay-max-fit",
        dest="delay_max_fit",
        type=float,
        help="Final delay (us) for frequency estimation (default: matches delay_max). "
        "Choose largest delay with low noise.",
        default=None,
    )
    parser.add_argument(
        "--no-source-decoupling-detection",
        dest="source_decoupling_detection",
        action="store_false",
        default=True,
        help="Disable detection of the delay required for source decoupling.",
    )
    parser.add_argument(
        "--skip-anneal-offset-verification",
        dest="skip_anneal_offset_verification",
        action="store_true",
        help="Skip the data analysis stage with anneal offsets applied.",
    )
    parser.add_argument(
        "--use-01-c-range",
        dest="use_01_c_range",
        action="store_true",
        help=(
            "Restrict generated schedule c-range to [0, 1]. This changes "
            "detector/source quench behavior."
        ),
    )
    parser.add_argument(
        "--no-symmetrize-c-bounds",
        dest="symmetrize_c_bounds",
        action="store_false",
        default=True,
        help="Disable symmetric c-bounds around target_c when building schedules.",
    )
    parser.add_argument(
        "--no-common-bounds",
        dest="use_common_bounds",
        action="store_false",
        default=True,
        help="Disable common c-bounds alignment across annealing lines.",
    )
    parser.add_argument(
        "--schedule-fn",
        dest="schedule_fn",
        type=str,
        help="Path to the annealing schedule Excel file (.xlsx). Should be matched to the solver.",
        default="09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    )
    parser.add_argument(
        "--save-figures",
        dest="save_figures",
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
        detector_lines=tuple(args.detector_lines),
        source_lines=tuple(args.source_lines),
        target_A=args.target_A,
        schedule_fn=args.schedule_fn,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        delay_min_fit=args.delay_min_fit,
        delay_max_fit=args.delay_max_fit,
        source_decoupling_detection=args.source_decoupling_detection,
        verify_anneal_offsets=not args.skip_anneal_offset_verification,
        apply_flux_bias_shim=args.apply_flux_bias_shim,
        use_01_c_range=args.use_01_c_range,
        symmetrize_c_bounds=args.symmetrize_c_bounds,
        use_common_bounds=args.use_common_bounds,
        save_figures=args.save_figures,
    )
