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
An example to show embedding for multicolor annealing.
"""

import argparse
import hashlib
import json
import os
import re

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import dimod
from dwave.system import DWaveSampler
from dwave.system.composites import ParallelEmbeddingComposite

from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.experimental.multicolor_anneal import (
    get_properties,
    make_tds_graph,
    make_tds_x_schedules,
    qubit_to_Advantage2_annealing_line,  # Per comments, requires modification subject to dwave-experimental/pull/52
    SOLVER_FILTER,
    standardize_schedule_endpoints,
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
    target_A: float,
    dAdc: float,
    Amin: float | None = None,
    Amax: float | None = None,
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
        frequencies: frequencies at which power provided.
        psd: power spectral density, the absolute discrete fourier transform
            value squared at each frequency.
        target_A: expected/desired peak position.
        dAdc: Approximate rate of change of A with c (anneal offset).
        Amin: A lower bound on the frequency range used in estimation,
        Amax: An upper bound on the frequency range used in estimation,

    Returns:
        Estimated error on c relative to the mean value for a collection of

    """

    # NB a symmetric window only works for frequencies in the range,
    # and some bias is introduced by use of a target.
    if Amin is None:
        Amin = target_A / 2
    if Amax is None:
        Amax = target_A * 1.5

    Afilter = np.logical_and(frequencies < Amax, frequencies > Amin)
    mean_A_est = np.sum(
        psd[:, Afilter] * frequencies[Afilter][np.newaxis, :], axis=1
    ) / np.sum(psd[:, Afilter], axis=1)
    mu = np.mean(mean_A_est)
    print("Standard deviation in A estimates", np.sqrt(np.var(mean_A_est)))
    print()
    dcs = (mean_A_est - mu) / dAdc
    return dcs


def artificial_data(
    delays: np.ndarray,
    A: float,
    decay_time: float,
    num_independent_samples: int = float("Inf"),
    prng: np.random.Generator | int | None = None,
):
    """Create an artificial data set

    y(t) = np.exp(-delays / decay_time) * np.cos(2* np.pi * A * delays)
    with variance of (1 - y(t)^2) in the measured state. Given independent
    and identically distributed samples we can model noise as normally
    distributed.

    Args:
        delays: time of measurement
        A: frequency
        decay_time: exponential envelope time scale
        num_independent_samples: number of samples to model
        prng: pseudo random number generator or seed.
    Returns:
        A model signal:
    """
    y = np.exp(-delays / decay_time) * np.cos(2 * np.pi * A * delays)
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
        sampler: A parallel embedding composite sampler, wrapping the qpu sampler.
        bqm: Binary Quadratic Model
        qpu_parameters: parameters passed to the QPU sampler.
        delays: detector x_schedule_delays
        line_detector: detector line.

    Returns:
        A numpy array of detector magnetizations

    """
    mean_Z_detector = []
    for delay in tqdm(delays):
        qpu_parameters["x_schedule_delays"][line_detector] = delay
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
    plt.title("All detector flux_biases")
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
    plt.figure("PWL_waveforms")
    plt.title("PWL waveforms")
    for line, schedule in enumerate(x_anneal_schedules):
        plt.plot(
            [x for x, _ in schedule], [y for _, y in schedule], label=f"Line {line}"
        )
    plt.plot(
        [x for x, _ in x_polarizing_schedule],
        [y for _, y in x_polarizing_schedule],
        label="Polarizing bias",
        linestyle="dashed",
        color="black",
    )
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Schedule value")
    plt.legend()


def _plot_tds_schedules(
    x_polarizing_schedule: list[list[float]],
    x_anneal_schedules: list[list[list[float]]],
):
    """Plots the piecewise linear schedules used

    Args:
        x_polarizing_biases: The polarization signal.
        x_anneal_schedules: The list of anneal schedules, one per line.
    """
    plt.figure()
    plt.title("PWL waveforms")
    for line, schedule in enumerate(x_anneal_schedules):
        plt.plot(
            [x for x, _ in schedule], [y for _, y in schedule], label=f"Line {line}"
        )
    plt.plot(
        [x for x, _ in x_polarizing_schedule],
        [y for _, y in x_polarizing_schedule],
        label="Polarizing bias",
        linestyle="dashed",
        color="black",
    )
    plt.xlabel("Time (microseconds)")
    plt.ylabel("Schedule value")
    plt.legend()


def _get_experiment_id(args):
    print(vars(args))
    vars_args = vars(args).copy()
    vars_args.pop(
        "save_figures", None
    )  # save_figures is not relevant to the experiment data, so we exclude it from the hash
    args_string = json.dumps(vars_args, sort_keys=True)
    return hashlib.sha256(args_string.encode("utf-8")).hexdigest()[:num_char]


def _fix_standard_c_range(anneal_schedules):
    for anneal_schedule in anneal_schedules:
        for tc in anneal_schedule:
            if tc[1] > 1.0:
                tc[1] = 1.0
            elif tc[1] < 0.0:
                tc[1] = 0.0


def main(
    cache_str: str | None = None,
    solver: dict | str | None = None,
    line_detector: int = 0,
    line_source: int = 3,
    seed: int | None = None,
    max_num_embeddings: int | None = None,
    target_c: float = 0.37,
    no_flux_biases: bool = False,
    no_anneal_offsets: bool = False,
    delay_min: float = 0.005,
    delay_max: float = 0.015,
    delay_min_fit: float | None = None,
    delay_max_fit: float | None = None,
    fn_schedule: str = "09-1317A-D_Advantage2_research1_4_annealing_schedule.xlsx",
    use_01_c_range: bool = False,
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
            A unique experimental idenfier. If not None a directory
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
            normalized control bias at which the target qubits are held
        no_flux_biases:
            When set to True, flux_biases are not modified. When False flux_biases
            are modified on detector qubits to achived zero expected magnetization at
            long delay.
        no_anneal_offsets:
            When set to True, anneal_offsets are not modified. When False anneal_offsets
            are modified so that the peak power-spectral density is peaked at a common
            value for all qubits. This peak values characterizes the frequency of the target
            qubit in simple well-calibratied models.
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

    Raises:
        ValueError: If the number of lines is less than 3, or
        if {detector_line, source_line} is not a size 2 subset of
        set(range(num_lines))
    """
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
    # Schedule based approximations, target_A and dA/dc are approximated.
    qpu_anneal_schedule = pd.read_excel(
        fn_schedule, sheet_name="Fast-Annealing Schedule"
    )
    plt.figure("Schedule")
    plt.title("Schedule")
    delta_vs_s = qpu_anneal_schedule[::-1]
    plt.plot(delta_vs_s["s"], delta_vs_s["A(s) (GHz)"], label="A(s)")
    plt.plot(delta_vs_s["s"], delta_vs_s["B(s) (GHz)"], label="B(s)")
    target_A = np.interp(
        1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
    )  # Expected frequency of detector magnetization oscillations
    target_B = np.interp(1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["B(s) (GHz)"])
    print("Schedule predictions: ", "A(c)", target_A, "B(c)", target_B)
    print()
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
        label=f"c={target_c}",
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

    qpu = DWaveSampler(solver=solver)
    zephyr_shape = qpu.properties["topology"]["shape"]
    exp_feature_info = get_properties(qpu)
    line_assignments = {
        n: al_idx for al_idx, al in enumerate(exp_feature_info) for n in al["qubits"]
    }
    num_lines = len(exp_feature_info)
    cmap = plt.colormaps.get_cmap("plasma")
    line_color = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=set(range(num_lines)) - {line_detector, line_source},
        target_c=target_c,
        detector_lines=(line_detector,),
        source_lines=(line_source,),
    )

    if use_01_c_range:
        _fix_standard_c_range(x_anneal_schedules)
    _plot_tds_schedules(
        x_polarizing_schedule,
        x_anneal_schedules,
    )
    x_schedule_delays = [0.0] * num_lines

    anneal_offsets = [0.0] * qpu.properties["num_qubits"]
    flux_biases = [0.0] * qpu.properties["num_qubits"]

    # See documented Larmour precession example
    qpu_parameters = dict(
        num_reads=500,
        answer_mode="raw",
        x_disable_filtering=True,
        x_schedule_delays=x_schedule_delays,
        x_anneal_schedules=x_anneal_schedules,
        x_polarizing_schedule=x_polarizing_schedule,
        flux_biases=flux_biases,
        anneal_offsets=anneal_offsets,
    )

    print(
        "Determine many T-D-S embeddings appropriate for parallel programming (see mca_embedding.py example)."
    )
    print()
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
    embs_by_line = {i: [] for i in range(num_lines)}
    for i, emb in enumerate(embs):
        q = emb[0][0]
        # The 6-line scheme is used, requires update after merge of this pull-eqest Excepts due to absence of num_lines support https://github.com/dwavesystems/dwave-experimental/pull/52
        # embs_by_line[qubit_to_Advantage2_annealing_line(q, zephyr_shape, num_lines=num_lines)].append(emb)
        # Applies to all instances of qubit_to_anneal_line
        embs_by_line[qubit_to_Advantage2_annealing_line(q, zephyr_shape)].append(emb)

    embs = [emb for i in range(num_lines) for emb in embs_by_line[i]]

    sampler = ParallelEmbeddingComposite(qpu, embeddings=embs)

    dt = 1 / target_A / 1000 / 4  # Appropriate scale for frequency resolution.
    delays = np.linspace(delay_min, delay_max, round((delay_max - delay_min) / dt) + 1)

    # Demonstrate some data for simple model y(t) = cos(2 pi A [t + t0]) exp(- [t + t0]/d):
    delays_ns = 5 * np.random.random() + 1000 * delays
    ld = len(delays_ns)
    frequencies = np.arange(ld) / dt / 1000 / ld
    decay_time_ns = 20
    for idx, A in enumerate([target_Aminus, target_A, target_Aplus]):
        for num_independent_samples in [100, float("Inf")]:
            signal = artificial_data(
                delays_ns,
                A,
                decay_time=decay_time_ns,
                num_independent_samples=num_independent_samples,
            )
            if num_independent_samples == float("Inf") and idx == 1:
                label = f"A={A:.3g}, no sample err."
            elif num_independent_samples == 100:
                label = f"A={A:.3g}"
            else:
                continue

            plt.figure("artificial_timeseries")
            plt.title("y=cos(2pi A t)exp(-t/T)+sampling error")
            plt.plot(delays_ns, signal, label=label)
            plt.xlabel("Time, microseconds")
            plt.ylabel(r"Magnetization, $\langle Z \rangle_{detector}$")
            plt.legend()

            plt.figure("artificial_psd")
            plt.title("Approx Lorentzian power spectral density ~ A/((f-A)^2 + A^2)")
            psd = np.abs(np.fft.fft(signal)) ** 2 / len(signal)
            plt.plot(frequencies[: ld // 2], psd[: ld // 2], label=label)
            plt.ylabel(rf"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
            plt.xlabel(r"Frequency ($\omega$), GHz")
            plt.legend()

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes()}, {e: -1 for e in S.edges()}
    )

    if not no_flux_biases:
        print(
            "Shim flux biases for zero detector magnetization in"
            " the limit of long delay (at equilibrium)."
        )
        print()
        fn_cache = f"cache/FB_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                flux_biases, flux_history, mag_history = pickle.load(f)
        else:
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
                if qubit_to_Advantage2_annealing_line(n, zephyr_shape) == line_detector
            }

            qpu_parameters["x_schedule_delays"][
                line_detector
            ] = 0.1  # Documented limit. TODO- grab from properties if possible.
            flux_biases, flux_history, mag_history = shim_flux_biases(
                bqm=bqm_embedded,
                sampler=qpu,
                sampling_params=qpu_parameters,
                shimmed_variables=shimmed_variables,
            )
            if cache_str:
                os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
                with open(fn_cache, "wb") as f:
                    pickle.dump((flux_biases, flux_history, mag_history), f)
        plot_shim(
            mag_history,
            flux_history,
        )
        qpu_parameters["flux_biases"] = flux_biases

    if save_figures:
        _save_open_figures("figures/", cache_str)
    plt.show()
    print(f"Collect data for {len(embs)} parallel embeddings")
    fn_cache = f"cache/AO_It0_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
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
        raise ValueError("Fit window is empty: t-fit range too small for target_A")

    frequencies = np.arange(ld) / dt / 1000 / ld
    psd = np.array(
        [
            np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
            for i in range(len(embs))
        ]
    ) / (last - first)

    print(
        "Plot real space data in 3 formats, and the power spectral density estimated by a discrete Fourier transform"
    )

    plt.figure("Timeseries")
    plt.title("Time series for several qubits using distinct target lines")
    plotted_embIs = set()
    for idx, emb in enumerate(embs):
        q = emb[0][0]
        plotted_embI = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
        if plotted_embI not in plotted_embIs:
            plt.plot(
                delays * 1000,
                mean_Z_detector[:, idx],
                color=line_color[plotted_embI],
                label=f"target line {plotted_embI}",
            )
            plotted_embIs.add(plotted_embI)
    plt.ylabel("Detector magnetizations")
    plt.xlabel("Detector delay, ns")
    plt.legend()
    plt.grid()

    plt.figure("Timeseries_divergent_colormap")
    plt.title("Real space magnetizations (divergent colormap)")
    plt.imshow(mean_Z_detector, vmin=-1, vmax=1, cmap="RdBu")
    yticks_dict = {
        first: f"{1000 * delays[first]:.3g}",
        last - 1: f"{1000 * delays[last-1]:.3g}",
    }
    yticks_dict.update(
        {0: str(1000 * delays[0]), mean_Z_detector.shape[0] - 1: str(1000 * delays[-1])}
    )
    plt.yticks(
        list(yticks_dict.keys()),
        list(yticks_dict.values()),
    )
    plt.xlabel("Target-Detector-Source embedding")
    plt.ylabel("Delay, nanoseconds")

    plt.figure("Timeseries_default_colormap")
    plt.title("Real space magnetizations (default colormap)")
    plt.imshow(mean_Z_detector[first:last, :])
    yticks_dictN = {
        0: f"{1000 * delays[first]:.3g}",
        last - first - 1: f"{1000 * delays[last-1]:.3g}",
    }
    plt.yticks(
        list(yticks_dictN.keys()),
        list(yticks_dictN.values()),
    )
    plt.xlabel("Target-Detector-Source embedding")
    plt.ylabel("Delay, nanoseconds")

    plt.figure("PSD")
    plt.title("Power associated to magnetization time series")
    lines_represented = set()
    for i, emb in enumerate(embs):
        q = emb[0][0]
        plotted_embI = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
        if plotted_embI in lines_represented:
            label = None
        else:
            label = f"target-qubit line={plotted_embI}"
            lines_represented.add(plotted_embI)
        plt.plot(
            frequencies[: ld // 2],
            psd[i, : ld // 2],
            color=line_color[plotted_embI],
            label=label,
        )
    plt.plot(
        [target_A, target_A],
        [0, np.max(psd)],
        color="black",
        linestyle="dashed",
        label="Schedule prediction",
    )
    plt.legend()
    plt.ylabel(rf"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
    plt.xlabel(r"Frequency ($\omega$), GHz")
    plt.grid(True)

    # Calculate anneal_offsets for synchronization
    if not no_anneal_offsets:
        anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding

        print("Collect data with anneal offset compensation of frequency variation")
        fn_cache = f"cache/AO_It1_{cache_str}.npy"
        if cache_str and os.path.isfile(fn_cache):
            mean_Z_detector = np.load(fn_cache)
        else:
            for emb, ao in zip(embs, anneal_offsets):
                qpu_parameters["anneal_offsets"][
                    emb[0][0]
                ] -= ao  # Apply correction to target on each embedding
            mean_Z_detector = run_parallel_experiment(
                sampler, bqm, qpu_parameters, delays, line_detector
            )
            if cache_str:
                np.save(fn_cache, mean_Z_detector)
        psd = np.array(
            [
                np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
                for i in range(len(embs))
            ]
        ) / (last - first)

        print(
            "Plot real space data in 3 formats, and the power spectral density estimated by a discrete Fourier transform"
        )

        plt.figure("Timeseries_after_anneal_offsets")
        plt.title("Time series after anneal_offsets")
        plotted_embIs = set()
        for i, emb in enumerate(embs):
            q = emb[0][0]
            plotted_embI = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
            if plotted_embI not in plotted_embIs:
                plt.plot(
                    delays * 1000,
                    mean_Z_detector[:, i],
                    color=line_color[plotted_embI],
                    label=f"target line {plotted_embI}",
                )
                plotted_embIs.add(plotted_embI)
        plt.ylabel("Detector magnetizations")
        plt.xlabel("Detector delay, ns")
        plt.legend()
        plt.grid()

        plt.figure("Timeseries_divergent_colormap_w_AO")
        plt.title("Real space magnetizations after anneal offsets")
        plt.imshow(mean_Z_detector, vmin=-1, vmax=1, cmap="RdBu")
        plt.yticks(
            list(yticks_dict.keys()),
            list(yticks_dict.values()),
        )
        plt.xlabel("Target-Detector-Source embedding")
        plt.ylabel("Delay, nanoseconds")

        plt.figure("Timeseries_default_colormap_w_AO")
        plt.title("Real space magnetizations after anneal offsets")
        plt.imshow(mean_Z_detector[first:last, :])
        plt.yticks(
            list(yticks_dictN.keys()),
            list(yticks_dictN.values()),
        )
        plt.xlabel("Target-Detector-Source embedding")
        plt.ylabel("Delay, nanoseconds")

        plt.figure("PSD_w_AO")
        plt.title("Power associated to magnetization time series after anneal offsets")
        lines_represented = set()
        for i, emb in enumerate(embs):
            q = emb[0][0]
            plotted_embI = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
            if plotted_embI in lines_represented:
                label = None
            else:
                label = f"target-qubit line={plotted_embI}"
                lines_represented.add(plotted_embI)
            plt.plot(
                frequencies[: ld // 2],
                psd[i, : ld // 2],
                color=line_color[plotted_embI],
                label=label,
            )
        plt.plot(
            [target_A, target_A],
            [0, np.max(psd)],
            color="black",
            linestyle="dashed",
            label="Schedule prediction",
        )
        plt.legend()
        plt.ylabel(rf"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$")
        plt.xlabel(r"Frequency ($\omega$), GHz")
        plt.grid(True)

        plt.figure("AnnealOffsets")
        anneal_offsets0 = anneal_offsets
        anneal_offsets = _calc_anneal_offsets(
            frequencies, psd, target_A, dAdc
        )  # Per embedding
        lines_represented = set()
        for i, emb in enumerate(embs):
            q = emb[0][0]
            plotted_embI = qubit_to_Advantage2_annealing_line(q, zephyr_shape)
            if plotted_embI in lines_represented:
                label = None
            else:
                label = f"target-qubit line={plotted_embI}"
                lines_represented.add(plotted_embI)
            plt.plot(
                anneal_offsets0[i],
                anneal_offsets[i],
                color=line_color[plotted_embI],
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
        help="Add this flag to save experimental data, and reload when available at command line parameters. Note that the QPU is identified only by the solver parameters - if graph_id changes new embeddings may be required.",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="Option to specify QPU solver, by default an experimental system supporting fast reverse anneal",
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
        help="Maximum number of embeddings to find. By default, all available embeddings are searched.",
        default=None,
    )
    parser.add_argument(
        "--target_c",
        type=float,
        help="The normalized phi_cjj value for the target. This should correspond to a frequency A(c_target) of approximately 1 to 3GHz for reasonable performance. The fn_schedule file can be used to infer an approximate relationship between A(target_c) and target_c.",
        default=0.37,  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--delay_min",
        type=float,
        help="Initial delay time (us) for data collection.",
        default=0.005,
    )
    parser.add_argument(
        "--delay_max",
        type=float,
        help="Final delay time (us) for data collection.",
        default=0.015,
    )
    parser.add_argument(
        "--delay_min_fit",
        type=float,
        help="Initial delay time (us) for frequency estimation, by default matches delay_min.  This is ideally chosen to match the smallest delay for which the signal is not polarized.",
        default=None,  # Matches delay_min by default
    )
    parser.add_argument(
        "--delay_max_fit",
        type=float,
        help="Final delay time (us) for frequency estimation, by default matches delay_max_fit. This is ideally chosen to match the largest delay for which the signal is not dominated by noise, or can be smaller to accommodate reduced runtime.",
        default=None,  # Matches delay_max by default
    )
    parser.add_argument(
        "--no_flux_biases",
        action="store_true",
        help="Add this flag to omit the flux_bias shimming calibration stage (simulaton run exclusively with flux_biases=[0.0]*num_qubits)",
    )
    parser.add_argument(
        "--no_anneal_offsets",
        action="store_true",
        help="Add this flag to omit the data analsis with anneal_offsets set  (simulaton run exclusively with anneal_offsets=[0.0]*num_qubits)",
    )
    parser.add_argument(
        "--use_01_c_range",
        action="store_true",
        help="Add this flag to use a schedule range restricted to [minC, maxC] = [0,1]. This lowers the detector and source quench rates, impacting fidelity and some other parameters. TODO later - support symmetrized crange (for better quenbch rate but maintaining delay regularization, or overshoot crange (for higher performance)",
    )
    parser.add_argument(
        "--save_figures",
        action="store_true",
        help="Add this flag to save figures generated during the experiment to a figures folder. The experiment hash is appended to figure filenames.",
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
        target_c=args.target_c,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        delay_min_fit=args.delay_min_fit,
        delay_max_fit=args.delay_max_fit,
        no_anneal_offsets=args.no_anneal_offsets,
        no_flux_biases=args.no_flux_biases,
        use_01_c_range=args.use_01_c_range,
        save_figures=args.save_figures,
    )
