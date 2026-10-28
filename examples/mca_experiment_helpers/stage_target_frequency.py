"""Stage 5 and verification: target-frequency alignment via anneal offsets."""

from .common import *


def run_anneal_offsets(ctx: MCAContext) -> MCAContext:
    """Estimate anneal_offsets that align target frequencies to ``target_A``.

    Collects a detector-magnetization time series, estimates the power-spectral
    density per embedding, and proposes per-qubit anneal_offsets that center the
    PSD peak on the target frequency. Produces the time series, heatmap, PSD and
    proposed-offset figures.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.

    Returns:
        The same ``ctx``, updated with the collected data, PSD and proposed
        ``anneal_offsets_target``.
    """
    compute_delay_grid(ctx, ctx.fit_window, rel_sampling_rate=2, rel_start=1.0)
    stage_idx = ctx.stage_idx
    verify_anneal_offsets = ctx.verify_anneal_offsets
    n_embs = ctx.n_embs
    schedule_fn = ctx.schedule_fn
    target_A = ctx.target_A
    cache_str = ctx.cache_str
    online = ctx.online
    sampler = ctx.sampler
    bqm = ctx.bqm
    sampling_params = ctx.sampling_params
    detector_lines = ctx.detector_lines
    delays = ctx.delays
    delay_min = ctx.delay_min
    delay_max = ctx.delay_max
    dt = ctx.dt
    embs = ctx.embs
    line_assignments = line_assignments_from_feature_info(ctx.exp_feature_info)
    line_color = ctx.line_color
    line_exemplars = ctx.line_exemplars
    colormap_type = ctx.colormap_type
    cumsum_by_line = ctx.cumsum_by_line
    dAdc = ctx.dAdc
    save_figures = ctx.save_figures

    print()
    print_once = (
        "Collecting data by default in the interval ~[0, T2] at a rate close to twice the Nyquist frequency "
        "associated to the target frequency allows the power-spectral density to be estimated in good agreement "
        "with a Lorentzian. The peak can be robustly estimated. "
        "Estimators of higher accuracy and efficiency are available exploiting phase information and "
        "collecting data at higher sampling rates, but this power spectral density method is "
        "sufficient to resolve some discrepancies on typical Advantage2 research processors."
    )

    stage_idx += 1
    print(
        f"Stage {stage_idx}: Compensate for per-qubit target-frequency variation to reach {target_A:.3g}GHz (with anneal_offsets, for all {n_embs} parallel embeddings)."
    )
    print(print_once)
    print(f"1-step anneal offset correction, parameterized by {schedule_fn}. ")

    fn_cache = f"cache/AO_It0_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
        if not online:
            raise RuntimeError("QPU not available, and no cached data found.")

        mean_Z_detector = run_parallel_experiment(
            sampler, bqm, sampling_params, detector_lines, delays=delays
        )
        if cache_str:
            np.save(fn_cache, mean_Z_detector)

    first = np.argmax(delays >= delay_min)
    last = np.argmax(delays >= delay_max) + 1
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

    # Plot data #
    timeseries_fig, (ax_first_iter_timeseries, ax_second_iter_timeseries) = (
        plt.subplots(1, 2, figsize=(12, 5), num="Timeseries", constrained_layout=True)
    )
    timeseries_fig.suptitle(
        "Time series for several qubits using distinct target lines"
    )
    ax_first_iter_timeseries.set_title("Before anneal-offset refinement")
    ax_second_iter_timeseries.set_title("After anneal-offset refinement")
    _plot_time_series(
        embs,
        line_assignments,
        mean_Z_detector,
        delays * 1000,
        line_color,
        plotted_emb_idxs=line_exemplars.values(),
        label_emb_idxs=line_exemplars.values(),
        ax=ax_first_iter_timeseries,
    )
    heatmap_after_axes = {}

    heatmap_fig, (ax_first_iter_heatmap, ax_second_iter_heatmap) = plt.subplots(
        2,
        1,
        figsize=(8, 10),
        num=f"Timeseries_{colormap_type}_colormap",
        constrained_layout=True,
    )
    # Add vertical padding so each subplot title stays legible.
    heatmap_fig.get_layout_engine().set(hspace=0.15)
    heatmap_fig.suptitle(f"Detector magnetization ({colormap_type} colormap)")
    ax_first_iter_heatmap.set_title("Before anneal-offset refinement")
    ax_second_iter_heatmap.set_title("After anneal-offset refinement")
    imshow_data(
        mean_Z_detector=mean_Z_detector,
        delays=delays,
        colormap_type=colormap_type,
        first=first,
        last=last,
        ax=ax_first_iter_heatmap,
        vertical_lines=[v - 0.5 for v in cumsum_by_line.values()],
    )
    heatmap_after_axes[colormap_type] = ax_second_iter_heatmap

    psd_fig, (ax_first_iter_psd, ax_second_iter_psd) = plt.subplots(
        1, 2, figsize=(12, 5), num="PSD", constrained_layout=True
    )
    psd_fig.suptitle("Power associated with magnetization time series")
    ax_first_iter_psd.set_title("Before anneal-offset refinement")
    ax_second_iter_psd.set_title("After anneal-offset refinement")
    _plot_time_series(
        embs,
        line_assignments,
        psd[:, : ld // 2].T,
        frequencies[: ld // 2],
        line_color,
        label_emb_idxs=line_exemplars.values(),
        xlabel=r"Frequency ($\omega$), GHz",
        ylabel=r"Power Spectral Density, $|\langle Z\rangle(\omega)|^2$",
        ax=ax_first_iter_psd,
    )
    ax_first_iter_psd.plot(
        [target_A, target_A],
        [0, np.max(psd)],
        color="black",
        linestyle="dashed",
        label="Schedule prediction",
    )
    ax_first_iter_psd.legend()

    # Calculate anneal_offsets for synchronization
    anneal_offsets_target = y = _calc_anneal_offsets(
        frequencies, psd, target_A, dAdc
    )  # Per embedding
    # anneal offsets can be realized line-wise by changing target_c,
    # or qubit-wise by modification of anneal_offset. We can
    # correct for the mean with a line_offset, and then qubit-wise
    # variation with the anneal offset.

    plt.figure("Proposed anneal_offsets")
    plt.plot(
        sorted(y),
        np.arange(len(y)) / len(y),
        label=f"RMS(It=1)={np.sqrt(np.mean(np.array(y)**2)):.3g}",
    )
    plt.xlabel("Proposed anneal offset")
    plt.ylabel("Cumulative distribution function")
    plt.legend()
    if not verify_anneal_offsets:
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next stages.")
        _apply_tight_layout()
        plt.show()

    ctx.stage_idx = stage_idx
    ctx.mean_Z_detector = mean_Z_detector
    ctx.first = first
    ctx.last = last
    ctx.frequencies = frequencies
    ctx.psd = psd
    ctx.anneal_offsets_target = anneal_offsets_target
    ctx.ax_second_iter_timeseries = ax_second_iter_timeseries
    ctx.ax_second_iter_psd = ax_second_iter_psd
    ctx.heatmap_after_axes = heatmap_after_axes
    return ctx


def run_verification(ctx: MCAContext) -> MCAContext:
    """Re-measure after applying anneal_offsets and verify the alignment.

    Applies the proposed ``anneal_offsets_target`` to the targets, re-collects
    the detector magnetization, re-estimates the PSD, and fills the "after"
    panels of the time series, heatmap and PSD figures alongside an
    offset-correlation plot.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.

    Returns:
        The same ``ctx``, updated with the verification data, PSD and refined
        ``anneal_offsets_target``.
    """
    stage_idx = ctx.stage_idx
    anneal_offsets_target = ctx.anneal_offsets_target
    cache_str = ctx.cache_str
    online = ctx.online
    embs = ctx.embs
    sampling_params = ctx.sampling_params
    sampler = ctx.sampler
    bqm = ctx.bqm
    detector_lines = ctx.detector_lines
    delays = ctx.delays
    first = ctx.first
    last = ctx.last
    ld = last - first
    frequencies = ctx.frequencies
    dAdc = ctx.dAdc
    target_A = ctx.target_A
    line_assignments = line_assignments_from_feature_info(ctx.exp_feature_info)
    line_color = ctx.line_color
    line_exemplars = ctx.line_exemplars
    colormap_type = ctx.colormap_type
    cumsum_by_line = ctx.cumsum_by_line
    ax_second_iter_timeseries = ctx.ax_second_iter_timeseries
    ax_second_iter_psd = ctx.ax_second_iter_psd
    heatmap_after_axes = ctx.heatmap_after_axes

    anneal_offsets0_target = anneal_offsets_target
    stage_idx += 1
    print()
    print(
        f"Stage {stage_idx}: Verify and refine the target-frequency alignment (with anneal_offsets, before and after adjustments)."
    )
    fn_cache = f"cache/AO_It1_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
        if not online:
            raise RuntimeError("QPU not available, and no cached data found.")
        for emb, ao in zip(embs, anneal_offsets_target):
            sampling_params["anneal_offsets"][
                emb[0][0]
            ] -= ao  # Apply correction to target on each embedding
        mean_Z_detector = run_parallel_experiment(
            sampler, bqm, sampling_params, detector_lines, delays=delays
        )
        if cache_str:
            np.save(fn_cache, mean_Z_detector)
    psd = np.array(
        [
            np.abs(np.fft.fft(mean_Z_detector[first:last, i])) ** 2
            for i in range(len(embs))
        ]
    ) / (last - first)
    y = anneal_offsets_target = _calc_anneal_offsets(
        frequencies, psd, target_A, dAdc
    )  # Per embedding

    for emb, ao in zip(embs, anneal_offsets_target):
        sampling_params["anneal_offsets"][
            emb[0][0]
        ] -= ao  # Apply correction to target on each embedding

    plt.figure("Proposed anneal_offsets")
    plt.plot(
        sorted(anneal_offsets_target),
        np.arange(len(anneal_offsets_target)) / len(anneal_offsets_target),
        label=f"RMS(It=2)={np.sqrt(np.mean(np.array(anneal_offsets_target)**2)):.3g}",
    )
    plt.legend()

    _plot_time_series(
        embs,
        line_assignments,
        mean_Z_detector,
        delays * 1000,
        line_color,
        plotted_emb_idxs=line_exemplars.values(),
        label_emb_idxs=line_exemplars.values(),
        ax=ax_second_iter_timeseries,
    )

    imshow_data(
        mean_Z_detector=mean_Z_detector,
        delays=delays,
        colormap_type=colormap_type,
        first=first,
        last=last,
        ax=heatmap_after_axes[colormap_type],
        vertical_lines=[v - 0.5 for v in cumsum_by_line.values()],
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
        ax=ax_second_iter_psd,
    )
    ax_second_iter_psd.plot(
        [target_A, target_A],
        [0, np.max(psd)],
        color="black",
        linestyle="dashed",
        label="Schedule prediction",
    )
    ax_second_iter_psd.legend()

    plt.figure("AnnealOffsets")
    legend_idxs = set(line_exemplars.values())
    for emb_idx, emb in enumerate(embs):
        q = emb[0][0]
        line_target = line_assignments[q]
        if emb_idx in legend_idxs:
            label = f"target-qubit line={line_target}"
        else:
            label = None
        plt.plot(
            anneal_offsets0_target[emb_idx],
            anneal_offsets_target[emb_idx],
            color=line_color[line_target],
            marker="x",
            label=label,
        )
    plt.xlabel("Estimated anneal-offset correction (baseline)")
    plt.ylabel("Estimated anneal-offset correction (after refinement)")
    plt.grid(True)
    plt.legend()

    print(
        f"Left/right subplots are divided into {len(cumsum_by_line)} ordered panels, each T-D-S line combination programmed in parallel. "
        "Inhomogeneity of T can impact frequency, whereas D and S variation impacts synchronization (delay). "
        "Ideally no-refinement of anneal offsets are required (left subplot anneal_offsets=0). If effective, "
        "anneal_offsets (and refined line delays) improve synchronization (right subplot). "
        "Note: both plots by default already incorporate a bulk line-delay alignment and shimming of flux_biases."
    )

    ctx.stage_idx = stage_idx
    ctx.anneal_offsets_target = anneal_offsets_target
    ctx.mean_Z_detector = mean_Z_detector
    ctx.psd = psd
    return ctx
