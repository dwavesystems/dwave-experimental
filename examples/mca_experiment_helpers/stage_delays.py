"""Stages 4 and 6: detect and refine source-decoupling delays.

Both stages share the same bisection-based decoupling-time estimator; the refine
stage additionally applies per-qubit detector anneal_offsets.
"""

from .common import *


def run_detect_delays(ctx: MCAContext) -> MCAContext:
    """Detect the source-decoupling delay for the bulk of parallel embeddings.

    When ``ctx.t_decoupled`` is None, estimate the delay at which the detector
    decouples from the source (and plot the detection), otherwise apply the
    user-provided scalar delay to every detector line. The resolved delays are
    added to ``sampling_params['x_schedule_delays']``.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.

    Returns:
        The same ``ctx``, updated with the per-line decoupling delays.
    """
    stage_idx = ctx.stage_idx
    t_decoupled = ctx.t_decoupled
    cache_str = ctx.cache_str
    n_embs = ctx.n_embs
    sampler = ctx.sampler
    bqm = ctx.bqm
    sampling_params = ctx.sampling_params
    detector_lines = ctx.detector_lines
    target_A = ctx.target_A
    line_assignments = line_assignments_from_feature_info(ctx.exp_feature_info)
    source_lines = ctx.source_lines
    by_det_src_pair = ctx.by_det_src_pair
    save_figures = ctx.save_figures
    online = ctx.online

    if t_decoupled is None:
        stage_idx += 1
        print()
        print(
            f"Stage {stage_idx}: Compensate for source-detector decoupling timing (with x_schedule_delays, for bulk of {n_embs} parallel embeddings)."
        )
        print(
            "Whilst the source is coupled a polarized signal is detected. "
            "Larmor precession proceeds from the point where decoupling occurs; "
            "determine this processor- and anneal-schedule-specific value."
        )

        fn_cache = f"cache/source_decoupling_{cache_str}.pkl"
        if cache_str and os.path.isfile(fn_cache):
            with open(fn_cache, "rb") as f:
                t_decoupled, t_mags, t_decoupled_by_qubit = pickle.load(f)
        else:
            if not online:
                raise RuntimeError("QPU not available, and no cached data found.")
            t_decoupled, t_mags, t_decoupled_by_qubit = estimate_decoupling_timescale(
                sampler=sampler,
                bqm=bqm,
                sampling_params=sampling_params,
                detector_lines=detector_lines,
                target_A=target_A * 1000,
                line_assignments=line_assignments,
                source_lines=source_lines,
                by_det_src_pair=by_det_src_pair,
            )
            if cache_str:
                with open(fn_cache, "wb") as f:
                    pickle.dump((t_decoupled, t_mags, t_decoupled_by_qubit), f)

        delays_arr = np.array([t for t, _ in t_mags])
        mags_arr = np.array([mag for _, mag in t_mags])
        if len(source_lines) == 1 and len(detector_lines) == 1:
            plt.figure("source_decoupling")
            plt.plot(
                delays_arr,
                mags_arr,
                linestyle="None",
                marker=".",
            )
            plt.xlabel("x_schedule_delay (microseconds)")
            plt.ylabel(r"Detector Magnetization, $\langle \sigma^z_i \rangle$")
            plt.title("Source Decoupling Detection")
            for line, t in t_decoupled.items():
                plt.axvline(
                    t,
                    color="r",
                    linestyle="--",
                    label=f"Estimated decoupling delay (line {line})",
                )
            plt.legend()
        else:
            # Per-qubit line assignments let us align each series by its own
            # source/detector decoupling delay: x = t + source_delay - detector_delay.
            emb_detector_lines = np.array(
                [
                    line_assignments[emb[("detector", 0)][0]]
                    for emb in sampler.embeddings
                ]
            )
            emb_source_lines = np.array(
                [line_assignments[emb[("source", 0)][0]] for emb in sampler.embeddings]
            )
            shifts = np.array(
                [
                    t_decoupled.get(src, 0.0) - t_decoupled[det]
                    for det, src in zip(emb_detector_lines, emb_source_lines)
                ]
            )
            fig, (ax_before, ax_after) = plt.subplots(
                1, 2, sharey=True, num="source_decoupling"
            )
            ax_before.plot(
                delays_arr,
                mags_arr,
                linestyle="None",
                marker=".",
            )
            ax_before.set_xlabel("x_schedule_delay (microseconds)")
            ax_before.set_ylabel(r"Detector Magnetization, $\langle \sigma^z_i \rangle$")
            ax_before.set_title("Source Decoupling Detection")
            for line, t in t_decoupled.items():
                ax_before.axvline(
                    t,
                    color="r",
                    linestyle="--",
                    label=f"Estimated decoupling delay (line {line})",
                )
            ax_before.legend()
            x_shifted = delays_arr[:, None] + shifts[None, :]
            ax_after.plot(
                x_shifted,
                mags_arr,
                linestyle="None",
                marker=".",
            )
            ax_after.axvline(
                0.0,
                color="r",
                linestyle="--",
                label="Aligned decoupling delay",
            )
            ax_after.set_xlabel(
                "x_schedule_delay + source_delay - detector_delay (microseconds)"
            )
            ax_after.set_title("Source Decoupling (delay-aligned)")
            ax_after.legend()

        print(
            "Detected decoupling times by detector line (or relative offset required on source line)",
            t_decoupled,
        )
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next stages.")
        _apply_tight_layout()
        plt.show()
    else:
        # A user-provided scalar delay applies to all detector lines; no
        # decoupling sweep is measured, so there is no timeseries to carry over.
        t_decoupled = {line: t_decoupled for line in detector_lines}
        t_mags = None
        t_decoupled_by_qubit = None

    for line, val in t_decoupled.items():
        sampling_params["x_schedule_delays"][line] += val

    ctx.stage_idx = stage_idx
    ctx.t_decoupled = t_decoupled
    ctx.t_mags = t_mags
    ctx.t_decoupled_by_qubit = t_decoupled_by_qubit
    return ctx


def run_refine_delays(
    ctx: MCAContext,
    *,
    refine_phase_accuracy: float = 0.05,
    refine_delay_t_min_factor: float = -2.0,
    refine_delay_t_max_factor: float = 4.0,
) -> MCAContext:
    """Refine source and detector delays after anneal-offset refinement.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.
        refine_phase_accuracy: Target phase accuracy (in oscillation cycles) for
            the per-qubit decoupling-delay refinement.
        refine_delay_t_min_factor: Lower bound of the refinement bisection search
            window, as a multiple of the target period ``1 / target_A``.
        refine_delay_t_max_factor: Upper bound of the refinement bisection search
            window, as a multiple of the target period ``1 / target_A``.

    Returns:
        The same ``ctx``, updated with the refined per-line and per-qubit
        decoupling delays.
    """
    stage_idx = ctx.stage_idx
    cache_str = ctx.cache_str
    sampler = ctx.sampler
    bqm = ctx.bqm
    sampling_params = ctx.sampling_params
    detector_lines = ctx.detector_lines
    target_A = ctx.target_A
    line_assignments = line_assignments_from_feature_info(ctx.exp_feature_info)
    source_lines = ctx.source_lines
    by_det_src_pair = ctx.by_det_src_pair
    embs = ctx.embs
    dt_dc = ctx.dt_dc
    online = ctx.online

    # Repeat the source-decoupling detection; the outcome differs because
    # of the updated anneal_offsets on the targets.
    stage_idx += 1
    print()
    print(
        f"Stage {stage_idx}: Refine source and detector delays (with x_schedule_delays and detector anneal_offsets)"
    )
    print(
        "We first reestablish decoupling time to high accuracy (now subject"
        " to some improved target frequency alignment on a plurality of qubits)."
    )

    if not online:
        raise RuntimeError("QPU not available, and no cached data found.")

    fn_cache = f"cache/source_decoupling_refined_{cache_str}.pkl"
    if cache_str and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            t_decoupled, t_mags, t_decoupled_by_qubit = pickle.load(f)
    else:
        t_decoupled, t_mags, t_decoupled_by_qubit = estimate_decoupling_timescale(
            sampler=sampler,
            bqm=bqm,
            sampling_params=sampling_params,
            detector_lines=detector_lines,
            target_A=target_A * 1000,
            line_assignments=line_assignments,
            source_lines=source_lines,
            by_det_src_pair=by_det_src_pair,
            phase_accuracy=refine_phase_accuracy,
            interpolate_final_interval=True,
            t_min=refine_delay_t_min_factor / target_A / 1000,
            t_max=refine_delay_t_max_factor / target_A / 1000,
        )
        if cache_str:
            with open(fn_cache, "wb") as f:
                pickle.dump((t_decoupled, t_mags, t_decoupled_by_qubit), f)

    plt.figure("Per-qubit delays")
    num_estimated = len(t_decoupled_by_qubit)
    plt.plot(
        sorted(t_decoupled_by_qubit.values()),
        np.arange(num_estimated) / num_estimated,
    )
    plt.xlabel("Per-detected qubit decoupling time, us")
    plt.ylabel(f"Cumulative Distribution Function")
    plt.title(
        f"Linear interpolation estimates for {num_estimated} of {len(sampler.embeddings)} cases"
    )

    delays_arr = np.array([t for t, _ in t_mags])
    mags_arr = np.array([mag for _, mag in t_mags])
    plt.figure("per_qubit_decoupling")
    plt.plot(
        delays_arr,
        mags_arr,
        linestyle="None",
        marker=".",
    )
    plt.xlabel("x_schedule_delay (microseconds)")
    plt.ylabel(r"Detector Magnetization, $\langle \sigma^z_i \rangle$")
    plt.title("Source Decoupling Detection")

    print(
        "Detected decoupling times by detector line (or relative offset required on source line)",
        t_decoupled,
    )
    print("We refine the x_schedule_delay per line accordingly.")
    for line, val in t_decoupled.items():
        sampling_params["x_schedule_delays"][line] += val

    print(
        "We then refine the anneal offset on detectors to mitigate for remaining per-qubit variation. "
        "c(t)=anneal_offset(t) is to a good approximation linear, assuming "
        f"dt/dc ~ {dt_dc}us we can correct a measured delay in one step."
    )
    mags_t0 = [
        run_parallel_experiment(sampler, bqm, sampling_params, detector_lines).ravel()
    ]
    if len(source_lines) == 1:
        t_decoupled[source_lines[0]] = 0.0
    for emb in embs:
        det_q = emb[("detector", 0)][0]
        src_q = emb[("source", 0)][0]
        if det_q in t_decoupled_by_qubit:
            t_decoupled_by_qubit[det_q] -= (
                t_decoupled[line_assignments[det_q]]
                - t_decoupled[line_assignments[src_q]]
            )
        # Residual part, not captured by line-wise correction.

    mags_t0.append(
        run_parallel_experiment(sampler, bqm, sampling_params, detector_lines).ravel()
    )
    relative_offset = np.mean(list(t_decoupled_by_qubit.values()))
    for emb in embs:
        det_q = emb[("detector", 0)][0]
        if det_q in t_decoupled_by_qubit:
            # TO DO - split evenly with src makes a bit more sense.
            det_q = emb[("detector", 0)][0]
            sampling_params["anneal_offsets"][det_q] = (
                -(t_decoupled_by_qubit[det_q] - relative_offset) / dt_dc
            )
    mags_t0.append(
        run_parallel_experiment(sampler, bqm, sampling_params, detector_lines).ravel()
    )
    plt.figure("Mags with/without synchronization")
    num_points = len(mags_t0[0])
    inter_quartiles = [np.diff(np.percentile(mags, [25, 75])) for mags in mags_t0]
    plt.plot(
        sorted(mags_t0[0]),
        np.arange(num_points) / num_points,
        label=f"Coarse x_schedule_delays (IQD={inter_quartiles[0]})",
    )
    plt.plot(
        sorted(mags_t0[1]),
        np.arange(num_points) / num_points,
        label=f"Refined x_schedule_delays (IQD={inter_quartiles[1]})",
    )
    plt.plot(
        sorted(mags_t0[2]),
        np.arange(num_points) / num_points,
        label=f"with additional anneal_offsets (IQD={inter_quartiles[2]})",
    )
    plt.xlabel("Magnetization at delay 0.")
    plt.ylabel("Cumulative distribution function")
    plt.legend()
    # Low data scenario, a bulk measure is safer since variability by qubit
    # is expected to be small.
    # Column i of each mags_t0 array corresponds to embs[i]; group by the
    # (source line, detector line) pair so dt_dc is reestimated per combination.
    emb_src_lines = np.array([line_assignments[emb[("source", 0)][0]] for emb in embs])
    emb_det_lines = np.array(
        [line_assignments[emb[("detector", 0)][0]] for emb in embs]
    )
    dt_dc_new = {}
    for src, det in sorted(set(zip(emb_src_lines.tolist(), emb_det_lines.tolist()))):
        mask = (emb_src_lines == src) & (emb_det_lines == det)
        m_base = mags_t0[1][mask]
        dt_dc_new[(src, det)] = (
            np.median(np.abs(m_base - np.mean(m_base)))
            / np.median(np.abs(m_base - mags_t0[2][mask]))
            * dt_dc
        )
    dt_dc_bulk = (
        np.median(np.abs(mags_t0[1] - np.mean(mags_t0[1])))
        / np.median(np.abs(mags_t0[1] - mags_t0[2]))
        * dt_dc
    )
    for emb in embs:
        det_q = emb[("detector", 0)][0]
        sampling_params["anneal_offsets"][det_q] *= dt_dc / dt_dc_bulk
    print("Revised estimation of dt/dc ", dt_dc_bulk, " used value ", dt_dc)
    print("The final anneal_offsets are adjusted to this new estimate.")

    ctx.stage_idx = stage_idx
    ctx.t_decoupled = t_decoupled
    ctx.t_mags = t_mags
    ctx.t_decoupled_by_qubit = t_decoupled_by_qubit
    return ctx
