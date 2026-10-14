"""Orchestrates the multi-color-annealing calibration-refinement stages.

This thin orchestrator threads a shared :class:`~mca_shim.common.MCAContext`
through the individual ``stage_*`` modules, preserving the original sequential
data flow while keeping each stage in its own small, readable script.
"""

from typing import Literal
from collections.abc import Iterable

from .common import MCAContext, _save_open_figures, _apply_tight_layout, plt
from . import (
    stage_target_frequency,
    stage_delays,
    stage_embeddings,
    stage_flux_biases,
    stage_loop,
    stage_schedule,
    stage_waveforms,
)


def main(
    cache_str: str | None = None,
    solver: dict | str | None = None,
    detector_lines: Iterable[int] | None = None,
    source_lines: Iterable[int] | None = None,
    target_lines: Iterable[int] | None = None,
    seed: int | None = None,
    max_num_embeddings: int | None = None,
    target_c: float | None = None,
    target_A: float | None = None,
    target_B: float | None = None,
    dAdc: float | None = None,
    td_shim_type: Literal[
        "None", "alternating", "detector_only", "sequential", "by_line_quench"
    ] = "detector_only",
    t_decoupled: float | None = None,
    by_det_src_pair: bool = True,
    num_anneal_offset_iterations: int = 2,
    schedule_fn: str = "09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    num_reads: int = 500,
    use_common_c_bounds: bool = True,
    use_overshoot: bool = True,
    save_figures: bool = False,
    T2: float = 0.0101,
    Jtd: float = 1.0,
    Jts: float = 1.0,
    Jtt: float = 0.1,
    loop_length: int | None = 16,
    preparation_orientation: float = 1.0,
    embedding_timeout: int = 60,
    colormap_type: Literal["divergent", "default"] = "divergent",
    dt_div_A_final: float = 0.25,
    refine_delays: bool = True,
    dt_dc: float = 2.3e-3,
    show_artificial_timeseries: bool = False,
    fit_window: float | None = None,
    loop_delay_range: float | None = None,
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
    calibration refinement of a detector flux bias to restore symmetry.

    Higher accuracy calibration refinement, and calibration refinement of target
    flux_biases may also be desirable, but are beyond the scope of the example.
    Note that we can use simple statistics to determine flux-bias assignment on a
    detector relative to a target. E.g. a) when decoupled from the source and
    detector a 1 qubit model frequency omega=root(A(s)^2 + B(s)^2 h^2) is a convex
    monotonic function of the linear field, b) when decoupled from the source the
    response of the detector magnetization to a flux_bias perturbation is maximized.

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
        target_lines:
            An iterable of integer indices of the target lines. When None,
            defaults to all lines that are neither detector nor source lines.
        seed:
            Random seed used for embedding generation.
        max_num_embeddings:
            Maximum number of embeddings to find. If None, search for all available embeddings.
        target_c:
            Normalized control bias at which the target qubits are held.
            target_c is inferred from target_A, or defaulted such that A(target_c)=B(target_c).
            Either target_A or target_c should be specified, not both.
        target_A:
            The expected qubit frequency in GHz.
            When None target_A is inferred from target_c (and the given processor schedule).
            Either target_A or target_c should be specified, not both.
        target_B:
            The expected qubit frequency in GHz for the B field.
            When None target_B is inferred from target_c (and the given processor schedule).
            target_B is not used to parameterize experiments.
        dAdc:
            Approximate rate of change of A(c) with the normalized control bias c
            near target_c (GHz per unit c), used to convert a frequency discrepancy
            into an anneal offset. If None, it is estimated from the schedule file.
        td_shim_type:
            When set to "None", flux_biases are not modified. When
            "detector_only", flux_biases are modified on detector qubits to
            achieve zero expected magnetization at long delay. When
            "by_line_quench", flux_biases are refined on every line assuming the
            required correction can be determined by independent quenches on each
            line. When "sequential" (by_line_quench followed by detector_only) or
            "alternating", flux_biases are modified on both target and
            detector qubits. The "alternating" method can diverge, particularly
            for fast quenches of sources and detectors, large |Jtd| or |Jts|,
            and small target_A. See :func:`~dwave.experimental.shimming.shim_tds_flux_biases`.
        t_decoupled:
            When None (default), estimate the delay required for the detector to
            decouple from the source before collecting timeseries data. When a
            float is provided, this detection stage is skipped and the value is
            applied to all detector lines.
        by_det_src_pair:
            When True (default), resolve the source-decoupling delay
            independently for each detector/source line pair. When False, a
            single bulk delay is shared across all lines. Has no effect when a
            single source and single detector line are present.
        num_anneal_offset_iterations:
            Number of anneal-offset refinement iterations to run. The first iteration
            estimates the required offsets; subsequent iterations re-measure and
            refine them. This verification stage is skipped when the value is 0.
            Anneal offsets are modified so that the peak power-spectral density is
            centered at a common target frequency for all qubits. This peak value
            characterizes the frequency of the target qubit in simple, well-calibrated
            models.
        schedule_fn: A schedule file that is used to estimate an appropriate sampling interval for delay
            time and an appropriate scale for anneal_offset synchronization. This should be matched to the
            solver.
        save_figures:
            When True, save generated figures to a ``figures`` folder.
        num_reads:
            The number of reads to perform for each measurement.
        use_common_c_bounds:
            When True, align the c-bounds of the generated schedules across annealing lines.
        use_overshoot:
            When True, use overshoot transitions for the source and detector quenches
            when building the multi-color annealing schedules. By default this is True
            to maximize the quench rate on source and detector.
        T2:
            Assumed decoherence (envelope) time scale used by the simple signal model
            (microseconds).
        Jtd: The coupling strength between target and detector qubits.
        Jts: The coupling strength between target and source qubits.
        Jtt: The coupling strength between coupled target qubits. Only used when
            ``loop_length`` is specified.
        loop_length: When a positive (even, >=4) value is given, target qubits are
            embedded in unfrustrated loops of this length coupled with ``Jtt``
            (defaults to 16). Pass ``1`` (or ``None``) to use independent T-D-S
            systems without a loop constraint (Larmor precession example), which
            also skips the final loop experiment stage.
        preparation_orientation: The orientation of the target qubit whilst coupled
            to the source.
        embedding_timeout: Time budget (seconds) for the embedding search.
        colormap_type: Colormap style used for the detector-magnetization
            heatmaps ("divergent" or "default").
        dt_div_A_final: Sampling rate for the final (loop) experiment. Defaults
            to the same value used in anneal_offset shimming (1/4), smaller values
            can be used for pretty plots.
        dt_dc: Decoupling time change as a function of anneal offset. This is primarily
            anneal_schedule (quench C-interval and rate) dependent. For the default
            overshoot waveform on Advantage2 processors dt_dc ~ 2.3ns.
            A correction is applied using dt_dc = 2.3ns, data recollected, and the final
            anneal offset applied using a corrected model.
        show_artificial_timeseries: When True, display an illustrative artificial
            (model) timeseries and power-spectral-density demonstration before the
            experimental target-frequency stage. Defaults to False.
        fit_window: Width of the delay data window (measured from the decoupling
            time ``delay_min``, spanning ``[delay_min, delay_min + fit_window]``)
            used to establish the power-spectral density. Values much larger than
            ``T2`` are noise dominated and add no value; runtime is proportional to
            the fit window. The same window is overloaded to determine a bound on the
            delay fitting window (by bisection search). When None, defaults to ``T2``.
        loop_delay_range: Width of the delay data window (measured from the
            decoupling time ``delay_min``) collected for the loop ``pi/2``-pulse
            propagation stage, sampled 4x more finely than ``fit_window`` to better
            resolve pulse propagation and synchronization limitations. When None,
            defaults to ``T2 / 4``.
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

    if loop_length < 3:
        # Sentinel for "no loop constraint": use independent T-D-S systems.
        loop_length = None

    if fit_window is None:
        fit_window = T2

    if loop_delay_range is None:
        loop_delay_range = T2 / 2

    print()
    print(
        "This example currently serves as a guide to available functionality, and "
        "some simple heuristics, in a calibration refinement context. "
        "It is not a general purpose or canonical methodology. "
        "Correction of flux_biases, x_schedule_delays and anneal_offsets are correlated, "
        "and a high performance outcome may require an application-specific iterative "
        "approach with a larger number of programmings and reads than used in these examples."
    )
    print()
    print(
        "The example proceeds sequentially through several stages. "
        "Success in each stage may be contingent on previous stages and a reasonable baseline calibration. "
        "The main stages are: "
        "embedding Target-Detector-Source systems and multi-color annealing sampling parameters; "
        "refining flux_biases to restore an anticipated Z2 symmetry at equilibrium; "
        "determining line delays that characterize the preparation time (decoupling from the source); "
        "determining anneal_offsets per qubit to improve the accuracy of the target "
        "qubit frequencies and preparation time."
    )
    print()
    print(
        "Note: To save experiment embeddings and data for replotting add the --use-cache flag."
    )

    ctx = MCAContext(
        cache_str=cache_str,
        solver=solver,
        detector_lines=detector_lines,
        source_lines=source_lines,
        target_lines=target_lines,
        seed=seed,
        max_num_embeddings=max_num_embeddings,
        target_c=target_c,
        target_A=target_A,
        target_B=target_B,
        dAdc=dAdc,
        td_shim_type=td_shim_type,
        t_decoupled=t_decoupled,
        by_det_src_pair=by_det_src_pair,
        num_anneal_offset_iterations=num_anneal_offset_iterations,
        schedule_fn=schedule_fn,
        num_reads=num_reads,
        use_common_c_bounds=use_common_c_bounds,
        use_overshoot=use_overshoot,
        save_figures=save_figures,
        T2=T2,
        Jtd=Jtd,
        Jts=Jts,
        Jtt=Jtt,
        loop_length=loop_length,
        preparation_orientation=preparation_orientation,
        embedding_timeout=embedding_timeout,
        colormap_type=colormap_type,
        dt_div_A_final=dt_div_A_final,
        refine_delays=refine_delays,
        dt_dc=dt_dc,
        show_artificial_timeseries=show_artificial_timeseries,
        fit_window=fit_window,
        loop_delay_range=loop_delay_range,
    )
    ctx.stage_idx = -1

    stage_schedule.run_schedule(ctx)
    stage_waveforms.run_waveforms(ctx)
    stage_embeddings.run_embeddings(ctx)
    if ctx.td_shim_type != "None":
        stage_flux_biases.run_flux_biases(ctx)
    stage_delays.run_detect_delays(ctx)
    if ctx.num_anneal_offset_iterations > 0:
        stage_target_frequency.run_anneal_offsets(ctx)
    if ctx.refine_delays:
        stage_delays.run_refine_delays(ctx)
    if ctx.num_anneal_offset_iterations > 1:
        stage_target_frequency.run_verification(ctx)

    if ctx.save_figures:
        _save_open_figures("figures", ctx.cache_str)
    _apply_tight_layout()
    plt.show()
    if ctx.loop_length:
        print(
            "Close figures to proceed to next stage: pi/2-pulse propagation."
        )
        stage_loop.run_loop(ctx)
