"""Stage 1: build multi-color annealing waveforms compatible with the QPU."""

from .common import *


def run_waveforms(ctx):
    stage_idx = ctx.stage_idx
    cache_str = ctx.cache_str
    solver = ctx.solver
    detector_lines = ctx.detector_lines
    source_lines = ctx.source_lines
    target_lines = ctx.target_lines
    target_c = ctx.target_c
    use_common_c_bounds = ctx.use_common_c_bounds
    use_overshoot = ctx.use_overshoot
    Jts = ctx.Jts
    Jtd = ctx.Jtd
    preparation_orientation = ctx.preparation_orientation
    num_reads = ctx.num_reads
    save_figures = ctx.save_figures

    stage_idx += 1
    print()
    print(
        f"Stage {stage_idx}: Build multi color annealing waveforms compatible with the "
        f"research QPU."
    )

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
    line_assignments = line_assignments_from_feature_info(exp_feature_info)
    num_lines = len(exp_feature_info.annealing_line_info)
    # 2/3 of lines reserved for the target (typical applications)
    if (detector_lines is None) != (source_lines is None):
        raise ValueError(
            "detector_lines and source_lines must either both be specified or both be None."
        )
    elif detector_lines is None:
        if num_lines <= 6:
            # 6 line convention
            detector_lines = (0,)
            source_lines = (num_lines // 2,)
        else:
            # 12 line convention
            detector_lines = (0, num_lines // 4 + 1)
            source_lines = (num_lines // 2, (3 * num_lines) // 4 + 1)

    if target_lines is None:
        target_lines = (
            set(range(num_lines)) - set(detector_lines) - set(source_lines)
        )
    if len(target_lines) == 0:
        raise ValueError("No target lines available.")

    cmap = plt.colormaps.get_cmap("plasma")

    line_color = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=target_lines,
        target_c=target_c,
        detector_lines=detector_lines,
        source_lines=source_lines,
        use_common_bounds=use_common_c_bounds,
        use_overshoot=use_overshoot,
        sign_polarization=-np.sign(Jts * Jtd * preparation_orientation),
        # depolarization_time_scale=50.0
    )
    _plot_tds_schedules(
        x_polarizing_schedule,
        x_anneal_schedules,
    )
    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next stages.")
    _apply_tight_layout()
    plt.show()

    x_schedule_delays = [0.0] * num_lines

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
        auto_scale=False,
    )

    ctx.stage_idx = stage_idx
    ctx.qpu = qpu
    ctx.online = online
    ctx.exp_feature_info = exp_feature_info
    ctx.detector_lines = detector_lines
    ctx.source_lines = source_lines
    ctx.target_lines = target_lines
    ctx.line_color = line_color
    ctx.x_anneal_schedules = x_anneal_schedules
    ctx.x_polarizing_schedule = x_polarizing_schedule
    ctx.x_schedule_delays = x_schedule_delays
    ctx.anneal_offsets = anneal_offsets
    ctx.flux_biases = flux_biases
    ctx.sampling_params = sampling_params
    return ctx
