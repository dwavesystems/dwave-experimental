"""Stage 1: build multi-color annealing waveforms compatible with the QPU."""

from .common import *


def run_waveforms(ctx: MCAContext) -> MCAContext:
    """Build multi-color annealing waveforms compatible with the QPU.

    Uses cached solver properties when available (no client call; the solver is
    then created on demand by a stage that must sample), otherwise connects to
    the solver. Resolves the detector, source and target lines, builds the
    T-D-S x-schedules and polarizing schedule, plots them, and initializes the
    shared ``sampling_params``.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.

    Returns:
        The same ``ctx``, updated with the QPU sampler, line assignments,
        schedules and ``sampling_params``.
    """
    stage_idx = ctx.stage_idx
    cache_str = ctx.cache_str
    solver = ctx.solver_name
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
        f"Stage {stage_idx}: Build multi-color annealing waveforms compatible with the "
        "research QPU."
    )

    qpu_fn = f"cache/qpu_{cache_str}.pkl"  # Only used when cache_str is set.
    cached_chip_id = None
    if cache_str and os.path.isfile(qpu_fn):
        # Cached properties suffice, so the client is not contacted here; the
        # first stage that must sample creates the solver (see connect_qpu).
        with open(qpu_fn, "rb") as f:
            properties, exp_feature_info = pickle.load(f)
        qpu = MockDWaveSampler(
            properties=properties,
            nodelist=properties["qubits"],
            edgelist=properties["couplers"],
        )
        cached_chip_id = properties.get("chip_id")
        print(
            f"Using cached solver properties (chip_id {cached_chip_id}; check matches "
            "schedule file). The solver is created on demand if a stage needs new data."
        )
        online = False
    else:
        if cache_str:
            print(f"Cache miss: {qpu_fn} not found. Creating the solver.")
        qpu = create_solver(solver, cache_empty=bool(cache_str))
        print(
            f"Solver connected to (check matches schedule file): {qpu.solver.identity}"
        )
        exp_feature_info = get_properties(qpu)
        if cache_str:
            os.makedirs("cache", exist_ok=True)
            with open(qpu_fn, "wb") as f:
                pickle.dump((qpu.properties, exp_feature_info), f)
        online = True
    if len(exp_feature_info) != 2:
        raise ValueError("Legacy format")
    line_assignments = line_assignments_from_feature_info(
        exp_feature_info.annealing_line_info
    )
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
    )
    _plot_tds_schedules(
        x_polarizing_schedule,
        x_anneal_schedules,
        detector_lines=detector_lines,
        source_lines=source_lines,
        target_lines=target_lines,
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
    ctx.cached_chip_id = cached_chip_id
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
