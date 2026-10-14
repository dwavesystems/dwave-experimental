"""Final stage: pi/2-pulse propagation around coupled target loops."""

from .common import *


def run_loop(ctx):
    # Build a loop-specific grid: a shorter window sampled 4x more finely than
    # the target-frequency stage to better resolve pi/2-pulse propagation.
    compute_delay_grid(ctx, ctx.loop_delay_range, rel_sampling_rate=4, rel_start=0.0)
    stage_idx = ctx.stage_idx
    loop_length = ctx.loop_length
    target_B = ctx.target_B
    Jtt = ctx.Jtt
    qpu = ctx.qpu
    embs_experiment = ctx.embs_experiment
    S_experiment = ctx.S_experiment
    Snode_to_tds = ctx.Snode_to_tds
    Jtd = ctx.Jtd
    Jts = ctx.Jts
    cache_str = ctx.cache_str
    dt_div_A_final = ctx.dt_div_A_final
    target_A = ctx.target_A
    delay_min = ctx.delay_min
    delay_max = ctx.delay_max
    sampling_params = ctx.sampling_params
    detector_lines = ctx.detector_lines
    delays = ctx.delays
    dt = ctx.dt
    colormap_type = ctx.colormap_type
    save_figures = ctx.save_figures

    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: pi/2 pulse propagation.")
    print(
        f"Target qubits are coupled with Jtt={Jtt} in loops of length {loop_length}. "
    )
    print(
        "A pi/2 pulse is applied to the (loop_length//2 indexed) qubit. "
        "targets are then measured in the (same) computational basis subject to delay. "
    )
    print("Delays are chosen by default over a 2-times smaller window with 2-times higher "
          "density relative to the grids used for the target-frequency stage, the initial "
          "time is offset smaller to demonstrate the stage with a single-qubit (per loop) "
          "coupled to the polarized source."
    )
    print(
        "We anticipate excitation propagation around the loop subject to interference. "
    )
    if target_B is not None:
        print(
            f"Single-site propagation occurs on a time scale  ~ 1/(2pi B(s) |Jtt|) = {1/np.abs(target_B * np.pi * 2 * Jtt):.3g}ns"
        )
    sampler_experiment = ParallelEmbeddingComposite(qpu, embeddings=embs_experiment)
    bqm_experiment = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S_experiment.nodes()},
        {
            e: Jtd
            for e in S_experiment.edges()
            if any(Snode_to_tds[v] == "detector" for v in e)
        }
        | {
            e: Jts
            for e in S_experiment.edges()
            if any((Snode_to_tds[v] == "source" and v[1] == loop_length // 2) for v in e)
        }
        | {
            e: Jtt
            for e in S_experiment.edges()
            if all(Snode_to_tds[v] == "target" for v in e)
        },
    )  # bqm restricted to decoupled source and target nodes
    fn_cache = f"cache/LoopExperiment_{cache_str}.npy"

    if dt_div_A_final != 0.25:
        experiment_dt = dt_div_A_final / target_A / 1000
        delays = np.linspace(
            delay_min,
            delay_max,
            round((delay_max - delay_min) / experiment_dt) + 1,
        )
        dt = delays[1] - delays[0]

    if cache_str and os.path.isfile(fn_cache):
        mean_Z_detector = np.load(fn_cache)
    else:
        mean_Z_detector = run_parallel_experiment(
            sampler_experiment,
            bqm_experiment,
            sampling_params,
            detector_lines,
            delays=delays,
        )
        if cache_str:
            np.save(fn_cache, mean_Z_detector)
    square_data = mean_Z_detector.reshape(mean_Z_detector.shape[0], -1)
    print(
        "Note: The x-axis is ordered by position on the ring, rather than target line assignment (per earlier plots). "
        "The excitation propagates in two directions (left and right, modulo the periodic boundary condition); "
        "Under ideal calibration refinement all panels become equal up to sampling error, and "
        "qubit variability beyond the scope of API controls. "
    )
    if len(embs_experiment) > 1:
        print(
            f"The plot is divided into {len(embs_experiment)} panels, reflecting independent (decoupled) rings that were programmed in parallel"
        )
    cols = square_data.shape[1]
    line_x = np.arange(0.5 + loop_length, cols - 0.5, loop_length)
    imshow_data(
        square_data,
        delays=delays,
        context_str=f"coupled loop length={loop_length}",
        vertical_lines=line_x,
        colormap_type=colormap_type,
    )
    plt.title(
        f"{len(embs_experiment)} target loops of length {loop_length}, each with pi/2-pulse at origin."
    )
    if save_figures:
        _save_open_figures("figures", cache_str)
    _apply_tight_layout()
    plt.show()

    ctx.stage_idx = stage_idx
    return ctx
