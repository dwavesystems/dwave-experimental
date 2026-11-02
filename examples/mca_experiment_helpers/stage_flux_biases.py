"""Stage 3: compensate static qubit body cross-talks via flux_biases."""

from .common import *


def run_flux_biases(ctx: MCAContext) -> MCAContext:
    """Compensate static qubit body cross-talks via flux_biases.

    Qubits whose refined flux_bias changed anomalously (more than 3 standard
    deviations from the mean change) are reported.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.

    Returns:
        The same ``ctx``, with the refined flux biases stored in
        ``ctx.sampling_params["flux_biases"]``.
    """
    stage_idx = ctx.stage_idx
    td_shim_type = ctx.td_shim_type
    cache_str = ctx.cache_str
    online = ctx.online
    embs = ctx.embs
    bqm = ctx.bqm
    qpu = ctx.qpu
    sampling_params = ctx.sampling_params
    target_lines = ctx.target_lines
    detector_lines = ctx.detector_lines
    source_lines = ctx.source_lines
    exp_feature_info = ctx.exp_feature_info
    target_c = ctx.target_c
    line_assignments = line_assignments_from_feature_info(
        exp_feature_info.annealing_line_info
    )
    save_figures = ctx.save_figures

    stage_idx += 1
    print()
    print(
        f"Stage {stage_idx}: Compensate for static qubit body cross talks (with flux_biases)."
    )
    print(f"Use the (CLI-configurable) {td_shim_type} method for flux_biases.")
    flux_biases_before = (
        None
        if sampling_params.get("flux_biases") is None
        else list(sampling_params["flux_biases"])
    )

    fn_cache = f"cache/FB_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            flux_biases, flux_history, mag_history = pickle.load(f)
        sampling_params["flux_biases"] = flux_biases
    else:
        if not online:
            connect_qpu(ctx)
            qpu = ctx.qpu

        bqm_embedded = dimod.BinaryQuadraticModel("SPIN").from_ising(
            {emb[n][0]: h for emb in embs for n, h in bqm.linear.items()},
            {
                tuple(emb[n][0] for n in e): J
                for emb in embs
                for e, J in bqm.quadratic.items()
            },
        )
        flux_biases, flux_history, mag_history = shim_tds_flux_biases(
            bqm=bqm_embedded,
            sampler=qpu,
            sampling_params=sampling_params,
            target_lines=set(target_lines),
            detector_lines=set(detector_lines),
            td_shim_type=td_shim_type,
            exp_feature_line_info=exp_feature_info.annealing_line_info,
            source_lines=set(source_lines),
            target_c=target_c,
            sampling_params_in_place=True,
        )
        if cache_str:
            with open(fn_cache, "wb") as f:
                pickle.dump((flux_biases, flux_history, mag_history), f)
    if flux_biases is not None:
        _report_outlier_changes(
            "flux_biases",
            flux_history.keys(),
            flux_biases_before,
            flux_biases,
            embs,
        )
    if td_shim_type != "detector_only":
        plot_shim(
            line_assignments=line_assignments,
            mag_history=mag_history,
            flux_history=flux_history,
        )
    else:
        plot_shim(
            mag_history=mag_history,
            flux_history=flux_history,
        )
    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next stages.")
    _apply_tight_layout()
    plt.show()

    ctx.stage_idx = stage_idx
    return ctx
