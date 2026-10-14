"""Stage 3: compensate static qubit body cross-talks via flux_biases."""

from .common import *


def run_flux_biases(ctx):
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
    line_assignments = ctx.line_assignments
    save_figures = ctx.save_figures
    x_polarizing_schedule = ctx.x_polarizing_schedule
    flux_biases = ctx.flux_biases

    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Compensate for static qubit body cross talks (with flux_biases).")
    print(f"Use the (CLI-configurable) {td_shim_type} method for flux_biases.")

    fn_cache = f"cache/FB_{cache_str}.npy"
    if cache_str and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            flux_biases, flux_history, mag_history = pickle.load(f)
    else:
        if not online:
            raise RuntimeError("QPU not available, and no cached data found.")

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
            exp_feature_line_info=exp_feature_info[1],
            source_lines=set(source_lines),
            target_c=target_c,
        )
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
            with open(fn_cache, "wb") as f:
                pickle.dump((flux_biases, flux_history, mag_history), f)
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
    sampling_params["flux_biases"] = flux_biases
    if save_figures:
        _save_open_figures("figures/", cache_str)
    print("Close figures to proceed to next stages.")
    _apply_tight_layout()
    plt.show()
    sampling_params["x_polarizing_schedule"] = x_polarizing_schedule


    ctx.stage_idx = stage_idx
    ctx.flux_biases = flux_biases
    ctx.sampling_params = sampling_params
    return ctx
