"""Stage 0: estimate target_A, target_c and dA/dc from the QPU schedule.

These parameterize the anneal-offset refinement and the accuracy threshold for
x_schedule_delay refinements.
"""

from .common import *


def run_schedule(ctx, *, dtarget_c=0.01):
    """Estimate target_A, target_c and dA/dc from the QPU schedule.

    Args:
        dtarget_c: Half-width (in normalized control bias c) of the symmetric
            finite difference used to estimate ``dAdc`` from the schedule file,
            i.e. ``dAdc = (A(target_c + dtarget_c) - A(target_c - dtarget_c)) /
            (2 * dtarget_c)``. Ignored when ``dAdc`` is supplied.
    """
    stage_idx = ctx.stage_idx
    schedule_fn = ctx.schedule_fn
    target_A = ctx.target_A
    target_c = ctx.target_c
    target_B = ctx.target_B
    dAdc = ctx.dAdc

    if schedule_fn is None:
        if target_A is None or target_c is None or dAdc is None:
            raise ValueError(
                "Schedule file is required if target_A, target_c, or dAdc is not specified."
            )
    else:
        stage_idx += 1
        print()
        print(
            f"Stage {stage_idx}: Estimate consistent target_A, target_c and dA/dc from the given qpu schedule. "
            "These are used to parameterize the anneal_offset refinement method, and the accuracy threshold for x_schedule_delays refinements."
        )
        print(f"Schedule file used: {schedule_fn} ")
        print(
            "For calibration refinement of anneal offsets, dA/dc must be a reasonable match in the vicinity of target_A."
        )
        qpu_anneal_schedule = pd.read_excel(
            schedule_fn, sheet_name="Fast-Annealing Schedule"
        )

        plt.figure("Schedule")
        plt.title("Annealing Schedule")
        delta_vs_s = qpu_anneal_schedule[::-1]
        plt.plot(delta_vs_s["s"], delta_vs_s["A(s) (GHz)"], label="A(s)")
        plt.plot(delta_vs_s["s"], delta_vs_s["B(s) (GHz)"], label="B(s)")
        if target_c is None:
            if target_A is None:
                target_c = np.interp(
                    0.0,
                    -delta_vs_s["B(s) (GHz)"] + delta_vs_s["A(s) (GHz)"],
                    delta_vs_s["s"],
                )  # Expected normalized control bias at which to hold the qubits
            else:
                target_c = np.interp(
                    target_A, delta_vs_s["A(s) (GHz)"], delta_vs_s["s"]
                )  # Expected normalized control bias at which to hold the qubits
        elif target_A is not None:
            raise ValueError(
                "Specification of both target_c and target_A is not permitted when also specifying schedule_fn."
            )
        if target_A is None:
            target_A = np.interp(
                1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
            )  # Expected frequency of detector magnetization oscillations
        target_B = np.interp(
            1 - target_c, 1 - delta_vs_s["s"], delta_vs_s["B(s) (GHz)"]
        )

        target_Aminus = np.interp(
            1 - (target_c - dtarget_c), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
        )
        target_Aplus = np.interp(
            1 - (target_c + dtarget_c), 1 - delta_vs_s["s"], delta_vs_s["A(s) (GHz)"]
        )
        if dAdc is None:
            dAdc = (target_Aplus - target_Aminus) / (2 * dtarget_c)
        plt.plot(
            [target_c, target_c],
            [0, np.max(delta_vs_s["A(s) (GHz)"])],
            label=f"c={target_c:.3g}",
        )
        plt.plot(
            [0, target_c - dtarget_c],
            [target_Aminus, target_Aminus],
            linestyle="dotted",
            color="black",
        )
        plt.plot(
            [target_c - dtarget_c, target_c - dtarget_c],
            [0, target_Aminus],
            linestyle="dotted",
            color="black",
        )
        plt.plot(
            [0, target_c + dtarget_c],
            [target_Aplus, target_Aplus],
            linestyle="dotted",
            color="black",
        )
        plt.plot(
            [target_c + dtarget_c, target_c + dtarget_c],
            [0, target_Aplus],
            linestyle="dotted",
            color="black",
        )
        plt.xlabel("Normalized control bias, c")
        plt.ylabel("Energy scale, GHz")
        plt.ylim([0, 2 * max(target_A, target_B)])
        plt.xlim([0, 1])
        plt.legend()
        print(
            f"target_c = {target_c:.3g} ",
            f"A(target_c) = {target_A:.3g} GHz ",
            f"B(target_c) = {target_B:.3g} GHz",
        )

    ctx.stage_idx = stage_idx
    ctx.target_c = target_c
    ctx.target_A = target_A
    ctx.target_B = target_B
    ctx.dAdc = dAdc
    return ctx
