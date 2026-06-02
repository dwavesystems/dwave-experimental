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

from copy import deepcopy
import math
from typing import Iterable, Literal

import networkx as nx

from dwave_networkx import zephyr_coordinates


__all__ = [
    "ScheduleError",
    "qubit_to_Advantage2_annealing_line",
    "make_tds_graph",
    "make_tds_intervals",
    "make_tds_x_anneal_schedules",
    "make_tds_x_polarizing_schedule",
    "make_tds_x_schedules",
    "standardize_schedule_endpoints",
]

Interval = tuple[float, float]
LineFeatureInfo = dict[str, float]
AnnealSchedule = list[list[float]]
AnnealSchedules = list[AnnealSchedule]


class ScheduleError(ValueError):
    """Raised when schedules are incompatible with sequencing constraints."""


def _round_sigfigs(
    value: float, sigfigs: int = 3, mode: Literal["up", "down"] = "down"
) -> float:
    """Round a value up or down to a fixed number of significant figures.

    This addresses an issue in the client, whereby some inbound properties
    have more significant figures than the client can reliably use.

    Args:
        value: The value to round.
        sigfigs: The number of significant figures to round to.
        mode: Whether to round up or down. "up" rounds away from zero, "down"
            rounds towards zero.
    Returns:
        The rounded value.
    """

    if value == 0:
        return 0.0
    if sigfigs <= 0:
        raise ValueError("sigfigs must be positive")
    if mode not in ("up", "down"):
        raise ValueError("mode must be 'up' or 'down'")
    scale = 10 ** (sigfigs - 1 - math.floor(math.log10(abs(value))))
    if mode == "up":
        return math.ceil(value * scale) / scale
    return math.floor(value * scale) / scale


def qubit_to_Advantage2_annealing_line(
    n: int | tuple[int, int, int, int, int],
    shape: tuple[int, ...],
    num_lines: int = 6,
) -> int:
    """Return the annealing line associated to an Advantage2 qubit

    Advantage2 processors can allow multicolor annealing based in
    some cases on a 6-line control scheme. Compatibility with this
    scheme should be confirmed using a solver API or release notes.
    Based on the Zephyr coordinate system (u,w,k,j,z), a qubit
    can be uniquely assigned a color. u denotes qubit orientation
    while j and z control aligned displacement on the processor. See also
    :func:`dwave_networkx.zephyr_graph` and
    :func:`dwave_networkx.zephyr_coordinates`.

    Args:
        n: Qubit label, as an integer, or a Zephyr coordinate as a 5-tuple.
        shape: Advantage2 processor shape, accessible as a solver
            property ``properties['topology']['shape']``.
        num_lines: number of annealing lines, may be 6 or 12.

    Returns:
        Integer annealing line assignment for Advantage2 processors
        using 6 or 12-annealing line control.

    Examples:
        Retrieve multi-color annealing line properties for a default solver, and
        if a 6 (or 12) color scheme is used, confirm the programmatic mapping is
        in agreement with the multicolor annealing properties on all qubits
        and lines.

        >>> from dwave.system import DWaveSampler
        >>> import dwave.experimental.multicolor_anneal as mca
        >>> qpu = DWaveSampler()            # doctest: +SKIP
        >>> annealing_lines = mca.get_properties(qpu)            # doctest: +SKIP
        >>> shape = qpu.properties['topology']['shape']          # doctest: +SKIP
        >>> num_lines = len(annealing_lines)            # doctest: +SKIP
        >>> assert(all(mca.qubit_to_Advantage2_annealing_line(n, shape, num_lines)==al_idx for al_idx, al in enumerate(annealing_lines) for n in al['qubits']))            # doctest: +SKIP

        To explicitly select a solver that supports advanced annealing
        features, such as multi-color annealing, see
        :data:`~dwave.experimental.fast_reverse_anneal.api.SOLVER_FILTER`.
    """

    if isinstance(n, tuple):
        u, _, _, j, z = n
    else:
        u, _, _, j, z = zephyr_coordinates(*shape).linear_to_zephyr(n)
    if num_lines == 6:
        return 3 * u + (1 - 2 * z - j) % 3
    elif num_lines == 12:
        return 6 * u + (z % 3 + 3 * j)
    else:
        raise ValueError("num_lines must be 6 or 12")


def make_tds_graph(
    target_graph: nx.Graph,
    detected_nodes: Iterable[int] | None = None,
    sourced_nodes: Iterable[int] | None = None,
) -> tuple[nx.Graph, dict[int | tuple[str, int], str]]:
    """Decorate a target graph with detectors and sources.

    We add single node source and detector branches to nodes of a target
    graph.

    Args:
        target_graph: A networkx target graph
        detected_nodes: An iterable on the target nodes, if None
            all target nodes.
        sourced_nodes: An iterable on the target nodes, if None
            all target nodes.

    Returns:
        A copy of the graph where each selected target node ``n`` is connected
        to ``('detector', n)`` and/or ``('source', n)``.

    Raises:
        ValueError: If detected_nodes or sourced_nodes are not in the graph
            target_graph.
    """

    if detected_nodes is None:
        detected_nodes = target_graph.nodes()
    elif not set(detected_nodes).issubset(target_graph.nodes()):
        raise ValueError("detected_nodes are not compatible with the target graph")

    if sourced_nodes is None:
        sourced_nodes = target_graph.nodes()
    elif not set(sourced_nodes).issubset(target_graph.nodes()):
        raise ValueError("sourced_nodes are not compatible with the target graph")

    node_to_tds = (
        {n: "target" for n in target_graph.nodes()}
        | {("source", n): "source" for n in sourced_nodes}
        | {("detector", n): "detector" for n in detected_nodes}
    )

    tds_graph = target_graph.copy()
    tds_graph.add_edges_from((n, ("detector", n)) for n in detected_nodes)
    tds_graph.add_edges_from((n, ("source", n)) for n in sourced_nodes)

    return tds_graph, node_to_tds


def make_tds_intervals(
    post_preparation_delay: float = 20.0,
    depolarization_time_scale: float = 2.0,
    *,
    depolarizing_time_scale: None | float = None,
    anneal_preparation_time_scale: None | float = None,
) -> tuple[Interval, Interval, Interval, float]:
    """Make default intervals for schedules construction.

    This routine sets up the timescales on which the source (and detector)
    lines are prepared in polarized (depolarized) states, the polarizing bias
    is removed, and the target is prepared.

    Defaulting is designed with a view to Larmour precession examples in
    :ref:`qpu_experimental_research_mca_example`
    and to allow replication of the
    `published experimental results <https://doi.org/10.48550/arXiv.2603.15534>`_.

    Args:
        post_preparation_delay: Time to wait in microseconds
            after line preparation before quenching. This is lower bounded
            by 0 but typically some longer time scale of order 10.0 us is
            recommended.
        depolarization_time_scale: Time before and after the depolarizing
            interval in which changes to the anneal schedule are prevented.
            This is lower bounded by the
            depolarizationAnnealScheduleRequiredDelay annealing-line property.
        depolarizing_time_scale: Step size for the polarizing schedule.
            It is lower bounded by minPolarizingTimeStep, but typically a
            larger value of order microseconds is desirable for robust
            preparation. Defaults to ``depolarization_time_scale`` when not
            set.
        anneal_preparation_time_scale: Step size for preparation of anneal
            schedules in polarized and depolarized regimes. If None, defaults
            to ``depolarizing_time_scale``. This is lower bounded by
            minAnnealingTimeStep annealing-line properties, but typically a
            larger value of order microseconds is desirable for robust
            preparation.

    Returns:
        A 4-tuple containing ``polarized_preparation_interval``,
        ``depolarization_interval``, ``depolarized_preparation_interval``,
        and ``quench_time``.
    """
    if depolarizing_time_scale is None:
        depolarizing_time_scale = depolarization_time_scale

    if anneal_preparation_time_scale is None:
        anneal_preparation_time_scale = depolarizing_time_scale

    polarized_preparation_interval = (0.0, anneal_preparation_time_scale)
    tp = polarized_preparation_interval[-1] + depolarization_time_scale
    depolarization_interval = (tp, tp + depolarizing_time_scale)
    ts = depolarization_interval[-1] + depolarization_time_scale
    depolarized_preparation_interval = (
        ts,
        ts + anneal_preparation_time_scale,
    )
    quench_time = depolarized_preparation_interval[-1] + post_preparation_delay

    return (
        polarized_preparation_interval,
        depolarization_interval,
        depolarized_preparation_interval,
        quench_time,
    )


def standardize_schedule_endpoints(
    x_anneal_schedules: AnnealSchedules,
    x_polarizing_schedule: AnnealSchedule | None = None,
    *,
    post_pwl_delay: float = 0.0,
    decimals: int = 0,
) -> tuple[AnnealSchedules, AnnealSchedule]:
    """Adapt anneal schedules to account for a delayed measurement.

    All schedule lengths (max time) are reset to accommodate the largest
    support time plus a delay. The final time is rounded up to the nearest
    microsecond.

    Args:
        x_anneal_schedules: List of anneal schedules, as might
            returned by :func:`make_tds_x_anneal_schedules`.
        x_polarizing_schedule: Anneal schedule, as might be returned by
            :func:`make_tds_x_polarizing_schedule`.
        post_pwl_delay: Delay in microseconds to apply to each anneal schedule.
            Inclusion of a delay on the order of microseconds prevents line
            desynchronization, filtering and other non-idealities from
            interfering with waveform completion.
        decimals: decimals to which the end point is rounded up. By default
            to the nearest microsecond.

    Returns:
        tuple consisting of adapted anneal and polarizing schedules
    """
    anneal_schedules = deepcopy(x_anneal_schedules)
    polarizing_schedule = deepcopy(x_polarizing_schedule)
    if post_pwl_delay < 0.0:
        raise ValueError("delay must be non-negative.")
    completion_time = max(pwl[-1][0] for pwl in anneal_schedules)
    if polarizing_schedule:
        completion_time = max(completion_time, polarizing_schedule[-1][0])
    completion_time = completion_time + post_pwl_delay
    if decimals is not None:
        completion_time = float(
            math.ceil(10**decimals * completion_time / 10**decimals)
        )

    for anneal_schedule in anneal_schedules:
        if anneal_schedule[-1][0] != completion_time:
            anneal_schedule += [[completion_time, anneal_schedule[-1][1]]]

    if polarizing_schedule and polarizing_schedule[-1][0] != completion_time:
        polarizing_schedule.append([completion_time, polarizing_schedule[-1][1]])

    return anneal_schedules, polarizing_schedule


def verify_schedules(
    exp_feature_info: list[LineFeatureInfo],
    x_anneal_schedules: AnnealSchedules | None = None,
    x_polarizing_schedule: AnnealSchedule | None = None,
    check_rounding: bool = True,
    term_time: float | None = None,
):
    """Verify that schedules are compatible with sequencing and min time steps.

    This routine checks that the schedules are compatible with sequencing
    and min time steps. It checks terminal values of the polarizing schedules
    match those of anneal_schedules, and all schedules begin at time 0.

    It does not exhaustively check all requirements. See Documentation.

    Args:
        exp_feature_info: List of dicts containing experimental feature
            information for each line, as returned by `get_properties`.
        x_anneal_schedules: List of anneal schedules, as might
            returned by :func:`make_tds_x_anneal_schedules`.
        x_polarizing_schedule: Anneal schedule, as might be returned by
            :func:`make_tds_x_polarizing_schedule`.
        check_rounding: Whether to check that all schedule times are almost
            multiples of the minimum time step for each line.
        term_time: Expected end time, if not given is inferred
            from the schedules, and should be consistent.

    Raises:
        ScheduleError: If schedules are malformed or incompatible with
            sequencing, endpoints, or minimum time-step constraints.
    """

    if x_anneal_schedules:
        min_time_steps = {
            line: exp_feature_info[line]["minAnnealingTimeStep"]
            for line in range(len(exp_feature_info))
        }
        for line in range(len(min_time_steps)):

            seq_times = [
                t + min_time_steps[line] * idx
                for idx, (t, _) in enumerate(x_anneal_schedules[line])
            ]
            if len(x_anneal_schedules[line]) < 2 or any(
                len(s) != 2 for s in x_anneal_schedules[line]
            ):
                raise ScheduleError(
                    f"Anneal schedule on line {line} must contain at least two [time, value] points."
                )
            if not math.isclose(seq_times[0], 0, abs_tol=1e-9):
                raise ScheduleError(
                    f"Anneal schedule on line {line} must start at time 0; got {seq_times[0]}."
                )
            if term_time is not None:
                if not math.isclose(
                    x_anneal_schedules[line][-1][0],
                    term_time,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ScheduleError(
                        f"Anneal schedule on line {line} ends at {x_anneal_schedules[line][-1][0]}, expected {term_time}."
                    )

            if sorted(seq_times) != seq_times:
                raise ScheduleError(
                    f"Anneal schedule on line {line} is non-increasing at specified time precision."
                )
            if check_rounding:
                for seq_time in seq_times:
                    ratio = seq_time / min_time_steps[line]
                    if not math.isclose(
                        ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ScheduleError(
                            f"Anneal schedule time {seq_time} on line {line} is not an almost multiple of minAnnealingTimeStep={min_time_steps[line]}."
                        )
            term_time = x_anneal_schedules[line][-1][0]

    if x_polarizing_schedule:
        min_time_step = exp_feature_info[0]["minPolarizingTimeStep"]
        if len(x_polarizing_schedule) < 2 or any(
            len(s) != 2 for s in x_polarizing_schedule
        ):
            raise ScheduleError(
                "Polarizing schedule must contain at least two [time, value] points."
            )
        seq_times = [
            t + min_time_step * idx for idx, (t, _) in enumerate(x_polarizing_schedule)
        ]
        if not math.isclose(seq_times[0], 0, abs_tol=1e-9):
            raise ScheduleError(
                f"Polarizing schedule must start at time 0; got {seq_times[0]}."
            )
        if term_time is not None:
            if not math.isclose(
                x_polarizing_schedule[-1][0], term_time, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ScheduleError(
                    f"Polarizing schedule ends at {x_polarizing_schedule[-1][0]}, expected {term_time}."
                )
        if sorted(seq_times) != seq_times:
            raise ScheduleError(
                "Polarizing schedule is non-increasing at specified time precision."
            )
        if check_rounding:
            for seq_time in seq_times:
                ratio = seq_time / min_time_step
                if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                    raise ScheduleError(
                        f"Polarizing schedule time {seq_time} is not an almost multiple of minPolarizingTimeStep={min_time_step}."
                    )


def make_tds_x_anneal_schedules(
    exp_feature_info: list[LineFeatureInfo],
    target_lines: Iterable[int],
    target_c: float,
    detector_lines: Iterable[int],
    *,
    polarized_preparation_interval: Interval | None = None,
    depolarized_preparation_interval: Interval | None = None,
    detector_quench_time: float | None = None,
    source_lines: Iterable[int] = tuple(),
    source_quench_time: float | None = None,
    use_common_bounds: bool = False,
    use_overshoot: bool = True,
    post_pwl_delay: float = 1.0,
) -> AnnealSchedules:
    """Set annealing schedules for target-detector-source experiments.

    Lines are designated as source, detector, target or neutral (unused).
    The polarizing schedule produced by :func:`make_tds_x_polarizing_schedule`
    is assumed to define a polarized state during ``polarized_preparation_interval``
    when ``source_lines`` is not empty.
    Source, detector and unused line qubits are quasistatically prepared
    during ``polarized_preparation_interval`` by setting the normalized
    control bias to ``maxC`` or ``minC`` as appropriate.
    The polarizing schedule is assumed to be turned off with a safe
    separation before ``depolarized_preparation_interval``.
    Target line qubits are then quasistatically prepared to ``target_c``.
    Source line qubits are then quenched to decouple them from the target.
    Detector line qubits are then quenched to measure the target.

    Defaulting is designed with a view to Larmour precession examples in
    :ref:`qpu_experimental_research_mca_example`
    and to allow replication of the
    `published experimental results <https://doi.org/10.48550/arXiv.2603.15534>`_.
    Modification of additional static programmable parameters such as
    ``couplings``, ``flux_biases``, and ``anneal_offsets`` is necessary as
    part of experimental setup. These can interact with optimal choices for
    the returned anneal schedules.

    Args:
        exp_feature_info: List of dicts containing experimental feature
            information for each line, as returned by `get_properties`.
        target_lines: Iterable of target line indices.
        target_c: Schedule value at which the target is held.
        detector_lines: Iterable of detector line indices.
        polarized_preparation_interval: Tuple ``(start, end)`` giving the
            interval during which a polarizing signal is present. During this
            interval, unused and detector lines are set to ``minC`` whereas
            source lines are set to ``maxC``. If None, defaults to the
            ``polarized_preparation_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
        detector_quench_time: Time at which to quench the detector line quench,
            in microseconds. If None, defaults to the ``quench_time`` returned
            by :func:`make_tds_intervals` with default arguments.
        source_lines: Iterable of source line indices.
        depolarized_preparation_interval: Tuple ``(start, end)`` for the
            preparation stage that occurs after the polarizing schedule is
            returned to zero. If None, defaults to the
            ``depolarized_preparation_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
        source_quench_time: Time at which to quench the source line
            from ``maxC`` to ``minC``, in microseconds. If None, defaults to
            ``detector_quench_time``. Setting ``source_quench_time`` equal to
            ``detector_quench_time`` is recommended as `x_schedule_delays` can
            be used for higher fidelity variation of the time difference.
        use_common_bounds: Parameters can vary by line. When True is used,
            a set of common compatible values define all lines.
        use_overshoot: Whether to use overshoot transitions for source and
            detector quenches.
        post_pwl_delay: Additional delay, in microseconds, used to extend the
            terminal values of all schedules to a common endpoint.
    Returns:
        A piecewise linear schedule for all lines.
    Raises:
        ValueError: If any of the input parameters are invalid or incompatible.
    """
    (
        polarized_preparation_interval0,
        _,
        depolarized_preparation_interval0,
        quench_time0,
    ) = make_tds_intervals()
    if not polarized_preparation_interval:
        polarized_preparation_interval = polarized_preparation_interval0
    if not depolarized_preparation_interval:
        depolarized_preparation_interval = depolarized_preparation_interval0
    if not detector_quench_time:
        detector_quench_time = quench_time0
    num_lines = len(exp_feature_info)
    all_lines = set(range(num_lines))
    source_lines = set(source_lines)
    detector_lines = set(detector_lines)
    target_lines = set(target_lines)
    tds_lines = source_lines | detector_lines | target_lines
    if len(tds_lines) != len(source_lines) + len(detector_lines) + len(target_lines):
        raise ValueError("Source, detector and target lines must be disjoint.")
    if not tds_lines.issubset(all_lines):
        raise ValueError(
            "Source, detector and target lines must be valid line indices."
        )

    maxCs = {
        line: _round_sigfigs(exp_feature_info[line]["maxC"], sigfigs=3, mode="down")
        for line in range(num_lines)
    }
    minCs = {
        line: _round_sigfigs(exp_feature_info[line]["minC"], sigfigs=3, mode="up")
        for line in range(num_lines)
    }
    maxCOvershoots = {
        line: _round_sigfigs(
            exp_feature_info[line]["maxCOvershoot"], sigfigs=3, mode="down"
        )
        for line in range(num_lines)
    }
    minCOvershoots = {
        line: _round_sigfigs(
            exp_feature_info[line]["minCOvershoot"], sigfigs=3, mode="up"
        )
        for line in range(num_lines)
    }
    min_time_steps = {
        line: exp_feature_info[line]["minAnnealingTimeStep"]
        for line in range(num_lines)
    }
    holdOvershootFors = {
        line: exp_feature_info[line].get("holdOvershootFor", 0)
        for line in range(num_lines)
    }

    if use_common_bounds:
        maxC = min(maxCs.values())
        minC = max(minCs.values())
        if minC >= maxC:
            raise ValueError(
                "Incompatible maxC and minC values across lines, cannot use common bounds."
            )
        maxCOvershoot = min(maxCOvershoots.values())
        minCOvershoot = max(minCOvershoots.values())
        if minCOvershoot >= maxCOvershoot:
            raise ValueError(
                "Incompatible maxCOvershoot and minCOvershoot values across lines, cannot use common bounds."
            )
        min_time_step = min(min_time_steps.values())  #  Should check this is uniform
        holdOvershootFor = max(holdOvershootFors.values())

        maxCs = {line: maxC for line in range(num_lines)}
        minCs = {line: minC for line in range(num_lines)}
        maxCOvershoots = {line: maxCOvershoot for line in range(num_lines)}
        minCOvershoots = {line: minCOvershoot for line in range(num_lines)}
        min_time_steps = {line: min_time_step for line in range(num_lines)}
        holdOvershootFors = {line: holdOvershootFor for line in range(num_lines)}

    times = []
    if len(source_lines) > 0:
        if polarized_preparation_interval is None:
            raise ValueError(
                "Must specify polarized_preparation_interval if source_lines is not empty."
            )
    if polarized_preparation_interval[1] - polarized_preparation_interval[0] < min(
        min_time_steps[l] for l in all_lines - target_lines
    ):
        raise ValueError(
            "polarized_preparation_interval must have duration compatible with min step ."
        )
    times += list(polarized_preparation_interval)
    if not target_lines:
        raise ValueError("At least one target line must be specified.")
    if not depolarized_preparation_interval:
        depolarized_preparation_interval = polarized_preparation_interval
    else:
        times += list(depolarized_preparation_interval)

    if depolarized_preparation_interval[1] - depolarized_preparation_interval[0] < min(
        min_time_steps[l] for l in target_lines
    ):
        raise ValueError(
            "depolarized_preparation_interval must have duration compatible with min step ."
        )
    if not detector_lines:
        raise ValueError("At least one detector line must be specified.")
    if not source_lines or source_quench_time is None:
        source_quench_time = detector_quench_time
    times.append(min(source_quench_time, detector_quench_time))
    times.append(max(source_quench_time, detector_quench_time))

    if not sorted(times) == times:
        raise ValueError(
            "Times must be in non-decreasing order:"
            " polarized preparation, depolarized preparation, then source/detector quenches."
        )
    if times[0] < 0:
        raise ValueError("Times must be non-negative.")

    # By default all lines are switched off during the preparation
    # window, except source lines which are switched on.
    anneal_schedules = [
        (
            [
                [polarized_preparation_interval[0], 0.0],
                [polarized_preparation_interval[1], maxCs[line]],
            ]
            if line in source_lines
            else [
                [polarized_preparation_interval[0], 0.0],
                [polarized_preparation_interval[1], minCs[line]],
            ]
        )
        for line in all_lines
    ]

    for line in target_lines:
        # Turned slowly to target value:
        anneal_schedules[line] = [
            [depolarized_preparation_interval[0], 0.0],
            [depolarized_preparation_interval[1], target_c],
        ]

    for line in source_lines:
        if use_overshoot:
            if holdOvershootFors[line] > 2 * min_time_steps[line]:
                anneal_schedules[line] += [
                    [
                        source_quench_time
                        - holdOvershootFors[line]
                        + min_time_steps[line],
                        maxCs[line],
                    ],
                    [
                        source_quench_time
                        - holdOvershootFors[line]
                        + 2 * min_time_steps[line],
                        maxCOvershoots[line],
                    ],
                    [source_quench_time, maxCOvershoots[line]],
                    [source_quench_time + min_time_steps[line], minCOvershoots[line]],
                    [
                        source_quench_time
                        + holdOvershootFors[line]
                        - min_time_steps[line],
                        minCOvershoots[line],
                    ],
                    [source_quench_time + holdOvershootFors[line], minCs[line]],
                ]
            else:
                anneal_schedules[line] += [
                    [
                        source_quench_time
                        - holdOvershootFors[line]
                        + min_time_steps[line],
                        maxCs[line],
                    ],
                    [source_quench_time, maxCOvershoots[line]],
                    [source_quench_time + min_time_steps[line], minCOvershoots[line]],
                    [source_quench_time + holdOvershootFors[line], minCs[line]],
                ]
        else:
            anneal_schedules[line] += [
                [source_quench_time, maxCs[line]],
                [source_quench_time + min_time_steps[line], minCs[line]],
            ]

    for line in detector_lines:
        if use_overshoot:
            if holdOvershootFors[line] > 2 * min_time_steps[line]:
                anneal_schedules[line] += [
                    [
                        detector_quench_time
                        - holdOvershootFors[line]
                        + min_time_steps[line],
                        minCs[line],
                    ],
                    [
                        detector_quench_time
                        - holdOvershootFors[line]
                        + 2 * min_time_steps[line],
                        minCOvershoots[line],
                    ],
                    [detector_quench_time, minCOvershoots[line]],
                    [detector_quench_time + min_time_steps[line], maxCOvershoots[line]],
                    [
                        detector_quench_time
                        + holdOvershootFors[line]
                        - min_time_steps[line],
                        maxCOvershoots[line],
                    ],
                    [detector_quench_time + holdOvershootFors[line], maxCs[line]],
                ]
            else:
                anneal_schedules[line] += [
                    [detector_quench_time - min_time_steps[line], minCs[line]],
                    [detector_quench_time, minCOvershoots[line]],
                    [detector_quench_time + min_time_steps[line], maxCOvershoots[line]],
                    [detector_quench_time + 2 * min_time_steps[line], maxCs[line]],
                ]
        else:
            anneal_schedules[line] += [
                [detector_quench_time, minCs[line]],
                [detector_quench_time + min_time_steps[line], maxCs[line]],
            ]

    # Set initial point to time 0 for all schedules.
    for line in all_lines:
        if anneal_schedules[line][0][0] != 0:
            anneal_schedules[line] = [[0.0, 0.0]] + anneal_schedules[line]

    # Create regular gapped end point.
    anneal_schedules, _ = standardize_schedule_endpoints(
        anneal_schedules, post_pwl_delay=post_pwl_delay
    )

    return anneal_schedules


def make_tds_x_polarizing_schedule(
    depolarization_interval: Interval = None,
    sign_polarization: Literal[-1, 1] = 1,
) -> AnnealSchedule:
    """Set polarizing schedules suitable for target detector source experiments.

    Creates a polarized signal on all qubits that is held for some
    period and then reduced to zero at a given depolarization time.

    Args:
        depolarization_interval: Tuple containing the start and end times of
            the depolarization interval, in microseconds. If None, defaults
            to the ``depolarization_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
        sign_polarization: Sign of the initial polarization, +1 or -1.

    Returns:
        A piecewise-linear polarizing schedule beginning at time 0 with
        given polarization, and evolving to polarization 0
        over the depolarization interval.
    """
    if depolarization_interval is None:
        _, depolarization_interval, _, _ = make_tds_intervals()
    elif (
        len(depolarization_interval) != 2
        or depolarization_interval[1] - depolarization_interval[0] <= 0
    ):
        raise ValueError("depolarization_interval must have a positive duration.")
    polarizing_schedule = [
        [0.0, sign_polarization],
        [depolarization_interval[0], sign_polarization],
        [depolarization_interval[1], 0],
    ]
    return polarizing_schedule


def make_tds_x_schedules(
    exp_feature_info: list[LineFeatureInfo],
    target_lines: Iterable[int],
    target_c: float,
    detector_lines: Iterable[int],
    source_lines: Iterable[int] = tuple(),
    *,
    post_preparation_delay: float = 20.0,
    depolarization_time_scale: float = 2.0,
    use_overshoot: bool = True,
    sign_polarization: Literal[-1, 1] = 1,
) -> tuple[AnnealSchedules, AnnealSchedule]:
    """Build synchronized anneal and polarizing schedules for TDS experiments.

    This helper composes interval construction, anneal schedule generation,
    polarizing schedule generation, endpoint alignment, and schedule
    verification into a single call.

    Args:
        exp_feature_info: List of dicts containing experimental feature
            information for each line, as returned by `get_properties`.
        target_lines: Iterable of target line indices.
        target_c: Schedule value at which the target is held.
        detector_lines: Iterable of detector line indices.
        source_lines: Iterable of source line indices.
        post_preparation_delay: Delay in microseconds between completion of
            depolarization and start of depolarized target preparation.
        depolarization_time_scale: Time scale for slow (quasi-static)
            modification of the polarizing signal and
            preparation of qubits to polarized/depolarized states.
        use_overshoot: Whether to use overshoot transitions for source and
            detector quenches.
        sign_polarization: Initial sign of the polarizing bias, +1 or -1.

    Returns:
        A tuple ``(x_anneal_schedules, x_polarizing_schedule)`` where
        ``x_anneal_schedules`` is a list of per-line piecewise-linear anneal
        schedules and ``x_polarizing_schedule`` is the corresponding global
        polarizing schedule.

    Raises:
        ValueError: If input line assignments or timing parameters are
            incompatible.
        ScheduleError: If generated schedules fail sequencing or rounding
            validation checks.
    """
    (
        polarized_preparation_interval,
        depolarization_interval,
        depolarized_preparation_interval,
        detector_quench_time,
    ) = make_tds_intervals(
        post_preparation_delay=post_preparation_delay,
        depolarization_time_scale=depolarization_time_scale,
    )
    x_anneal_schedules = make_tds_x_anneal_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=target_lines,
        depolarized_preparation_interval=depolarized_preparation_interval,
        detector_lines=detector_lines,
        detector_quench_time=detector_quench_time,
        source_lines=source_lines,
        polarized_preparation_interval=polarized_preparation_interval,
        target_c=target_c,
        post_pwl_delay=0.0,
        use_overshoot=use_overshoot,
    )
    x_polarizing_schedule = make_tds_x_polarizing_schedule(
        depolarization_interval=depolarization_interval,
        sign_polarization=sign_polarization,
    )
    x_anneal_schedules, x_polarizing_schedule = standardize_schedule_endpoints(
        x_anneal_schedules,
        x_polarizing_schedule,
        post_pwl_delay=depolarization_time_scale,
    )
    verify_schedules(
        exp_feature_info,
        x_anneal_schedules=x_anneal_schedules,
        x_polarizing_schedule=x_polarizing_schedule,
    )

    return x_anneal_schedules, x_polarizing_schedule


if __name__ == "__main__":
    from dwave.system import DWaveSampler
    from dwave.experimental.multicolor_anneal.api import get_properties

    print("Module code added temporarily for testing purposes.")
    print("To be moved in part to tests and examples.")

    qpu = DWaveSampler(solver="Advantage2_system1_x_internal")
    exp_feature_info = get_properties(qpu)
    x_anneal_schedules, x_polarizing_schedule = make_tds_x_schedules(
        exp_feature_info=exp_feature_info,
        target_lines=(0,),
        target_c=0.37,
        detector_lines=(1,),
        source_lines=(2,),
    )
    # Adapt polarizing schedule
    import matplotlib.pyplot as plt

    plt.figure()
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
    plt.title("Example Anneal Schedules")
    plt.legend()
    plt.show()
