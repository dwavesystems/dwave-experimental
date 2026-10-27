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

import math
from collections.abc import Collection, Iterable
from copy import deepcopy
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Literal, NamedTuple, TypeAlias
from warnings import warn

import networkx as nx

from dwave.graphs import zephyr_coordinates

__all__ = [
    "ScheduleError",
    "ExpFeatureInfo",
    "Interval",
    "LineFeatureInfo",
    "AnnealSchedule",
    "XAnnealSchedules",
    "qubit_to_Advantage2_annealing_line",
    "line_assignments_from_feature_info",
    "make_tds_graph",
    "make_tds_intervals",
    "make_tds_x_anneal_schedules",
    "make_tds_x_polarizing_schedule",
    "make_tds_x_schedules",
    "parse_exp_feature_line_info",
    "standardize_schedule_endpoints",
    "verify_schedules",
]

Interval: TypeAlias = tuple[float, float]
LineFeatureInfo: TypeAlias = dict[str, Any]
# A list of [time, value] support points; may be a partial or full schedule.
AnnealSchedule: TypeAlias = list[list[float]]
XAnnealSchedules: TypeAlias = list[AnnealSchedule]


class ExpFeatureInfo(NamedTuple):
    """Multicolor-annealing properties, as returned by :func:`get_properties`.

    Attributes:
        polarizing_line_info: Annealing-line-independent polarizing properties.
        annealing_line_info: Per-annealing-line properties, in ascending order
            of annealing-line index.
    """

    polarizing_line_info: dict[str, Any]
    annealing_line_info: list[LineFeatureInfo]


class ScheduleError(ValueError):
    """Raised when schedules are incompatible with sequencing constraints."""


def _round_sigfigs(
    value: float, sigfigs: int = 3, mode: Literal["up", "down"] = "down"
) -> float:
    """Round a value up or down to a fixed number of significant figures.

    This addresses an issue in the client, whereby some inbound properties
    have more significant figures than the D-Wave Solver API can reliably use.

    Args:
        value: The value to round.
        sigfigs: The number of significant figures to round to.
        mode: Whether to round up or down.

    Returns:
        The rounded value.
    """
    if value == 0:
        return 0.0
    if sigfigs <= 0:
        raise ValueError("sigfigs must be positive")
    if mode not in ("up", "down"):
        raise ValueError("mode must be 'up' or 'down'")
    # Round from the decimal string so binary-float representation artifacts do
    # not perturb values already at the requested precision (Decimal(0.1) would).
    d = Decimal(str(value))
    exp = d.adjusted() - (sigfigs - 1)
    rounding = ROUND_CEILING if mode == "up" else ROUND_FLOOR
    return float(d.quantize(Decimal(1).scaleb(exp), rounding=rounding))


def qubit_to_Advantage2_annealing_line(
    n: int | tuple[int, int, int, int, int],
    shape: tuple[int, ...] | None = None,
    num_lines: int = 6,
) -> int:
    """Return the annealing line associated to an Advantage2 qubit

    Advantage2 processors can allow multicolor annealing based in
    some cases on a 6 or 12-line control scheme. Compatibility with this
    scheme should be confirmed using a solver API or release notes.
    Based on the Zephyr coordinate system (u,w,k,j,z), a qubit
    can be uniquely assigned a color. u denotes qubit orientation
    while j and z control aligned displacement on the processor. See also
    :func:`dwave.graphs.zephyr_graph` and
    :class:`dwave.graphs.zephyr_coordinates`.

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
        >>> exp_feature_info = mca.get_properties(qpu)            # doctest: +SKIP
        >>> annealing_lines = exp_feature_info.annealing_line_info   # doctest: +SKIP
        >>> shape = qpu.properties['topology']['shape']          # doctest: +SKIP
        >>> num_lines = len(annealing_lines)            # doctest: +SKIP
        >>> assert all(            # doctest: +SKIP
        ...     mca.qubit_to_Advantage2_annealing_line(n, shape, num_lines) == al_idx
        ...     for al_idx, al in enumerate(annealing_lines)
        ...     for n in al['qubits']
        ... )

        To explicitly select a solver that supports advanced annealing
        features, such as multi-color annealing, see
        :data:`~dwave.experimental.fast_reverse_anneal.api.SOLVER_FILTER`.
    """

    if isinstance(n, tuple):
        u, _, _, j, z = n
    else:
        if shape is None:
            raise ValueError("shape is required when n is an integer qubit label.")
        u, _, _, j, z = zephyr_coordinates(*shape).linear_to_zephyr(n)
    if num_lines == 6:
        return 3 * u + (1 - 2 * z - j) % 3
    elif num_lines == 12:
        return 6 * u + (z % 3 + 3 * j)
    else:
        raise ValueError("num_lines must be 6 or 12")


def line_assignments_from_feature_info(
    exp_feature_info: ExpFeatureInfo,
) -> dict[int, int]:
    """Map each qubit to its annealing-line index.

    Builds a lookup from qubit label to annealing-line index using the per-line
    ``qubits`` lists of ``exp_feature_info``. This is the properties-driven
    counterpart to :func:`qubit_to_Advantage2_annealing_line`, which instead
    derives the line from a qubit's Zephyr coordinate.

    Args:
        exp_feature_info: Multicolor-annealing properties, as returned by
            :func:`~dwave.experimental.multicolor_anneal.api.get_properties`.
            An :class:`ExpFeatureInfo` named tuple whose ``annealing_line_info``
            field is a list of per-annealing-line dictionaries in ascending
            order of annealing-line index, each with a ``qubits`` entry.

    Returns:
        A dictionary mapping each qubit label to its annealing-line index.

    Examples:
        >>> from dwave.system import DWaveSampler
        >>> import dwave.experimental.multicolor_anneal as mca
        >>> exp_feature_info = mca.get_properties(DWaveSampler())   # doctest: +SKIP
        >>> line_assignments = mca.line_assignments_from_feature_info(
        ...     exp_feature_info
        ... )            # doctest: +SKIP
    """
    exp_feature_info = ExpFeatureInfo(*exp_feature_info)
    return {
        n: al_idx
        for al_idx, al in enumerate(exp_feature_info.annealing_line_info)
        for n in al["qubits"]
    }


def make_tds_graph(
    target_graph: nx.Graph,
    detected_nodes: Collection[int] | None = None,
    sourced_nodes: Collection[int] | None = None,
) -> tuple[nx.Graph, dict]:
    """Decorate a target graph with detectors and sources.

    We add single node source and detector branches to nodes of a target
    graph.

    Args:
        target_graph: A networkx target graph
        detected_nodes: A collection of target nodes, if None all target nodes.
        sourced_nodes: A collection of target nodes, if None all target nodes.

    Returns:
        A copy of the graph where each selected target node ``n`` is connected
        to ``('detector', n)`` and/or ``('source', n)``.

    Raises:
        ValueError: If detected_nodes or sourced_nodes are not in the graph
            target_graph.
    """

    if detected_nodes is None:
        detected_nodes = target_graph.nodes()
    else:
        # Materialize once so a single-pass iterable is not exhausted by the
        # validation below, leaving the edge construction silently empty.
        detected_nodes = set(detected_nodes)
        if not detected_nodes.issubset(target_graph.nodes()):
            raise ValueError("detected_nodes are not compatible with the target graph")

    if sourced_nodes is None:
        sourced_nodes = target_graph.nodes()
    else:
        sourced_nodes = set(sourced_nodes)
        if not sourced_nodes.issubset(target_graph.nodes()):
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
    buffering_time_scale: float = 2.0,
    *,
    depolarizing_time_scale: float | None = None,
    anneal_preparation_time_scale: float | None = None,
    skip_explicit_polarization: bool = True,
) -> tuple[Interval, Interval, Interval, Interval, float]:
    """Make default intervals for schedule construction.

    This routine sets up the timescales on which the source (and detector)
    lines are prepared in polarized (depolarized) states, the polarizing bias
    is removed, and the target is prepared.

    Defaulting is designed with a view to Larmor precession examples in
    :ref:`qpu_experimental_research_mca_example`
    and to allow replication of the
    `published experimental results <https://doi.org/10.48550/arXiv.2603.15534>`_.

    Args:
        post_preparation_delay: Time to wait in microseconds
            after line preparation before quenching. This is lower bounded
            by 0 but typically some longer time scale of order 10.0 us is
            recommended.
        buffering_time_scale: Time before and after the depolarizing
            interval in which changes to the anneal schedule are prevented.
            This is lower bounded by the
            depolarizationAnnealScheduleRequiredDelay annealing-line property.
        depolarizing_time_scale: Step size for the polarizing schedule.
            It is lower bounded by ``minPolarizingTimeStep``, but typically a
            larger value of order microseconds is desirable for robust
            preparation. Defaults to ``buffering_time_scale`` when not
            set.
        anneal_preparation_time_scale: Step size for preparation of anneal
            schedules in polarized and depolarized regimes. If None, defaults
            to ``depolarizing_time_scale``. This is lower bounded by
            minAnnealingTimeStep annealing-line properties, but typically a
            larger value of order microseconds is desirable for robust
            preparation.
        skip_explicit_polarization: Whether to skip the polarization interval entirely.
            If False a time interval is specified to characterize the x_anneal_schedule
            change from neutral [t_1, 0] to polarized (t_p, +/-1); other stages are
            delayed. If True the value (0.0, 0.0) is returned and a polarized
            initialization is assumed. Defaults to True.

    Returns:
        A 5-tuple containing ``polarization_interval``, ``polarized_preparation_interval``,
        ``depolarization_interval``, ``depolarized_preparation_interval``,
        and ``quench_time``.
    """
    if depolarizing_time_scale is None:
        depolarizing_time_scale = buffering_time_scale

    if anneal_preparation_time_scale is None:
        anneal_preparation_time_scale = depolarizing_time_scale

    if skip_explicit_polarization:
        polarization_interval = (0.0, 0.0)
        t0 = 0.0
    else:
        polarization_interval = (0.0, 0.0 + depolarizing_time_scale)
        t0 = polarization_interval[-1] + buffering_time_scale

    polarized_preparation_interval = (t0, t0 + anneal_preparation_time_scale)
    tp = polarized_preparation_interval[-1] + buffering_time_scale

    depolarization_interval = (tp, tp + depolarizing_time_scale)
    ts = depolarization_interval[-1] + buffering_time_scale

    depolarized_preparation_interval = (ts, ts + anneal_preparation_time_scale)
    quench_time = depolarized_preparation_interval[-1] + post_preparation_delay

    return (
        polarization_interval,
        polarized_preparation_interval,
        depolarization_interval,
        depolarized_preparation_interval,
        quench_time,
    )


def standardize_schedule_endpoints(
    x_anneal_schedules: XAnnealSchedules,
    x_polarizing_schedule: AnnealSchedule | None = None,
    *,
    post_pwl_delay: float = 0.0,
    decimals: int | None = 0,
    end_buffer: float = 1.0,
) -> tuple[XAnnealSchedules, AnnealSchedule | None]:
    """Adapt anneal schedules to accommodate a common end point.

    All schedule lengths (max time) are reset to a single common endpoint: the
    largest support time across ``x_anneal_schedules`` and
    ``x_polarizing_schedule``, extended by at least ``end_buffer`` (and by
    ``post_pwl_delay`` if larger) and rounded up to the nearest microsecond.
    Terminating every PWL schedule with this shared flat hold ensures a common
    endpoint and smoother termination of the realized signals.

    Args:
        x_anneal_schedules: List of anneal schedules, as might be
            returned by :func:`make_tds_x_anneal_schedules`.
        x_polarizing_schedule: Anneal schedule, as might be returned by
            :func:`make_tds_x_polarizing_schedule`.
        post_pwl_delay: Delay in microseconds to apply to each anneal schedule.
            Inclusion of a delay on the order of microseconds prevents line
            desynchronization, filtering and other non-idealities from
            interfering with waveform completion.
        decimals: Decimals to which the end point is rounded up. By default
            to the nearest microsecond.
        end_buffer: Minimum terminal hold, in microseconds, that every schedule
            holds its final value for at the common endpoint. Schedules that
            already share the common endpoint and satisfy this hold are returned
            unchanged, keeping repeated calls idempotent. Any non-negative
            value, including 0, is accepted.

    Returns:
        tuple consisting of adapted anneal and polarizing schedules
    """
    anneal_schedules = deepcopy(x_anneal_schedules)
    polarizing_schedule = deepcopy(x_polarizing_schedule)
    if post_pwl_delay < 0.0:
        raise ValueError("delay must be non-negative.")
    if end_buffer < 0.0:
        raise ValueError("end_buffer must be non-negative.")

    schedules = list(anneal_schedules)
    if polarizing_schedule:
        schedules.append(polarizing_schedule)

    max_final = max(pwl[-1][0] for pwl in schedules)

    def has_end_buffer(pwl: AnnealSchedule) -> bool:
        return (
            len(pwl) >= 2
            and pwl[-1][1] == pwl[-2][1]
            and pwl[-1][0] - pwl[-2][0] >= end_buffer - 1e-9
        )

    # No change needed only when every schedule already shares the common
    # endpoint and holds its final value for at least end_buffer; this keeps
    # repeated calls (e.g. the internal standardization in
    # make_tds_x_schedules) idempotent.
    if all(
        has_end_buffer(pwl) and math.isclose(pwl[-1][0], max_final, abs_tol=1e-9)
        for pwl in schedules
    ):
        return anneal_schedules, polarizing_schedule

    # Common endpoint: the latest support time extended by the requested delay,
    # but by at least end_buffer so every schedule gets a terminal hold, then
    # rounded up so all schedules share the same final support value.
    completion_time = max_final + max(post_pwl_delay, end_buffer)
    if decimals is not None:
        scale = 10**decimals
        completion_time = math.ceil(completion_time * scale) / scale

    def extend(pwl: AnnealSchedule) -> None:
        if pwl[-1][0] == completion_time:
            return
        # Move an existing terminal hold rather than add a collinear point.
        if len(pwl) >= 2 and pwl[-1][1] == pwl[-2][1] and pwl[-1][0] < completion_time:
            pwl[-1][0] = completion_time
        else:
            pwl.append([completion_time, pwl[-1][1]])

    for anneal_schedule in anneal_schedules:
        extend(anneal_schedule)
    if polarizing_schedule:
        extend(polarizing_schedule)

    return anneal_schedules, polarizing_schedule


def _pwl_value(pwl: AnnealSchedule, t: float) -> float:
    """Linearly interpolate a piecewise-linear schedule at time ``t``."""
    if t <= pwl[0][0]:
        return pwl[0][1]
    if t >= pwl[-1][0]:
        return pwl[-1][1]
    for (t0, v0), (t1, v1) in zip(pwl, pwl[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v1
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pwl[-1][1]


def verify_schedules(
    exp_feature_info: ExpFeatureInfo,
    x_anneal_schedules: XAnnealSchedules | None = None,
    x_polarizing_schedule: AnnealSchedule | None = None,
    check_rounding: bool = True,
    term_time: float | None = None,
) -> None:
    """Verify that schedules are compatible with sequencing and min time steps.

    This routine checks that the schedules are compatible with sequencing
    and min time steps. It checks terminal values of the polarizing schedules
    match those of anneal_schedules, that all schedules begin at time 0, and
    that all schedules share a common (almost equal) endpoint.

    It does not exhaustively check all requirements. See Documentation.

    Args:
        exp_feature_info: Multicolor-annealing properties, as returned by
            :func:`~dwave.experimental.multicolor_anneal.api.get_properties`.
            An :class:`ExpFeatureInfo` named tuple whose ``polarizing_line_info``
            field is a dictionary describing the polarizing line and whose
            ``annealing_line_info`` field is a list of per-annealing-line
            dictionaries.
        x_anneal_schedules: List of anneal schedules, as might be
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
    exp_feature_info = ExpFeatureInfo(*exp_feature_info)

    # All provided schedules must share a common endpoint, as produced by
    # standardize_schedule_endpoints.
    endpoints = [schedule[-1][0] for schedule in (x_anneal_schedules or []) if schedule]
    if x_polarizing_schedule:
        endpoints.append(x_polarizing_schedule[-1][0])
    if endpoints and any(
        not math.isclose(t, endpoints[0], rel_tol=1e-9, abs_tol=1e-9)
        for t in endpoints
    ):
        raise ScheduleError(
            f"Schedule endpoints are not synchronized: {endpoints}."
        )

    if x_anneal_schedules:
        exp_feature_line_info = exp_feature_info.annealing_line_info
        if len(x_anneal_schedules) != len(exp_feature_line_info):
            raise ScheduleError(
                f"Expected {len(exp_feature_line_info)} anneal schedules "
                f"(one per line); got {len(x_anneal_schedules)}."
            )
        min_time_steps = {
            line: exp_feature_line_info[line]["minAnnealingTimeStep"]
            for line in range(len(exp_feature_line_info))
        }
        for line, min_time_step in min_time_steps.items():
            if len(x_anneal_schedules[line]) < 2 or any(
                len(s) != 2 for s in x_anneal_schedules[line]
            ):
                raise ScheduleError(
                    f"Anneal schedule on line {line} must contain at least "
                    "two [time, value] points."
                )
            seq_times = [
                t - min_time_step * idx
                for idx, (t, _) in enumerate(x_anneal_schedules[line])
            ]
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
                        f"Anneal schedule on line {line} ends at "
                        f"{x_anneal_schedules[line][-1][0]}, expected {term_time}."
                    )

            if any(
                later < earlier - 1e-9
                for earlier, later in zip(seq_times, seq_times[1:])
            ):
                raise ScheduleError(
                    f"Anneal schedule on line {line} does not maintain the "
                    "minimum time-step spacing."
                )
            if check_rounding:
                for seq_time in seq_times:
                    ratio = seq_time / min_time_steps[line]
                    if not math.isclose(
                        ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ScheduleError(
                            f"Anneal schedule time {seq_time} on line {line} is "
                            "not an almost multiple of "
                            f"minAnnealingTimeStep={min_time_steps[line]}."
                        )
            term_time = x_anneal_schedules[line][-1][0]

    if x_polarizing_schedule:
        min_time_step = exp_feature_info.polarizing_line_info["minPolarizingTimeStep"]
        depolarization_delay = exp_feature_info.polarizing_line_info.get(
            "depolarizationAnnealScheduleRequiredDelay", 0.0
        )  # Unreported delay cannot be enforced; 0.0 imposes no extra constraint.
        polarization_change_delay = exp_feature_info.polarizing_line_info.get(
            "polarizationChangeAnnealScheduleRequiredDelay", 0.0
        )  # Expected standard feature from October 28 2026, best practice regardless.
        if len(x_polarizing_schedule) < 2 or any(
            len(s) != 2 for s in x_polarizing_schedule
        ):
            raise ScheduleError(
                "Polarizing schedule must contain at least two [time, value] points."
            )
        seq_times = [
            t - min_time_step * idx for idx, (t, _) in enumerate(x_polarizing_schedule)
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
                    f"Polarizing schedule ends at {x_polarizing_schedule[-1][0]}, "
                    f"expected {term_time}."
                )
        if any(
            later < earlier - 1e-9
            for earlier, later in zip(seq_times, seq_times[1:])
        ):
            raise ScheduleError(
                "Polarizing schedule does not maintain the minimum time-step spacing."
            )
        if check_rounding:
            for seq_time in seq_times:
                ratio = seq_time / min_time_step
                if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                    raise ScheduleError(
                        f"Polarizing schedule time {seq_time} is not an almost "
                        f"multiple of minPolarizingTimeStep={min_time_step}."
                    )
        if x_anneal_schedules:

            def _anneal_variation(a: float, b: float) -> int | None:
                """Return the first anneal line that varies on ``[a, b]``, if any."""
                for line, schedule in enumerate(x_anneal_schedules):
                    value_at_a = _pwl_value(schedule, a)
                    if not (
                        math.isclose(
                            _pwl_value(schedule, b),
                            value_at_a,
                            rel_tol=1e-9,
                            abs_tol=1e-9,
                        )
                        and all(
                            math.isclose(pt_v, value_at_a, rel_tol=1e-9, abs_tol=1e-9)
                            for pt_t, pt_v in schedule
                            if a - 1e-9 <= pt_t <= b + 1e-9
                        )
                    ):
                        return line
                return None

            for (t0, v_prev), (t1, v) in zip(
                x_polarizing_schedule, x_polarizing_schedule[1:]
            ):
                if v == v_prev:
                    continue
                # Anneal schedules must not vary while the polarization changes,
                # padded by the required settling delay on either side.
                line = _anneal_variation(
                    t0 - polarization_change_delay, t1 + polarization_change_delay
                )
                if line is not None:
                    raise ScheduleError(
                        f"Variation on {line} within {polarization_change_delay} of "
                        f"the polarization change over ({t0}, {t1})."
                    )
                # Reaching 0 from any nonzero value is a depolarization and
                # additionally requires the anneal schedules to hold through the
                # depolarization delay, including multi-segment/non-unit ramps.
                # By design this required delay applies only on the window after
                # the depolarization event; the pre-event side is governed by the
                # separate polarization-change delay above, not by this one.
                if v == 0 and v_prev != 0:
                    line = _anneal_variation(t1, t1 + depolarization_delay)
                    if line is not None:
                        raise ScheduleError(
                            f"Variation on {line} within {depolarization_delay} of "
                            f"the depolarization event at time {t1}."
                        )


def parse_exp_feature_line_info(
    exp_feature_line_info: list[LineFeatureInfo],
    use_common_bounds: bool,
    symmetrize_c_bounds: bool,
    sigfigs: int = 3,
    standard_01_c_range: bool = False,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
]:
    """Parse line feature metadata into normalized per-line schedule parameters.

    This helper extracts the per-line bounds and timing parameters used by
    schedule construction and optionally transforms them:

    - rounds C and C-overshoot bounds to a fixed number of significant figures,
        this prevents client side errors.
    - optionally computes one set of bounds compatible with all lines.
    - optionally enforces symmetric bounds by setting ``min = -max``.
    - optionally forces both C and C-overshoot ranges to ``[0, 1]``.

    Args:
        exp_feature_line_info: List of per-annealing-line experimental feature
            dictionaries. Typically the ``annealing_line_info`` field of the
            structure returned
            by :func:`~dwave.experimental.multicolor_anneal.api.get_properties`.
        use_common_bounds: When True, replace per-line bounds with values that
            are jointly valid across all lines. This applies to time
            steps as well as c-ranges. False allows faster quench rates for
            high performance applications, but delays need to be considered
            more carefully between lines and QPUs.
        symmetrize_c_bounds: When True, enforce symmetric bounds line-by-line
            (``minC = -maxC`` and ``minCOvershoot = -maxCOvershoot``) after any
            common-bound reduction.
        sigfigs: Significant figures used when rounding C and overshoot bounds.
        standard_01_c_range: When True, C and C-overshoot ranges are set to
            ``[0, 1]`` for all lines. In this mode, ``use_common_bounds`` and
            ``symmetrize_c_bounds`` are ignored for C ranges.

    Returns:
        A 6-tuple containing:

        1. ``maxCs``: maximum anneal value per line.
        2. ``minCs``: minimum anneal value per line.
        3. ``maxCOvershoots``: maximum overshoot value per line.
        4. ``minCOvershoots``: minimum overshoot value per line.
        5. ``min_time_steps``: minimum annealing time step per line.
        6. ``holdOvershootFors``: overshoot hold durations per line.

    Raises:
        ValueError: If any line has non-ordered bounds after rounding,
            i.e. if ``minCOvershoot <= minC < maxC <= maxCOvershoot`` is not
            satisfied.
        ValueError: If ``use_common_bounds`` is requested but no compatible
            shared ``minC/maxC`` or ``minCOvershoot/maxCOvershoot`` interval
            exists across lines.
    """

    num_lines = len(exp_feature_line_info)

    min_time_steps = {
        line: efi["minAnnealingTimeStep"]
        for line, efi in enumerate(exp_feature_line_info)
    }
    holdOvershootFors = {
        line: efi.get("holdOvershootFor", 0.0)
        for line, efi in enumerate(exp_feature_line_info)
    }
    if use_common_bounds:
        # The shared step must satisfy every line, so take the largest minimum.
        min_time_step = max(min_time_steps.values())
        min_time_steps = {line: min_time_step for line in range(num_lines)}
        holdOvershootFor = max(holdOvershootFors.values())
        holdOvershootFors = {line: holdOvershootFor for line in range(num_lines)}

    if standard_01_c_range:
        maxCs = {line: 1.0 for line in range(num_lines)}
        maxCOvershoots = {line: 1.0 for line in range(num_lines)}
        minCs = {line: 0.0 for line in range(num_lines)}
        minCOvershoots = {line: 0.0 for line in range(num_lines)}
    else:
        maxCs = {
            line: _round_sigfigs(efi["maxC"], sigfigs=sigfigs, mode="down")
            for line, efi in enumerate(exp_feature_line_info)
        }
        minCs = {
            line: _round_sigfigs(efi["minC"], sigfigs=sigfigs, mode="up")
            for line, efi in enumerate(exp_feature_line_info)
        }
        maxCOvershoots = {
            line: _round_sigfigs(efi["maxCOvershoot"], sigfigs=sigfigs, mode="down")
            for line, efi in enumerate(exp_feature_line_info)
        }
        minCOvershoots = {
            line: _round_sigfigs(efi["minCOvershoot"], sigfigs=sigfigs, mode="up")
            for line, efi in enumerate(exp_feature_line_info)
        }
        invalid_lines = [
            line
            for line in range(num_lines)
            if not (
                minCOvershoots[line]
                <= minCs[line]
                < maxCs[line]
                <= maxCOvershoots[line]
            )
        ]
        if invalid_lines:
            raise ValueError(
                "minCOvershoot, minC, maxC and maxCOvershoot are not ordered on "
                f"line(s) {invalid_lines}."
            )

    if use_common_bounds and not standard_01_c_range:
        maxC = min(maxCs.values())
        maxCs = {line: maxC for line in range(num_lines)}
        minC = max(minCs.values())
        minCs = {line: minC for line in range(num_lines)}
        if minC >= maxC:
            raise ValueError(
                "Incompatible maxC and minC values across lines, cannot use common bounds."
            )
        maxCOvershoot = min(maxCOvershoots.values())
        maxCOvershoots = {line: maxCOvershoot for line in range(num_lines)}
        minCOvershoot = max(minCOvershoots.values())
        minCOvershoots = {line: minCOvershoot for line in range(num_lines)}
        if minCOvershoot >= maxCOvershoot:
            raise ValueError(
                "Incompatible maxCOvershoot and minCOvershoot values across "
                "lines, cannot use common bounds."
            )

    if symmetrize_c_bounds and not standard_01_c_range:
        maxCs = {
            line: min(-v1, v2)
            for line, (v1, v2) in enumerate(zip(minCs.values(), maxCs.values()))
        }
        maxCOvershoots = {
            line: min(-v1, v2)
            for line, (v1, v2) in enumerate(
                zip(minCOvershoots.values(), maxCOvershoots.values())
            )
        }
        minCs = {line: -c for line, c in maxCs.items()}
        minCOvershoots = {line: -c for line, c in maxCOvershoots.items()}
        degenerate_lines = [
            line for line in range(num_lines) if maxCs[line] <= minCs[line]
        ]
        if degenerate_lines:
            warn(
                "Symmetrizing C bounds produced a degenerate (empty) C range on "
                f"line(s) {degenerate_lines}."
            )

    return (
        maxCs,
        minCs,
        maxCOvershoots,
        minCOvershoots,
        min_time_steps,
        holdOvershootFors,
    )


def _quench_segments(
    quench_time: float,
    step: float,
    hold: float,
    prep_end: float,
    start_value: float,
    end_value: float,
    start_overshoot: float | None,
    end_overshoot: float,
    use_overshoot: bool,
    include_start_overshoot: bool = True,
) -> AnnealSchedule:
    """Return the PWL support points for a single line's quench.

    Produces the mirror-symmetric waveform shared by the source, detector, and
    detected-target quenches: an optional overshoot past ``end_value`` on the
    settling side (and past ``start_value`` on the approach side), with a
    ``hold``-microsecond plateau at the overshoot value when ``hold`` is at
    least three times the time ``step``. A narrower ``hold`` cannot carry two
    plateau support points spaced by ``step`` (they would be ``hold - 2 * step``
    apart), so it falls back to a single overshoot point, keeping every support
    time at least ``step`` apart as required by the sequencing checks.

    Args:
        quench_time: Time at which the quench occurs.
        step: Minimum time step for the line.
        hold: Overshoot hold duration (``holdOvershootFor``).
        prep_end: Preparation-interval end, used to clamp the pre-quench
            overshoot start so support points never step backwards in time.
        start_value: Value held before the quench.
        end_value: Value settled to after the quench.
        start_overshoot: Overshoot value on the approach side; ignored when
            ``include_start_overshoot`` is False.
        end_overshoot: Overshoot value on the settling side.
        use_overshoot: Whether to overshoot the endpoints.
        include_start_overshoot: Whether to drive through ``start_overshoot``
            before the quench. False for lines already held at an arbitrary
            value (detected targets) that overshoot only on the settling side.

    Returns:
        The ``[time, value]`` support points to append after the held value.
    """
    if not use_overshoot:
        if include_start_overshoot:
            return [
                [quench_time, start_value],
                [quench_time + step, end_value],
            ]
        return [[quench_time + step, end_value]]

    if not include_start_overshoot:
        if hold >= 3 * step:
            return [
                [quench_time + step, end_overshoot],
                [quench_time + hold - step, end_overshoot],
                [quench_time + hold, end_value],
            ]
        return [
            [quench_time + step, end_overshoot],
            [quench_time + 2 * step, end_value],
        ]

    assert start_overshoot is not None  # guaranteed by include_start_overshoot
    if hold >= 3 * step:
        overshoot_start = max(quench_time - hold, prep_end)
        return [
            [overshoot_start + step, start_value],
            [overshoot_start + 2 * step, start_overshoot],
            [quench_time, start_overshoot],
            [quench_time + step, end_overshoot],
            [quench_time + hold - step, end_overshoot],
            [quench_time + hold, end_value],
        ]
    return [
        [quench_time - step, start_value],
        [quench_time, start_overshoot],
        [quench_time + step, end_overshoot],
        [quench_time + 2 * step, end_value],
    ]


def make_tds_x_anneal_schedules(
    exp_feature_line_info: list[LineFeatureInfo],
    target_lines: Iterable[int] = tuple(),
    detector_lines: Iterable[int] = tuple(),
    *,
    target_c: float | dict[int, float] | None = None,
    polarized_preparation_interval: Interval | None = None,
    depolarized_preparation_interval: Interval | None = None,
    detector_quench_time: float | None = None,
    source_lines: Iterable[int] = tuple(),
    source_quench_time: float | None = None,
    use_common_bounds: bool = False,
    symmetrize_c_bounds: bool = False,
    use_standard_01_c_range: bool = False,
    use_overshoot: bool | dict[str, bool] = True,
    post_pwl_delay: float = 1.0,
    quench_step_sizes: float | dict[int, float] | None = None,
    detected_target_lines: Iterable[int] = tuple(),
) -> XAnnealSchedules:
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

    Defaulting is designed with a view to Larmor precession examples in
    :ref:`qpu_experimental_research_mca_example`
    and to allow replication of the
    `published experimental results <https://doi.org/10.48550/arXiv.2603.15534>`_.
    Modification of additional static programmable parameters such as
    ``couplings``, ``flux_biases``, and ``anneal_offsets`` is necessary as
    part of experimental setup. These can interact with optimal choices for
    the returned anneal schedules.

    Args:
        exp_feature_line_info: List of per-annealing-line experimental feature
            dictionaries. Typically the ``annealing_line_info`` field of the
            structure returned
            by :func:`~dwave.experimental.multicolor_anneal.api.get_properties`.
        target_lines: Iterable of target line indices.
        detector_lines: Iterable of detector line indices.
        target_c: Schedule value at which the target is held. This must be specified
            if target_lines is not empty. If a dictionary format is provided, the
            held value can vary by line - it should be keyed by line number, and must
            provide an entry for every target line and every detected target line. If
            it is not specified lines are prepared to Cmin.
        polarized_preparation_interval: Tuple ``(start, end)`` giving the
            interval during which a polarizing signal is present. During this
            interval, unused and detector lines are set to ``minC`` whereas
            source lines are set to ``maxC``. If None, defaults to the
            ``polarized_preparation_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
        depolarized_preparation_interval: Tuple ``(start, end)`` for the
            preparation stage that occurs after the polarizing schedule is
            returned to zero. If None, defaults to the
            ``depolarized_preparation_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
        detector_quench_time: Time at which to quench the detector line quench,
            in microseconds. If None, defaults to the ``quench_time`` returned
            by :func:`make_tds_intervals` with default arguments.
        source_lines: Iterable of source line indices.
        source_quench_time: Time at which to quench the source line
            from ``maxC`` to ``minC``, in microseconds. If None, defaults to
            ``detector_quench_time``. Setting ``source_quench_time`` equal to
            ``detector_quench_time`` is recommended as ``x_schedule_delays`` can
            be used for higher fidelity variation of the time difference.
        use_common_bounds: Parameters can vary by line. When True is used,
            a set of common compatible values define all lines. False allows
            faster quench rates for high performance applications, but delays
            need to be considered more carefully between lines and QPUs.
        symmetrize_c_bounds: Whether to enforce maxC = -minC across all lines.
            Defaults to False. False allows faster quench rates for high
            performance applications, but delays need to be considered more
            carefully between lines and QPUs.
        use_standard_01_c_range: Whether to ignore exp_feature_line_info C bounds and
            use C and C-overshoot ranges ``[0, 1]`` for all lines.
        use_overshoot: Whether to use overshoot transitions for source and
            detector quenches. When a bool, the same setting is applied to both
            source and detector quenches. A mapping with keys ``"source"`` and
            ``"detector"`` may be provided instead to control overshoot for
            source and detector quenches independently; missing keys default to
            ``True``.
        post_pwl_delay: Additional delay, in microseconds, used to extend the
            terminal values of all schedules to a common endpoint.
        quench_step_sizes: Time in microseconds for detector or source quench.
            Specified as a dictionary (one per line) or a single value applied
            to all lines. Defaults to min step size permissible
            according to exp_feature_info by line.
        detected_target_lines: Target lines that are held at non-zero Cmin,
            and then read out by the detection process.

    Returns:
        A piecewise linear schedule for all lines.

    Raises:
        ValueError: If any of the input parameters are invalid or incompatible.

    Examples:
        Construct anneal schedules from solver-derived line properties for a
        single target, detector, and source line:

        >>> from dwave.system import DWaveSampler
        >>> import dwave.experimental.multicolor_anneal as mca
        >>> exp_feature_info = mca.get_properties(DWaveSampler())        # doctest: +SKIP
        >>> annealing_line_info = exp_feature_info.annealing_line_info                    # doctest: +SKIP
        >>> x_anneal_schedules = mca.make_tds_x_anneal_schedules(
        ...     exp_feature_line_info=annealing_line_info,
        ...     target_lines={0},
        ...     target_c=0.5,
        ...     detector_lines={1},
        ...     source_lines={2},
        ... )            # doctest: +SKIP
    """
    (
        _,
        polarized_preparation_interval0,
        _,
        depolarized_preparation_interval0,
        quench_time0,
    ) = make_tds_intervals()
    if polarized_preparation_interval is None:
        polarized_preparation_interval = polarized_preparation_interval0
    if depolarized_preparation_interval is None:
        depolarized_preparation_interval = depolarized_preparation_interval0
    if detector_quench_time is None:
        # Derive the quench from the (possibly caller-supplied) preparation end so
        # custom intervals compose, rather than always using the all-default quench.
        detector_quench_time = depolarized_preparation_interval[1] + (
            quench_time0 - depolarized_preparation_interval0[1]
        )
    if isinstance(use_overshoot, bool):
        use_overshoot_source = use_overshoot_detector = use_overshoot
    else:
        use_overshoot_source = use_overshoot.get("source", True)
        use_overshoot_detector = use_overshoot.get("detector", True)
    num_lines = len(exp_feature_line_info)
    all_lines = set(range(num_lines))
    source_lines = set(source_lines)
    detector_lines = set(detector_lines)
    target_lines = set(target_lines)
    detected_target_lines = set(detected_target_lines)
    tds_lines = source_lines | detector_lines | target_lines | detected_target_lines
    if len(tds_lines) != (
        len(source_lines)
        + len(detector_lines)
        + len(target_lines)
        + len(detected_target_lines)
    ):
        raise ValueError("Source, detector and target lines must be disjoint.")
    if not tds_lines.issubset(all_lines):
        raise ValueError(
            "Source, detector and target lines must be valid line indices."
        )

    (
        maxCs,
        minCs,
        maxCOvershoots,
        minCOvershoots,
        min_time_steps,
        holdOvershootFors,
    ) = parse_exp_feature_line_info(
        exp_feature_line_info,
        use_common_bounds=use_common_bounds,
        symmetrize_c_bounds=symmetrize_c_bounds,
        standard_01_c_range=use_standard_01_c_range,
    )
    if not isinstance(target_c, dict):
        if target_c is None:
            target_c = minCs
        else:
            target_c = {line: target_c for line in all_lines}
    times = []
    non_target_lines = all_lines - target_lines
    if non_target_lines and polarized_preparation_interval[
        1
    ] - polarized_preparation_interval[0] < max(
        min_time_steps[line] for line in non_target_lines
    ):
        raise ValueError(
            "polarized_preparation_interval must have duration compatible with min step."
        )
    times += list(polarized_preparation_interval)
    times += list(depolarized_preparation_interval)

    if len(target_lines) > 0 and depolarized_preparation_interval[
        1
    ] - depolarized_preparation_interval[0] < max(
        min_time_steps[line] for line in target_lines
    ):
        raise ValueError(
            "depolarized_preparation_interval must have duration compatible with min step."
        )
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
            [depolarized_preparation_interval[1], target_c[line]],
        ]

    if quench_step_sizes is not None:
        if not isinstance(quench_step_sizes, dict):
            quench_step_sizes = {line: quench_step_sizes for line in all_lines}
        if any(
            quench_step_sizes[line] < min_time_steps[line] for line in quench_step_sizes
        ):
            raise ValueError(
                "Quench step sizes must be at least the minimum time step for each line."
            )
        min_time_steps = {**min_time_steps, **quench_step_sizes}

    for line in source_lines:
        anneal_schedules[line] += _quench_segments(
            quench_time=source_quench_time,
            step=min_time_steps[line],
            hold=holdOvershootFors[line],
            prep_end=polarized_preparation_interval[1],
            start_value=maxCs[line],
            end_value=minCs[line],
            start_overshoot=maxCOvershoots[line],
            end_overshoot=minCOvershoots[line],
            use_overshoot=use_overshoot_source,
        )
    for line in detected_target_lines:
        # Prepare like a target line, read out like a detector.
        anneal_schedules[line] = [
            [depolarized_preparation_interval[0], 0.0],
            [depolarized_preparation_interval[1], target_c[line]],
            [detector_quench_time, target_c[line]],
        ]
        anneal_schedules[line] += _quench_segments(
            quench_time=detector_quench_time,
            step=min_time_steps[line],
            hold=holdOvershootFors[line],
            prep_end=depolarized_preparation_interval[1],
            start_value=target_c[line],
            end_value=maxCs[line],
            start_overshoot=None,
            end_overshoot=maxCOvershoots[line],
            use_overshoot=use_overshoot_detector,
            include_start_overshoot=False,
        )

    for line in detector_lines:
        anneal_schedules[line] += _quench_segments(
            quench_time=detector_quench_time,
            step=min_time_steps[line],
            hold=holdOvershootFors[line],
            prep_end=polarized_preparation_interval[1],
            start_value=minCs[line],
            end_value=maxCs[line],
            start_overshoot=minCOvershoots[line],
            end_overshoot=maxCOvershoots[line],
            use_overshoot=use_overshoot_detector,
        )

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
    polarization_interval: Interval | None = None,
    depolarization_interval: Interval | None = None,
    sign_polarization: Literal[-1, 1, 0] = 1,
) -> AnnealSchedule:
    """Set polarizing schedules suitable for target detector source experiments.

    Creates a polarized signal on all qubits that is held for some
    period and then reduced to zero at a given depolarization time.

    Args:
        polarization_interval: Tuple containing the start and end times of
            the polarization interval, in microseconds. If None, defaults
            to the ``polarization_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
            Note if the zero interval is given, the waveform is specified as
            starting in a polarized state (0.0, sign_polarization), rather
            than progressing from 0 to sign_polarization over the given interval.
        depolarization_interval: Tuple containing the start and end times of
            the depolarization interval, in microseconds. If None, defaults
            to the ``depolarization_interval`` returned by
            :func:`make_tds_intervals` with default arguments.
        sign_polarization: Sign of the initial polarization, +1 or -1. If 0
            then no polarizing signal is applied, but the interval-wise
            pattern of PWL construction doesn't change.

    Returns:
        A piecewise-linear polarizing schedule beginning at time 0 with
        given polarization, and evolving to polarization 0
        over the depolarization interval.

    Examples:
        Build a polarizing schedule using a depolarization interval.
        >>> from dwave.system import DWaveSampler
        >>> from dwave.experimental.multicolor_anneal import (
        ...     get_properties, make_tds_intervals,
        ...     make_tds_x_polarizing_schedule,
        ... )
        >>> exp_feature_info = get_properties(DWaveSampler())   # doctest: +SKIP
        >>> delay = exp_feature_info.polarizing_line_info[
        ...     'depolarizationAnnealScheduleRequiredDelay'
        ... ]            # doctest: +SKIP
        >>> _, _, depolarization_interval, _, _ = make_tds_intervals(
        ...     depolarizing_time_scale=delay,
        ... )            # doctest: +SKIP
        >>> x_polarizing_schedule = make_tds_x_polarizing_schedule(
        ...     depolarization_interval=depolarization_interval,
        ... )            # doctest: +SKIP
    """
    if depolarization_interval is None:
        default_polarization, _, default_depolarization, _, _ = make_tds_intervals()
        depolarization_interval = default_depolarization
        if polarization_interval is None:
            polarization_interval = default_polarization
    elif polarization_interval is None:
        # Only a depolarization interval was given: start polarized.
        polarization_interval = (0.0, 0.0)

    if polarization_interval[0] == polarization_interval[1] == 0:
        intervals = (depolarization_interval,)
    else:
        if depolarization_interval[0] <= polarization_interval[1]:
            raise ValueError(
                "polarization_interval must end before depolarization_interval starts."
            )
        intervals = (polarization_interval, depolarization_interval)
    if any(
        len(interval) != 2 or interval[1] - interval[0] <= 0
        for interval in intervals
    ):
        raise ValueError("intervals must have positive durations.")
    polarizing_schedule = []
    if polarization_interval[0] != polarization_interval[1]:
        polarizing_schedule += [
            [polarization_interval[0], 0],
            [polarization_interval[1], sign_polarization],
        ]

    polarizing_schedule += [
        [depolarization_interval[0], sign_polarization],
        [depolarization_interval[1], 0],
    ]

    if polarizing_schedule[0][0] > 0:
        polarizing_schedule = [[0.0, polarizing_schedule[0][1]]] + polarizing_schedule
    return polarizing_schedule


def make_tds_x_schedules(
    exp_feature_info: ExpFeatureInfo,
    target_lines: Iterable[int],
    detector_lines: Iterable[int],
    source_lines: Iterable[int] = tuple(),
    *,
    target_c: float | dict[int, float] | None = None,
    post_preparation_delay: float = 20.0,
    depolarization_time_scale: float = 2.0,
    use_common_bounds: bool = False,
    use_01_c_range: bool = False,
    symmetrize_c_bounds: bool = False,
    use_overshoot: bool | dict[str, bool] = True,
    sign_polarization: Literal[-1, 1, 0] = 1,
    skip_explicit_polarization: bool = True,
) -> tuple[XAnnealSchedules, AnnealSchedule | None]:
    """Build synchronized anneal and polarizing schedules for TDS experiments.

    This helper composes interval construction, anneal schedule generation,
    polarizing schedule generation, endpoint alignment, and schedule
    verification into a single call.

    .. note::
        Schedule support times are derived from the chosen time scales (such as
        ``post_preparation_delay`` and ``depolarization_time_scale``) and each
        line's ``minAnnealingTimeStep``. The final verification requires every
        support time to be an (almost) integer multiple of the relevant
        ``minAnnealingTimeStep``, so the time scales must be commensurate with
        every line's minimum time step. For a solver whose ``minAnnealingTimeStep``
        does not divide the default time scales, supply commensurate values (for
        example a ``depolarization_time_scale`` that is an integer multiple of
        the step); otherwise a :class:`ScheduleError` is raised deliberately to
        avoid submitting an unrealizable program.

    Args:
        exp_feature_info: Multicolor-annealing properties, as returned by
            :func:`~dwave.experimental.multicolor_anneal.api.get_properties`.
            An :class:`ExpFeatureInfo` named tuple whose ``polarizing_line_info``
            field is a dictionary describing the polarizing line and whose
            ``annealing_line_info`` field is a list of per-annealing-line
            dictionaries.
        target_lines: Iterable of target line indices.
        detector_lines: Iterable of detector line indices.
        source_lines: Iterable of source line indices.
        target_c: Schedule value at which the target is held. Can be a float
            (applied to all target lines) or a dictionary mapping target line
            indices to their respective schedule values. target_c must be
            specified if target_lines is not empty.
        post_preparation_delay: Delay in microseconds between completion of
            depolarization and start of depolarized target preparation.
        depolarization_time_scale: Time scale for slow (quasi-static)
            modification of the polarizing signal and
            preparation of qubits to polarized/depolarized states.
        use_common_bounds: When True, replace per-line bounds with values that
            are jointly valid across all lines. This applies to time
            steps as well as c-ranges. False allows faster quench rates for
            high performance applications, but delays need to be considered
            more carefully between lines and QPUs.
        use_01_c_range: Whether to force C and C-overshoot bounds to ``[0, 1]``
            when constructing anneal schedules.
        symmetrize_c_bounds: Whether to enforce symmetric per-line C-bounds,
            i.e. ``minC = -maxC`` and ``minCOvershoot = -maxCOvershoot``.
            Defaults to False. False allows faster quench rates for high
            performance applications, but delays need to be considered more
            carefully between lines and QPUs.
        use_overshoot: Whether to use overshoot transitions for source and
            detector quenches. When a bool, the same setting is applied to both
            source and detector quenches. A mapping with keys ``"source"`` and
            ``"detector"`` may be provided instead to control overshoot for
            source and detector quenches independently; missing keys default to
            ``True``.
        sign_polarization: Initial sign of the polarizing bias, +1, -1 or 0.
        skip_explicit_polarization: Whether to skip the polarization interval
            entirely when constructing intervals via :func:`make_tds_intervals`.
            Defaults to True.

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
    exp_feature_info = ExpFeatureInfo(*exp_feature_info)
    (
        polarization_interval,
        polarized_preparation_interval,
        depolarization_interval,
        depolarized_preparation_interval,
        detector_quench_time,
    ) = make_tds_intervals(
        post_preparation_delay=post_preparation_delay,
        buffering_time_scale=depolarization_time_scale,
        skip_explicit_polarization=skip_explicit_polarization,
    )
    x_anneal_schedules = make_tds_x_anneal_schedules(
        exp_feature_line_info=exp_feature_info.annealing_line_info,
        target_lines=target_lines,
        depolarized_preparation_interval=depolarized_preparation_interval,
        detector_lines=detector_lines,
        detector_quench_time=detector_quench_time,
        source_lines=source_lines,
        polarized_preparation_interval=polarized_preparation_interval,
        target_c=target_c,
        post_pwl_delay=depolarization_time_scale,
        use_common_bounds=use_common_bounds,
        use_standard_01_c_range=use_01_c_range,
        symmetrize_c_bounds=symmetrize_c_bounds,
        use_overshoot=use_overshoot,
    )
    x_polarizing_schedule = make_tds_x_polarizing_schedule(
        polarization_interval=polarization_interval,
        depolarization_interval=depolarization_interval,
        sign_polarization=sign_polarization,
    )
    # make_tds_x_anneal_schedules already standardized the anneal schedules;
    # this call brings the polarizing schedule onto the same common endpoint.
    # standardize_schedule_endpoints is idempotent and moves terminal holds in
    # place, so re-standardizing adds no redundant breakpoints.
    x_anneal_schedules, x_polarizing_schedule = standardize_schedule_endpoints(
        x_anneal_schedules,
        x_polarizing_schedule,
        post_pwl_delay=0.0,
    )
    verify_schedules(
        exp_feature_info,
        x_anneal_schedules=x_anneal_schedules,
        x_polarizing_schedule=x_polarizing_schedule,
    )

    return x_anneal_schedules, x_polarizing_schedule
