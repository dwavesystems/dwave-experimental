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
from typing import Iterable, Literal

import networkx as nx

from dwave_networkx import zephyr_coordinates


__all__ = ["qubit_to_Advantage2_annealing_line", "make_tds_graph"]


def qubit_to_Advantage2_annealing_line(
    n: int | tuple, shape: tuple, num_lines: int = 6
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
            property properties['topology']['shape']
        num_lines: number of annealing lines, may be 6 or 12.

    Returns:
        Integer annealing line assignment for Advantage2 processors
        using 6 or 12-annealing line control.

    Examples:
        Retrieve MCA annealing lines' properties for a default solver, and
        if a 6 (or 12) color scheme is used confirm the programmatic mapping is
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
    detected_nodes: list[int] | None = None,
    sourced_nodes: list[int] | None = None,
) -> tuple[nx.Graph, dict]:
    """Decorate a target graph with detectors and sources.

    We add single node source and detector branches to nodes of a target
    graph.

    Args:
        target_graph: A networkx target graph
        detector_nodes: An iterable on the target nodes, if None
            all target nodes.
        source_nodes: An iterable on the target nodes, if None
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


def make_default_intervals(
    post_polarization_delay: float = 20.0,
    polarization_schedule_step_size: None | float = None,
    anneal_schedule_step_size: None | float = None,
):
    """Make default intervals for schedules construction.

    This routine sets up the quasi-static timescales on which the
    source is prepared in a polarized state, the polarizing bias
    is removed and the target is prepared.

    Defaulting is designed with a view to Larmour precession examples in
    :ref:`qpu_experimental_research_mca_example`
    and to allow replication of the
    `published experimental results <https://doi.org/10.48550/arXiv.2603.15534>`_.

    Args:
        post_polarization_delay: Time to wait in microseconds
            after polarization before starting variation of the target line.
        polarization_schedule_step_size: Step size for the polarization
            schedule. If None, defaults to ``post_polarization_delay``.
        anneal_schedule_step_size: Step size for the anneal schedules. If
            None, defaults to ``post_polarization_delay``.

    Returns:
        A 3-tuple containing ``polarized_preparation_interval``,
        ``depolarization_interval``, and
        ``depolarized_preparation_interval``.
    """
    if polarization_schedule_step_size is None:
        polarization_schedule_step_size = post_polarization_delay
    if anneal_schedule_step_size is None:
        anneal_schedule_step_size = post_polarization_delay

    polarized_preparation_interval = (0.0, anneal_schedule_step_size)
    tp = polarized_preparation_interval[-1] + polarization_schedule_step_size
    depolarization_interval = (tp, tp + polarization_schedule_step_size)
    ts = depolarization_interval[-1] + post_polarization_delay
    depolarized_preparation_interval = (
        ts,
        ts + anneal_schedule_step_size,
    )

    return (
        polarized_preparation_interval,
        depolarization_interval,
        depolarized_preparation_interval,
    )


def standardize_schedule_endpoints(
    anneal_schedules: list[list[list[float]]], delay: float = 0.0
):
    """Adapt anneal schedules to account for a delayed measurement.

    All schedule lengths (max time) are reset to accommodate the largest
    support time plus a delay.

    Args:
        anneal_schedules: List of anneal schedules, as returned by
            :func:`make_tds_x_anneal_schedules`.
        delay: Delay in microseconds to apply to each anneal schedule.
            Inclusion of a delay on the order of microseconds prevents line
            desynchronization, filtering and other non-idealities from
            interfering with waveform completion.

    Returns:
        List of adapted anneal schedules.
    """
    anneal_schedules = deepcopy(anneal_schedules)
    if delay < 0.0:
        raise ValueError("delay must be non-negative.")
    completion_time = max(pwl[-1][0] + delay for pwl in anneal_schedules)
    for anneal_schedule in anneal_schedules:
        if anneal_schedule[-1][0] != completion_time:
            anneal_schedule += [[completion_time, anneal_schedule[-1][1]]]

    return anneal_schedules


def make_tds_x_anneal_schedules(
    exp_feature_info: list,
    target_lines: Iterable[int],
    target_c: float,
    detector_lines: Iterable[int],
    polarized_preparation_interval: tuple[float, float],
    detector_quench_time: float,
    *,
    source_lines: Iterable[int] = tuple(),
    depolarized_preparation_interval: tuple[float, float] | None = None,
    source_quench_time: float = None,
    use_common_bounds: bool = True,
    use_overshoot: bool = False,
    post_pwl_delay: float = 1.0,
) -> list[list[list[float]]]:
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
            information for each line, as returned by `getProperties`.
        target_lines: Iterable of target line indices.
        target_c: Schedule value at which the target is held.
        detector_lines: Iterable of detector line indices.
        polarized_preparation_interval: Tuple ``(start, end)`` giving the
            interval during which a polarizing signal is present. During this
            interval, unused and detector lines are set to ``minC`` whereas
            source lines are set to ``maxC``.
        detector_quench_time: Time at which to quench the detector line quench,
            in microseconds.
        source_lines: Iterable of source line indices.
        depolarized_preparation_interval: Tuple ``(start, end)`` for the
            preparation stage that occurs after the polarizing schedule is
            returned to zero. This can be set to None, in which case all lines
            are prepared during ``polarized_preparation_interval``. Targets are
            set to ``target_c`` during this interval; other lines are unchanged.
        source_quench_time: Time at which to quench the source line
            from ``maxC`` to ``minC``, in microseconds. If None, defaults to
            ``detector_quench_time``. Setting ``source_quench_time`` equal to
            ``detector_quench_time`` is recommended as `x_schedule_delays` can
            be used for higher fidelity variation of the time difference.
        use_common_bounds: Parameters can vary by line. When True is used,
            a set of common compatible values define all lines.
        use_overshoot: Whether to use overshoot transitions for source and
            detector quenches. This mode is not yet implemented.
        post_pwl_delay: Additional delay, in microseconds, used to extend the
            terminal values of all schedules to a common endpoint.

    Returns:
        A piecewise linear schedule for all lines.
    """

    num_lines = len(exp_feature_info)
    print(num_lines)
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

    if use_common_bounds:
        maxC = min(exp_feature_info[line]["maxC"] for line in range(num_lines))
        minC = max(exp_feature_info[line]["minC"] for line in range(num_lines))
        maxCs = {line: maxC for line in range(num_lines)}
        minCs = {line: minC for line in range(num_lines)}
        min_time_step = max(
            exp_feature_info[line]["minAnnealingTimeStep"] for line in range(num_lines)
        )
        min_time_steps = {line: min_time_step for line in range(num_lines)}
    else:
        maxCs = {line: exp_feature_info[line]["maxC"] for line in range(num_lines)}
        minCs = {line: exp_feature_info[line]["minC"] for line in range(num_lines)}
        min_time_steps = {
            line: exp_feature_info[line]["minAnnealingTimeStep"]
            for line in range(num_lines)
        }

    # Check that times are compatible with sequencing and min time steps.
    times = []
    if len(source_lines) > 0:
        if polarized_preparation_interval is None:
            raise ValueError(
                "Must specify polarized_preparation_interval if source_lines is not empty."
            )
        if polarized_preparation_interval[1] - polarized_preparation_interval[0] < min(
            min_time_steps[l] for l in source_lines
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
            " source preparation, target preparation, source/detector quenches."
        )
    if times[0] < 0:
        raise ValueError("Times must be non-negative.")

    # By default all lines are switched off during the preparation
    # window. This default is overwritten or augmented according
    # to the line class.
    anneal_schedules = [
        [
            [polarized_preparation_interval[0], 0.0],
            [polarized_preparation_interval[1], minCs[line]],
        ]
        for line in all_lines
    ]

    for line in target_lines:
        # Turned slowly to target value:
        anneal_schedules[line] = [
            [depolarized_preparation_interval[0], 0.0],
            [depolarized_preparation_interval[1], target_c],
        ]

    for line in source_lines:
        # Polarized, rather than depolarized, in the preparation window
        anneal_schedules[line] = [
            [polarized_preparation_interval[0], 0.0],
            [polarized_preparation_interval[1], maxCs[line]],
        ]
        if use_overshoot:
            raise ValueError("Not yet implemented (draft pull request), see Aking code")
        else:
            anneal_schedules[line] += [
                [source_quench_time, maxCs[line]],
                [source_quench_time + min_time_steps[line], minCs[line]],
            ]

    for line in detector_lines:
        if use_overshoot:
            raise ValueError("Not yet implemented (draft pull request), see Aking code")
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
    anneal_schedules = standardize_schedule_endpoints(
        anneal_schedules, post_pwl_delay
    )

    return anneal_schedules


def make_tds_x_polarizing_schedule(
    sign_polarization: Literal[-1, 1] = 1,
    depolarization_time: float = 6.0,
    polarizing_time_step: float = 1.0,
):
    """Set polarizing schedules suitable for target detector source experiments.

    Creates a polarized signal on all qubits that is held for some
    period and then reduced to zero at a given depolarization time.

    Args:
        sign_polarization: Sign of the initial polarization, +1 or -1.
        depolarization_time: Time at which the polarizing schedule
            is returned to 0, in microseconds.
        polarizing_time_step: Time required to reach zero from the
            initial polarization, in microseconds. This should be at least the
            minimum polarization time step supported by the solver.

    Returns:
        A piecewise-linear polarizing schedule beginning at time 0 with
        ``sign_polarization`` and returning to 0 at ``depolarization_time``.
    """
    if depolarization_time < 2 * polarizing_time_step:
        raise ValueError(
            "depolarization_time must be greater than or equal to 2 * polarizing_time_step."
        )
    polarization_schedule = [
        [0, sign_polarization],
        [depolarization_time - polarizing_time_step, sign_polarization],
        [depolarization_time, 0],
    ]
    return polarization_schedule


if __name__ == "__main__":
    from dwave.system import DWaveSampler
    from dwave.experimental.multicolor_anneal.api import get_properties

    print("Module code added temporarily for testing purposes.")
    print("To be moved in part to tests and examples.")
    quasistatic_time_step = 1.0
    post_polarization_delay = 20.0
    polarized_preparation_interval = (2.0, 2.0 + quasistatic_time_step)
    depolarization_time = (
        polarized_preparation_interval[1] + 2.0 + quasistatic_time_step
    )
    depolarized_preparation_interval = (
        depolarization_time + post_polarization_delay,
        depolarization_time + post_polarization_delay + quasistatic_time_step,
    )
    detector_quench_time = depolarized_preparation_interval[1] + quasistatic_time_step
    x_polarizing_schedule = make_tds_x_polarizing_schedule(
        sign_polarization=-1,
        depolarization_time=depolarization_time,
        polarizing_time_step=quasistatic_time_step,
    )
    print(x_polarizing_schedule)

    qpu = DWaveSampler(solver="Advantage2_system1_x_internal")
    exp_feature_info = get_properties(qpu)

    x_anneal_schedules = make_tds_x_anneal_schedules(
        exp_feature_info,
        target_lines=(0,),
        depolarized_preparation_interval=depolarized_preparation_interval,
        detector_lines=(1,),
        detector_quench_time=detector_quench_time,
        source_lines=(2,),
        polarized_preparation_interval=polarized_preparation_interval,
        target_c=0.37,
        post_pwl_delay=quasistatic_time_step,
    )
    print(x_anneal_schedules)

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
    assert len(x_anneal_schedules) == 6
    plt.show()
