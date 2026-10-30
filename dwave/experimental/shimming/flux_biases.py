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

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Literal

import dimod
import numpy as np
from dimod.typing import Bias, Variable

from dwave.experimental.multicolor_anneal import (
    AnnealSchedule,
    LineFeatureInfo,
    XAnnealSchedules,
    line_assignments_from_feature_info,
    make_tds_x_anneal_schedules,
)
from dwave.experimental.multicolor_anneal.utils import _round_sigfigs

__all__ = [
    "shim_flux_biases",
    "shim_linewise_flux_biases",
    "shim_tds_flux_biases",
    "qubit_freezeout_alpha_phi",
]


def qubit_freezeout_alpha_phi(
    eff_temp_phi: float = 0.112,
    flux_associated_variance: float = 1 / 1024,
    estimator_variance: float = 1 / 256,
    unit_conversion: float = 1.148e-3,
):
    r"""Determine the learning rate for independent qubits.

    For a qubit offset by :math:`\Phi_{0i}` that is to be corrected by a choice
    of :ref:`flux bias <parameter_qpu_flux_biases>` :math:`\Phi`, a model of
    single-qubit freezeout dictates that magnetization
    :math:`<s_i> = tanh(\frac{\Phi_{0i} + \Phi_i}{T})`, where :math:`T` is the
    effective temperature.

    Assume the unshimmed magnetization to be zero-mean distributed with small
    variance, :math:`\Delta_1`, and a standard sampling-based estimator with
    variance :math:`\Delta_2 = \frac{1}{\text{num_reads}}`. You can then
    determine an update to the flux, :math:`\Phi = l <s_i>_{\text{data}}`, where
    the learning rate :math:`l = T \frac{\Delta_1}{\Delta_1 + \Delta_2}`. This
    update is optimal in the sense that it minimizes the expected square
    magnetization.

    For correlated spin systems and/or experiments that are not well described
    by thermal freezeout, a data-driven approach is recommended to determining
    the schedule and related parameters. The freezeout (Boltzmann) distribution
    can be extended to correlated models, wherein the covariance matrix plays a
    role in determining the optimal learning rate. A single qubit rate can
    remain a good approximation given weakly correlated spins.

    Args:
        eff_temp_phi:
            Effective (unitless) temperature at freezeout. This
            can be determined from current device parameters.
        flux_associated_variance:
            The expected variance of the magnetization (:math:`m`) due to flux
            offset.
        estimator_variance:
            The expected variance in the magnetization estimate,
            :math:`\frac{1-m^2}{\text{num_reads}}`.
        unit_conversion:
            Conversion from units of :ref:`h <parameter_qpu_h>` to units of
            :math:`\Phi` can be determined from published device parameters. See
            :func:`~dwave.system.temperatures.h_to_fluxbias`.
    Returns:
        An appropriate scale for the learning rate, minimizing the expected
        square magnetization.

    Example:
        Determining an :math:`\alpha_\Phi` appropriate for forward anneal of a
        weakly coupled system, ``Advantage_system4.1`` based on published
        parameters. Note that defaults (by contrast) are determined based on
        published values for ``Advantage2_system1.3``.

        >>> from dwave.experimental.shimming import qubit_freezeout_alpha_phi
        ...
        >>> alpha_phi = qubit_freezeout_alpha_phi(
        ...     eff_temp_phi=0.198,
        ...     flux_associated_variance=1/1024,
        ...     estimator_variance=1/256,
        ...     unit_conversion=1.647e-3)

    """
    return (
        unit_conversion
        * eff_temp_phi
        * flux_associated_variance
        / (flux_associated_variance + estimator_variance)
    )


def shim_flux_biases(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    *,
    sampling_params: dict[str, Any] | None = None,
    shimmed_variables: Collection[Variable] | None = None,
    learning_schedule: Sequence[float] | None = None,
    convergence_test: Callable[[dict, dict], bool] | None = None,
    symmetrize_experiments: bool = True,
    flux_biases_break_symmetry: Literal[True, False] = False,
    sampling_params_updates: Sequence[dict[str, Any]] | None = None,
    beta_hypergradient: float = 0.4,
    num_steps: int = 10,
    alpha: float | None = None,
    exp_weights_per_update: Mapping[Variable, Sequence[float]] | None = None,
    sampling_params_in_place: bool = True,
) -> tuple[list[Bias], dict, dict]:
    r"""Return flux biases that minimize magnetization for symmetry-preserving
    experiments.

    You can refine calibration for specific QPU experiments by modifying your
    QPU programming. The :ref:`flux bias <parameter_qpu_flux_biases>` parameter
    compensates for low-frequency environmental spins that couple into qubits,
    distorting the target distribution. Although you can modify either
    :ref:`flux bias <parameter_qpu_flux_biases>` or :ref:`h <parameter_qpu_h>`
    to restore symmetry of the sampled distribution, flux biases more accurately
    eliminate common forms of low-frequency noise.

    Assuming the magnetization (expectation for measured spins, or sign of the
    persistent current) is a smooth, monotonic function of the qubit body's
    magnetic fluxes (:ref:`flux bias <parameter_qpu_flux_biases>`), you can
    determine parameters that achieve a target magnetization, :math:`m`, by
    iterating :math:`\Phi(t+1) \leftarrow \Phi(t) - L(t) (<s> - m)`, where
    :math:`L(t)` is an iteration-dependent map, :math:`<s>` is the expected
    magnetization, :math:`m` is the target magnetization, and :math:`\Phi` are
    the programmed flux biases.

    By default :math:`L(t)` is uniform with respect to programmed qubits, and
    determined by a
    `hypergradient descent method <https://doi.org/10.48550/arXiv.1703.04782>`_.
    Alternatively you can specify the learning rate as a list, in which
    case the hypergradient method is not used.

    Symmetry can be broken by the choice of initial condition in reverse
    annealing, non-zero :math:`h`, :ref:`parameter_polarizing_schedule`, or
    non-zero flux biases over unshimmed fluxes. You can collect data for two
    experiments, with inverted symmetry breaking between them, anticipating zero
    magnetization in the experimental average rather than per experiment.
    Shimming based on this symmetrized data set is expected to determine a good
    shim for both experiments, assuming a weak dependence of noise on the
    symmetry-breaking field.

    Where strong correlations, or strong symmetry-breaking effects, are present
    in an experiment, the sampled distribution may contain insufficient
    information to independently shim all degrees of freedom. Shims are expected
    to be a smooth function of annealing parameters such as
    :ref:`annealing time <parameter_qpu_annealing_time>`,
    :ref:`anneal schedule <parameter_qpu_anneal_schedule>`, and
    Hamiltonian parameters. You can use shims inferred in smoothly related
    models as approximations (or initial conditions) for searches in target
    models.

    If the provided learning rate or learning schedule is too large, it is
    possible to exceed the bounds of allowed values for the flux-bias offsets.

    Args:
        bqm: A :class:`~dimod.binary.BinaryQuadraticModel`.
        sampler: A :class:`~dwave.system.samplers.DWaveSampler`.
        sampling_params: Parameters of the
            :class:`~dwave.system.samplers.DWaveSampler`. Note that if
            ``sampling_params`` contains
            :ref:`flux biases <parameter_qpu_flux_biases>`, these are treated as
            an initial condition. They are edited in place only when
            ``sampling_params_in_place`` is ``True``. Choose a value for the
            :ref:`parameter_qpu_num_reads` parameter in conjunction with your
            chosen schedule. Note that the :ref:`parameter_qpu_initial_state`
            parameter, if provided, is assumed to be specified according to the
            Ising model convention (:math:`\pm 1`, and :math:`-3` for inactive).
        shimmed_variables: A collection of variables to shim; by default the shimmed variables
            are inferred from ``exp_weights_per_update`` or set to all elements
            in :attr:`~dimod.binary.BinaryQuadraticModel.variables`.
        learning_schedule: A sequence of gradient-descent prefactors. When not
            provided, prefactors are determined by a hypergradient-descent
            method parameterized by the ``alpha``, ``beta_hypergradient``, and
            ``num_steps`` arguments.
        convergence_test: A callable that takes the history of magnetizations and
            flux biases as input, returning ``True`` to exit the search, and
            ``False`` otherwise. By default, all stages specified in the
            ``learning_schedule`` argument are completed.
        symmetrize_experiments: If True, performs a test to determine symmetry
            breaking in the experiment: a non-zero
            :ref:`parameter_qpu_initial_state` for reverse anneal, non-zero
            :math:`h`, or non-zero :ref:`parameter_qpu_flux_biases` when
            ``flux_biases_break_symmetry`` is True. If any of these are present,
            magnetization is
            inferred by averaging over two experiments with symmetry-breaking
            elements inverted. The shim averages the symmetrically related
            experiments to achieve zero magnetization.
        flux_biases_break_symmetry: If True, non-zero flux biases on variables
            outside ``shimmed_variables`` trigger symmetrized experiments.
            Defaults to False, in which case flux biases do not trigger
            symmetrization. If flux_biases outside of the shimmed_variables set
            are being used for calibration refinement, this value should be True.
            If flux_biases are being used to intentionally apply symmetry breaking
            fields the value should be False.
        sampling_params_updates: Where you require averaging across many
            experiments, you can specify a list of updates. Each element in your
            list is a dictionary that updates the ``sampling_params`` argument.
            Experiments are averaged over these sampling-parameter updates to
            determine the magnetization used in shimming. The original value of
            the updated sampling parameter is ignored. The
            :ref:`parameter_qpu_flux_biases` parameter must not be an
            updated parameter. See the
            `examples directory <https://github.com/dwavesystems/dwave-experimental/tree/main/examples>`_
            for use cases.
        beta_hypergradient: Controls the learning rate evolution for the
            hypergradient-descent method, enabling improved performance through
            customizing for the annealing protocol and QPU. Supported values are
            in range :math:`(0,1)`. Ignored if you specify the
            ``learning_schedule`` argument.
        num_steps: Number of steps taken by the hypergradient-descent method if
            you do not specify a ``learning_schedule`` from which to infer.
            Default is 10 if neither is specified.
        alpha: Initial learning rate for the hypergradient-descent method,
            enabling improved performance through customizing for the annealing
            protocol and QPU. Supported values are positive, real floats with a
            typical scale you can determine using the
            :func:`.qubit_freezeout_alpha_phi` function. By default, initialized
            using the :func:`.qubit_freezeout_alpha_phi` function. Ignored if
            you specify the ``learning_schedule`` argument.
        exp_weights_per_update: A weighted sum of magnetizations determines the
            update applied to each variable. Shimmed variables are provided as
            the keys, with a sequence of per-experiment weights as the value.
            When provided, keys must match ``shimmed_variables`` and each
            weight sequence must have length
            ``num_signed_experiments * len(sampling_params_updates)``.
            By default, a sum of experimental outcomes is used (unit weight per
            experiment).
        sampling_params_in_place: If ``True``, the provided ``sampling_params``
            is mutated in place, with its
            :ref:`flux biases <parameter_qpu_flux_biases>` updated to the
            shimmed values. Default is ``True``. When ``False``, the caller's
            ``sampling_params`` is left unmodified. Ignored when
            ``sampling_params`` is ``None``, as there is no caller dict to
            mutate.

    Returns:
        A tuple consisting of 3 parts:
            1.  Flux biases in a list using the :ref:`parameter_qpu_flux_biases`
                format (for use by a
                :class:`~dwave.system.samplers.DWaveSampler` sampler).
            2.  History of flux-bias assignments per shimmed component.
            3.  History of magnetizations per shimmed component.

    Example:
        See the
        `examples <https://github.com/dwavesystems/dwave-experimental/tree/main/examples>`_
        `test <https://github.com/dwavesystems/dwave-experimental/tree/main/tests>`_
        directories for additional use cases.

        Shim degenerate qubits at constant learning rate and solver defaults.
        The learning schedule and number of reads is for demonstration only, and
        has not been optimized.

        >>> import numpy as np
        >>> import dimod
        >>> from dwave.system import DWaveSampler
        >>> from dwave.experimental.shimming import shim_flux_biases, qubit_freezeout_alpha_phi
        ...
        >>> qpu = DWaveSampler()        # doctest: +SKIP
        >>> bqm = dimod.BQM.from_ising({q: 0 for q in qpu.nodelist}, {})    # doctest: +SKIP
        >>> alpha_phi = qubit_freezeout_alpha_phi()  # Unoptimized to the experiment, for demonstration purposes.
        >>> ls = [alpha_phi]*5
        >>> sp = {'num_reads': 2048, 'auto_scale': False}
        >>> fb, fb_history, mag_history = shim_flux_biases(bqm,
        ...     qpu,
        ...     sampling_params=sp,
        ...     learning_schedule=ls)     # doctest: +SKIP
        ...
        >>> print(f"RMS magnetization by iteration: {np.sqrt(np.mean([np.array(v)**2 for v in mag_history.values()], axis=0))}") # doctest: +SKIP

        To explicitly select a solver that supports advanced annealing features, such as fast reverse anneal, see
        :attr:`~dwave.experimental.fast_reverse_anneal.api.SOLVER_FILTER`.
    """

    # Natural candidates for future feature enhancements:
    # - Use standard stochastic gradient descent methods, such as ADAM, perhaps with
    # hypergradient descent to eliminate learning rate choice inefficiencies.
    # - Allow shimming of linear combinations of fluxes, e.g. to control for a
    # known (or desired) correlation structure.
    # Note: the purpose of shimming should not be to learn parameters of general
    # graph-restricted Boltzmann machines, we should modify our plugin for this
    # purpose.

    # Preserve the caller's dict for the optional in-place flux_biases update;
    # all internal mutation happens on a copy so the only caller-visible change
    # is the flux_biases assignment performed at the end.
    caller_sampling_params = sampling_params
    if sampling_params is None:
        # No caller dict to mutate, so the in-place option is ignored.
        sampling_params = {}
        sampling_params_in_place = False
    elif sampling_params_updates is not None:
        # Each iteration applies sampling_params.update(spu), which would
        # overwrite caller keys; deep copy so only flux_biases is written back.
        sampling_params = deepcopy(sampling_params)
    else:
        # Shallow copy to avoid mutating the caller's dict; initial_state is
        # copied because it is negated in place during symmetrization.
        sampling_params = dict(sampling_params)
        if "initial_state" in sampling_params:
            sampling_params["initial_state"] = deepcopy(
                sampling_params["initial_state"]
            )

    if "flux_biases" in sampling_params:
        flux_biases = sampling_params.pop("flux_biases")
        if len(flux_biases) != sampler.properties["num_qubits"]:
            raise ValueError("flux_biases length incompatible with the sampler")
        # Copy so the in-place arithmetic below never mutates a caller-owned list.
        flux_biases = deepcopy(flux_biases)
    else:
        flux_biases = [0] * sampler.properties["num_qubits"]

    if shimmed_variables is None:
        if exp_weights_per_update is not None:
            shimmed_variables = list(exp_weights_per_update.keys())
        else:
            shimmed_variables = bqm.variables
    elif len(shimmed_variables) == 0:
        raise ValueError("shimmed_variables should not be empty")

    # Validate inferred and provided variables alike against the bqm.
    if not set(shimmed_variables).issubset(bqm.variables):
        raise ValueError("Invalid shimmed variables")

    # Shimming targets zero magnetization in the SPIN convention regardless of
    # symmetrization, so convert BINARY BQMs up front.
    if bqm.vartype is dimod.BINARY:
        bqm = bqm.change_vartype(dimod.SPIN, inplace=False)

    unshimmed_variables: set[Variable] = set()
    if symmetrize_experiments:
        unshimmed_variables = set(bqm.variables).difference(shimmed_variables)
        if flux_biases_break_symmetry:
            fbnonzero = any(flux_biases[v] != 0 for v in unshimmed_variables)
        else:
            fbnonzero = False
        hnonzero = any(bqm.linear.values())
        if hnonzero:
            bqm = bqm.copy()
        reverseanneal = "initial_state" in sampling_params
        polarizedmca = "x_polarizing_schedule" in sampling_params and any(
            v != 0 for _, v in sampling_params["x_polarizing_schedule"]
        )
    else:
        fbnonzero = hnonzero = reverseanneal = polarizedmca = False
    num_signed_experiments = 1 + int(
        reverseanneal or hnonzero or fbnonzero or polarizedmca
    )

    if sampling_params_updates is None:
        # By default, a single experimental setting:
        sampling_params_updates = [{}]
    else:
        # Although there are scenarios where some flux_biases are
        # set whilst others are shimmed, support for this is beyond
        # the scope of this function.
        if any("flux_biases" in sp for sp in sampling_params_updates):
            raise ValueError(
                "flux_biases should not be explicitly set "
                "within sampling_params_updates."
            )
    num_experiments = num_signed_experiments * len(sampling_params_updates)
    if exp_weights_per_update is not None:
        if set(exp_weights_per_update.keys()) != set(shimmed_variables):
            raise ValueError(
                "exp_weights_per_update should have the same keys as shimmed_variables"
            )
        if any(
            len(weights) != num_experiments
            for weights in exp_weights_per_update.values()
        ):
            raise ValueError(
                "Each exp_weights_per_update value should have length "
                f"num_experiments ({num_experiments})."
            )

    use_hypergradient = learning_schedule is None
    if not use_hypergradient:
        num_steps = len(learning_schedule)
        alpha = 0.0  # replaced each step from learning_schedule
    else:
        if alpha is None:
            alpha = qubit_freezeout_alpha_phi()
        if not (0 < beta_hypergradient < 1):
            raise ValueError("beta_hypergradient should be in the (0,1) interval")
        if not (alpha > 0):
            raise ValueError("alpha should be positively valued")

    if convergence_test is None:
        convergence_test = lambda x, y: False

    # Shaped like magnetizations; overwritten each hypergradient step before it
    # is read (only used when step > 0).
    last_mags = np.zeros(len(shimmed_variables))
    flux_bias_history = {v: [flux_biases[v]] for v in shimmed_variables}
    mag_history = {v: [] for v in bqm.variables}
    for step in range(num_steps):
        # Possible feature enhancement for intermediate num_experiments:
        # following loops are parallelizable, call sample() asynchronously.
        for spu in sampling_params_updates:
            sampling_params.update(spu)
            for _ in range(num_signed_experiments):
                if reverseanneal:
                    for i in bqm.variables:
                        sampling_params["initial_state"][i] *= -1
                if hnonzero:
                    for i in bqm.variables:
                        bqm.linear[i] *= -1
                if fbnonzero:
                    for i in unshimmed_variables:
                        flux_biases[i] *= -1
                if polarizedmca:
                    sampling_params["x_polarizing_schedule"] = [
                        (t, -v) for t, v in sampling_params["x_polarizing_schedule"]
                    ]
                ss = sampler.sample(bqm, flux_biases=flux_biases, **sampling_params)
                all_mags = np.sum(
                    ss.record.sample * ss.record.num_occurrences[:, np.newaxis], axis=0
                ) / np.sum(ss.record.num_occurrences)

                for idx, v in enumerate(ss.variables):
                    mag_history[v].append(all_mags[idx])

        if convergence_test(mag_history, flux_bias_history):
            # The data is not used to update the flux_biases
            # This can be included as part of the test evaluation (if required)
            break
        if exp_weights_per_update is None:
            exp_av_mags = {
                v: np.sum(mag_history[v][-num_experiments:]) for v in shimmed_variables
            }
        else:
            exp_av_mags = {
                v: np.sum(
                    [
                        mag_history[v][-num_experiments + exp_idx] * weight
                        for exp_idx, weight in enumerate(weights)
                    ]
                )
                for v, weights in exp_weights_per_update.items()
            }
        if use_hypergradient:
            magnetizations = np.array([exp_av_mags[v] for v in shimmed_variables])
            if step > 0:
                norm = np.linalg.norm(magnetizations) * np.linalg.norm(last_mags)
                if math.isclose(norm, 0):
                    # When magnetization norms are zero the paper method is ill defined.
                    # One could choose to convergence test to exit at zero magnetization
                    # as an alternative.
                    alpha *= 1 - beta_hypergradient
                else:
                    alpha *= (
                        1
                        + beta_hypergradient * np.dot(magnetizations, last_mags) / norm
                    )
            last_mags = magnetizations
        else:
            alpha = learning_schedule[step]

        for v in shimmed_variables:
            flux_biases[v] -= alpha * exp_av_mags[v]
            flux_bias_history[v].append(flux_biases[v])

    if sampling_params_in_place and caller_sampling_params is not None:
        caller_sampling_params["flux_biases"] = flux_biases

    return flux_biases, flux_bias_history, mag_history


def _extend_history(target: dict, source: dict) -> None:
    """Append ``source`` list values onto existing ``target`` entries."""
    for k, v in source.items():
        target.setdefault(k, []).extend(v)


def _check_no_duplicate_lines(name: str, lines: Collection[int]) -> None:
    """Raise if a line-set collection contains duplicate entries.

    It's not enough for documentation to make that clear sometimes.
    """
    lines = list(lines)
    if len(lines) != len(set(lines)):
        raise ValueError(f"{name} should not contain duplicate entries")


def shim_linewise_flux_biases(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    exp_feature_line_info: list[LineFeatureInfo],
    *,
    sampling_params: dict[str, Any] | None = None,
    shimmed_variables: Collection[Variable] | None = None,
    lines: Iterable[int] | None = None,
    target_c: float | None = None,
    sampling_params_in_place: bool = False,
    learning_schedule: Sequence[float] | None = None,
    convergence_test: Callable[[dict, dict], bool] | None = None,
    beta_hypergradient: float = 0.4,
    num_steps: int = 10,
    alpha: float | None = None,
) -> tuple[list[Bias], dict, dict]:
    """Shim flux_biases by per-line quench

    Controlling all lines except one (prepared as a detected target) to Cmin
    (minimal persistent current),
    flux biases are limited to single-qubits.
    A flux setting achieving zero magnetization can then be established by the
    standard procedure.

    This method assumes flux biases serve the purpose of calibration refinement
    only and do not break the symmetry of the system. The method operates on a
    strictly :math:`Z_2` symmetric system; this is enforced by safely popping
    any ``x_polarizing_schedule`` from ``sampling_params`` before calling
    shim_flux_biases with `symmetrize_experiments=False`. Some methods generalize
    but the aim is to protect against strange edge cases.

    Args:
        bqm: A :class:`~dimod.binary.BinaryQuadraticModel` describing the
            coupled system.
        sampler: A :class:`~dwave.system.samplers.DWaveSampler`.
        exp_feature_line_info: Per-line experiment feature information. Used to
            map variables to annealing lines and to parameterize the per-line
            ``x_anneal_schedules``.
        sampling_params: Base sampling parameters passed to the sampler. The
            ``x_anneal_schedules`` and ``x_polarizing_schedule`` entries are
            replaced per line, and ``flux_biases`` are carried forward across
            lines. If not specified, then defaults are used.
        shimmed_variables: Variables to shim. Each variable is updated only
            during the line to which it is assigned.
        lines: Annealing lines to shim, one at a time. Defaults to all lines in
            ``exp_feature_line_info``.
        target_c: Value at which line is prepared before quenching.
        sampling_params_in_place: If ``True``, the provided ``sampling_params``
            is mutated in place, with its
            :ref:`flux biases <parameter_qpu_flux_biases>` updated to the
            shimmed values. Default is ``False``. Ignored when
            ``sampling_params`` is ``None``.
        learning_schedule: Passed through to :func:`.shim_flux_biases`.
        convergence_test: Passed through to :func:`.shim_flux_biases`.
        beta_hypergradient: Passed through to :func:`.shim_flux_biases`.
        num_steps: Passed through to :func:`.shim_flux_biases`.
        alpha: Passed through to :func:`.shim_flux_biases`.

    Returns:
        A tuple of three parts mirroring the return value of
        :func:`.shim_flux_biases`:

        1.  Flux biases in the :ref:`parameter_qpu_flux_biases` format
            for use with QPU sampling.
        2.  History of flux-bias assignments per shimmed component.
        3.  History of magnetizations per shimmed component.
    """
    if lines is None:
        lines = range(len(exp_feature_line_info))
    else:
        lines = list(lines)
        _check_no_duplicate_lines("lines", lines)
    if sampling_params is None:
        # No caller dict to mutate, so the in-place option is ignored.
        sampling_params = {'num_reads': 1000}
        sampling_params_in_place = False
    _sampling_params = deepcopy(sampling_params)
    _sampling_params.pop("x_anneal_schedules", None)
    _sampling_params.pop("x_polarizing_schedule", None)
    popped_fb = _sampling_params.pop("flux_biases", None)
    flux_biases: list[Bias] = (
        [0] * sampler.properties["num_qubits"] if popped_fb is None else popped_fb
    )
    if shimmed_variables is None:
        shimmed_variables = list(bqm.variables)
    elif not set(shimmed_variables).issubset(bqm.variables):
        raise ValueError("shimmed_variables should be a subset of bqm.variables")
    line_assignments = line_assignments_from_feature_info(exp_feature_line_info)
    dict_fb_all = {}
    dict_mag_all = {}
    viable_lines = {
        line_assignments[q] for q in bqm.variables if q in line_assignments
    }
    for line in sorted(viable_lines.intersection(lines)):
        line_shimmed_variables = [
            v for v in shimmed_variables if line_assignments.get(v) == line
        ]
        if not line_shimmed_variables:
            continue
        x_anneal_schedules = make_tds_x_anneal_schedules(
            exp_feature_line_info=exp_feature_line_info,
            detector_lines=set(),
            target_lines=set(),
            source_lines=set(),
            detected_target_lines={line},
            target_c=target_c,
        )
        _sampling_params.update({"x_anneal_schedules": x_anneal_schedules})
        if flux_biases is not None:
            _sampling_params["flux_biases"] = flux_biases

        flux_biases, dict_fb, dict_mag = shim_flux_biases(
            bqm,
            sampler,
            sampling_params=_sampling_params,
            symmetrize_experiments=False,
            shimmed_variables=line_shimmed_variables,
            learning_schedule=learning_schedule,
            convergence_test=convergence_test,
            beta_hypergradient=beta_hypergradient,
            num_steps=num_steps,
            alpha=alpha,
        )
        _extend_history(dict_fb_all, dict_fb)
        _extend_history(dict_mag_all, {k: dict_mag[k] for k in dict_fb})

    if sampling_params_in_place and flux_biases is not None:
        sampling_params["flux_biases"] = flux_biases
    return flux_biases, dict_fb_all, dict_mag_all


def _neutralize_line_schedule(
    schedule: AnnealSchedule, min_c: float, no_polarizing_schedule: bool = False,
) -> AnnealSchedule:
    """Flatten a single line's schedule to a constant ``min_c`` hold.

    Leading zero-valued support points are kept; from the first point whose
    value deviates from 0 onward, every value is replaced by ``min_c``. Only
    existing support points are revalued and their times are untouched, so no
    new interval of variability is introduced and every step size is preserved
    — the schedule remains valid.

    We need only be careful with intervals of change if polarizing schedules are
    in use.
    """
    neutralized = []
    neutralize = False
    for t, k in schedule:
        if k != 0 or no_polarizing_schedule:
            neutralize = True
        neutralized.append([t, min_c if neutralize else k])
    return neutralized


def _neutralize_lines(
    exp_feature_line_info: list[LineFeatureInfo],
    unused_lines: Iterable[int],
    x_anneal_schedules: XAnnealSchedules,
    no_polarizing_schedule: bool = False
) -> XAnnealSchedules:
    """Set non-target/detector lines to a neutral (rounded minC) hold.

    ``minC`` is rounded toward zero to the client-safe precision via
    :func:`~dwave.experimental.multicolor_anneal.utils._round_sigfigs`.
    """
    for line in unused_lines:
        min_c = _round_sigfigs(exp_feature_line_info[line]["minC"], mode="up")
        x_anneal_schedules[line] = _neutralize_line_schedule(
            x_anneal_schedules[line], min_c, no_polarizing_schedule=no_polarizing_schedule
        )
    return x_anneal_schedules


def shim_tds_flux_biases(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    target_lines: Collection[int],
    detector_lines: Collection[int],
    exp_feature_line_info: list[LineFeatureInfo],
    *,
    source_lines: Iterable[int] | None = None,
    by_line_pair: bool = True,
    sampling_params: dict[str, Any] | None = None,
    shimmed_variables: Collection[Variable] | None = None,
    target_c: float | None = None,
    num_reads: int = 500,
    td_shim_type: Literal[
        "by_line_quench", "detector_only", "sequential", "alternating"
    ] = "alternating",
    sampling_params_in_place: bool = True,
    learning_schedule: Sequence[float] | None = None,
    convergence_test: Callable[[dict, dict], bool] | None = None,
    beta_hypergradient: float = 0.4,
    num_steps: int = 10,
    alpha: float | None = None,
) -> tuple[list[Bias], dict, dict]:
    """Shim flux biases on target, detector and source annealing lines.

    Several tds shims are supported on the basis of unsourced (equilibrium)
    measurements.
    Fitting shims to a time series model is an alternative not supported.

    The tds methods assume flux biases serve the purpose of calibration
    refinement only and do not break the symmetry of the system. The methods
    operate on a strictly :math:`Z_2` symmetric system; this is enforced by
    safely popping any ``x_polarizing_schedule`` from ``sampling_params`` before
    calling shim_flux_biases with `symmetrize_experiments=False`. Some methods generalize
    but the aim is to protect against strange edge cases.

    Source line qubits are decoupled from the target by holding them at small
    ``c`` (the rounded ``minC``). This models the expected state at equilibrium,
    long after the source has depolarized.

    Detector only method:
    The source qubits, and qubits on unused lines, can be decoupled (set to Cmin).
    Then assuming a target consists of degenerate qubits, the detected magnetization
    is expected to be a monotonic function of a static flux bias through the body.
    A unique value of the flux biases on detectors achieves zero magnetization,
    this can be sought by stochastic gradient descent.
    If there are no target_lines or shimmed variables contains no target qubits,
    then sequential and alternating methods fall back to the detector only method.

    Sequential method:
    Run first a line-wise quench on sources, targets and detectors. Then the
    detector only method. Note the parameters passed through to shim_flux_biases
    are the same in both stages.

    Alternating method:
    Coupled qubits can act as complementary detectors. If we assume the required
    calibration refinement does not depend on the target/detector waveforms
    for example that bias is determined by static flux offsets at a mid
    point of the dynamics, then we can iterate a pair of experiments (role
    of detector) to determine a calibration refinement both of the target
    and detector system.
    We assume flux biases required are weakly dependent on the choice
    of ``x_anneal_schedules``. We can measure the coupled system(s)
    of qubits alternating the role of detector. Assuming
    :math:`d< m_D/dPhi_D > - sign(J_{SD}) d<m_D>/dPhi_S > = 0`,
    we can pursue paired experiments and update only detector qubit fluxes
    in each with a convergence guarantee. Convergence requires the
    associated Jacobian matrix to be convergent - this is guaranteed
    only if the lines are well synchronized (same quench rate, and
    same frequency and Ip on targets, independent of orientation).
    Otherwise divergence can be anticipated in some cases.
    Two experiments per iteration are run with detector and
    target roles alternated (via swapped ``x_anneal_schedules``;
    ``x_schedule_delays`` are left untouched), and the update for each variable
    uses only the experiment in which it plays the detector role. Delays need
    not be swapped: the target is assumed quasi-static, so for a well-defined
    (canonical) quench the delays are irrelevant to the measured statistics.
    When ``shimmed_variables`` contains only detector variables, no alternation
    is performed and shimming reduces to a single-experiment call to
    :func:`.shim_flux_biases`.
    The role swap uses a single representative schedule per role, so all
    ``target_lines`` must share one ``x_anneal_schedules`` entry and all
    ``detector_lines`` must share one (a :class:`ValueError` is raised
    otherwise). The target and detector schedules must additionally be mutually
    compatible, meaning interchangeable under the role swap, as produced by
    :func:`~dwave.experimental.multicolor_anneal.make_tds_x_anneal_schedules`
    with ``use_common_bounds=True``.

    To shim the source requires coupling into the target. Assuming degeneracy on
    targets and detectors is achieved. We can operate in a regime with the source
    coupled into the target. The magnetization measured on the detector can then
    be assumed to be proportional to a flux bias error on the source, which is
    correctable. This involves measuring the detector, but applying an offset to
    the source.

    Args:
        bqm: A :class:`~dimod.binary.BinaryQuadraticModel` describing the
            coupled target-detector(-source) system. A source line is optional.
        sampler: A :class:`~dwave.system.samplers.DWaveSampler`.
        target_lines: Indices of annealing lines whose qubits act as targets
            (the system whose state we wish to measure). By default bqm
            variables on these lines are shimmed, shimmed_variables can be
            used to narrow the set, and if no target lines are included
            a simpler shim on detector qubits only is performed.
        detector_lines: Indices of annealing lines whose qubits act as
            detectors. By default all bqm qubits on the detector lines are
            shimmed, but the set can be reduced using the ``shimmed_variables``
            parameter.
        exp_feature_line_info: Per-line experiment feature information. Used to
            map each variable (qubit index) to its annealing line index, and,
            when ``x_anneal_schedules`` is not supplied in ``sampling_params``,
            to parameterize it. Also used to determine settings for neutralizing lines in
            some methods.
        source_lines: Indices of annealing lines whose qubits act as sources.
            Used by the ``"sequential"`` and ``"by_line_quench"`` strategies to
            include source lines in the line-wise quench. Optional.
        by_line_pair: If ``True``, the shimming is performed independently
            for each combination of target and detector line.
        sampling_params: Base sampling parameters passed to the sampler.
            Entries the caller omits are filled with defaults: missing
            ``x_anneal_schedules`` is built from ``exp_feature_line_info`` (and
            ``target_c``), missing ``num_reads`` defaults to the ``num_reads``
            argument, and missing ``x_disable_filtering`` defaults to ``True``.
            Any ``x_polarizing_schedule`` is removed to keep the system
            :math:`Z_2` symmetric. For ``td_shim_type="alternating"``, the
            ``x_anneal_schedules`` must be identical across all ``target_lines``
            and identical across all ``detector_lines``, and the target and
            detector schedules must be mutually compatible (interchangeable
            under the role swap), as produced by
            :func:`~dwave.experimental.multicolor_anneal.make_tds_x_anneal_schedules`
            with ``use_common_bounds=True``.
        sampling_params_in_place = False
        shimmed_variables: Variables to shim; validated before any sampling
            against the lines the chosen ``td_shim_type`` shims:
            ``"detector_only"`` requires a subset of ``detector_lines``
            variables; ``"alternating"`` a subset of ``target_lines`` or
            ``detector_lines`` variables; ``"sequential"`` and
            ``"by_line_quench"`` additionally accept ``source_lines`` variables
            (shimmed in the line-wise stage). By default, variables assigned to
            ``detector_lines`` (and, for ``"alternating"``, also
            ``target_lines``); the line-wise stage instead defaults to every bqm
            variable, regardless of whether it is on a source, target or detector
            line. The final shim stage updates only ``detector_lines`` variables,
            so for ``"sequential"`` the full set is used in the line-wise first
            stage and then narrowed to the ``detector_lines`` subset for the
            second stage.
        target_c: If ``x_anneal_schedules`` is not supplied in
           ``sampling_params``, this is used to parameterize it for
           ``target_lines``.
        num_reads: If ``num_reads`` is not supplied in ``sampling_params``, this
            is used to set the default number of reads per iterative stage.
            Larger values result in lower variance (better) convergence.
        td_shim_type: The shimming strategy for target-detector pairs at
            equilibrium (with source depolarized). One of ``"detector_only"``
            (shim detector qubits only), ``"by_line_quench"`` (shim every line
            independently by single-line quench), ``"sequential"`` (a line-wise
            quench followed by the detector-only shim), or ``"alternating"``
            (alternate the detector and target roles to shim both).
        sampling_params_in_place: If ``True``, the provided ``sampling_params``
            is mutated in place, with its
            :ref:`flux biases <parameter_qpu_flux_biases>` updated to the final
            shimmed values. Default is ``True``. Ignored when
            ``sampling_params`` is ``None``.
        learning_schedule: Passed through to :func:`.shim_flux_biases`.
        convergence_test: Passed through to :func:`.shim_flux_biases`.
        beta_hypergradient: Passed through to :func:`.shim_flux_biases`.
        num_steps: Passed through to :func:`.shim_flux_biases`.
        alpha: Passed through to :func:`.shim_flux_biases`.
    Returns:
        A tuple of three parts mirroring the return value of
        :func:`.shim_flux_biases`:

        1.  Flux biases in the :ref:`parameter_qpu_flux_biases` format
            for use with QPU sampling.
        2.  History of flux-bias assignments per shimmed component
            across iterations.
        3.  History of magnetizations per variable across experiments
            and iterations.

        Multi-stage shimming processes (such as ``"sequential"``) append the
        per-stage histories, so entries may not align across variables by
        iteration when the shimmed variables differ between stages.
    """
    if not detector_lines:
        raise ValueError(
            "There must be detector lines to shim, but detector_lines is empty"
        )

    _check_no_duplicate_lines("detector_lines", detector_lines)
    _check_no_duplicate_lines("target_lines", target_lines)
    if source_lines is not None:
        source_lines = list(source_lines)
        _check_no_duplicate_lines("source_lines", source_lines)

    line_assignments = line_assignments_from_feature_info(exp_feature_line_info)
    num_lines = len(exp_feature_line_info)

    # Preserve the caller's dict for the optional in-place flux_biases update.
    caller_sampling_params = sampling_params
    if sampling_params is None:
        sampling_params = {}
        sampling_params_in_place = False
    else:
        sampling_params = deepcopy(sampling_params)
    sampling_params.pop("x_polarizing_schedule", None)
    # Fill any sampling parameters the caller omitted with defaults built from
    # the line info; x_polarizing_schedule is dropped above for Z2 symmetry.
    if "x_anneal_schedules" not in sampling_params:
        sampling_params["x_anneal_schedules"] = make_tds_x_anneal_schedules(
                    exp_feature_line_info=exp_feature_line_info,
                    target_lines=target_lines,
                    target_c=target_c,
                    detector_lines=detector_lines,
                    use_common_bounds=(td_shim_type == "alternating"),
                    symmetrize_c_bounds=(td_shim_type == "alternating"),
                )  # For consistency under line swapping.
    if "num_reads" not in sampling_params:
        sampling_params["num_reads"] = num_reads
    if "x_disable_filtering" not in sampling_params:
        sampling_params["x_disable_filtering"] = True
    detector_set = set(detector_lines)
    target_set = set(target_lines)
    dt_lines = detector_set | target_set
    shim_lines = set(detector_lines)
    if not (
        dt_lines.issubset(range(num_lines))
        and len(dt_lines) == len(detector_set) + len(target_set)
    ):
        raise ValueError(
            "detector_lines and target_lines should be disjoint subsets of set(range(num_lines))"
        )

    # Validate before any sampling so an invalid set is rejected without wasted
    # QPU calls, against the lines the chosen strategy actually shims.
    if td_shim_type in ("sequential", "by_line_quench"):
        # The line-wise stage also shims source lines.
        shimmable_lines = dt_lines | (set(source_lines) if source_lines else set())
    elif td_shim_type == "detector_only":
        shimmable_lines = detector_set
    else:  # "alternating"
        shimmable_lines = dt_lines
    if shimmed_variables is not None and not set(shimmed_variables).issubset(
        v for v in bqm.variables if line_assignments.get(v) in shimmable_lines
    ):
        raise ValueError(
            "shimmed_variables should be a subset of the bqm variables assigned "
            f"to the lines shimmed by td_shim_type='{td_shim_type}'"
        )

    if td_shim_type != "detector_only":
        if td_shim_type == "alternating":
            flux_biases = sampling_params.pop("flux_biases", None)
            dict_fb_all = {}
            dict_mag_all = {}
            shim_lines = dt_lines
        elif td_shim_type == "sequential" or td_shim_type == "by_line_quench":
            tds_lines = dt_lines.copy()
            if source_lines is not None:
                tds_lines |= set(source_lines)
            flux_biases, dict_fb_all, dict_mag_all = shim_linewise_flux_biases(
                exp_feature_line_info=exp_feature_line_info,
                bqm=bqm,
                sampler=sampler,
                sampling_params=sampling_params,
                shimmed_variables=shimmed_variables,
                lines=tds_lines,
                target_c=target_c,
                learning_schedule=learning_schedule,
                convergence_test=convergence_test,
                beta_hypergradient=beta_hypergradient,
                num_steps=num_steps,
                alpha=alpha,
            )
            if td_shim_type == "by_line_quench":
                if (
                    sampling_params_in_place
                    and caller_sampling_params is not None
                    and flux_biases is not None
                ):
                    caller_sampling_params["flux_biases"] = flux_biases
                return flux_biases, dict_fb_all, dict_mag_all
        else:
            raise ValueError(
                'Invalid td_shim_type. Expected "alternating", "sequential" or "by_line_quench".'
            )
    else:
        flux_biases = sampling_params.pop("flux_biases", None)
        dict_fb_all = {}
        dict_mag_all = {}

    viable_shimmed_variables = set(
        v for v in bqm.variables if line_assignments.get(v) in shim_lines
    )  # Only detectors and targets are shimmed henceforth
    unused_lines = set(range(num_lines)).difference(dt_lines)
    # x_polarizing_schedule is popped above, so there is no polarizing schedule.
    sampling_params["x_anneal_schedules"] = _neutralize_lines(
        exp_feature_line_info,
        unused_lines,
        sampling_params["x_anneal_schedules"],
        no_polarizing_schedule=True,
    )

    # The final shim stage updates only shim_lines variables (detector lines,
    # plus target lines for the alternating method). shimmed_variables were
    # validated against the full target-or-detector domain above and forwarded
    # in full to the line-wise first stage of the sequential method; here they
    # are narrowed to the shim_lines subset for the remaining stage.
    if shimmed_variables is None:
        shimmed_variables = viable_shimmed_variables
    else:
        shimmed_variables = set(
            v for v in shimmed_variables if line_assignments.get(v) in shim_lines
        )
    use_target_variables = any(
        line_assignments.get(v) in target_lines for v in shimmed_variables
    )

    if td_shim_type == "alternating" and use_target_variables:
        # Any qubit outside of the target and detector lines is
        # explicitly decoupled. This is desirable for the alternating method.
        # otherwise it makes sense to leave the full Hamiltonian programmed
        # to properly account for cross talks.
        bqm = bqm.copy(deep=True)
        bqm.remove_variables_from(
            set(bqm.variables).difference(viable_shimmed_variables)
        )
        if by_line_pair:
            # degenerate detector lines interfere post-quench, therefore process separately.
            for target_line in target_lines:
                for detector_line in detector_lines:
                    if flux_biases is not None:
                        sampling_params["flux_biases"] = flux_biases
                    shimmed_variables_dt = set(
                        v
                        for v in shimmed_variables
                        if line_assignments[v] in {target_line, detector_line}
                    )
                    if not shimmed_variables_dt:
                        continue
                    flux_biases, dict_fb, dict_mag = shim_tds_flux_biases(
                        bqm=bqm,
                        sampler=sampler,
                        target_lines={target_line},
                        detector_lines={detector_line},
                        by_line_pair=False,
                        sampling_params=sampling_params,
                        shimmed_variables=shimmed_variables_dt,
                        exp_feature_line_info=exp_feature_line_info,
                        learning_schedule=learning_schedule,
                        convergence_test=convergence_test,
                        beta_hypergradient=beta_hypergradient,
                        num_steps=num_steps,
                        alpha=alpha,
                    )
                    _extend_history(dict_fb_all, dict_fb)
                    _extend_history(dict_mag_all, {k: dict_mag[k] for k in dict_fb})
            if (
                sampling_params_in_place
                and caller_sampling_params is not None
                and flux_biases is not None
            ):
                caller_sampling_params["flux_biases"] = flux_biases
            return flux_biases, dict_fb_all, dict_mag_all
        else:
            # Alternate between detector and target quench. The swap below
            # applies one representative line's schedule to every line of the
            # other role, which is only valid when all lines of a role share an
            # identical schedule. x_schedule_delays are left untouched.
            for role_name, role_lines in (
                ("target_lines", target_lines),
                ("detector_lines", detector_lines),
            ):
                role_schedules = [
                    sampling_params["x_anneal_schedules"][line] for line in role_lines
                ]
                if any(s != role_schedules[0] for s in role_schedules):
                    raise ValueError(
                        "td_shim_type='alternating' requires identical "
                        f"x_anneal_schedules across all {role_name}."
                    )
            x_anneal_schedules_reversed = deepcopy(
                sampling_params["x_anneal_schedules"]
            )
            dl = next(iter(detector_lines))
            for line in target_lines:
                # Deep copy so collapsed lines do not share one mutable object.
                x_anneal_schedules_reversed[line] = deepcopy(
                    sampling_params["x_anneal_schedules"][dl]
                )
            tl = next(iter(target_lines))
            for line in detector_lines:
                x_anneal_schedules_reversed[line] = deepcopy(
                    sampling_params["x_anneal_schedules"][tl]
                )

            exp_weights_per_update = {
                v: [1.0, 0.0] if line_assignments.get(v) in detector_lines else [0.0, 1.0]
                for v in shimmed_variables
            }

            sampling_params_updates = [
                {"x_anneal_schedules": sampling_params["x_anneal_schedules"]},
                {"x_anneal_schedules": x_anneal_schedules_reversed},
            ]
    else:
        # Simple shim of detectors
        sampling_params_updates = None
        exp_weights_per_update = None

    if flux_biases is not None:
        sampling_params["flux_biases"] = flux_biases

    flux_biases, dict_fb, dict_mag = shim_flux_biases(
        bqm,
        sampler,
        sampling_params=sampling_params,
        symmetrize_experiments=False,
        sampling_params_updates=sampling_params_updates,
        exp_weights_per_update=exp_weights_per_update,
        shimmed_variables=shimmed_variables,
        learning_schedule=learning_schedule,
        convergence_test=convergence_test,
        beta_hypergradient=beta_hypergradient,
        num_steps=num_steps,
        alpha=alpha,
    )
    _extend_history(dict_fb_all, dict_fb)
    _extend_history(dict_mag_all, dict_mag)
    if (
        sampling_params_in_place
        and caller_sampling_params is not None
        and flux_biases is not None
    ):
        caller_sampling_params["flux_biases"] = flux_biases
    return flux_biases, dict_fb_all, dict_mag_all
