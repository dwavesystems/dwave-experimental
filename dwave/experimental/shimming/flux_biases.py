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
    LineFeatureInfo,
    XAnnealSchedules,
    make_tds_x_anneal_schedules,
)

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
            an initial condition and edited in place. Choose a value for the
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
            By default, a mean of experimental outcomes is used.

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

    if sampling_params is None:
        sampling_params = {}

    if "flux_biases" in sampling_params:
        flux_biases = sampling_params.pop("flux_biases")
        if len(flux_biases) != sampler.properties["num_qubits"]:
            raise ValueError("flux_biases length incompatible with the sampler")
        pop_fb = True
    else:
        flux_biases = [0] * sampler.properties["num_qubits"]
        pop_fb = False

    if shimmed_variables is None:
        if exp_weights_per_update is not None:
            shimmed_variables = list(exp_weights_per_update.keys())
        else:
            shimmed_variables = bqm.variables
    else:
        if len(shimmed_variables) == 0:
            raise ValueError("shimmed_variables should not be empty")
        elif not set(shimmed_variables).issubset(bqm.variables):
            raise ValueError("Invalid shimmed variables")

    if symmetrize_experiments:
        unshimmed_variables = set(bqm.variables).difference(shimmed_variables)
        if flux_biases_break_symmetry:
            fbnonzero = any(flux_biases[v] != 0 for v in unshimmed_variables)
        else:
            fbnonzero = False
        if bqm.vartype is dimod.BINARY:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
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
    else:
        if alpha is None:
            alpha = qubit_freezeout_alpha_phi()
        if not (0 < beta_hypergradient < 1):
            raise ValueError("beta_hypergradient should be in the (0,1) interval")
        if not (alpha > 0):
            raise ValueError("alpha should be positively valued")

    if convergence_test is None:
        convergence_test = lambda x, y: False

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
            # Switching this to np.mean has been considered, but it makes only a
            # superficial difference when the other hyperparameters (e.g. alpha,
            # learning_schedule) are properly tuned or defaulted.
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

    if pop_fb:
        sampling_params["flux_biases"] = flux_biases

    return flux_biases, flux_bias_history, mag_history


def _extend_history(target: dict, source: dict) -> None:
    """Append ``source`` list values onto existing ``target`` entries."""
    for k, v in source.items():
        target.setdefault(k, []).extend(v)


def shim_linewise_flux_biases(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler,
    exp_feature_line_info: list[LineFeatureInfo],
    *,
    sampling_params: dict[str, Any] | None = None,
    shimmed_variables: Collection[Variable] | None = None,
    lines: Iterable[int] | None = None,
    target_c: float | None = None,
    **kwargs,
) -> tuple[list[Bias], dict, dict]:
    """Shim flux_biases by per-line quench

    Controlling all lines except one (prepared as a detected target) to Cmin
    (minimal persistent current),
    flux biases are limited to single-qubits.
    A flux setting achieving zero magnetization can then be established by the
    standard procedure.

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
        **kwargs: Additional keyword arguments passed through to
            :func:`.shim_flux_biases`, such as ``learning_schedule``,
            ``convergence_test``, ``beta_hypergradient``, ``num_steps``, and
            ``alpha``.

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
    if sampling_params is None:
        sampling_params = {'num_reads': 1000}
    line_assignments = {
        q: i
        for i, line_info in enumerate(exp_feature_line_info)
        for q in line_info["qubits"]
    }
    _sampling_params = deepcopy(sampling_params)
    _sampling_params.pop("x_anneal_schedules", None)
    _sampling_params.pop("x_polarizing_schedule", None)
    _sampling_params["x_schedule_delays"] = [0.0] * len(exp_feature_line_info)
    flux_biases = _sampling_params.pop("flux_biases", None)
    if shimmed_variables is None:
        shimmed_variables = list(bqm.variables)
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
            **kwargs,
        )
        _extend_history(dict_fb_all, dict_fb)
        _extend_history(dict_mag_all, {k: dict_mag[k] for k in dict_fb})

    return flux_biases, dict_fb_all, dict_mag_all


def _neutralize_lines(
    exp_feature_line_info: list[LineFeatureInfo],
    unused_lines: Iterable[int],
    x_anneal_schedules: XAnnealSchedules,
) -> XAnnealSchedules:
    """Set unused lines in the x_anneal_schedules to neutral (Cmin) values."""
    for line in unused_lines:
        x_anneal_schedules[line] = [
            [t, exp_feature_line_info[line]["minC"]]
            for t, _ in x_anneal_schedules[line]
        ]
        x_anneal_schedules[line][0][0] = 0.0  # Requirement
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
    symmetrize_experiments: bool = False,
    shimmed_variables: Collection[Variable] | None = None,
    decouple_tar_and_det: bool | None = None,
    target_c: float | None = None,
    num_reads: int = 500,
    td_shim_type: Literal[
        "by_line_quench", "detector_only", "sequential", "alternating"
    ] = "alternating",
    **kwargs,
) -> tuple[list[Bias], dict, dict]:
    """Shim flux biases on target, detector and source annealing lines.

    Several tds shims are supported on the basis of unsourced (equilibrium)
    measurements.
    Fitting shims to a time series model is an alternative not supported.

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
    detector only method.

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
    target roles alternated (via swapped ``x_anneal_schedules`` and
    ``x_schedule_delays``), and the update for each variable uses only the
    experiment in which it plays the detector role. When ``shimmed_variables``
    contains only detector variables, no alternation is performed and shimming
    reduces to a single-experiment call to :func:`.shim_flux_biases`.

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
            when ``sampling_params`` is not provided, to parameterize
            ``x_anneal_schedules``.
        source_lines: Indices of annealing lines whose qubits act as sources.
            Used by the ``"sequential"`` and ``"by_line_quench"`` strategies to
            include source lines in the line-wise quench. Optional.
        by_line_pair: If ``True``, the shimming is performed independently
            for each combination of target and detector line.
        sampling_params: Base sampling parameters passed to the sampler.
            If not specified, then defaults are used. A minimal set of
            functional parameters should include ``x_anneal_schedules`` and
            ``num_reads``.
        symmetrize_experiments: If ``True``, symmetry-breaking elements in the
            experiment are inverted for a second run and magnetizations are
            averaged. Passed through to :func:`.shim_flux_biases` for the
            optimization. Default is ``False``. Must be ``False`` when
            ``td_shim_type="alternating"``, which instead resolves symmetry by
            alternating the detector and target roles.
        shimmed_variables: Variables to shim. By default, variables assigned to
            ``detector_lines`` (and, for the ``"alternating"`` strategy, also
            ``target_lines``). If provided, they must be a subset of variables
            assigned to ``target_lines`` or ``detector_lines``.
        decouple_tar_and_det: If ``True``, a copy of ``bqm`` restricted to
            variables assigned to ``target_lines`` or ``detector_lines`` is
            used for shimming. This decouples the target-detector system
            from any residual couplings to qubits on other annealing lines.
            The caller's ``bqm`` is not modified. Default is ``None``, which
            decouples only for the ``"alternating"`` strategy.
        target_c: If ``sampling_params`` is not provided, this is used
           to parameterize x_anneal_schedules for ``target_lines``.
        num_reads: If ``sampling_params`` is not provided, this is used to
            set the default number of reads per iterative stage. Larger
            values result in lower variance (better) convergence.
        td_shim_type: The shimming strategy for target-detector pairs at
            equilibrium (with source depolarized). One of ``"detector_only"``
            (shim detector qubits only), ``"by_line_quench"`` (shim every line
            independently by single-line quench), ``"sequential"`` (a line-wise
            quench followed by the detector-only shim), or ``"alternating"``
            (alternate the detector and target roles to shim both). The
            ``"alternating"`` strategy requires ``symmetrize_experiments=False``.
        **kwargs: Additional keyword arguments passed through to
            :func:`.shim_flux_biases`, such as ``learning_schedule``,
            ``convergence_test``, ``beta_hypergradient``, ``num_steps``, and
            ``alpha``.
    Returns:
        A tuple of three parts mirroring the return value of
        :func:`.shim_flux_biases`:

        1.  Flux biases in the :ref:`parameter_qpu_flux_biases` format
            for use with QPU sampling.
        2.  History of flux-bias assignments per shimmed component
            across iterations.
        3.  History of magnetizations per variable across experiments
            and iterations.
    """
    if not detector_lines:
        raise ValueError(
            "There must be detector lines to shim, but detector_lines is empty"
        )

    if td_shim_type == "alternating" and symmetrize_experiments:
        # Alternation resolves symmetry via the detector-role weighting, which
        # assumes one experiment per role; per-experiment symmetrization would
        # double the experiment count and is incompatible with those weights.
        raise ValueError(
            "td_shim_type='alternating' requires symmetrize_experiments=False."
        )

    line_assignments = {
        q: i
        for i, line_info in enumerate(exp_feature_line_info)
        for q in line_info["qubits"]
    }

    if sampling_params is None:
        # A symmetric default schedule is effective. Dependence on the detailed
        # form (use of overshoot, etc.) is expected to be a perturbative
        # effect on the required flux biases.
        if td_shim_type == "alternating":
            # Requires interchangeable schedules for self-consistency.
            use_common_bounds = True
            symmetrize_c_bounds = True
        else:
            # Defaults are fine
            use_common_bounds = False
            symmetrize_c_bounds = False
        sampling_params = dict(
            x_anneal_schedules=make_tds_x_anneal_schedules(
                exp_feature_line_info=exp_feature_line_info,
                target_lines=target_lines,
                target_c=target_c,
                detector_lines=detector_lines,
                use_common_bounds=use_common_bounds,
                symmetrize_c_bounds=symmetrize_c_bounds,
            ),  # For consistency under line swapping.
            num_reads=num_reads,
            x_disable_filtering=True,
        )
    else:
        sampling_params = deepcopy(sampling_params)
    num_lines = len(exp_feature_line_info)
    if "x_schedule_delays" not in sampling_params:
        sampling_params["x_schedule_delays"] = [0.0] * num_lines
    sampling_params.pop("x_polarizing_schedule", None)

    dt_lines = set(detector_lines) | set(target_lines)
    shim_lines = set(detector_lines)
    if not (
        dt_lines.issubset(range(num_lines))
        and len(dt_lines) == len(detector_lines) + len(target_lines)
    ):
        raise ValueError(
            "detector_lines and target_lines should be disjoint subsets of set(range(num_lines))"
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
                lines=tds_lines,
                target_c=target_c,
                **kwargs,
            )
            if td_shim_type == "by_line_quench":
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
    sampling_params["x_anneal_schedules"] = _neutralize_lines(
        exp_feature_line_info, unused_lines, sampling_params["x_anneal_schedules"]
    )

    if shimmed_variables is None:
        shimmed_variables = viable_shimmed_variables
    else:
        if not set(shimmed_variables).issubset(viable_shimmed_variables):
            raise ValueError(
                "shimmed_variables should be a subset of variables assigned to target_lines or detector_lines"
            )
    use_target_variables = any(
        line_assignments.get(v) in target_lines for v in shimmed_variables
    )

    if (
        decouple_tar_and_det
        or (decouple_tar_and_det is None and td_shim_type == "alternating")
    ) and not (
        by_line_pair and use_target_variables and td_shim_type == "alternating"
    ):
        # Any qubit outside of the target and detector lines is
        # explicitly decoupled. This is desirable for the alternating method.
        # otherwise it makes sense to leave the full Hamiltonian programmed
        # to properly account for cross talks.
        bqm = bqm.copy(deep=True)
        bqm.remove_variables_from(
            set(bqm.variables).difference(viable_shimmed_variables)
        )
    if td_shim_type == "alternating" and use_target_variables:
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
                        symmetrize_experiments=symmetrize_experiments,
                        shimmed_variables=shimmed_variables_dt,
                        decouple_tar_and_det=True,  # decouples this pair
                        exp_feature_line_info=exp_feature_line_info,
                        **kwargs,
                    )
                    _extend_history(dict_fb_all, dict_fb)
                    _extend_history(dict_mag_all, {k: dict_mag[k] for k in dict_fb})
            return flux_biases, dict_fb_all, dict_mag_all
        else:
            # Alternate between detector and target quench.
            x_schedule_delays_reversed = deepcopy(sampling_params["x_schedule_delays"])
            x_anneal_schedules_reversed = deepcopy(
                sampling_params["x_anneal_schedules"]
            )
            dl = next(iter(detector_lines))
            for line in target_lines:
                x_schedule_delays_reversed[line] = sampling_params["x_schedule_delays"][
                    dl
                ]
                x_anneal_schedules_reversed[line] = sampling_params[
                    "x_anneal_schedules"
                ][dl]
            tl = next(iter(target_lines))
            for line in detector_lines:
                x_schedule_delays_reversed[line] = sampling_params["x_schedule_delays"][
                    tl
                ]
                x_anneal_schedules_reversed[line] = sampling_params[
                    "x_anneal_schedules"
                ][tl]

            exp_weights_per_update = {
                v: [1.0, 0.0] if line_assignments.get(v) in detector_lines else [0.0, 1.0]
                for v in shimmed_variables
            }

            sampling_params_updates = [
                {
                    "x_anneal_schedules": sampling_params["x_anneal_schedules"],
                    "x_schedule_delays": sampling_params["x_schedule_delays"],
                },
                {
                    "x_anneal_schedules": x_anneal_schedules_reversed,
                    "x_schedule_delays": x_schedule_delays_reversed,
                },
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
        symmetrize_experiments=symmetrize_experiments,
        sampling_params_updates=sampling_params_updates,
        exp_weights_per_update=exp_weights_per_update,
        shimmed_variables=shimmed_variables,
        **kwargs,
    )
    _extend_history(dict_fb_all, dict_fb)
    _extend_history(dict_mag_all, dict_mag)
    return flux_biases, dict_fb_all, dict_mag_all
