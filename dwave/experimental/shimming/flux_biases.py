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
from typing import Any, Optional, Iterable, Callable

import numpy as np

import dimod
from dimod.typing import Variable, Bias

__all__ = ["shim_flux_biases", "qubit_freezeout_alpha_phi"]


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
    the learning rate :math:`l = T \frac{\Delta_1}{Delta_1 + Delta_2}`. This
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
            Effective (unitless) inverse temperature at freezeout. This
            can be determined from current device parameters.
        flux_associated_variance:
            The expected variance of the magnetization (:math:`m`) due to flux
            offset.
        estimator_variance:
            The expected variance in the magnetization estimate,
            :math:`\frac{1-m^2}{\text{num_reads}}`.
        flux_scale:
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
    sampling_params: Optional[dict[str, Any]] = None,
    shimmed_variables: Optional[Iterable[Variable]] = None,
    learning_schedule: Optional[Iterable[float]] = None,
    convergence_test: Optional[Callable] = None,
    symmetrize_experiments: bool = True,
    sampling_params_updates: Optional[list] = None,
    beta_hypergradient: float = 0.4,
    num_steps: int = 10,
    alpha: Optional[float] = None,
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
    annealing, non-zero :math:`h`, :ref:`parameter_polarizing_schedules`, or
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
            an initial condition and edited in place. Chose a value for the
            :ref:`parameter_qpu_num_reads` parameter in conjunction with your
            chosen schedule. Note that, the :ref:`parameter_qpu_initial_state`
            parameter, if provided, is assumed to be specified according the
            Ising model convention (:math:`\pm 1`, and :math:`-3` for inactive).
        shimmed_variables: A list of variables to shim; by default all elements
            in :attr:`~dimod.binary.BinaryQuadraticModel.variables`.
        learning_schedule: An iterable of gradient-descent prefactors. When not
            provided, prefactors are determined by a hypergradient-descent
            method parameterized by the ``alpha``, ``beta_hypergradient``, and
            ``num_steps`` arguments.
        convergence_test: A callable that take the history of magnetizations and
            flux biases as input, returning ``True`` to exit the search, and
            ``False`` otherwise. By default, all stages specified in the
            ``learning_schedule`` argument are completed.
        symmetrize_experiments: If True, performs a test to determine symmetry
            breaking in the experiment: a non-zero
            :ref:`parameter_qpu_initial_state` for reverse anneal, non-zero
            :math:`h`, or non-zero :ref:`parameter_qpu_flux_biases` (on some
            unshimmed variables). If any of these are present, magnetization is
            inferred by averaging over two experiments with symmetry-breaking
            elements inverted. The shim averages the symmetrically related
            experiments to achieve zero magnetization.
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
        >>> qpu = DWaveSampler()
        >>> bqm = dimod.BQM.from_ising({q: 0 for q in qpu.nodelist}, {})
        >>> alpha_phi = qubit_freezeout_alpha_phi()  # Unoptimized to the experiment, for demonstration purposes.
        >>> ls = [alpha_phi]*5
        >>> sp = {'num_reads': 2048, 'auto_scale': False}
        >>> fb, fb_history, mag_history = shim_flux_biases(bqm,
        ...     qpu,
        ...     sampling_params=sp,
        ...     learning_schedule=ls)
        ...
        >>> print(f"RMS magnetization by iteration: {np.sqrt(np.mean([np.array(v)**2 for v in mag_history.values()], axis=0))}") # doctest: +SKIP
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
        # All variables of the model
        shimmed_variables = bqm.variables
    else:
        if len(shimmed_variables) == 0:
            raise ValueError("shimmed_variables should not be empty")
        elif not set(shimmed_variables).issubset(bqm.variables):
            raise ValueError("Invalid shimmed variables")

    if symmetrize_experiments:
        unshimmed_variables = set(bqm.variables).difference(shimmed_variables)
        fbnonzero = any(flux_biases[v] != 0 for v in shimmed_variables)
        if bqm.vartype is dimod.BINARY:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        hnonzero = any(bqm.linear.values())
        if hnonzero:
            bqm = bqm.copy()
        reverseanneal = "initial_state" in sampling_params
        polarizedmca = "x_polarizing_schedules" in sampling_params and any(
            v != 0 for wfm in sampling_params["x_polarizing_schedules"] for _, v in wfm
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
                "flux_biases should not be explicitely set"
                "within sampling_params_updates."
            )
    num_experiments = num_signed_experiments * len(sampling_params_updates)

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
        # following loops are parallelizable, call sample() asyncrhonously.
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
                    sampling_params["x_polarizing_schedules"] = [
                        [(t, -v) for t, v in wfm]
                        for wfm in sampling_params["x_polarizing_schedules"]
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

        if use_hypergradient:
            magnetizations = np.array(
                [np.mean(mag_history[v][-num_experiments:]) for v in shimmed_variables]
            )
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
            flux_biases[v] -= alpha * sum(mag_history[v][-num_experiments:])
            flux_bias_history[v].append(flux_biases[v])

    if pop_fb:
        sampling_params["flux_biases"] = flux_biases

    return flux_biases, flux_bias_history, mag_history
