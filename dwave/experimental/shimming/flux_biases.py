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

    Assume a qubit is offset by phi_0, which is to be corrected
    by some choice of flux_bias phi. A model of single qubit freezeout dictates
    that magnetization <s_i> = tanh((phi_{0i} + phi_i)/ T),
    where T is the effective temperature.

    We assume the unshimmed magnetization to be zero mean distributed
    with small variance Delta_1. Assume that we use the standard sampling
    based estimator with variance Delta_2 = 1/num_reads. We can then
    determine an update the flux as phi = l <s_i>_data, where the
    learning rate l = T Delta_1 /(Delta_1 + Delta_2). This update is
    optimal in the sense it minimizes the expected square magnetization.

    For correlated spin systems and/or experiments not well described by
    thermal freezeout it is recommended a data-driven approach is taken
    to determining the schedule and related parameters. The freezeout
    (Boltzmann) distribution can be extended to correlated models, wherein
    the covariance matrix plays a role in determining optimal learning rate.
    A single qubit rate can remain a good approximation given weakly
    correlated spins.

    Args:
        eff_temp_phi:
            Effective (unitless) inverse temperature at freezeout. This
            can be determined from current device parameters
        flux_associated_variance:
            The expected variance of the magnetization (m) due to flux
            offset.
        estimator_variance:
            The expected variance in the magnetization estimate,
            (1-m^2) divided by number of independent reads.
        flux_scale:
            Conversion from units of h to units of phi. See
            dwave.system.temperatures.h_to_phi. This can be determined
            from published device parameters.
    Returns:
        An appropriate scale for the learning rate, minimizing the expected
        square magnetization.

    Example:
        Determining an alpha_phi appropriate for forward anneal of a weakly
        coupled system Advantage_system4.1 based on published parameters.
        Note that defaults (by contrast) are determined based on published
        values for Advantage2_system1.3

        >>> alpha_phi = qubit_freezeout_alpha_phi(eff_temp_phi=0.198, flux_associated_variance=1/1024, estimator_variance=1/256, unit_conversion=1.647e-3)

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
) -> tuple[list[Bias], dict, dict]:
    r"""Return flux_biases achieving  <s_i> = 0 for symmetry preserving
    experiments.

    Calibration can be improved for specific QPU protocols by modification of
    QPU programmings. The flux_bias parameter can compensate for
    low-frequency environmental spins that couple into qubits, distorting
    the target distribution. Although modification of either flux_biases and h might
    be used to restore symmetry of the sampled distribution, flux biases
    more accurately eliminate common forms of low frequency noise.

    Assuming the magnetization (expectation for the
    measured spins (sign of persistent current) to be a smooth monotonic
    function of the qubit body magnetic fluxes (`flux_bias`), we can
    determine parameters achieving a target magnetization (m) by iteration of
    :\math:`\Phi(t+1) \leftarrow Phi(t) - L(t) (<s> - m)`. Where
    L(t) is an iteration-dependent map, <s> is the expected magnetization, m
    is the target magnetization and Phi are the programmed flux biases.

    Symmetry can be broken by the choice of initial condition in reverse
    annealing, non-zero h, or non-zero flux biases over unshimmed
    fluxes. We can collect data for two experiments, where the symmetry
    breaking is inverted - we can anticipate zero magnetization not per
    experiment but in the experimental average. Shimming based on this
    symmetrized data set is expected to determine a good shim for both
    experiments, assuming a weak dependence of noise on the symmetry breaking
    field.

    Where strong correlations, or strong symmetry breaking effects, are present in
    an experiment, the sampled distribution may contain insufficient information to
    independently shim all degrees of freedom. Shims are expected to be a smooth
    function of annealing parameters such as annealing time, anneal schedule, and
    Hamiltonian parameters. Shims inferred in smoothly related models can be used
    as approximations (or initial conditions) for searches in related models.

    Args:
       bqm: A dimod binary quadratic model.
       sampler: A DWaveSampler.
       sampling_params: Parameters of the DWaveSampler. Note that if sampling_params
           contains flux_biases, these are treated as an initial condition and
           edited in place.
           num_reads should be appropriately chosen in conjunction with the
           schedule. Note that, initial_states if provided is assumed to
           be specified according the Ising model convention (+/-1, and -3 for inactive).
       shimmed_variables: A list of variables to shim, by default all elements in
           bqm.variables.
       learning_schedule: An iterable of gradient descent prefactors. By default, a
           single step is taken. Experimental-timescale fluctuations in the flux
           noise (characterized by a 1/f spectrum) as well as sampling error
           linked to estimator variance, limit the accuracy attainable. Per standard
           gradient descent approaches one can consider a constant learning rate,
           or one that decreases with t. See examples.
       convergence_test: A callable taking the history of magnetizations and flux_biases
           as input, returning True to exit the search, and False otherwise. By default,
           all stages specified in the learning_schedule are completed.
       symmetrize_experiments: If True a test is performed to determine symmetry breaking
           in the experiment: a non-zero initial_state for reverse anneal, non-zero h,
           or non-zero flux_bias (on some unshimmed variables). If any of these are present
           the magnetization is inferred by averaging over two experiments (with symmetry
           breaking elements inverted). We shim so that the average of the symmetrically
           related experiments has zero magnetization.

    Returns:
        A tuple consisting of 3 parts:
        1. flux_biases in a list format suitable as a DWaveSampler argument.
        2. A history of flux_bias assignments per shimmed component.
        3. A history of magnetizations per shimmed component.

    Example:
        See examples/ and tests/ for additional use cases.
    
        Shim degenerate qubits at constant learning rate and solver defaults.
        The learning schedule and num_reads is for demonstration only, and has not been optimized.

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
        >>> fb, fb_history, mag_history = shim_flux_biases(bqm, qpu, sampling_params=sp, learning_schedule=ls)
        ...
        >>> print(f'Root mean-square magnetization by iteration:', np.sqrt(np.mean([np.array(v)**2 for v in mag_history.values()], axis=0)))
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

    if symmetrize_experiments:
        unshimmed_variables = set(bqm.variables).difference(shimmed_variables)
        fbnonzero = any(flux_biases[v] != 0 for v in shimmed_variables)
        if bqm.vartype is dimod.BINARY:
            bqm = bqm.change_vartype(dimod.SPIN, inplace=False)
        hnonzero = any(bqm.linear.values())
        if hnonzero:
            bqm = bqm.copy()
        reverseanneal = "initial_state" in sampling_params
    else:
        fbnonzero = hnonzero = reverseanneal = False
    num_experiments = 1 + int(reverseanneal or hnonzero or fbnonzero)

    if learning_schedule is None:
        learning_schedule = [qubit_freezeout_alpha_phi()]

    if convergence_test is None:
        convergence_test = lambda x, y: False

    flux_bias_history = {v: [flux_biases[v]] for v in shimmed_variables}
    mag_history = {v: [] for v in bqm.variables}

    for lr in learning_schedule:
        for _ in range(num_experiments):
            if reverseanneal:
                for i in bqm.variables:
                    sampling_params["initial_state"][i] *= -1
            if hnonzero:
                for i in bqm.variables:
                    bqm.linear[i] *= -1
            if fbnonzero:
                for i in unshimmed_variables:
                    flux_biases[i] *= -1

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

        for v in shimmed_variables:
            flux_biases[v] = flux_biases[v] - lr * sum(
                mag_history[v][-num_experiments:]
            )
            flux_bias_history[v].append(flux_biases[v])

    if pop_fb:
        sampling_params["flux_biases"] = flux_biases

    return flux_biases, flux_bias_history, mag_history
