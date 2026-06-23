# Copyright 2026 D-Wave
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
"""
Detector parameterization example in line with methods of
https://arxiv.org/abs/2211.13227.

Created by Rahul Deshpande, reformatted by Jack Raymond.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt
from dwave.system.temperatures import fluxbias_to_h
from qutip import QobjEvo, qeye, sigmax, sigmaz, tensor
from tqdm import tqdm


def make_schedule(
    schedule_file: str = "09-1317A-F_Advantage2_research1_annealing_schedule.xlsx",
):
    """Import global (fast) anneal schedule

    Args:
        schedule_file: Path to excel annealing schedule.

    Returns:
        A, B: The energy scales A and B in GHz as functions of the normalized
        annealing bias c.
    """
    schedule_path = Path(schedule_file)
    if not schedule_path.is_absolute():
        schedule_path = Path(__file__).resolve().parent / schedule_path

    schedule = pd.read_excel(schedule_path, sheet_name="Fast-Annealing Schedule")
    cp = schedule["c (normalized)"].values
    Ap = np.array(schedule["A(s) (GHz)"].values)
    Bp = np.array(schedule["B(s) (GHz)"].values)

    def A(c):
        """Energy-scale, A, as function of the normalized annealing bias, c"""
        return np.interp(c, cp, Ap)

    def B(c):
        """Energy-scale, B, as function of the normalized annealing bias, c"""
        return np.interp(c, cp, Bp)

    return A, B


def ramp_0_1(t: float, start_time: float = 0.0, ramp_time: float = 10 / 6):
    """Map time to normalized annealing bias, c, assuming a linear ramp.

    Args:
        t: Time to map to c.
        start_time: Time at which ramp starts.
        ramp_time: Duration of the ramp.
    """
    if t < start_time:
        return 0.0
    if t > start_time + ramp_time:
        return 1.0
    return (t - start_time) / ramp_time


def make_Qevo_TD(
    c_target: float,
    start_time: float,
    ramp_time: float,
    det_fb: float = 0.0,
    J_det: float = 1.0,
    c_freezeout: float = 1.0,
    schedule_file: str = "09-1317A-F_Advantage2_research1_annealing_schedule.xlsx",
) -> QobjEvo:
    """Make the time-dependent Hamiltonian for the readout simulation.

    Args:
        c_target: Target qubit normalized annealing bias at which to read out.
        start_time: Time at which ramp starts.
        ramp_time: Duration of the ramp.
        det_fb: Detector flux bias.
        J_det: Detector coupling strength.
        c_freezeout: Freezeout parameter.
        schedule_file: Path to excel annealing schedule.
    """
    fb_scale_factor = fluxbias_to_h()
    A, B = make_schedule(schedule_file)

    def det_delta(t: float) -> float:
        c_det = c_freezeout * ramp_0_1(t, start_time, ramp_time)
        return A(c_det)

    def targ_det_J(t: float) -> float:
        c_det = c_freezeout * ramp_0_1(t, start_time, ramp_time)
        return np.sqrt(B(c_det) * B(c_target)) * J_det

    def det_h(t: float) -> float:
        c_det = c_freezeout * ramp_0_1(t, start_time, ramp_time)
        return np.sqrt(B(c_det)) * fb_scale_factor * det_fb

    ops = [
        A(c_target) * tensor([sigmax(), qeye(2)]),
        [tensor([qeye(2), sigmax()]), det_delta],
        [tensor([sigmaz(), sigmaz()]), targ_det_J],
    ]
    if det_fb != 0:
        ops.append([tensor([qeye(2), sigmaz()]), det_h])
    qevo = QobjEvo(ops)

    return 2 * np.pi * qevo


def simulate_readout(
    c_target: float,
    start_time: float,
    ramp_time: float,
    s_dephasing: qt.Qobj,
    s_ptrace: qt.Qobj,
    s_prep: qt.Qobj,
    h_rand: qt.Qobj,
    det_fb: float = 0.0,
    J_det: float = 1.0,
    c_freezeout: float = 1.0,
    schedule_file: str = "09-1317A-F_Advantage2_research1_annealing_schedule.xlsx",
    propagator_nsteps: int = 10000,
    show_plot: bool = False,
):
    """Simulate unitary evolution

    Args:
        c_target: Target qubit normalized annealing bias at which to read out.
        start_time: Time at which ramp starts.
        ramp_time: Duration of the ramp.
        s_dephasing: Dephasing superoperator.
        s_ptrace: Partial trace superoperator.
        s_prep: Preparation superoperator.
        h_rand: Small random Hamiltonian to break degeneracies.
        det_fb: Detector flux bias.
        J_det: Detector-target coupling strength.
        c_freezeout: Freezeout parameter.
        schedule_file: Path to excel annealing schedule.
        propagator_nsteps: Maximum ODE integration steps for the propagator.
        show_plot: Whether to plot the measurement map.
    """
    qevo = (
        make_Qevo_TD(
            c_target, start_time, ramp_time, det_fb, J_det, c_freezeout, schedule_file
        )
        + h_rand
    )

    # Convert to measurement superoperator
    prop = qt.propagator(
        qevo,
        start_time + ramp_time,
        options={"nsteps": propagator_nsteps},
    )
    meas_map = s_dephasing * s_ptrace * qt.to_super(prop) * s_prep

    # Normalize singleton tensor dims so QuTiP can treat this as a 1-qubit channel.
    meas_map = qt.Qobj(
        meas_map.data,
        dims=[[[2], [2]], [[2], [2]]],
        superrep="super",
    )
    meas_map_pauli = qt.core.superop_reps.to_superpauli(meas_map).tidyup(1e-3)
    if show_plot:
        qt.hinton(meas_map)
        plt.show()

    return meas_map_pauli


def main(
    target_c=0.3,
    schedule_file: str = "09-1317A-F_Advantage2_research1_annealing_schedule.xlsx",
    fb_range=1.5e-4,
    *,
    start_time=0.0,
    det_anneal_time=8,
    target_det_J=1,
    numpts_fb=51,
    h_perturbation=1e-7,
    c_freezeout=1.0,
    dephasing_time=1000.0,
    dephasing_nsteps=100000,
    propagator_nsteps=10000,
    example_det_fb=1.25e-4,
):
    """Main function to run the readout simulation example.

    Reproduces qualitatively a subset of results in
    https://arxiv.org/abs/2211.13227

    Args:
        target_c: Target qubit s/c at which to read out.
        start_time: Time at which the detector ramp starts.
        det_anneal_time: Duration of the detector anneal in ns.
        target_det_J: Coupling strength between target and detector.
        numpts_fb: Number of flux bias points to simulate.
        fb_range: Range of flux bias to simulate in micro Phi0 (symmetric around 0).
        schedule_file: Path to excel annealing schedule.
        h_perturbation: Scale of random Hamiltonian perturbation to break degeneracies in GHz.
        c_freezeout: Freezeout scaling applied to the detector anneal parameter.
        dephasing_time: Evolution time used to build the dephasing superoperator.
        dephasing_nsteps: Maximum ODE integration steps for dephasing propagator construction.
        propagator_nsteps: Maximum ODE integration steps for readout propagator construction.
        example_det_fb: Detector flux bias used for the final example measurement-map plot.
    """

    print(
        "Calculate the operator achieved at readout parameterized by Bloch sphere angles, as a function of the detector flux bias."
    )
    # Dephasing superoperator
    s_dephasing = qt.propagator(
        qt.liouvillian(0.0 * sigmaz(), [sigmaz()]),
        dephasing_time,
        options={"nsteps": dephasing_nsteps},
    ).tidyup(1e-3)

    # Superoperator to force detector initial state being |+>
    q = qt.tensor(qt.identity(2), (qt.ket("0") + qt.ket("1")).unit())
    s_prep = qt.to_super(q)

    # Partial trace superoperator
    s_ptrace = qt.tensor_contract(qt.to_super(qt.identity([2, 2])), (0, 2))

    # Small random h to break degeneracies
    rng = np.random.default_rng()
    h_rand = h_perturbation * (
        rng.random() * tensor([sigmaz(), qeye(2)])
        + rng.random() * tensor([qeye(2), sigmaz()])
    )

    det_fbs = np.linspace(-fb_range, fb_range, numpts_fb)
    readout_sims = []

    for det_fb in tqdm(det_fbs):
        res = simulate_readout(
            target_c,
            start_time,
            det_anneal_time,
            s_dephasing,
            s_ptrace,
            s_prep,
            h_rand,
            det_fb,
            target_det_J,
            c_freezeout=c_freezeout,
            schedule_file=schedule_file,
            propagator_nsteps=propagator_nsteps,
        )
        res_array = res[3, 1:]
        readout_sims.append(res_array)

    readout_sims = np.array(readout_sims)
    phis = np.angle(readout_sims[:, 2] + 1j * readout_sims[:, 1])
    thetas = np.angle(
        readout_sims[:, 0] + 1j * np.linalg.norm(readout_sims[:, [1, 2]], axis=1)
    )

    plt.figure()
    plt.title("Detector parameterization as function of the flux bias")
    plt.plot(det_fbs, (phis - phis[numpts_fb // 2]) / np.pi, label=r"$\phi$")
    plt.plot(det_fbs, thetas / np.pi, label=r"$\theta$")

    plt.legend()
    plt.xlabel(r"Detector flux bias, $\mu \Phi_0$")
    plt.ylabel(r"Angle, radians/$\pi$")

    # Example of mapping energy eigenbasis of target to flux basis of detector
    print(
        "Calculate the mapping of the operator on the target, to an operation on the detector at the end of the anneal."
    )

    simulate_readout(
        target_c,
        start_time,
        det_anneal_time,
        s_dephasing,
        s_ptrace,
        s_prep,
        h_rand,
        example_det_fb,
        target_det_J,
        c_freezeout=c_freezeout,
        schedule_file=schedule_file,
        propagator_nsteps=propagator_nsteps,
        show_plot=True,
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detector parameterization example in line with methods of https://arxiv.org/abs/2211.13227."
    )
    parser.add_argument(
        "--target-c",
        type=float,
        default=0.3,
        help="Target qubit s/c at which to read out (default: 0.3). ",
    )
    parser.add_argument(
        "--schedule-file",
        type=str,
        default="09-1317A-F_Advantage2_research1_annealing_schedule.xlsx",
        help="Path to excel annealing schedule (default: 09-1317A-F_Advantage2_research1_annealing_schedule.xlsx).",
    )
    parser.add_argument(
        "--fb-range",
        type=float,
        default=1.5e-4,
        help="Flux-bias sweep range, symmetric about zero (default: 1.5e-4)",
    )
    args = parser.parse_args()
    main(
        target_c=args.target_c,
        fb_range=args.fb_range,
        schedule_file=args.schedule_file,
    )
