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
"""Multi-color-annealing calibration-refinement command-line entry point.

The implementation is split into small per-stage modules under the
``mca_shim_AO_FB_helpers`` package. This file defines the command-line interface
and delegates to :func:`~mca_shim_AO_FB_helpers.runner.main`.
"""

import argparse

from mca_shim_AO_FB_helpers.common import SOLVER_FILTER, _get_experiment_id
from mca_shim_AO_FB_helpers.runner import main


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Target-detector-source model embedding demo with optional flux-bias, "
            "anneal offset and x_schedule_delay refinement. "
            "Higher accuracy methods are possible with experimental insight and enhanced data."
        )
    )
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help=(
            "Cache and reload experiment artifacts keyed by CLI parameters. "
            "If solver graph_id changes, cached embeddings may be invalid."
        ),
    )
    parser.add_argument(
        "--solver-name",
        dest="solver_name",
        type=str,
        help="QPU solver name. Default research system with fast reverse anneal.",
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--detector-lines",
        dest="detector_lines",
        type=int,
        nargs="+",
        help="Detector lines (one or more integer indices).",
        default=None,  # First vertical qubit line
    )
    parser.add_argument(
        "--source-lines",
        dest="source_lines",
        type=int,
        nargs="+",
        help="Source lines (one or more integer indices).",
        default=None,  # First horizontal qubit line under 6-line control
    )
    parser.add_argument(
        "--target-lines",
        dest="target_lines",
        type=int,
        nargs="+",
        help="Target lines (one or more integer indices).",
        default=None,  # Default: all lines that are neither detector nor source lines
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for embedding generation.",
        default=None,
    )
    parser.add_argument(
        "--max-num-embeddings",
        dest="max_num_embeddings",
        type=int,
        help="Max embeddings to find (default: all available).",
        default=None,
    )
    parser.add_argument(
        "--target-A",
        dest="target_A",
        type=float,
        help="Expected qubit frequency (GHz). "
        "(default A(target_c) such that A(target_c)=B(target_c), which is approximately 2 GHz)",
        default=None,
    )
    parser.add_argument(
        "--schedule-fn",
        dest="schedule_fn",
        type=str,
        help="Path to the annealing schedule Excel file (.xlsx). Should be matched to the solver.",
        default="09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    )
    parser.add_argument(
        "--Jts",
        dest="Jts",
        type=float,
        help="Coupling strength between target and source qubits. "
        "The AFM value maximizing the effective quench rate is used. "
        "Note that, an FM value (-2.0) can achieve higher quench rates still but at risk of enhanced control errors",
        default=1.0,
    )
    parser.add_argument(
        "--Jtd",
        dest="Jtd",
        type=float,
        help="Coupling strength between target and detector qubits. "
        "The AFM value maximizing the effective quench rate is used. "
        "Note that, an FM value (-2.0) can achieve higher quench rates still but at risk of enhanced control errors",
        default=1.0,
    )
    parser.add_argument(
        "--Jtt",
        dest="Jtt",
        type=float,
        help="Coupling strength between target and target qubits. The parameter is ignored unless a loop length is specified. "
        "Note that a sufficiently weak coupling limit [B(s_target) J_tt << A(s_target)] is required for independent sourcing and detection of qubits.",
        default=-0.1,
    )
    parser.add_argument(
        "--loop-length",
        dest="loop_length",
        type=int,
        help="Length of the loop model (even and >=4), defaulting to 16. "
        "T-D-S systems compatible with coupled parallel unfrustrated loops are embedded. "
        "Specifying a value 1 results in a Larmour precession demonstration for independent "
        "T-D-S systems with no final loop experiment stage, this allows more T-D-S embeddings.",
        default=16,
    )
    parser.add_argument(
        "--flux-bias-method",
        dest="td_shim_type",
        type=str,
        choices=[
            "None",
            "detector_only",
            "alternating",
            "sequential",
            "by_line_quench",
        ],
        default="detector_only",
        help="Flux-bias calibration refinement mode: 'None' disables calibration refinement; "
        "'detector_only' refines detector qubit flux_biases to achieve zero measured magnetization; "
        "'by_line_quench' refines flux biases on all lines by assuming the required correction can be determined by independent quenches on every line; "
        "'sequential' combines by_line_quench and detector_only; "
        "'alternating' alternates detector/target roles to refine detector and target flux_biases "
        "(but this can cause divergence, particularly at small frequencies and "
        "when the qubit state realized by target_c varies significantly by line or qubit).",
    )
    parser.add_argument(
        "--t-decoupling",
        dest="t_decoupled",
        type=float,
        default=None,
        help="Delay required on the detector relative to the source in order that the target is decoupled.",
    )
    parser.add_argument(
        "--delay-by-det-src-pair",
        dest="by_det_src_pair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve source-decoupling delays independently per detector/source line pair "
        "(default). Use --no-delay-by-det-src-pair to estimate a single bulk delay shared "
        "across all lines. This setting has no impact if only "
        "one source and one detector line is present.",
    )
    parser.add_argument(
        "--num_anneal_offset_iterations",
        dest="num_anneal_offset_iterations",
        type=int,
        help="Number of anneal-offset iterations to perform. Note that a second iteration is useful to verify the first iteration (thus 2 by default).",
        default=2,
    )
    parser.add_argument(
        "--common-c-bounds",
        dest="use_common_c_bounds",
        action="store_true",
        default=False,
        help="Enable common c-bounds alignment across annealing lines. "
        "This ensures quench rates are nominally independent of the line choices."
        "This is required by some non-default shimming methods.",
    )
    parser.add_argument(
        "--no-overshoot",
        dest="use_overshoot",
        action="store_false",
        default=True,
        help="Disable overshoot transitions for the source and detector quenches.",
    )
    parser.add_argument(
        "--disable-save-figures",
        dest="save_figures",
        action="store_false",
        help="Disable saving figures to figures/ folder with hash-based names.",
    )
    parser.add_argument(
        "--no-refine-delays",
        dest="refine_delays",
        action="store_false",
        default=True,
        help="Repeat source-decoupling delay detection after anneal-offset refinement, "
        "This is then applied to synchronize detector/sources on a per qubit basis.",
    )
    parser.add_argument(
        "--show-artificial-timeseries",
        dest="show_artificial_timeseries",
        action="store_true",
        default=False,
        help="Display an illustrative artificial (model) timeseries and power-spectral-density "
        "demonstration before the experimental target-frequency stage.",
    )

    parser.add_argument(
        "--fit-window",
        dest="fit_window",
        type=float,
        default=None,
        help="Data window size used to establish the power-spectral density. "
        "Defaults to T2 (a decoherence time typical of Advantage2 processors). "
        "Values much larger "
        "than T2 are noise dominated and add no value; runtime is proportional to the "
        "fit window. The same window is overloaded to determine a bound on the delay "
        "fitting window (by bisection search).",
    )

    parser.add_argument(
        "--loop-delay-range",
        dest="loop_delay_range",
        type=float,
        default=None,
        help="Defaults to T2/4, but with data collected at delays 4x more finely "
        "spaced to better resolve pi/2 pulse propagation and synchronization limitations.",
    )

    args = parser.parse_args()
    if args.use_cache:
        cache_str = _get_experiment_id(args, num_char=8)
    else:
        cache_str = None
    main(
        cache_str=cache_str,
        solver=args.solver_name,
        detector_lines=args.detector_lines,
        source_lines=args.source_lines,
        target_lines=args.target_lines,
        target_A=args.target_A,
        schedule_fn=args.schedule_fn,
        t_decoupled=args.t_decoupled,
        by_det_src_pair=args.by_det_src_pair,
        num_anneal_offset_iterations=args.num_anneal_offset_iterations,
        td_shim_type=args.td_shim_type,
        use_common_c_bounds=args.use_common_c_bounds,
        use_overshoot=args.use_overshoot,
        save_figures=args.save_figures,
        Jts=args.Jts,
        Jtd=args.Jtd,
        Jtt=args.Jtt,
        loop_length=args.loop_length,
        max_num_embeddings=args.max_num_embeddings,
        refine_delays=args.refine_delays,
        show_artificial_timeseries=args.show_artificial_timeseries,
        fit_window=args.fit_window,
        loop_delay_range=args.loop_delay_range,
    )
