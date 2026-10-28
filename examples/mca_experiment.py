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
``mca_experiment_helpers`` package. This file defines the command-line interface
and delegates to :func:`~mca_experiment_helpers.runner.main`.
"""

import argparse

from mca_experiment_helpers.common import SOLVER_FILTER, _get_experiment_id
from mca_experiment_helpers.runner import main


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Target-detector-source model embedding demo with optional flux-bias, "
            "anneal offset and x_schedule_delay refinement. "
            "Higher accuracy methods are possible with experimental insight and enhanced data."
        )
    )
    # Stage control: presence/absence of refinement stages.
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
        "--refine-frequencies",
        dest="refine_frequencies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate and apply per-qubit anneal_offsets to synchronize target "
        "frequencies (default). Use --no-refine-frequencies to skip the "
        "anneal-offset frequency-refinement stage.",
    )
    parser.add_argument(
        "--verify-anneal-offsets",
        dest="verify_anneal_offsets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-measure after applying anneal_offsets to verify the frequency "
        "alignment (a second iteration is useful to verify the first). Use "
        "--no-verify-anneal-offsets to skip verification. Has no effect when "
        "--no-refine-frequencies is set.",
    )
    parser.add_argument(
        "--no-refine-delays",
        dest="refine_delays",
        action="store_false",
        default=True,
        help="Disable repeating source-decoupling delay detection after "
        "anneal-offset refinement (which otherwise synchronizes detectors and "
        "sources on a per-qubit basis).",
    )
    parser.add_argument(
        "--loop-length",
        dest="loop_length",
        type=int,
        help="Length of the loop model (even and >=4), defaulting to 16. "
        "T-D-S systems compatible with coupled parallel unfrustrated loops are embedded. "
        "Specifying a value 1 results in a Larmor precession demonstration for independent "
        "T-D-S systems with no final loop experiment stage, this allows more T-D-S embeddings.",
        default=16,
    )
    # Solver and matching schedule.
    parser.add_argument(
        "--solver-name",
        dest="solver_name",
        type=str,
        help="QPU solver name. Default research system with fast reverse anneal.",
        default=SOLVER_FILTER,
    )
    parser.add_argument(
        "--schedule-fn",
        dest="schedule_fn",
        type=str,
        help="Path to the annealing schedule Excel file (.xlsx). Should be matched to the solver.",
        default="09-1323A-D_Advantage2_system4_annealing_schedule.xlsx",
    )
    # Embedding selection.
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
        "--max-num-embeddings",
        dest="max_num_embeddings",
        type=int,
        help="Max embeddings to find (default: all available).",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for embedding generation.",
        default=None,
    )
    # Target frequency.
    parser.add_argument(
        "--target-A",
        dest="target_A",
        type=float,
        help="Expected qubit frequency (GHz). "
        "(default A(target_c) such that A(target_c)=B(target_c), which is approximately 2 GHz)",
        default=None,
    )
    # Waveform option.
    parser.add_argument(
        "--no-overshoot",
        dest="use_overshoot",
        action="store_false",
        default=True,
        help="Disable overshoot transitions for the source and detector quenches.",
    )
    # Caching and output.
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Cache and reload experiment artifacts keyed by CLI parameters "
            "(default). Use --no-use-cache to disable. "
            "If solver graph_id changes, cached embeddings may be invalid."
        ),
    )
    parser.add_argument(
        "--save-figures",
        dest="save_figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save generated figures to the figures/ folder with hash-based "
        "names (default). Use --no-save-figures to disable.",
    )

    args = parser.parse_args()
    if args.use_cache:
        cache_str = _get_experiment_id(args, num_char=8)
    else:
        cache_str = None
    main(
        td_shim_type=args.td_shim_type,
        refine_frequencies=args.refine_frequencies,
        verify_anneal_offsets=args.verify_anneal_offsets,
        refine_delays=args.refine_delays,
        loop_length=args.loop_length,
        solver_name=args.solver_name,
        schedule_fn=args.schedule_fn,
        detector_lines=args.detector_lines,
        source_lines=args.source_lines,
        target_lines=args.target_lines,
        max_num_embeddings=args.max_num_embeddings,
        seed=args.seed,
        target_A=args.target_A,
        use_overshoot=args.use_overshoot,
        save_figures=args.save_figures,
        cache_str=cache_str,
    )
