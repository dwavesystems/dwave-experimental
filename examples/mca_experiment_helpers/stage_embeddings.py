"""Stage 2: find Target-Detector-Source embeddings for parallel programming."""

from .common import *


def run_embeddings(
    ctx: MCAContext, *, embedding_timeout: int = 60, create_plot: bool = True
) -> MCAContext:
    """Find T-D-S embeddings for parallel programming.

    Args:
        ctx: Shared context carrying configuration and computed state between
            stages.
        embedding_timeout: Time budget (seconds) for the embedding search.
        create_plot: When ``True`` (default), draw the parallel T-D-S embeddings
            on the processor graph (see ``mca_embedding.py``).

    Returns:
        The same ``ctx``, updated with the embeddings, per-line groupings and
        the parallel-embedding sampler.
    """
    stage_idx = ctx.stage_idx
    qpu = ctx.qpu
    line_assignments = line_assignments_from_feature_info(
        ctx.exp_feature_info.annealing_line_info
    )
    detector_lines = ctx.detector_lines
    source_lines = ctx.source_lines
    target_lines = ctx.target_lines
    loop_length = ctx.loop_length
    cache_str = ctx.cache_str
    solver = ctx.solver_name
    max_num_embeddings = ctx.max_num_embeddings
    seed = ctx.seed
    J_td = ctx.J_td
    J_ts = ctx.J_ts
    line_color = ctx.line_color
    save_figures = ctx.save_figures

    stage_idx += 1
    print()
    print(f"Stage {stage_idx}: Find T-D-S embeddings for parallel programming")
    print("(see mca_embedding.py example).")
    print(
        "Finding embeddings is NP-hard in general. The routine attempts to find many "
        f"such embeddings that can be programmed in parallel. If no embedding is found "
        f"within the (CLI configurable) timeout of {embedding_timeout} seconds, the search is terminated."
    )

    T = qpu.to_networkx_graph()

    def _target_assignments(n: int) -> str:
        """Classify a qubit as "detector", "source", or "target" by its line."""
        line = line_assignments[n]
        if line in detector_lines:
            return "detector"
        elif line in source_lines:
            return "source"
        elif line in target_lines:
            return "target"
        else:
            return "unused"

    Tnode_to_tds = {n: _target_assignments(n) for n in qpu.nodelist}

    target_graph = nx.Graph()
    target_graph.add_node(0)
    # Embeddings are reprocessed, to be treated as independent.
    S, Snode_to_tds = make_tds_graph(target_graph)
    if loop_length is not None:
        target_graph_experiment = nx.Graph()
        target_graph_experiment.add_nodes_from(range(loop_length))
        target_graph_experiment.add_edges_from(
            (i, (i + 1) % loop_length) for i in range(loop_length)
        )
        S_experiment, Snode_to_tds = make_tds_graph(target_graph_experiment)
    else:
        S_experiment = S

    if cache_str:
        # Identifier depending only on parameters that affect the embedding
        # search (the lines below), so cached embeddings are reused when other
        # parameters are varied.
        embedding_cache_str = hashlib.sha256(
            json.dumps(
                {
                    "solver": "DefaultSolver" if solver == SOLVER_FILTER else solver,
                    "detector_lines": sorted(detector_lines),
                    "source_lines": sorted(source_lines),
                    "target_lines": sorted(target_lines),
                    "loop_length": loop_length,
                    "max_num_embeddings": max_num_embeddings,
                    "seed": seed,
                    "embedding_timeout": embedding_timeout,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        fn_cache = f"cache/emb_{embedding_cache_str}.pkl"
        os.makedirs(os.path.dirname(fn_cache), exist_ok=True)
    else:
        fn_cache = None  # Caching disabled: no cache file is sought or written.

    if fn_cache is not None and os.path.isfile(fn_cache):
        with open(fn_cache, "rb") as f:
            embs_experiment = pickle.load(f)
        if len(embs_experiment) == 0:
            raise RuntimeError(
                f"The cached embeddings in {fn_cache} are empty; delete the file "
                "to repeat the embedding search."
            )
    else:
        subgraph_kwargs = dict(
            node_labels=(Snode_to_tds, Tnode_to_tds),
            as_embedding=True,
            timeout=embedding_timeout,
        )
        embs_experiment = find_multiple_embeddings(
            S_experiment,
            T,
            max_num_emb=max_num_embeddings,
            embedder_kwargs=subgraph_kwargs,
            one_to_iterable=True,
            seed=seed,
            timeout=embedding_timeout,
        )
        if len(embs_experiment) == 0:
            raise RuntimeError(
                "No embeddings were found, try a larger embedding_timeout, simpler target (smaller or no num_loops) or new line combination."
            )
        if fn_cache is not None:
            with open(fn_cache, "wb") as f:
                pickle.dump(embs_experiment, f)

    if loop_length is not None:
        print(
            f"{len(embs_experiment)} independent length-{loop_length} target loops were found, where each target qubit is connected to both a source and a detector. "
            f"In the calibration refinement stages these are treated as {len(embs_experiment)}x{loop_length} independent T-D-S systems."
        )
        embs = _to_independent_tds(embs_experiment)
    else:
        embs = embs_experiment

    print(
        f"{len(embs)} T-D-S placements were found; each is calibrated with parallelized data collection. "
        "These are ordered by (target line, detector line, source line) for purposes of visualization. "
        "For time series plots an exemplar is plotted to capture each line combination. "
    )
    embs_by_line = defaultdict(list)
    for _, emb in enumerate(embs):

        embs_by_line[tuple(line_assignments[c[0]] for c in emb.values())].append(emb)

    embs = [emb for key in sorted(embs_by_line) for emb in embs_by_line[key]]
    sum_by_line = {key: len(embs_by_line[key]) for key in sorted(embs_by_line)}
    cumsum_by_line = {
        key: sum(sum_by_line[k] for k in sorted(sum_by_line) if k <= key)
        for key in sorted(sum_by_line)
    }
    line_exemplars = {key[0]: cumsum_by_line[key] - 1 for key in embs_by_line}
    print("Number of embeddings by t-d-s line combination:", sum_by_line)
    n_embs = len(embs)

    sampler = ParallelEmbeddingComposite(qpu, embeddings=embs)

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising(
        {n: 0 for n in S.nodes()},
        {e: J_td for e in S.edges() if any(Snode_to_tds[v] == "detector" for v in e)}
        | {e: J_ts for e in S.edges() if any(Snode_to_tds[v] == "source" for v in e)},
    )  # bqm restricted to decoupled source and target nodes

    if create_plot:
        # Colour used qubits by annealing line (as in mca_embedding.py); unused
        # qubits are greyed out. S_experiment is the ring (loop) or single target.
        used_nodes = {v[0] for emb in embs_experiment for v in emb.values()}
        node_color = [
            line_color[line_assignments[n]] if n in used_nodes else "grey"
            for n in T.nodes()
        ]
        plt.figure(f"Parallel embedding of {len(embs_experiment)} T-D-S systems")
        draw_parallel_embeddings(
            T, embeddings=embs_experiment, S=S_experiment, node_color=node_color, cmap=None
        )
        if save_figures:
            _save_open_figures("figures/", cache_str)
        print("Close figures to proceed to next stages.")
        _apply_tight_layout()
        plt.show()

    ctx.stage_idx = stage_idx
    ctx.T = T
    ctx.Tnode_to_tds = Tnode_to_tds
    ctx.S = S
    ctx.Snode_to_tds = Snode_to_tds
    ctx.S_experiment = S_experiment
    ctx.embs_experiment = embs_experiment
    ctx.embs = embs
    ctx.embs_by_line = embs_by_line
    ctx.sum_by_line = sum_by_line
    ctx.cumsum_by_line = cumsum_by_line
    ctx.line_exemplars = line_exemplars
    ctx.n_embs = n_embs
    ctx.sampler = sampler
    ctx.bqm = bqm
    return ctx
