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

"""
This module contains an automorphism composite that can make use of
either the Schreier-Sims representation of the automorphism_generation.py
module, or a (generator, degree) format where the generator is a dictionary.
This generator should later be moved to dwave.preprocessing.composites to
work alongside SpinReversalTransformComposite - since it serves a closely
related purpose.

This module also specifies the generators for defect free Chimera, Pegasus
and Zephyr graphs either in a format suitable for sampling by sequential
application of the generators, or in the form of strongest sets. This might
be later moved to dwave_networkx to sit alongside other code related to
processor-graph theoretic properties.
"""


from typing import Sequence, Any, Literal
from itertools import product

import networkx as nx
import dimod
import numpy as np

from dimod import ComposedSampler
from dwave_networkx import pegasus_graph, pegasus_coordinates
from dwave.experimental.automorphism import (
    schreier_rep,
    sample_automorphisms as sample_automorphisms_u_vector,
)

__all__ = [
    "AutomorphismComposite",
    "chimera_generators",
    "pegasus_generators",
    "zephyr_generators",
    "sample_automorphisms_listtuple",
    "listtuple_to_arrays",
    "reduce_generator_by_node_set",
]


def listtuple_to_arrays(
    listtuple: list[tuple[dict, int]], node_to_idx: dict
) -> list[list[np.ndarray[np.intp]]]:
    """Unwrap (generator, degree) pairs into an array format.

    Note, the array format is accepted by the
    :meth:`~dwave.experimental.automorphism.automorphism_generatation.sample_automorphisms`.
    It is more general than the generator pair format, so an
    inverse mapping is not possible in general.

    Args:
        listtuple: A list of generator, degree pairs. Each generator
            is defined by a dictionary.
        node_to_idx: A mapping from node labels of the graph to
            contiguous integers labeled from zero.

    Output:
        A list of lists of numpy arrays, suitable for use by
        :meth:`~dwave.experimental.automorphism.automorphism_generatation.sample_automorphisms`.
    """

    node_set = set(node_to_idx)
    listtuple = [
        (reduce_generator_by_node_set(g, node_set), n) for g, n in listtuple
    ]  # Compress listtuple w.r.t. mapped nodes.
    listtuple = [p for p in listtuple if p[0]]  # Remove anything redundant
    llarray = [
        [np.arange(len(node_to_idx), dtype=np.intp) for _ in range(n - 1)]
        for _, n in listtuple
    ]
    for larray, (g, degree) in zip(llarray, listtuple):
        g = {node_to_idx[k]: node_to_idx[v] for k, v in g.items()}
        for k, v in g.items():
            larray[0][k] = v  # First element is standard generator
        for idx in range(degree - 2):
            parray = larray[idx]  # same generator applied up to n-1 times.
            for k, v in g.items():
                larray[idx + 1][parray[k]] = parray[v]
    return llarray


def chimera_generators(
    m: int,
    n: int | None = None,
    t: int = 4,
    generator_type: Literal["redundant", "strongest"] = "redundant",
) -> list[tuple[dict, int]]:
    """Return generators for Chimera graphs.

    Args:
       m: number or rows
       n: number of columns
       t: tile parameter
       generator_type: The type of generator, either `redundant`,
          to specify a set suitable for sequential sampling, or
          'strongest' to generate a strongest set. The size of the
          group is given by the product of degrees in the redunant
          representation if m!=n or t<2. And is given by the product
          divided by 2 for the case t>1 and m=n.

    Returns:
        A list of tuples. Each tuple includes a generator, and the
        degree of the generator.
    """

    if n is None:
        n = m
    if n == m:
        # Verticals to horizontals
        diagonal = [
            (
                {
                    (i, j, u, k): (i, j, 1 - u, k)
                    for i in range(m)
                    for j in range(n)
                    for u in range(2)
                    for k in range(t)
                },
                2,
            )
        ]
    else:
        diagonal = []
    if m > 1:
        vertical = [
            (
                {
                    (i, j, u, k): (m - 1 - i, j, u, k)
                    for i in range(m)
                    for j in range(n)
                    for u in range(2)
                    for k in range(t)
                },
                2,
            )
        ]
    else:
        vertical = []
    if n > 1 and (m != n or generator_type == "redundant"):
        horizontal = [
            (
                {
                    (i, j, u, k): (i, n - 1 - j, u, k)
                    for i in range(m)
                    for j in range(n)
                    for u in range(2)
                    for k in range(t)
                },
                2,
            )
        ]
    else:
        horizontal = []
    # Shore Permutations correlated by row/column
    # - must be ordered: (1,..,k) then (1,,k-1) .. (1,2) etc. fully mixes (permutation)
    if generator_type == "redundant":
        vert_shores = [
            (
                {
                    (i, j, 0, k): (i, j, 0, (k + 1) % kp)
                    for i in range(m)
                    for k in range(kp)
                },
                kp,
            )
            for kp in range(t, 1, -1)
            for j in range(n)
        ]
        horiz_shores = [
            (
                {
                    (i, j, 1, k): (i, j, 1, (k + 1) % kp)
                    for j in range(n)
                    for k in range(kp)
                },
                kp,
            )
            for kp in range(t, 1, -1)
            for i in range(m)
        ]
    else:
        vert_shores = [
            (
                {
                    (i, j, 0, k + o1): (i, j, 0, k + o2)
                    for i in range(m)
                    for o1, o2 in [(0, 1), (1, 0)]
                },
                2,
            )
            for k in range(t, 1, -1)
            for j in range((n + 1) // 2)
        ]
        if m != n:
            horiz_shores = [
                (
                    {
                        (i, j, 1, k + o1): (i, j, 1, k + o2)
                        for j in range(n)
                        for o1, o2 in [(0, 1), (1, 0)]
                    },
                    2,
                )
                for k in range(t, 1, -1)
                for i in range((m + 1) // 2)
            ]
        else:
            horiz_shores = []
    return diagonal + vertical + horizontal + vert_shores + horiz_shores


def _flip_zephyr(u: int, w: int, k: int, j: int, z: int, orient: bool, m: int) -> tuple:
    if orient:
        ue = u
    else:
        ue = 1 - u
    return (
        u,
        ue * (2 * m - w) + (1 - ue) * w,
        k,
        ue * j + (1 - ue) * (1 - j),
        ue * z + (1 - ue) * (m - 1 - z),
    )


def zephyr_generators(
    m: int, t: int = 4, generator_type: Literal["redundant", "strongest"] = "redundant"
) -> list[tuple[dict, int]]:
    """Create generators for zephyr graphs

    Args:
        m: Grid parameter for the Zephyr lattice.
        t: Tile parameter for the Zephyr lattice.
        generator_type: The type of generator, either `redundant`,
          to specify a set suitable for sequential sampling, or
          'strongest' to generate a strongest set. The size of the
          group is given by the product of degrees in the redundant.

    Returns:
        A list of tuples. Each tuple includes a generator, and the
        degree of the generator.
    """

    diagonal = [
        (
            {
                (u, w, k, j, z): (1 - u, w, k, j, z)
                for u in range(2)
                for w in range(2 * m + 1)
                for k in range(t)
                for j in range(2)
                for z in range(m)
            },
            2,
        )
    ]
    vertical = [
        (
            {
                (u, w, k, j, z): _flip_zephyr(u, w, k, j, z, True, m)
                for u in range(2)
                for w in range(2 * m + 1)
                for k in range(t)
                for j in range(2)
                for z in range(m)
            },
            2,
        )
    ]
    if generator_type != "strongest":
        horizontal = [
            (
                {
                    (u, w, k, j, z): _flip_zephyr(u, w, k, j, z, False, m)
                    for u in range(2)
                    for w in range(2 * m + 1)
                    for k in range(t)
                    for j in range(2)
                    for z in range(m)
                },
                2,
            )
        ]
    else:
        horizontal = []
    # Shore Permutations correlated by row/column
    # - must be ordered: (1,..,k) then (1,,k-1) .. (1,2) etc. fully mixes (permutation)
    if generator_type != "strongest":
        shores = [
            (
                {
                    (u, w, k, j, z): (u, w, (k + 1) % kp, j, z)
                    for z in range(m)
                    for j in range(2)
                    for k in range(kp)
                },
                kp,
            )
            for kp in range(t, 1, -1)
            for u in range(2)
            for w in range(2 * m + 1)
        ]
    else:
        shores = [
            (
                {
                    (u, w, k, j, z): (u, w, (k + 1) % kp, j, z)
                    for z in range(m)
                    for j in range(2)
                    for k in range(kp)
                },
                kp,
            )
            for kp in range(t, 1, -1)
            for u in range(1)  # one orientation suffices
            for w in range(m + 1)  # half way suffices
        ]

    return diagonal + vertical + horizontal + shores


def pegasus_generators(
    m: int, generator_type: Literal["redundant", "strongest"] = "redundant"
) -> list[tuple[dict, int]]:
    """Create generators for pegasus (fabric_only, for m>1)

    Args:
        m: Grid parameter for the Pegasus lattice.
        generator_type: The type of generator, either `redundant`,
          to specify a set suitable for sequential sampling, or
          'strongest' to generate a strongest set. The size of the
          group is given by the product of degrees in the redundant.

    Returns:
        A list of tuples. Each tuple includes a generator, and the
        degree of the generator.
    """
    diagonal = [
        (
            {
                (u, w, k, z): (1 - u, m - 1 - w, 11 - k, m - 2 - z)
                for u in range(2)
                for w in range(m)
                for k in range(12)
                for z in range(m - 1)
            },
            2,
        )
    ]
    # Odd-pairs
    odd_pairs = [
        (
            {
                (u, w, koff + kin, z): (u, w, koff + (kin + 1) % 2, z)
                for z in range(m - 1)
                for kin in range(2)
            },
            2,
        )
        for koff in range(0, 12, 2)
        for u in range(2)
        for w in range(m)
    ]

    # Constrain to the standard (fabric_only) representation (the largest component):
    nonfabric = {
        (u, m - 1, k, z) for u in range(2) for k in range(10, 12) for z in range(m - 1)
    } | {(u, 0, k, z) for u in range(2) for k in range(2) for z in range(m - 1)}
    fabric = {
        c for c in product(range(2), range(m), range(12), range(m - 1))
    }.difference(nonfabric)
    pruned_generators = []
    for g in diagonal + odd_pairs:
        new = reduce_generator_by_node_set(g[0], node_set=fabric)
        if new:
            pruned_generators.append((new, g[1]))

    return pruned_generators


def sample_automorphisms_listtuple(
    generators_listtuple: list[tuple[dict, int]],
    *,
    prng: np.random.Generator | int | None = None,
    mapping: dict | None = None,
):
    """Sample an automorphism by ordered application of generators

    An abelian group allows for generators to be sampled sequentially to fairly
    sample the space of all automorphisms.
    Generators for Chimera, Pegasus and Zephyr graphs are not abelian, but can
    be ordered to allow for similar fair sampling. See the respective generator methods
    of this module.
    For more general cases it is recommended to fairly sample by the Schreier-Sims
    method of automorphism_generation.py module.

    Generators are specified as dictionaries (1:1 mappings), with some known
    degree e.g. {2: 4, 4: 2}, degree 2

    Args:
        generators_listtuple: A list of generator, generator degree tuples, to be
            sampled sequentially.
        prng: A numpy pseudo random number generator, or seed used to initialize
            the PRNG.
        mapping: An initial permutation to which generators are applied, by default
            the identity. Note the mapping is modified in place.
    """

    prng = np.random.default_rng(prng)
    if mapping is None:
        vars_set = set(n for g in generators_listtuple for n in g[0].keys())
        mapping = {n: n for n in vars_set}
    for generator, degree in generators_listtuple:
        for _ in range(prng.integers(degree)):
            mapping.update({k: mapping[v] for k, v in generator.items()})

    return mapping


def reduce_generator_by_node_set(generator_dict: dict, node_set: set):
    """Restrict a generator to a closed subspace defined by node_set

    Note that the returned generators will only be valid for some graph
    in general if the node_set are fully connected.

    Args:
        generator_dict: An automorphism as a dictionary.
        node_set: A set of variables (defining the subgraph).

    Returns:
        a generator restricted to the subspace defined by the node_set,
        or an empty dictionary if the node_set is not closed under the
        generator.

    .. note:: The generator keys need only specify a subset of
        variables in the node_set. Variables not specified in the generator
        (dictionary format) are assumed to map 1:1.

    """

    node_set = node_set.intersection(generator_dict)
    if not node_set:
        return {}
    reduced_node_set = node_set.intersection(generator_dict[n] for n in node_set)
    while reduced_node_set != node_set:
        node_set = reduced_node_set
        reduced_node_set = node_set.intersection(generator_dict[n] for n in node_set)
    return {n: generator_dict[n] for n in node_set}


class AutomorphismComposite(ComposedSampler):
    """Composite for applying automorphic preprocessing.

    The ordering of variables is permuted subject to a given pattern of
    generators or an inherited sampler structure.
    This can be useful to mitigate for noise, control errors or other
    symmetry-breaking features of the child sampler.

    .. note::
        If you are configuring an anneal schedule, be mindful that this
        composite does not recognize the ``initial_state`` parameter
        used by dimod's :class:`~dwave.system.samplers.DWaveSampler` for
        reverse annealing (composites do not generally process all keywords
        of child samplers) and does not permute any of the configured initial
        states.
    .. note::
        If the generators or graph are not specified, it is by default inferred
        from the graph of the structured child solver. Attention in the choice
        of generators, or explicit specification of a target subgraph, can
        enhance the variety of automorphisms available.
        When the child graph is not structured it is assumed unconstrainted,
        and the space of automorphisms is assumed to be that of a fully
        connected graph (all permutations).

    Args:
        sampler: A `dimod` sampler object.

        seed: As passed to :func:`numpy.random.default_rng`.

        generators_listtuple: A set of permutations compatible with the child
            strcture. A list where each element is a tuple of generator (dict)
            and integer cycle length. This allows for uniform sampling of some
            graphs. If generators_listtuple is None (by default) a schreir_context
            is used.

        generators_u_vector: A dwave.experimental.SchreirContext object. This allows
            fair sampling of any graph. The SchreirContext for an arbitrary
            graph can be created using dwave.experimental. If a schreir_context
            is None, and generators_listtuple is None, then the schreir_context
            compatible with the child sampler structure is created by default.
            If both a generators_listtuple, and a schreir_context are provided,
            the generators_listtuple is ignored.

        G: A `nx.Graph` from which automorphisms are to be inferred when neither
            generators_listtuple or generators_u_vector are provided. If G is
            not specified, the child sampler must be a `dimod.StructuredSampler`
            and the graph is assumed from the given nodelist and edgelist.

        idx_to_node: A mapping from variables to sequential integer labels, required
            only when automorphisms are determined from `generators_u_vector`.

    Examples:
        This example composes a dimod ExactSolver sampler with the AutomorphismComposite
        then uses it to sample an Ising problem.

        >>> from dimod import ExactSolver
        >>> from dwave.preprocessing.composites import AutomorphismComposite
        >>> base_sampler = ExactSolver()
        >>> generators_listtuple = [({'a': 'b', 'b':'a'}, 2)]
        >>> composed_sampler = AutomorphismComposite(base_sampler, generators_listtuple)
        ... # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.first.sample
        {'a': -1, 'b': -1}

    """

    _children: list[dimod.core.Sampler]
    _parameters: dict[str, Sequence[str]]
    _properties: dict[str, Any]

    def __init__(
        self,
        child: dimod.core.Sampler,
        *,
        seed: np.random.Generator | int | None = None,
        generators_listtuple: list[tuple[dict, int]] | None = None,
        generators_u_vector: list[list[np.ndarray[np.intp]]] = None,
        G: nx.Graph = None,
        idx_to_node: dict = None,
    ):
        self._child = child
        self.rng = np.random.default_rng(seed)
        self.generators_listtuple = generators_listtuple
        self.generators_u_vector = generators_u_vector
        self.idx_to_node = idx_to_node
        if (
            generators_u_vector is None
            and generators_listtuple is None
            and (G is not None or isinstance(child, dimod.Structured))
        ):
            if G is None:
                G = nx.Graph()
                G.add_nodes_from(child.nodelist)
                G.add_edges_from(child.edgelist)
            if self.idx_to_node is None:
                self.idx_to_node = {idx: n for idx, n in enumerate(G.nodes)}
            else:
                if set(G.nodes()) != set(self.idx_to_node.values()):
                    raise ValueError(
                        "idx_to_node values are incompatible with G node labels"
                    )
                if set(range(G.number_of_nodes())) != set(self.idx_to_node.keys()):
                    raise ValueError("idx_to_node should have contiguous integer keys")
            result = schreier_rep(
                nx.relabel_nodes(G, {n: idx for idx, n in self.idx_to_node.items()}),
            )
            self.generators_u_vector = result.u_vector
        elif generators_u_vector is not None:
            if idx_to_node is None:
                raise ValueError(
                    "idx_to_node must be specified in combination with generators_u_vector"
                )
            else:
                # Could check contiguous keys and match of values to structured sampler, user errors
                pass
        else:
            pass  # Unconstrained (permutations) are assumed

    @property
    def children(self) -> list[dimod.core.Sampler]:
        try:
            return self._children
        except AttributeError:
            pass

        self._children = children = [self._child]
        return children

    @property
    def parameters(self) -> dict[str, Sequence[str]]:
        try:
            return self._parameters
        except AttributeError:
            pass

        self._parameters = parameters = dict(automorphism_variables=tuple())
        parameters.update(self._child.parameters)
        return parameters

    @property
    def properties(self) -> dict[str, Any]:
        try:
            return self._properties
        except AttributeError:
            pass

        self._properties = dict(child_properties=self._child.properties)
        return self._properties

    class _SampleSets:
        def __init__(self, samplesets: list[dimod.SampleSet]):
            self.samplesets = samplesets

        def done(self) -> bool:
            return all(ss.done() for ss in self.samplesets)

    @staticmethod
    def _reorder_variables(
        sampleset: dimod.SampleSet, order: dimod.variables.Variables
    ) -> dimod.SampleSet:
        """Return a sampleset with the given variable order."""
        if sampleset.variables == order:
            return sampleset

        # .index(...) is O(1) for dimod's Variables objects so this isn't too bad
        sampleset_order = sampleset.variables
        reorder = [sampleset_order.index(v) for v in order]

        return dimod.SampleSet.from_samples(
            (sampleset.record.sample[:, reorder], order),
            sort_labels=False,
            vartype=sampleset.vartype,
            info=sampleset.info,
            **sampleset.data_vectors,
        )

    @dimod.decorators.nonblocking_sample_method
    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        mappings: list[dict] | None = None,
        num_automorphisms: int | None = None,
        **kwargs,
    ):
        """Sample from the binary quadratic model.

        Args:
            bqm: Binary quadratic model to be sampled from.

            mappings:
                A list of mappings in the form of dictionaries.
                Each dictionary defines a permutation over a
                subset of variables. If mappings is provided and
                length 0, then a sampleset is returned subject
                to no automorphism.

            num_automorphisms:
                When mappings is not given, specifies the number of mappings to
                apply (create). If mappings is provided, it
                is inferred as :code:`len(mappings)`, otherwise it is defaulted
                to 1.
                A value of ``0`` will result in sampling of an unmapped problem.
                If mappings is None the mappings are generated randomly using the
                `generators_listtuple` class variable.

        Returns:
            A sample set. Note that for a sampler that returns ``num_reads`` samples,
            the sample set will contain ``num_reads*num_automorphisms`` samples.

        Examples:
            This example runs 10 automorphisms of a two variable QUBO problem.

            >>> from dimod import ExactSolver
            >>> from dwave.preprocessing.composites import EmbeddingComposite
            >>> base_sampler = ExactSolver()
            >>> composed_sampler = EmbeddingComposite(base_sampler)
            ...
            >>> Q = {('a', 'a'): -1, ('b', 'b'): -1, ('a', 'b'): 2}
            >>> response = composed_sampler.sample_qubo(Q,
            ...               num_automorphisms=10)
            >>> len(response)
            40
        """
        sampler = self._child

        if num_automorphisms is None:
            if mappings is not None:
                num_automorphisms = len(mappings)
            else:
                num_automorphisms = 1

        # No SRTs, so just pass the problem through
        if not num_automorphisms or not bqm.num_variables:
            sampleset = sampler.sample(bqm, **kwargs)
            # yield twice because we're using the @nonblocking_sample_method
            yield sampleset  # this one signals done()-ness
            yield sampleset  # this is the one actually used by the user
            return

        # Check or generate mappings.
        if mappings is not None:
            # Given permutation
            if len(mappings) != num_automorphisms:
                raise ValueError(
                    "len(mappings) should match num_automorphisms when not None"
                )
        elif self.generators_listtuple is not None:
            # Generator compatible (uniform random) permutation on all variables
            mapping = {v: v for v in bqm.variables}
            mappings = [
                sample_automorphisms_listtuple(
                    self.generators_listtuple, prng=self.rng, mapping=mapping
                )
                for _ in range(num_automorphisms)
            ]
        elif self.generators_u_vector is not None:
            arrays = sample_automorphisms_u_vector(
                self.generators_u_vector, num_samples=num_automorphisms
            )
            mappings = [
                {self.idx_to_node[k]: self.idx_to_node[v] for k, v in enumerate(array)}
                for array in arrays
            ]
        else:
            # Random permutation (no generator constraint) on all variables
            var_list = list(bqm.variables)
            mappings = []
            for n in range(num_automorphisms):
                self.rng.shuffle(var_list)
                mappings.append({v1: v2 for v1, v2 in zip(bqm.variables, var_list)})

        samplesets: list[dimod.SampleSet] = []
        for mapping in mappings:
            _bqm = bqm.copy()  # Costly, but assumed not to be a bottleneck
            _bqm.relabel_variables(mapping)
            samplesets.append(sampler.sample(_bqm, **kwargs))
        # Yield a view of the samplesets that reports done()-ness
        yield self._SampleSets(samplesets)

        # Relabel variables on samplesets
        for mapping, ss in zip(mappings, samplesets):
            ss.relabel_variables({v: k for k, v in mapping.items()})

        # Reorder the variables of all the returned samplesets to match our
        # original BQM
        samplesets = [self._reorder_variables(ss, bqm.variables) for ss in samplesets]

        if num_automorphisms == 1:
            # If one sampleset, return full information
            # (info returned in full)
            yield samplesets[0]
        else:
            # finally combine all samplesets together
            yield dimod.concatenate(samplesets)
