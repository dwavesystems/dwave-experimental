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

import hashlib
from numbers import Integral

import dimod
import numpy as np

from dwave.experimental.lattice_utils.experiment.experiment import Experiment

__all__ = ["SpinGlassExperiment"]


class SpinGlassExperiment(Experiment):
    """Experiment that cycles deterministic spin-glass disorder instances."""

    def _make_logical_bqms(self) -> list[dimod.BQM]:
        """Make one seeded spin-glass BQM per QPU embedding."""
        num_instances = self.param["num_random_instances"]
        if (
            isinstance(num_instances, bool)
            or not isinstance(num_instances, Integral)
            or num_instances < 1
        ):
            raise ValueError("num_random_instances must be a positive integer.")

        num_embeddings = (
            len(self.lattice.embedding_list) if hasattr(self.lattice, "embedding_list") else 1
        )

        bqms = []
        for embedding_index in range(num_embeddings):
            seed = num_embeddings * (self.run_index % num_instances) + embedding_index
            bqm = self.lattice.make_bqm(seed=seed)
            bqms.append(bqm)

        return bqms

    def _get_relative_data_path(self) -> str:
        """Make a subdirectory name for a sampler call's data."""
        base = super()._get_relative_data_path()
        embeddings = getattr(self.lattice, "embedding_list", None)
        if embeddings is None:
            embedding_id = "unembedded"
            num_embeddings = 1
        else:
            embeddings_array = np.asarray(embeddings, dtype="<i8")
            num_embeddings = len(embeddings_array)
            hash_input = str(embeddings_array.shape).encode("ascii") + embeddings_array.tobytes()
            digest = hashlib.sha256(hash_input).hexdigest()[:16]
            embedding_id = f"emb{num_embeddings}_{digest}"

        return f"{base}/instances{self.param['num_random_instances']}/{embedding_id}"
