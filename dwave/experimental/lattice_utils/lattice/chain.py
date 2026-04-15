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

from collections.abc import Iterator

from dwave.experimental.lattice_utils.lattice import Lattice

__all__ = ['Chain']

class Chain(Lattice):

    def __init__(self, **kwargs):
        periodic: tuple[bool, ...] = kwargs.pop("periodic", (True,))
        self.geometry_name: str = "Chain"
        self.num_spins = kwargs["dimensions"][0]
        super().__init__(periodic=periodic, **kwargs)

    def generate_edges(self) -> Iterator[tuple[int, int]]:
        """Yield edges for a 1D chain lattice."""
        n = self.dimensions[0]
        for i in range(n - 1):
            yield (i, i + 1)

        if self.periodic[0] and n > 1:
            yield (n - 1, 0)
