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

from collections.abc import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

def bootstrap(
    array: NDArray,
    repetitions: int = 200,
    bootstrap_function: Callable[[NDArray], float] = np.nanmedian,
    seed: int | None = None,
    skipnan: bool = True,
) -> list[float]:
    """Compute bootstrap estimates of a statistic."""
    array = np.asarray(np.atleast_1d(array)).ravel()
    if skipnan:
        array = array[~np.isnan(array)]
        if len(array) == 0:
            return [np.nan] * repetitions

    output = []
    if len(array) > 0:
        for inds in generate_bootstrap_indices(array.size, repetitions, seed=seed):
            output.append(bootstrap_function(array[inds]))

    return output


def generate_bootstrap_indices(
    size: int,
    repetitions: int,
    seed: int | None = None,
) -> Iterator[NDArray]:
    """Generate resampled indices."""
    np.random.seed(seed)
    for _ in range(repetitions):
        inds = np.random.choice(range(size), replace=True, size=size)
        yield inds


def confidence_interval(array: NDArray, width: float = 0.95) -> tuple[float, float, float]:
    """Ravel and take the quantiles; return median and error bar lengths."""
    x = np.asarray(array).ravel()
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    x.sort()
    low = x[int(np.floor((1 - width) / 2 * x.size))]
    high = x[int(np.floor((1 - (1 - width) / 2) * x.size))]
    med = np.median(x)

    return med, med - low, high - med
