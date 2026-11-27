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

from collections.abc import Generator
import io
import sqlite3
import zlib
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@contextmanager
def open_database(path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite database connection with initialization for embedding sets.

    Args:
        path: Path to the SQLite database file.

    Yields:
        sqlite3.Connection: An active database connection.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=5.0)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        with connection:
            if version == 0:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"Unsupported database version: {version}")
            if version == 0:
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
                if tables.fetchone() is not None:
                    raise ValueError("Refusing to initialize an unrelated database.")
                connection.execute(
                    "CREATE TABLE embedding_sets ("
                    "embedding_set_id INTEGER PRIMARY KEY, "
                    "lattice_key TEXT NOT NULL, "
                    "geometry_name TEXT NOT NULL, "
                    "dimensions TEXT NOT NULL, "
                    "periodic TEXT NOT NULL, "
                    "sampler_name TEXT, "
                    "exclude_qubits TEXT NOT NULL DEFAULT '[]', "
                    "num_embeddings INTEGER NOT NULL CHECK (num_embeddings > 0), "
                    "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                    "data BLOB NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX embedding_sets_by_lattice "
                    "ON embedding_sets (lattice_key)"
                )
                connection.execute("PRAGMA user_version = 1")
        with connection:
            yield connection

def encode_array(array: NDArray[np.generic]) -> bytes:
    """Encode a NumPy array as a compressed byte string.

    Args:
        array: The NumPy array to encode.

    Returns:
        bytes: The compressed byte representation of the array.
    """
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return zlib.compress(buffer.getvalue())

def decode_array(compressed_data: bytes) -> NDArray[np.generic]:
    """Decode a compressed byte string back into a NumPy array.

    Args:
        compressed_data: The compressed byte representation of the array.

    Returns:
        NDArray[np.generic]: The decoded NumPy array.
    """
    return np.load(io.BytesIO(zlib.decompress(compressed_data)), allow_pickle=False)
