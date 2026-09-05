"""Bounded process-local cache for dataset metadata and read-only mappings."""

import os
from collections import OrderedDict
from pathlib import Path
from typing import TypeVar

import numpy as np

_Key = TypeVar("_Key")
_Array = TypeVar("_Array", bound=np.ndarray)
_Signature = tuple[int, int, int, int]


class DatasetFileCache:
    """Cache NumPy file objects without carrying live handles across processes."""

    def __init__(self, max_arrays: int = 128, max_maps: int = 32) -> None:
        """Initialize empty caches with independently bounded entry counts."""
        if max_arrays < 0 or max_maps < 0:
            msg = "Dataset file cache limits must be non-negative."
            raise ValueError(msg)
        self.max_arrays = max_arrays
        self.max_maps = max_maps
        self._initialize()

    def _initialize(self) -> None:
        self._pid = os.getpid()
        self._arrays: OrderedDict[tuple[str, bool], tuple[_Signature, np.ndarray]] = (
            OrderedDict()
        )
        self._maps: OrderedDict[
            tuple[str, str, tuple[int, ...]], tuple[_Signature, np.memmap],
        ] = OrderedDict()

    def _ensure_process(self) -> None:
        if self._pid != os.getpid():
            self._initialize()

    @staticmethod
    def _path_and_signature(path: Path) -> tuple[str, _Signature]:
        absolute_path = str(path.expanduser().resolve())
        stat = Path(absolute_path).stat()
        return absolute_path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    @staticmethod
    def _lookup(
        entries: OrderedDict[_Key, tuple[_Signature, _Array]],
        key: _Key,
        signature: _Signature,
    ) -> _Array | None:
        entry = entries.get(key)
        if entry is None:
            return None
        cached_signature, value = entry
        if cached_signature != signature:
            del entries[key]
            return None
        entries.move_to_end(key)
        return value

    @staticmethod
    def _store(
        entries: OrderedDict[_Key, tuple[_Signature, _Array]],
        key: _Key,
        signature: _Signature,
        value: _Array,
        limit: int,
    ) -> _Array:
        entries[key] = (signature, value)
        entries.move_to_end(key)
        while len(entries) > limit:
            entries.popitem(last=False)
        return value

    def load_array(self, path: Path, *, allow_pickle: bool) -> np.ndarray:
        """Load or reuse an array while its backing-file signature is unchanged."""
        self._ensure_process()
        absolute_path, signature = self._path_and_signature(path)
        key = (absolute_path, allow_pickle)
        cached = self._lookup(self._arrays, key, signature)
        if cached is not None:
            return cached
        value = np.load(absolute_path, allow_pickle=allow_pickle)
        return self._store(self._arrays, key, signature, value, self.max_arrays)

    def open_map(self, path: Path, *, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        """Open or reuse a read-only map for a path, dtype, and shape."""
        self._ensure_process()
        absolute_path, signature = self._path_and_signature(path)
        normalized_dtype = np.dtype(dtype).str
        normalized_shape = tuple(shape)
        key = (absolute_path, normalized_dtype, normalized_shape)
        cached = self._lookup(self._maps, key, signature)
        if cached is not None:
            return cached
        value = np.memmap(absolute_path, dtype=dtype, mode="r", shape=normalized_shape)
        return self._store(self._maps, key, signature, value, self.max_maps)

    def clear(self) -> None:
        """Release the cache's references without closing exposed objects."""
        self._arrays.clear()
        self._maps.clear()

    def __getstate__(self) -> dict[str, int]:
        """Serialize limits without live arrays or memory maps."""
        return {"max_arrays": self.max_arrays, "max_maps": self.max_maps}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Restore limits into empty caches owned by the current process."""
        self.max_arrays = state["max_arrays"]
        self.max_maps = state["max_maps"]
        self._initialize()
