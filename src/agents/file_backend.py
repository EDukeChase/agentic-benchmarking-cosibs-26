"""The agents' file tools, made safe for the very large data tables.

The file tools (read_file, edit_file, ...) run inside the pipeline's own process, not in the
agent's sandboxed code process. The library's `read` loads the whole file and then splits it into
lines, which takes about three times the file size in memory. One read_file call on the 6.5 GB
observations table is therefore enough to get the whole container killed at its memory limit.

Files up to the size limit behave exactly as in the library. For a bigger text file, `read` returns
the requested line window by streaming, so the same call works with a small, constant amount of
memory. Bigger non-text files and edits of bigger files get a short error that points the agent to
execute_python.
"""

from __future__ import annotations

import os
from itertools import islice
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import EditResult, FileData, ReadResult
from deepagents.backends.utils import _get_file_type

MAX_WHOLE_FILE_MB = 10


class BoundedFilesystemBackend(FilesystemBackend):
    """FilesystemBackend that never loads a file bigger than max_file_size_mb into memory."""

    def __init__(self, root_dir: str | Path | None = None, virtual_mode: bool | None = None,
                 max_file_size_mb: int = MAX_WHOLE_FILE_MB) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=virtual_mode, max_file_size_mb=max_file_size_mb)

    def _is_too_big(self, file_path: str) -> bool:
        try:
            path = self._resolve_path(file_path)
            return path.is_file() and path.stat().st_size > self.max_file_size_bytes
        except (OSError, RuntimeError, ValueError):
            return False  # let the library report the real problem (or refuse the path)

    def _too_big_message(self, file_path: str, what: str) -> str:
        limit_mb = self.max_file_size_bytes // (1024 * 1024)
        return (f"Error: '{file_path}' is larger than {limit_mb} MB, so {what}. "
                "Use execute_python to work with a file this size.")

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if not self._is_too_big(file_path):
            return super().read(file_path, offset, limit)
        if _get_file_type(file_path) != "text":
            return ReadResult(error=self._too_big_message(file_path, "it cannot be read with this tool"))
        try:
            path = self._resolve_path(file_path)
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                window = list(islice(handle, offset, offset + limit))
        except (OSError, RuntimeError, UnicodeDecodeError) as exc:
            return ReadResult(error=f"Error reading file '{file_path}': {exc}")
        if not window:
            return ReadResult(error=f"Line offset {offset} exceeds file length")
        return ReadResult(file_data=FileData(content="".join(window), encoding="utf-8"))

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if self._is_too_big(file_path):
            return EditResult(error=self._too_big_message(file_path, "it cannot be edited with this tool"))
        return super().edit(file_path, old_string, new_string, replace_all)
