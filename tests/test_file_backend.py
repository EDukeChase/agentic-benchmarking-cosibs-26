import tempfile
import tracemalloc
import unittest
from pathlib import Path

from deepagents.backends import FilesystemBackend

from src.agents.file_backend import BoundedFilesystemBackend


def make_backend(root: str, limit_mb: int = 1) -> BoundedFilesystemBackend:
    return BoundedFilesystemBackend(root_dir=root, virtual_mode=True, max_file_size_mb=limit_mb)


class BoundedReadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_lines(self, name: str, count: int) -> None:
        with open(self.root / name, "w", encoding="utf-8") as handle:
            for i in range(count):
                handle.write(f"row{i},value{i},padding-padding-padding-padding\n")

    def test_small_file_is_identical_to_the_library(self):
        self.write_lines("small.csv", 50)
        ours = make_backend(str(self.root)).read("/small.csv", 10, 5)
        theirs = FilesystemBackend(root_dir=str(self.root), virtual_mode=True).read("/small.csv", 10, 5)
        self.assertEqual(ours, theirs)

    def test_big_file_returns_the_same_window_as_the_library(self):
        self.write_lines("big.csv", 40_000)  # about 1.9 MB, over the 1 MB limit
        ours = make_backend(str(self.root)).read("/big.csv", 120, 30)
        theirs = FilesystemBackend(root_dir=str(self.root), virtual_mode=True).read("/big.csv", 120, 30)
        self.assertIsNone(ours.error)
        self.assertEqual(ours.file_data["content"], theirs.file_data["content"])

    def test_big_file_read_uses_little_memory(self):
        self.write_lines("big.csv", 400_000)  # about 19 MB
        backend = make_backend(str(self.root))
        tracemalloc.start()
        try:
            result = backend.read("/big.csv", 0, 2000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertIsNone(result.error)
        self.assertLess(peak, 2 * 1024 * 1024)

    def test_offset_past_the_end_of_a_big_file_is_an_error(self):
        self.write_lines("big.csv", 40_000)
        result = make_backend(str(self.root)).read("/big.csv", 1_000_000, 10)
        self.assertIsNotNone(result.error)
        self.assertIn("exceeds", result.error)

    def test_big_binary_file_is_refused_with_a_pointer_to_execute_python(self):
        (self.root / "big.png").write_bytes(b"\0" * (2 * 1024 * 1024))
        result = make_backend(str(self.root)).read("/big.png")
        self.assertIn("execute_python", result.error)

    def test_big_file_cannot_be_edited(self):
        self.write_lines("big.csv", 40_000)
        result = make_backend(str(self.root)).edit("/big.csv", "row1,", "row9,")
        self.assertIn("execute_python", result.error)
        self.assertIn("row1,", (self.root / "big.csv").read_text().splitlines()[1])

    def test_small_file_can_still_be_edited(self):
        (self.root / "notes.txt").write_text("alpha beta\n")
        result = make_backend(str(self.root)).edit("/notes.txt", "alpha", "gamma")
        self.assertIsNone(result.error)
        self.assertEqual((self.root / "notes.txt").read_text(), "gamma beta\n")

    def test_missing_file_gives_the_library_error(self):
        result = make_backend(str(self.root)).read("/nope.csv")
        self.assertIn("not found", result.error)

    def test_a_path_outside_the_root_is_refused_as_before(self):
        with self.assertRaises(ValueError):
            FilesystemBackend(root_dir=str(self.root), virtual_mode=True).read("/../outside.csv")
        with self.assertRaises(ValueError):
            make_backend(str(self.root)).read("/../outside.csv")


if __name__ == "__main__":
    unittest.main()
