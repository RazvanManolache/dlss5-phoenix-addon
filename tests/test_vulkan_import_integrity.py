"""Imported sources remain pinned across clean LF and CRLF checkouts."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from build_vulkan import imported_source_matches


class ImportIntegrity(unittest.TestCase):
    def test_line_endings_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"source.c"
            for original in (b"int count = 1;\n// notice\n",b"int count = 1;\r\n// notice\r\n"):
                digest=hashlib.sha256(original).hexdigest()
                for newline in (b"\n",b"\r\n"):
                    candidate=original.replace(b"\r\n",b"\n").replace(b"\n",newline)
                    path.write_bytes(candidate)
                    self.assertTrue(imported_source_matches(path,digest))
                    path.write_bytes(candidate.replace(b"1;",b"2;"))
                    self.assertFalse(imported_source_matches(path,digest))
                    path.write_bytes(candidate.replace(b"notice",b"different notice"))
                    self.assertFalse(imported_source_matches(path,digest))

    def test_binary_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"binary"
            original=b"\0\r\n"
            digest=hashlib.sha256(original).hexdigest()
            path.write_bytes(original)
            self.assertTrue(imported_source_matches(path,digest))
            path.write_bytes(b"\0\n")
            self.assertFalse(imported_source_matches(path,digest))


if __name__=="__main__": unittest.main()
