"""Formatting must not disable a required build adaptation."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "source_patches", Path(__file__).resolve().parents[1] / "tools/source_patches.py"
)
patches = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patches)


class SourcePatchTests(unittest.TestCase):
    def test_formatted_condition_and_comment_preserve_unique_match(self):
        source = "// if (!ready) return 0;\nif (!ready)\n    return 0;\n"
        result = patches.replace_code(
            source, "if (!ready) return 0;", "if (!ready) return 1;", expected=1
        )
        self.assertIn("// if (!ready) return 0;", result)
        self.assertIn("if (!ready) return 1;", result)

    def test_call_wrapping_and_trailing_comma(self):
        source = "    result = call(\n        left,\n        right,\n    )\n"
        result = patches.replace_code(
            source, "result = call(left,right)", "result = other(left,right)", expected=1
        )
        self.assertEqual(result, "    result = other(left,right)\n")

    def test_string_contents_are_not_code(self):
        source = 'const char* label = "if (!ready) return 0;";'
        self.assertEqual(patches.code_count(source, "if (!ready) return 0;"), 0)

    def test_python_floor_division_is_part_of_anchor(self):
        source = "hidden = byte_count // 2 // channels\n"
        result = patches.replace_code(
            source,
            "hidden = byte_count // 2 // channels",
            "hidden = explicit_width",
            expected=1,
        )
        self.assertEqual(result, "hidden = explicit_width\n")

    def test_python_comments_do_not_match(self):
        source = "# hidden = byte_count // 2\nhidden = byte_count // 2\n"
        self.assertEqual(patches.code_count(source, "hidden = byte_count // 2"), 1)

    def test_missing_and_ambiguous_matches_fail(self):
        for source in ("unrelated();", "call(); call();"):
            with self.assertRaises(RuntimeError):
                patches.replace_code(source, "call();", "replacement();", expected=1)


if __name__ == "__main__":
    unittest.main()
