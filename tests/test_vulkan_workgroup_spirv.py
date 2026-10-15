"""Approximate shaders may use FP16, but never acquire subgroup dependencies."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from generate_standalone_vulkan import check_workgroup_shader


class ShaderCapabilities(unittest.TestCase):
    def check(self, capability, approximate=False):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'shader.spv'
            path.write_bytes(struct.pack('<8I',0x07230203,0x10600,0,1,0,(2<<16)|17,capability,0))
            # Remove the unused final word; the scanner checks full instruction lengths.
            path.write_bytes(path.read_bytes()[:-4])
            check_workgroup_shader(path,allow_half=approximate)

    def test_half_is_explicit_opt_in(self):
        with self.assertRaises(RuntimeError):self.check(9)
        self.check(9,approximate=True)

    def test_subgroups_rejected_in_both_modes(self):
        for capability in range(61,69):
            for approximate in [False,True]:
                with self.assertRaises(RuntimeError):self.check(capability,approximate)

    def test_int16_arithmetic_stays_rejected(self):
        for approximate in [False,True]:
            with self.assertRaises(RuntimeError):self.check(22,approximate)


if __name__=='__main__':unittest.main()
