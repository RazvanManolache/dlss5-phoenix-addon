"""Production catalogue consistency and invalid-input checks; no GPU required."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "production_builder", ROOT / "tools/production_build_native_kernels.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ProductionInputsTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = json.loads((ROOT / builder.CATALOGUE).read_text())
        self.graph = json.loads((ROOT / builder.GRAPH).read_text())

    def validate(self, catalogue=None, graph=None, *, check_sources=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / builder.MODEL).mkdir(parents=True)
            (root / builder.CATALOGUE).write_text(
                json.dumps(self.catalogue if catalogue is None else catalogue)
            )
            (root / builder.GRAPH).write_text(json.dumps(self.graph if graph is None else graph))
            return builder.production_inputs(root, check_sources=check_sources)

    def test_production_interface_is_complete(self):
        catalogue, graph = self.validate()
        self.assertEqual(len(catalogue["kernels"]), 41)
        self.assertEqual(len(graph["layers"]), 154)
        self.assertEqual(len(graph["tensors"]), 201)

    def test_unknown_graph_kernel_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        graph["layers"][0]["kernel"] = "missing"
        with self.assertRaisesRegex(ValueError, "graph and kernel catalogue differ"):
            self.validate(graph=graph)

    def test_duplicate_kernel_is_rejected(self):
        catalogue = copy.deepcopy(self.catalogue)
        catalogue["kernels"][1]["name"] = catalogue["kernels"][0]["name"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate(catalogue=catalogue)

    def test_source_escape_is_rejected(self):
        catalogue = copy.deepcopy(self.catalogue)
        catalogue["kernels"][0]["source"] = "../outside.cu"
        with self.assertRaisesRegex(ValueError, "inside the production kernel folder"):
            self.validate(catalogue=catalogue)

    def test_missing_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing production source"):
            self.validate(check_sources=True)

    def test_invalid_thread_block_is_rejected(self):
        catalogue = copy.deepcopy(self.catalogue)
        catalogue["kernels"][0]["block"] = [1024, 2, 1]
        with self.assertRaisesRegex(ValueError, "thread block"):
            self.validate(catalogue=catalogue)

    def test_invalid_assembler_optimization_is_rejected(self):
        for value in (True, 0, 4, "1"):
            with self.subTest(value=value):
                catalogue = copy.deepcopy(self.catalogue)
                catalogue["kernels"][0]["assembler_optimization"] = value
                with self.assertRaisesRegex(ValueError, "assembler optimization"):
                    self.validate(catalogue=catalogue)

    def test_ptx_build_inputs_and_unsupported_options_are_rejected(self):
        catalogue = copy.deepcopy(self.catalogue)
        catalogue["kernels"][0]["source"] = catalogue["kernels"][0]["source"][:-3] + ".ptx"
        with self.assertRaisesRegex(ValueError, "production kernel folder"):
            self.validate(catalogue=catalogue)
        catalogue = copy.deepcopy(self.catalogue)
        catalogue["kernels"][0]["ptx_options"] = {"texture_mode": "unified", "max_registers": 128}
        with self.assertRaisesRegex(ValueError, "PTX options"):
            self.validate(catalogue=catalogue)

    def test_generated_boundary_ptx_keeps_the_sampler_mode_and_register_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kernel.ptx"
            source = ".version 9.1\n.target sm_120\n.visible .entry y34(\n.param .align 8 .b8 args[264]\n)\n.maxntid 32\n{ ret; }\n"
            options = {"texture_mode": "independent", "max_registers": 128}
            path.write_text(source)
            builder.configure_boundary_ptx(path, "sm_120", "y34", options)
            result = path.read_text()
            self.assertIn(".target sm_120, texmode_independent", result)
            self.assertIn(".maxntid 32\n.maxnreg 128\n", result)
            self.assertIn("args[264]", result)
            for invalid in (
                source.replace("y34", "other"),
                source + source,
                source.replace("sm_120", "sm_90"),
            ):
                path.write_text(invalid)
                with self.assertRaisesRegex(ValueError, "Unexpected generated"):
                    builder.configure_boundary_ptx(path, "sm_120", "y34", options)

    def test_build_keeps_production_flags_and_does_not_claim_inference_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / builder.MODEL).mkdir(parents=True)
            (root / builder.CATALOGUE).write_text(json.dumps(self.catalogue))
            (root / builder.GRAPH).write_text(json.dumps(self.graph))
            compiler = root / "toolchain/nvcc.exe"
            compiler.parent.mkdir()
            compiler.touch()
            compiler.with_name("ptxas.exe").touch()
            for row in self.catalogue["kernels"]:
                source = root / row["source"]
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("test source")
            commands = []

            def simulate_compilation(command, **kwargs):
                if command[0] == "cmd":
                    recipe = Path(command[-1])
                    commands.append(recipe.read_text())
                    if "-ptx " in commands[-1]:
                        entry = re.search(r"--entries (\w+)", commands[-1]).group(1)
                        (recipe.parent / "kernel.ptx").write_text(
                            ".version 9.1\n.target sm_120\n.visible .entry "
                            + entry
                            + "()\n.maxntid 32\n{ret;}\n"
                        )
                    else:
                        (recipe.parent / "kernel.cubin").write_bytes(b"test module")
                else:
                    commands.append(" ".join(command))
                    Path(command[command.index("-o") + 1]).write_bytes(b"test module")

            with (
                patch.object(builder, "ROOT", root),
                patch("sys.argv", ["builder", "test", "--compiler", str(compiler)]),
                patch.object(builder.subprocess, "check_output", return_value="C:/VS"),
                patch.object(builder.subprocess, "run", side_effect=simulate_compilation),
                patch("builtins.print"),
            ):
                builder.main()
            manifest = json.loads((root / "build/test/native_manifest.json").read_text())
            self.assertTrue(manifest["passed"])
            self.assertEqual(len(manifest["kernels"]), 41)
            self.assertEqual(len(commands), 43)
            self.assertTrue(all("-O3" in command for command in commands))
            lower_optimization = sum(row.get("assembler_optimization", 3) == 1
                                     for row in self.catalogue["kernels"])
            self.assertEqual(sum("-Xptxas=-v,-O3" in command for command in commands),
                             39 - lower_optimization)
            self.assertEqual(sum("-Xptxas=-v,-O1" in command for command in commands),
                             lower_optimization)
            self.assertEqual(sum("--suppress-debug-info" in command for command in commands), 41)
            self.assertEqual(sum("-ptx " in command for command in commands), 2)
            for generated in (root / "build/test").rglob("kernel.ptx"):
                self.assertIn("texmode_independent", generated.read_text())
                self.assertIn(".maxnreg 128", generated.read_text())
            self.assertEqual(manifest["selection_status"], "compiled")
            self.assertTrue(manifest["validation_pending"])
            self.assertFalse(manifest["model_weights_included"])
            self.assertEqual(
                json.loads((root / "build/test/native_graph.json").read_text()), self.graph
            )


if __name__ == "__main__":
    unittest.main()
