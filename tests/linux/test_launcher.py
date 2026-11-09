"""No GPU, model or game required."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "tools/phoenix_linux.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_independent_game_device(self):
        original = {"VK_DRIVER_FILES": "/drivers/a.json:/drivers/b.json"}
        selected = "vulkan:" + "AB"*16
        env = launcher.make_environment(Path("/package"), Path("/config"), original, selected)
        self.assertEqual(env["PHOENIX_GAME_DEVICE"], selected.lower())
        self.assertEqual(env["VK_DRIVER_FILES"], original["VK_DRIVER_FILES"])
        self.assertNotIn("PHOENIX_GAME_DEVICE", original)
        for invalid in (None, "vulkan:" + "z"*32, "vulkan:" + "a"*30 + "  ", "cpu"):
            with self.assertRaises(ValueError):
                launcher.validate_device(invalid)

    def test_preserves_loader_configuration(self):
        original = {"VK_LAYER_PATH": "/custom", "VK_ADD_LAYER_PATH": "/extra", "VK_INSTANCE_LAYERS": "overlay:" + launcher.LAYER, "HOME": "/home/user"}
        before = original.copy()
        env = launcher.make_environment(Path("/package with spaces"), Path("/private/runtime.json"), original)
        self.assertEqual(original, before)
        self.assertEqual(env["VK_LAYER_PATH"], "/package with spaces/share/phoenix:/custom")
        self.assertEqual(env["VK_ADD_LAYER_PATH"], "/package with spaces/share/phoenix:/extra")
        self.assertEqual(env["VK_INSTANCE_LAYERS"], launcher.LAYER + ":overlay")
        self.assertEqual(env["HOME"], before["HOME"])

    def test_config_and_no_implicit_device_or_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "external model"
            model.mkdir()
            path = root / "config.json"
            config = {"backend": "vulkan", "device": "vulkan:" + "a"*32, "weight_source": {"kind": "model_folder", "path": str(model)}}
            path.write_text(json.dumps(config))
            manifest = {"backends": ["cpu"]}
            with self.assertRaisesRegex(ValueError, "not included"):
                launcher.runtime_config(root, path, manifest)
            manifest["backends"].append("vulkan")
            result = launcher.runtime_config(root, path, manifest)
            self.assertEqual(result["runtime_root"], str(root / "lib"))
            self.assertNotIn("allow_unvalidated", result)
            self.assertEqual(json.loads(path.read_text()), config)
            config["device"] = "vulkan:" + "z"*32
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "hexadecimal"):
                launcher.runtime_config(root, path, manifest)

    def test_checksum_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "lib.so"
            path.write_bytes(b"payload")
            manifest = {"sha256": {"lib.so": launcher.hashlib.sha256(b"payload").hexdigest()}}
            launcher.verify(root, manifest)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                launcher.verify(root, manifest)
            manifest = {"sha256": {"../outside": "0"*64}}
            with self.assertRaisesRegex(ValueError, "unsafe"):
                launcher.verify(root, manifest)

    def test_native_cuda_runtime_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            path = root / "config.json"
            config = {"backend": "native-cuda", "device": "cuda:" + "AB"*16,
                      "weight_source": {"kind": "model_folder", "path": str(model)}}
            path.write_text(json.dumps(config))
            result = launcher.runtime_config(root, path, {"backends": ["native-cuda"]})
            self.assertEqual(result["device"], "cuda:" + "ab"*16)
            env = launcher.make_environment(root, path, {}, "vulkan:" + "c"*32, result["backend"])
            self.assertEqual(env["PHOENIX_RUNTIME_LIBRARY"], str(root / "lib/libphoenix_native_cuda.so"))
            self.assertEqual(env["PHOENIX_GAME_DEVICE"], "vulkan:" + "c"*32)


if __name__ == "__main__":
    unittest.main()
