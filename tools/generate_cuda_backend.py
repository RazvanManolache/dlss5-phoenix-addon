"""Generate the CUDA specialization of the shared HIP/CUDA graph backend."""
import argparse
from pathlib import Path


def generate():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / "native_runtime/backend_gpu.cpp").read_text()
    # Match the established Windows portable-backend specialization exactly.
    source = source.replace("hip", "cuda").replace("cudaUUID uuid", "cudaUUID_t uuid")
    prefix = '#define PHX_CUDA\n#define DEVICE_NAME "cuda-cpp"\n#define DEVICE_PREFIX "cuda:"\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prefix + source)


if __name__ == "__main__":
    generate()
