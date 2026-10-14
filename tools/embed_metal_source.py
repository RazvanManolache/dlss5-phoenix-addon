"""Embed generated shader text so a standalone Metal runner has no shader folder."""
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    delimiter = "PHOENIX_MSL"
    if f'){delimiter}"' in text:
        raise ValueError("Shader contains embedding delimiter")
    args.output.write_text(
        '#pragma once\nstatic constexpr char phoenix_metal_source[] = R"'
        + delimiter + '(' + text + ')' + delimiter + '";\n', encoding="utf-8")
