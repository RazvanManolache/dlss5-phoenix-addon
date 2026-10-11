# AMD shader interface provenance

- Upstream: [mochizuki0323/DLSSNR-AMD](https://github.com/mochizuki0323/DLSSNR-AMD).
- Pinned import revision: `9e4574e1812d7d0632d6db02721a20c9617be434`.
- License: [MIT](LICENSE), copyright (c) 2026 mochizuki0323.
- Attribution and integration scope: [NOTICE](NOTICE).

The two files under `windows/shaders/rdna4/include/` originate from this
revision. The readability update applies local formatting and a modification
notice; executable tokens and original copyright comments are retained.
The upstream host application, model files and binaries are not included.

`import_manifest.json` now records checksums of the current local input files
in `files[].sha256`, so build-time integrity checks validate this subset.
`files[].pre_readability_sha256` preserves each previous recorded digest;
the pinned upstream revision above remains the origin reference.

## Readability refresh (2026-10-10)

The current runtime sources were copied from the validated DLSS5 Phoenix
readability pass. Original license text, copyright notices and the complete
upstream NOTICE remain retained. Publication-specific adaptation headers are
preserved. No model weights, NVIDIA binaries or extraction tools were added.
