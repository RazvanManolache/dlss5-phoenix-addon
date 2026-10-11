# Intel runtime source provenance

- Upstream: [Uzbekunknown/dlss-nr-on-intel](https://github.com/Uzbekunknown/dlss-nr-on-intel).
- Pinned import revision: `0b43a76a3b8e34ad88859dedfd2751b5028be982`.
- License: [Apache License 2.0](LICENSE), copyright 2026 uzbekunknown and
  the DLSS-NR on Intel Xe2 contributors.
- Upstream attribution, including its own dependencies: [NOTICE](NOTICE).

The 25 files under `src/gpu/` originate from the pinned revision. The
readability update applies local formatting and modification notices while
retaining executable tokens and original copyright comments.

`src/ref/nr_model.py` is modified for DLSS5 Phoenix: the safetensors disk loader,
its dtype table and the `from_safetensors` constructor are removed; callers
supply logical tensors in memory. The module description reflects that input.
The imported local file also differs in some punctuation encoding. A prominent
file header now identifies the adaptation. No executable code changed during
this publication transfer.

The complete upstream NOTICE is retained verbatim below a publication-scope
preface, replacing the shortened local integration notice. References inside
the upstream notice describe the upstream repository; omitted tools and files
are not supplied here. LICENSE is retained byte for byte.

`import_manifest.json` now records checksums of the current local input files
in `files[].sha256`, so build-time integrity checks validate this subset.
`files[].pre_readability_sha256` preserves each previous recorded digest;
the pinned upstream revision above remains the origin reference.

## Readability refresh (2026-10-10)

The current runtime sources were copied from the validated DLSS5 Phoenix
readability pass. Original license text, copyright notices and the complete
upstream NOTICE remain retained. Publication-specific adaptation headers are
preserved. No model weights, NVIDIA binaries or extraction tools were added.
