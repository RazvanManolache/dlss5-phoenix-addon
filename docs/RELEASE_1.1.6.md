# Phoenix DLSS5 1.1.6

- Fixed a rendering rejection when Phoenix appearance was combined with super resolution or a custom working scale. Ordinary style and intensity composition now runs on the neural canvas before reconstruction.
- Kept Normal and Super resolution usable with custom color controls. Resolution and quality buttons update the active working scale without clearing color settings.
- Renamed the Image section to **Color and scaling** and its switch to **Custom color and scaling**. The panel shows the working scale and offers spatial reconstruction while custom color processing is active.
- Added rendering and UI regressions for these combinations on both addon architectures.

Existing configuration keys remain compatible. Model weights are separate downloads. Engine-supplied UI, control and distortion maps retain their matching-dimensions contract.

[Validation scope](VALIDATION_1.1.6.md). This is a focused scaling and controls fix; it does not claim a general FP8 temporal-stability improvement.
