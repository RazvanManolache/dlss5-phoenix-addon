# A stronger photorealistic makeover

[User guide](USER_GUIDE.md) · [Color and processing chain](PROCESSING_CHAIN.md)

For a stronger transformation, start with this recipe, then compare the same
frame with Input. It was selected by inspecting seven recipes with both
Original and Lightweight FP8, through both Phoenix and NVIDIA: 28 completed
outputs from one fixed SDR scene. It is a starting point, not a universal best
setting or a motion-stability result. A second comparison then checked 16 color
and scaling recipes with each model/output combination: 64 more completed
outputs at 100%, 67%, 50% and 150% scale, with appearance held fixed.

| Control | Starting value |
| --- | --- |
| Neural model | Original |
| Appearance overrides | On |
| Overall intensity | 1 |
| Artistic style | Natural |
| Override strengths | On |
| Local tone / Local structure | 2 / 2 |
| Override skin and guides / Automatic skin mask | On / On |
| Skin structure | 2 |
| Resolution | Normal / 100% |
| Network passes / Transformer blocks / Per-block repetitions | 1 / 8 / 1 each |
| Custom input encoding / NR signal encoding | Auto / sRGB |
| Exposure multiplier / Use exposure value | 1 / On |
| Diffuse white | Automatic |

Original with Natural style gave the stronger facial and lighting change in
the inspected scene. Default retained more of the original warm lighting;
Cinematic is a different grade, not a higher-quality network. FP8 was more
restrained on this input and should be compared separately.

If skin becomes too exaggerated, first lower structure and skin strength
toward 1.5 and 1.3. Those are tuning directions, not guaranteed improvements.
Keep tone higher if the stronger lighting change is the part you prefer.
Intensity already reaches full blend at 1; setting it to 2 does not add more.

For maximum source detail, keep 100% processing. Resampling filters have little
to do at equal dimensions. If you need Super resolution, start with Quality,
Area downsampling and Bicubic upsampling; use Bilinear if Bicubic rings around
hair or text. RGB residual reconstruction retains full-resolution source
texture. Processed-image reconstruction carries more of the makeover but can
soften detail at reduced scales. In the fixed-frame scaling comparison, 50%
processed-image reconstruction was visibly softer than RGB residual. Area
avoided the strongest aliasing from nearest-neighbor reduction. Linear filtering
with Area/Bicubic and RGB residual is a useful reduced-scale starting point;
100% remained the preferred reference for the stronger detailed makeover.

Input encoding describes the actual source. Leave Auto on unless you know the
metadata is wrong; choosing HDR10 for SDR is incorrect color interpretation.
The 0.7x and 1.3x exposure trials changed the look but did not give a clearly
better balance than 1x in the inspected scene. Keep exposure neutral while
judging tone and structure. In the additional signal comparisons, PQ SDR
exaggerated facial contrast and texture; Linear SDR introduced visible color
shifts. sRGB was the better starting point on this SDR input. This signal
choice is separate from the scaling filter domain. Linear-light filtering is
a useful reference for resizing; it is not an extra makeover strength.

Use valid motion for moving scenes, leave forced per-frame/periodic resets off,
and keep completed-frame protection on for helper routes. Judge hair, eyes,
thin geometry and camera cuts in motion before saving a per-game preset.
