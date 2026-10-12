# OZ-9600 LCD drawing modes

On 13 October 2026, an uploaded native probe measured all eight low raster-mode
values on a physical OZ-9600 following the ROM's LH1553F controller branch.
Each test first wrote and read back destination AA, then applied source 3C to
one vertical eight-pixel group with mask FF. These patterns cover all four
per-pixel source/destination Boolean pairs.

| Mode | Operation | Result |
| --- | --- | --- |
| 0 | SET source | 3C |
| 1 | destination XOR source | 96 |
| 2 | destination AND source | 28 |
| 3 | destination OR source | BE |
| 4 | SET inverted source | C3 |
| 5 | destination XOR inverted source | 69 |
| 6 | destination AND inverted source | 82 |
| 7 | destination OR inverted source | EB |

Bit 2 inverts the source before the operation. The shared Rust controller and
Python reference now implement these measured rules. The previous mode-2 OR
and combined-result inversion assumptions disagreed with modes 2, 6 and 7.

The probe returned normally and restored the tested pixels, LCD controls,
coordinates and its card allocation. One interior vertical group does not
qualify clipping, other orientations, busy timing or other controller variants.
The core regression uses the physical result table; existing real-ROM bitmap
and highlight tests retain their own mask and stack checks. Schedule's cache
failure remains a separate unresolved issue.
