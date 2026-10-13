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


## Physical window readback controls

A baseline FF was followed by SET 00 under an X-excluding or partial-Y window.
Reads under those descriptors returned00. Restoring the original descriptor,
before rewriting pixels, revealed FF or C3. Reads therefore mask the pixels
outside the inclusive window, while writes preserve those backing pixels.

The extra Y-end high-byte bit1 was previously treated as a global bypass.
Physical X-exclusion and paired Y-minimum/Y-maximum controls disprove that
assumption. With source3C, both Y-outside pairs retainedFF, and every clipped
read returned00, with that bit either clear or set. Its readback is retained;
it is ignored for these ordinary clipping checks. Other possible roles remain
unqualified. The ordinary Y field stays8 low bits plus high-byte bit0.

Rust/Python regressions use those independent physical values. Component and
parity fixtures now program a full `(0,0)..(335,239)` descriptor explicitly.
The original-window control returned normally and restored its descriptor,
pixels, registers and card allocation. Timing, negative/overflow coordinates,
other controller variants and glass boundaries remain open.

## Related firmware window policies

The measured inclusive read/write contract is the shared controller's default
for OZ-9600. PV-F1, PI-3000 and PA-9500 explicitly retain
`firmware_window_candidate`: their inherited window behavior allows the extra
Y-end bit to address their working store and reads backing pixels outside the
selected write window. This remains firmware compatibility evidence, not a
physical measurement of those controllers. Geometry and earlier-window write
protection remain owned by each model.

PV-F1's native full-area descriptor keeps Y end239 with the extra bit while its
rectangle contract reaches working row288 of its292-row store. Applying the
OZ window rule directly clipped that consumer. The explicit policy restores the
native rectangle result while retaining the measured Boolean operations.

Normal PV-F1 Calendar uses mode2 and global-phase AA/55 pattern bytes to dim
its disabled Modify label. Its corrected pixels equal independently preserved
enabled-label pixels AND that mask. All325 changes lie inside the label; every
other Calendar pixel equals the prior frame. Historical frames/expectations are
preserved; updated rendering regressions retain strict application, CPU, data,
retention and exact-frame checks.
