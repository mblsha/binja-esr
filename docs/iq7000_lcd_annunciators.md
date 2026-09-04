# IQ-7000 fixed LCD segments

The 96×64 matrix does not include the fixed symbols at the right of the glass.
The renderer follows the supplied physical reference: inverse BATT, CARD, EDIT,
SHIFT, CAPS, star/boxed S, single musical note/bell, up/down, left/right.
The fixed symbols use continuous vector paths, rasterized at the requested
output resolution, independently of the crisp matrix pixel grid. Their shapes
follow the reference, but their dimensions are not a physical measurement.

## Decoding and evidence

Four RAM-backed LCD bytes drive the segments. Firmware workspace bytes
`1FDA3..1FDA6` are diagnostic only: ORing them into the display would resurrect
cleared symbols before the firmware synchronizes them.

| LCD byte | Mask | Symbol | Evidence for physical name |
| --- | --- | --- | --- |
| 6160 | 80 | inverse BATT | Strong hypothesis, external-status path |
| 6160 | 40 | CARD | Strong hypothesis, card/storage state |
| 6160 | 20 | EDIT | ROM/editor UI corroborated |
| 6160 | 10 | SHIFT | ROM/keyboard UI corroborated |
| 6160 | 08 | CAPS | ROM/keyboard UI corroborated |
| 6160 | 04 | star: displayed data is secret | ROM record flag |
| 6160 | 02 | musical note: key beep | Strong hypothesis, settings |
| 6160 | 01 | up | Paired scroll-bit hypothesis |
| 6161 | 04 | boxed S: secret mode | ROM secret-state paths |
| 6161 | 02 | bell: alarm | Strong hypothesis, alarm/settings |
| 6161 | 01 | down | ROM scrolling UI corroborated; provisional glass assignment |
| 61E0 | 80 | left | Tentative paired-bit ordering |
| 61E1 | 80 | right | Tentative paired-bit ordering |

Addresses and masks above are hexadecimal. No new hardware one-hot mapping is
claimed. The reference photograph establishes appearance, not electrical bit
assignments. Captures carry `mapping_status: "rom-derived-hypothesis-v1"`;
unknown bits are reported but never assigned a symbol.

## Capture contract

- Native `--capture-png`, WASM `lcd_capture()`, and the live web canvas share
  Rust decoding and rasterization. IQ layout is 122×64 logical units:
  96 matrix columns, a 2-column gap, and a 24-column segment panel.
- Function Runner: `const frame = await e.lcd.capture()` returns
  `{ kind, cols, rows, pixel_format, pixel_scale, pixels, annunciators }`.
  IQ defaults to `pixel_scale: 4`, or 488×256 pixels, also used by the live
  canvas. `e.lcd.capture({ scale: 3 })` returns 366×192; scales 1–16 are
  supported. Save using returned geometry without scaling the image again.
- `pixel_format` is `"gray8"`: pixels are final grayscale intensities,
  0 black and 192 LCD grey, with intermediate values for antialiased segment
  edges. They are **not palette indices or booleans**. Inactive segments blend
  completely into the background; the matrix retains exact integer pixels.
  There are no host-font or image-asset dependencies.
- `annunciators` retains workspace/shadow bytes, unmapped masks,
  desynchronization, named flags, and the confidence marker.
- `lcd_pixels()` / `e.lcd.pixels()` and `lcd_geometry()` remain matrix-only
  for existing pixel analyses. The separate `lcd_annunciator_bytes()` API
  retains the four shadows.
- Reads are silent: capture does not step the CPU, invoke bus callbacks, or
  increment memory-read counters. Segment-only changes are sent to the live
  display even when matrix pixels stay unchanged.
- PC-E500 keeps its 240×32 matrix geometry without IQ symbols; full captures
  use the same black-on-grey intensities, defaulting to scale 1.

## Validation

Core tests cover each of the thirteen bits independently, bounded vector
placement, matrix preservation, clears, stale workspace state, unknown bits,
silent reads, owned captures, PNG dimensions, subpixel edges, scale validation,
and PC-E500 isolation.
Browser tests check live-worker/capture agreement. Their explicitly labelled
all-segments fixture writes test values and is not a ROM/hardware proof.
The optional real-ROM browser test uses only normal ROM execution and key
events to capture MEMO with CAPS/key beep, then with SHIFT also active.
An optional private comparison script calls the ROM's weekly-demo setup and
render routines without stubs, then checks every matrix pixel against the raw
framebuffer. It proves that demo's rendering, not navigation through the UI.
