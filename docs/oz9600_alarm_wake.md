# OZ-9600 external alarm wake experiment

`experimental-external-wake` adds automatic alarm wake to the v1 execution
basis. An enabled external interrupt (`ISR.EXI`, `IMR.EXM`) can leave OFF through
the ordinary CPU wake path. The RTC and gate-array masks still control whether
that input is asserted. The physical ON contact, SSR polling and ONKI latch
remain separate. The ROM handles the notification and subsequent dismissal.

Select **OZ-9600 v1 + alarm wake experiment** before loading a ROM bundle in the
browser. The native runner accepts the same profile:

```sh
cargo build --release --locked --manifest-path sc62015/native-window/Cargo.toml
sc62015/native-window/target/release/oz9600-window \
  --rom /absolute/path/verified.ozrom \
  --profile experimental-external-wake \
  --state /absolute/path/alarm.ozsession
```

The core CLI accepts `--oz9600-profile experimental-external-wake`. Browser and
native session files record this policy. A full session made with another
profile is rejected before replacement; to start a new CPU under this experiment,
export its RAM/RTC battery image and load that image with the new profile.
Keep the original full session for recovery of unfinished work.

The experiment inherits v1's initial BP=D0 assumption for empty SRAM, advancing
RTC, ON-edge and architectural interrupt-mask policies. Its nominal RTC rate
is 1,024,000 scheduler timing units per second. These remain provisional hardware
assumptions. Strict and earlier profiles retain their previous OFF behavior;
`provisional-v1` remains the browser default.

## Evidence and limits

Synthetic regression tests exercise both batch and cooperative execution.
Disabling the RTC, gate-array or CPU external mask keeps the machine OFF.
Wake consumes an idle boundary, then normal execution delivers EXI. No physical
ON contact or ONKI bit is created by the alarm.

Private verified-ROM checks reach the Daily Alarm notification automatically,
retain it in the wait control, and dismiss it with ordinary ON input. Production
browser trials also recover Welcome and the preceding Calendar application.
Browser exports restore identical controller pixels and RAM/RTC through both
WASM and the native frontend. Fresh readers recover the stored organizer records.
These are software checks with an explicit wake hypothesis, rather than a
measurement of ESR-P electrical behavior.

The firmware preserves EXM and RTC gate selection when preparing OFF and has
an external-cause recovery path. This motivates the experiment; it does not
establish the complete silicon wake contract. Physical signal polarity, latency,
RTC register semantics and clock calibration still need qualification.

The Schedule cache race during RTC callbacks and missed quick tablet contacts
remain open. The wake experiment does not repair those paths. A short versus
long Clock contact still changes navigation in a preserved mixed-record trial.
The existing input and recovery controls remain necessary when investigating it.
