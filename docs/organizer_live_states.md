# PV-F1 live states and records

In **Saved records**, **Save live state** pauses the machine and downloads a
`.pvstate` file. **Restore live state** resumes from the same paused instruction
boundary, screen, in-progress editor and queued input. Use Resume or the paused
step controls to continue. The matching firmware and exact emulator WASM build
are required. Export a normal backup as well for migration between builds.

The state contains CPU and legacy-card execution, RAM, LCD, RTC/clock phases,
interrupt/timer state, UART queues and framing, audio capture, controller and
recognition state, mounted card backing, learned templates and host input
queues. Immutable firmware is verified and loaded separately; the recognition
model travels inside the file. Restore constructs and validates a new machine
before replacing the active machine. Invalid files leave it intact. Active
recording and native tracing/debug-capture operations must finish first.

This differs from a `.pvbackup`, which preserves durable backing and starts a
fresh CPU, and an `.orgsession`, which replays delivered contacts from a fresh
restart. A live state does not reproduce the browser's DOM focus or already
played speaker output. Browser audio/serial delivery epochs restart on restore.
It contains private records and mounted media, just like a backup.

The checked `PVSTATE1` envelope stores browser settings and input ownership,
firmware/WASM identities, model bytes and the native `PVLIVE01` payload. Both
layers use SHA-256 and size/extent checks. Native state uses compressed JSON;
its version and private Rust layout are not a cross-version interchange ABI.

## Record explorer

Open **Record explorer and transfer**, then load a native CSV transfer or a
lossless JSON export. **Read serial capture** opens bytes already received by
the emulator's serial console. Search examines every row and field, with
Unicode width normalization; the table shows the first 100 matching rows.
Downloads always include all rows, regardless of the current search.

Contacts and business cards can export vCard 4, Schedule can export iCalendar,
and Memo can export UTF-8 text. Native CSV and lossless JSON preserve the
original CP932 bytes, unknown columns and fields. JSON also contains a readable
view and an integrity hash; edited readable text is rejected rather than
silently ignored. Modern text formats are convenient projections and cannot
replace the lossless export. Calendar times stay local/floating, and all-day
dates stay dates; no timezone or completion-time information is invented.
The mappings follow [vCard 4](https://www.rfc-editor.org/rfc/rfc6350) and
[iCalendar](https://www.rfc-editor.org/rfc/rfc5545).

To import, open **Receive** in the matching native application and start it on
the organizer display. Select a complete native CSV or lossless JSON file,
then **Import to Receive**. The bridge streams the original bytes through the
emulated UART and firmware parser, respecting native flow control. Completion
means the ROM closed Receive after the whole extent was queued; inspect the
native display for application-level results. Failure or cancellation restores
the complete pre-import machine. No host code writes application records or
replaces firmware file services. Keep an exported backup when replacing data.

## Browser compatibility checks

`PCE500_E2E_ENGINES=chromium,firefox,webkit` selects focused Playwright projects;
the default remains Chromium. Tests use one shared emulator build. These are
engine integration checks, not physical Safari/mobile device or hardware
fidelity measurements. `PCE500_E2E_HEADED=1` permits a visible diagnostic launch.
