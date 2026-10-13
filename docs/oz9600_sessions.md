# OZ-9600 session and recovery files

Native `--state` and browser automatic saving preserve the running CPU,
interrupt/timer state, keyboard/UART queues, RAM/RTC, tablet conversion state,
GPIO/mapper controls and the complete 336 × 240 controller image. Reopening
resumes the saved scheduler boundary, including unfinished edits. Frontends
release restored physical contacts through normal input transitions because
old host input owners no longer exist. Queued input and latched IRQ/ADC state
remain available to the ROM.

Use **Export full session** in the browser for a portable `.ozsession` file.
Native `--state /path/work.ozsession` reads and writes that same format. Native
`--retained` imports either full sessions or legacy `.ozbat` RAM/RTC images.
Browser **Import backup** identifies the format from its bytes. RAM/RTC backups
retain their existing interface and start a fresh CPU; they recover stored
records. A healthy Reset starts a fresh CPU and discards unfinished editor
context. Reset after a runtime fault restores the last committed checkpoint,
including its saved editor context when a full session is available.

Browser storage commits the RAM/RTC image and full session in one IndexedDB
transaction. The pair must agree before loading. Both frontends construct and
validate a fresh candidate before replacing the current runtime. Failed writes
keep the last complete file/transaction; failed imports keep the current runtime.
Missing storage denotes first launch. Present empty or damaged files block
startup and remain available for recovery/export.

The browser exposes rejected RAM and session files even when startup fails.
Import a known-good backup to recover. Native failed startup shows Retry (F5),
Copy original (F8) and Close (Escape). Copy uses a distinct recovery filename.
Headless failures return immediately. Cooperating browser tabs/native processes
exclude a second writer. Sudden termination can lose work since the last save.

## Envelope, version 1

| Offset | Size | Content |
| --- | --- | --- |
| 0 | 8 | ASCII `OZRUN01` followed by NUL |
| 8 | 64 | Same fixed-ROM/bank identity as the RAM/RTC image |
| 72 | 32 | SHA-256 of the compressed payload |
| 104 | remainder | One gzip member containing schema-1 typed JSON |

Encoded size is bounded at 2 MiB; decoded JSON at 8 MiB. Restore validates
lengths, hashes, schema, CPU/profile, timers, keyboard, UART, tablet and LCD
layout before mutation. Immutable firmware is reconstructed from the verified
bundle. Process-local call-stack stamps and bank-view caches are reconstructed.
The Python envelope codec provides independent transport validation; typed
runtime restoration is implemented by the Rust core.

Full sessions currently support the built-in organizer. Card state, external
host callbacks, active ROM Function Runner calls and failed runtimes require
their existing separate backups/recovery controls. Physical reset/clock/IRQ
accuracy and automatic alarm wake remain provisional. Saved-session restoration
continues prior execution; it does not establish a hardware cold-boot result.

## CPU-contract update, 11 October 2026

Physical OZ-9600 probes through ordinary stock-OZ-707 BASIC CALLs established
software setting of every ISR source in the tested direct, BP-relative and OR
stores. The device now uses replacement ISR writes, low-byte ADD/SUB sources
and coupled external predecrement block transfers independently of its reset
or clock profile. The explicit `experimental-isr-clear-only` and
`experimental-isr-mti-writable` profiles remain available for historical
comparisons. Other machine models retain their existing defaults.

Older v1, ON-edge and advancing-RTC full sessions automatically upgrade their
MTI-only ISR implementation option to replacement. Import preserves all guest
registers, pending latches, memory, peripherals and the saved scheduler boundary;
only this implementation option changes. The explicit ISR experiment and other
clock/wake/profile mismatches still reject before replacing the live runtime.
RAM/RTC backup format and identity checks are unchanged.

These measurements do not establish power-on BP, instruction timing, nested
interrupt acceptance or alarm wake. The Schedule cache-eviction counterexample
still reproduces with the corrected ISR store rule.

Legacy `OZRUN01` sessions saved before configurable LCD geometry retain their
original336×240 backing and disabled ordered-window experiment when those
metadata fields are absent. Explicit geometry is validated; an invalid shape
rejects the candidate without replacing a live session. This format migration
preserves CPU/RAM/RTC and does not change guest execution or hardware defaults.
