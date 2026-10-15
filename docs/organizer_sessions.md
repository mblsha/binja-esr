# Organizer session recording

PV-F1 and OZ-9600 share **Session recording and replay** in the browser.
Select **Restart and record** to capture the current saved backing and restart
the organizer at a reproducible boundary. Use Resume or the paused step control,
use the normal device controls, and select **Add screen checkpoint** at useful
points. **Finish and export recording** releases held inputs and downloads an
`.orgsession` file. Import it with **Replay recording**; replay checks every
recorded screen and pauses when finished. Pause cancels a running replay.

The file includes the model/profile, firmware bundle SHA-256, exact WASM core
SHA-256, initial execution settings, saved RAM/RTC, and actual delivered matrix,
ON and raw pen contacts at completed scheduler boundaries. PV-F1 also includes
the mounted program or memory card, its write protection and recognition model.
Screen checkpoints hash the complete native monochrome LCD, including storage
outside the main viewport. The envelope is checksummed and bounded in size.

Replay requires the same firmware and emulator build. A checksum, model,
firmware or build failure is rejected before replacing the machine. A screen
mismatch stops at its first failing checkpoint and reports the boundary and
expected/actual hashes. Cancellation or failure releases replay-owned contacts.
Replay replaces the current machine with the recorded backing; export a backup
first if you need to keep the current session. Automatic saving then follows
the normal browser policy.

Recording begins with a fresh restart from saved backing, not a live CPU
snapshot. It records contacts after host assistance and queuing, so it reproduces
the electrical input actually delivered to the ROM. It does not preserve the
browser's original pointer timing, host focus changes, DOM state or audio output.
Debug scripts, external serial input, reset and firmware/media replacement are
blocked while recording; finish or discard before using those controls. Serial
peers and external accessories are outside this format. The file contains user
records and mounted card contents: share it with the same care as a backup.

The recorder changes no guest application queues, PC, RAM or interrupt state.
All execution and physical contacts still traverse the shared Rust runtime.
Host hold/gap assistance is a browser usability policy; these checks do not
establish physical controller timing.

PV-F1 assisted virtual matrix keys serialize distinct taps with a 200,000
boundary hold and 50,000 boundary gap. Pen assistance uses the same hold/gap
and preserves movement at 20,000 boundary intervals. Disable assisted taps for
raw virtual matrix levels. ON bypasses queued typing. Allow the firmware to
finish startup or wake recovery before app input; delivering a contact does not
guarantee that a busy ROM accepts it.
