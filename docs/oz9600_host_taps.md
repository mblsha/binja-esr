# OZ-9600 host tap serialization

The browser and native runner retain complete quick clicks while the verified
ROM still has an earlier tablet event pending. Previously, a second click could
reach the ADC correctly and still be skipped by the ROM's occupied-event-slot
branch. In the qualified To Do case, that left `revised taskalpha task` after
Clear Field was missed; the queued click produces `revised task`.

The read-only helper follows the current software-context pointer at `1FF47`
and checks the C8/C9/CA records at offsets 00/0B/1A. Verified F0 routines DA326,
DA37D and DA3E7 enqueue those records. The helper only reads existing RAM. The
ROM continues to create, consume and acknowledge its events. No guest memory,
program counter, interrupt or ADC result is repaired by host assistance.

The adapters retain the 40,000-boundary minimum, insert a 40,000-boundary released
gap between clicks, and bound the queue to 16 complete taps. A requested release
also waits for a completed coordinate pair and idle panel drive, with at most
400,000 additional boundaries. This avoids splitting the next acquisition merely
because an earlier pair was sampled. Cancellation overrides all assistance.
Only ordinary scheduler execution advances these policies; input while paused
runs no CPU. Raw replay and native `--minimum-contact-boundaries 0` bypass them.

These software policies depend on the verified ROM ABI and the provisional
memory model. They do not qualify panel voltages, ADC latency, ESR-P clocks or
physical touch timing. Schedule's RTC/cache race and physical alarm wake remain
separate open issues. Pending host taps are cancelled on fault/focus loss and
are not persisted as part of a guest full session.
