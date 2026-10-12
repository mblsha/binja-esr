# OZ-9600 IRQ recognition experiment

The `experimental-irq-recognition` profile adds a CPU recognition latch to the v1 composition. The measured software MTI requests and interrupt-mask enables execute one complete following instruction before entry. Clearing IMR or ISR in that instruction does not cancel the CPU frame. The native dispatcher may then find no eligible callback. Repeated writes coalesce. Hardware also demonstrated software STI nesting within an MTI callback and restoration of the tested foreground register/stack context.

The frame contains PC, F and IMR after the following instruction. The implementation keeps recognition separate from pending status, preserves it across full-session export/reload, and clears it on reset or accepted entry. Nested bookkeeping is adjusted only in this explicit experiment. Default profiles retain their prior scheduling policy.

The native CLI accepts `--oz9600-profile experimental-irq-recognition` together with `--model oz9600 --rom VERIFIED.ozrom`. The WASM profile parser accepts the same profile string. Select the matching profile before loading a session. The independent Python policy reference describes the same latch and source-priority decisions; Rust executes the OZ CPU.

Running sessions from this profile use body schema2 in the existing OZRUN01 envelope. Older readers require schema1 and reject them before discarding the recognized request. Existing profiles still export schema1; loading a session with a different recognition policy fails before changing the candidate machine. Battery RAM/RTC images remain independent of CPU recognition state.

Physical receipts and ordinary ROM probe sources are in the private test repository under `docs/oz-9600/reference/cpu_irq_*_20261012`. Four fresh CPU frames and controlled callback counts/foreground markers match the acquired cancellation/repeated-write probe. A final live ISR sample differs (modeled EXI|STI versus hardware zero); its RTC/timer phase is unqualified and is not folded into the measured contract.

The exact fast Schedule failing witness and static-calendar control remain reproducible. This model change produces the same failing framebuffer as the original witness; it does not repair the cache overwrite. No guest memory/register patch or forced execution jump is used. The fast diagnostic explicitly changes only the CPU implementation option after restoring its exact saved guest.

Electrical/asynchronous arrival, source selection changes within the recognized interval, HALT/OFF behavior, bare-RETI edge cases and true reset-state values remain unqualified. Keep this profile opt-in while those contracts and IRQ-time workspace mapping are investigated.
