# PI-3000 native runtime

The `pi3000` Cargo feature adds a PI-3000 board adapter to the shared
CoreRuntime. It reuses the native CPU, matrix LCD controller, RTC reference
and bounded ESR-H interpreter. PI ROM mapping, controller geometry and input
ports come from the acquired PI firmware. This is an implementation in
progress; foreground calibration, ordinary input, RTC/power, cards and browser
integration remain open.

The normal factory accepts only a `PIROM01` bundle of the verified fixed ROM
and 32 complete selector views (00..0F and F0..FF). It does not accept a mapped
1 MiB dump as startup RAM. The private acquisition tool packs the bundle;
ROM data is not distributed with this source tree.

```sh
cargo run --manifest-path sc62015/core/Cargo.toml --features pi3000 \
  --bin pce500 -- --model pi-3000 --rom /path/to/pi-3000.pirom --steps 20
```

The factory starts fresh scratch/high RAM and leaves periodic timers disabled
pending calibration. Unsupported bus operations stop and poison the runtime.
An unknown instruction mapping fails preflight; an unknown data access stops
after its committed instruction. Generic card overlays are not applied.
Restart constructs a new board; v4 snapshots do not represent its devices.

`BootstrapProfile::Strict` is the default native-only/unknown-auxiliary control.
`Compatibility` opts into the firmware-derived gate/page handoff.
`AuxiliaryAbsent` additionally supplies status02 and retains auxiliary requests
without completion or response bytes. The firmware can take its own timeout
path; this does not implement the auxiliary device.

The default ISR replacement policy currently triggers an early IRQ before
reset has installed the handler workspace. An explicitly selected clear-only
ISR experiment reaches a four-cross calibration frame after two million
scheduler boundaries, including 29 legacy instructions. Initial CPU and ISR
semantics remain unqualified; model selection does not infer that experiment.

`configure_pi3000_tablet_route` selects one of the two ADC branches in
FE:DBE41. `Integrated` routes control/data to 08B80/08BC0; `External` routes
to 08800/08C00. `set_pi3000_raw_contact` supplies ten-bit samples and contact
state. It neither populates firmware event queues nor asserts an unqualified
interrupt. The route cannot change once selected. Both original ROM branches
return independently chosen raw coordinates with balanced stacks.

ADC completion-on-first-read, held split-byte samples, binary contact pressure,
LCD sampling sync and physical route selection remain provisional. The
controller frame is 240×336 native storage; the physical panel specification
is 239×168 and scan orientation is still being recovered.
