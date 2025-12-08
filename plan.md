Rust/Python overlay parity plan
- Done: MemoryImage overlay support with handler/data backing, PC-aware overlay read/write logs, RuntimeBus PC propagation, parity unit tests for handler reads, data writes, fallback behavior, and Perfetto overlay-name tagging regression test.
- Done: Convenience helpers mirroring Python overlays (`add_ram_overlay`, `add_rom_overlay`, `load_memory_card`) with size validation matching Python memory card ranges.
