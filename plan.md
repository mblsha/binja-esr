Rust/Python overlay parity plan
- Done: MemoryImage now supports external overlays with handler/data backing, PC-aware overlay read/write logs, and RuntimeBus passes PC into overlay paths; unit tests cover handler reads, data writes, and fallback behaviour.
- Remaining: Add convenience helpers mirroring Python’s overlay APIs (`add_ram`, `add_rom`, `load_memory_card`) and propagate overlay names into Perfetto traces for parity with Python logs.
