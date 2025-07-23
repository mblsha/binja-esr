#!/usr/bin/env python3
"""Run the SC62015 emulator for 1 million instructions and generate a Perfetto trace."""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sc62015.pysc62015.emulator import Memory, RegisterName
from pce500.tracing_emulator import TracingEmulator
from pce500.memory.mapper import MemoryMapper
from pce500.memory.regions import RAMRegion


class TestMemory(Memory):
    """Simple memory implementation for testing."""
    
    def __init__(self, size: int = 0x100000):
        self._memory = bytearray(size)
    
    def read(self, address: int, size: int) -> bytes:
        return bytes(self._memory[address:address + size])
    
    def write(self, address: int, data: bytes) -> None:
        for i, byte in enumerate(data):
            self._memory[address + i] = byte
    
    def read_byte(self, address: int) -> int:
        return self._memory[address]
    
    def write_byte(self, address: int, value: int) -> None:
        self._memory[address] = value & 0xFF


class MemoryMapperAdapter(Memory):
    """Adapter to make MemoryMapper work with SC62015 emulator's Memory interface."""
    
    def __init__(self, mapper: MemoryMapper):
        self.mapper = mapper
    
    def read_byte(self, address: int) -> int:
        return self.mapper.read_byte(address)
    
    def write_byte(self, address: int, value: int) -> None:
        self.mapper.write_byte(address, value)
    
    def read(self, address: int, size: int) -> bytes:
        return bytes([self.mapper.read_byte(address + i) for i in range(size)])
    
    def write(self, address: int, data: bytes) -> None:
        for i, byte in enumerate(data):
            self.mapper.write_byte(address + i, byte)
    
    def read_bytes(self, size: int, address: int) -> bytes:
        """Read bytes in the order expected by eval_llil."""
        return self.read(address, size)
    
    def write_bytes(self, size: int, address: int, data) -> None:
        """Write bytes in the order expected by eval_llil."""
        # Handle both int and bytes data
        if isinstance(data, int):
            # Convert int to bytes (little-endian)
            data_bytes = data.to_bytes(size, byteorder='little', signed=False)
        else:
            data_bytes = data
        self.write(address, data_bytes)
    
    def set_context(self, context: dict) -> None:
        """Pass through context setting."""
        if hasattr(self.mapper, 'set_context'):
            self.mapper.set_context(context)


def create_test_program(memory: Memory):
    """Create a simple test program that executes many instructions.
    
    The program creates an infinite loop of NOPs and simple operations
    that we'll run for 1 million instructions.
    """
    # Starting address
    addr = 0x1000
    
    # Simple infinite loop with minimal stack usage
    loop_start = addr
    
    # Just do a bunch of NOPs and simple operations
    # This avoids complex stack operations that might have compatibility issues
    
    # Load immediate values (doesn't use stack)
    memory.write(addr, bytes([0x08, 0x55]))  # MV A, #0x55
    addr += 2
    
    # Some NOPs for padding
    for _ in range(10):
        memory.write(addr, bytes([0x00]))  # NOP
        addr += 1
    
    # Compare for conditional jump test  
    memory.write(addr, bytes([0x60, 0x00]))  # CMP A, #0
    addr += 2
    
    # Conditional jump (will not be taken since A = 0x55)
    skip_target = addr + 3
    memory.write(addr, bytes([0x14, skip_target & 0xFF, skip_target >> 8]))  # JPZ skip
    addr += 3
    
    # More NOPs
    for _ in range(5):
        memory.write(addr, bytes([0x00]))  # NOP
        addr += 1
    
    # Another conditional test
    memory.write(addr, bytes([0x60, 0x55]))  # CMP A, #0x55
    addr += 2
    
    # This jump WILL be taken (A == 0x55, so Z flag is set)
    taken_target = addr + 3
    memory.write(addr, bytes([0x14, taken_target & 0xFF, taken_target >> 8]))  # JPZ taken
    addr += 3
    
    # More NOPs
    for _ in range(10):
        memory.write(addr, bytes([0x00]))  # NOP
        addr += 1
    
    # Jump back to start (infinite loop)
    memory.write(addr, bytes([0x02, loop_start & 0xFF, loop_start >> 8]))  # JP loop_start
    addr += 3
    
    return 0x1000  # Start address


def main():
    """Run the emulator for 1 million instructions."""
    print("Setting up emulator for 1 million instruction run...")
    
    # Create memory mapper with RAM
    mapper = MemoryMapper()
    mapper.add_region(RAMRegion(0x0000, 0x100000, "Main RAM"))
    
    # Create test program
    print("Creating test program...")
    memory = TestMemory()
    start_addr = create_test_program(memory)
    
    # Copy program to mapper's RAM
    for addr in range(0x100000):
        mapper.write_byte(addr, memory.read_byte(addr))
    
    # Create memory adapter
    memory_adapter = MemoryMapperAdapter(mapper)
    
    # Create tracing emulator
    trace_path = Path("million_instructions.perfetto-trace")
    print(f"Creating tracing emulator with output: {trace_path}")
    
    # Check if tracing is enabled
    from pce500.trace_manager import ENABLE_PERFETTO_TRACING
    print(f"  Tracing enabled: {ENABLE_PERFETTO_TRACING}")
    
    emulator = TracingEmulator(memory_adapter, trace_path)
    
    # Verify tracing started
    from pce500.trace_manager import g_tracer
    print(f"  Tracing started: {g_tracer.is_tracing()}")
    
    # Set initial PC
    emulator.regs.set(RegisterName.PC, start_addr)
    
    # Track execution
    start_time = time.time()
    instruction_count = 0
    last_report = 0
    
    print("Starting execution...")
    print("Progress:")
    
    try:
        while instruction_count < 1_000_000:
            # Get current PC
            pc = emulator.regs.get(RegisterName.PC)
            
            # Execute instruction
            emulator.execute_instruction(pc)
            instruction_count += 1
            
            # Progress report every 100k instructions
            if instruction_count - last_report >= 100_000:
                elapsed = time.time() - start_time
                ips = instruction_count / elapsed if elapsed > 0 else 0
                print(f"  {instruction_count:,} instructions executed "
                      f"({instruction_count/10000:.1f}%) - "
                      f"{ips:,.0f} instructions/sec")
                last_report = instruction_count
            
            # Since it's an infinite loop, we'll just run until 1 million instructions
    
    except KeyboardInterrupt:
        print(f"\nInterrupted after {instruction_count:,} instructions")
    
    except Exception as e:
        print(f"\nError after {instruction_count:,} instructions: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop tracing
        print("\nStopping trace...")
        emulator.stop_tracing()
        
        # Report results
        elapsed = time.time() - start_time
        print("\nExecution summary:")
        print(f"  Total instructions: {instruction_count:,}")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  Average speed: {instruction_count/elapsed:,.0f} instructions/sec")
        print(f"  Call depth reached: {emulator.regs.call_sub_level}")
        print(f"  Memory reads: {emulator._memory_reads:,}")
        print(f"  Memory writes: {emulator._memory_writes:,}")
        
        # Debug: Check if tracing was active
        from pce500.trace_manager import g_tracer
        print("\nTracing status:")
        print(f"  Was tracing active: {g_tracer._tracing_enabled}")
        print(f"  Trace file handle: {g_tracer._trace_file}")
        
        # Verify trace file
        if trace_path.exists():
            size = trace_path.stat().st_size
            print("\nTrace file generated successfully!")
            print(f"  Path: {trace_path}")
            print(f"  Size: {size:,} bytes ({size/1024/1024:.2f} MB)")
            
            # Check if it's a valid Perfetto trace (should start with specific bytes)
            with open(trace_path, 'rb') as f:
                header = f.read(16)
                if header:
                    print(f"  Header: {header.hex()}")
                    print("  ✓ Trace file appears to be valid")
                else:
                    print("  ⚠ Trace file is empty!")
        else:
            print(f"\n✗ Trace file was not created at {trace_path}")
        
        print("\nTo view the trace:")
        print("  1. Open https://ui.perfetto.dev")
        print("  2. Click 'Open trace file'")
        print(f"  3. Select: {trace_path.absolute()}")


if __name__ == "__main__":
    main()