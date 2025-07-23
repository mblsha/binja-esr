#!/usr/bin/env python3
"""Test complete tracing implementation with all features."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable Perfetto tracing
os.environ['PCE500_ENABLE_TRACING'] = '1'

from pce500.simple_emulator import SimplifiedPCE500Emulator
from sc62015.pysc62015.emulator import RegisterName


def create_test_rom():
    """Create a ROM with various control flow instructions."""
    rom_data = bytearray(256 * 1024)
    
    # Set reset vector at 0xFFFFD to point to 0xC0000
    rom_data[0x3FFFD] = 0x00  # Low byte
    rom_data[0x3FFFE] = 0x00  # Middle byte  
    rom_data[0x3FFFF] = 0x0C  # High byte (0xC0000)
    
    # Set interrupt vector at 0xFFFFA to point to 0xC0100
    rom_data[0x3FFFA] = 0x00  # Low byte
    rom_data[0x3FFFB] = 0x01  # Middle byte
    rom_data[0x3FFFC] = 0x0C  # High byte (0xC0100)
    
    # Main program at 0xC0000
    addr = 0x0000
    
    # NOP
    rom_data[addr] = 0x00
    addr += 1
    
    # CALL to function at 0xC0050
    rom_data[addr] = 0x04      # CALL opcode
    rom_data[addr+1] = 0x50    # Low byte
    rom_data[addr+2] = 0xC0    # High byte
    addr += 3
    
    # JP to 0xC0020
    rom_data[addr] = 0x0E      # JP opcode
    rom_data[addr+1] = 0x20    # Low byte
    rom_data[addr+2] = 0xC0    # High byte
    addr += 3
    
    # Function at 0xC0050
    addr = 0x0050
    
    # CALLF to far function at 0x0C0080
    rom_data[addr] = 0x05      # CALLF opcode
    rom_data[addr+1] = 0x80    # Low byte
    rom_data[addr+2] = 0x00    # Mid byte
    rom_data[addr+3] = 0x0C    # High byte (20-bit address)
    addr += 4
    
    # RET
    rom_data[addr] = 0x06      # RET opcode
    addr += 1
    
    # Far function at 0xC0080
    addr = 0x0080
    
    # Test conditional jump - JRZ (jump if zero)
    rom_data[addr] = 0x08      # JRZ opcode
    rom_data[addr+1] = 0x02    # Offset (+2)
    addr += 2
    
    # NOP (skipped if Z flag is set)
    rom_data[addr] = 0x00
    addr += 1
    
    # RETF
    rom_data[addr] = 0x07      # RETF opcode
    addr += 1
    
    # Target at 0xC0020
    addr = 0x0020
    
    # Test relative jump - JR
    rom_data[addr] = 0x0D      # JR opcode
    rom_data[addr+1] = 0xFE    # Offset (-2, jump back 2 bytes)
    addr += 2
    
    # Interrupt handler at 0xC0100
    addr = 0x0100
    
    # NOP in interrupt
    rom_data[addr] = 0x00
    addr += 1
    
    # RETI
    rom_data[addr] = 0x01      # RETI opcode
    addr += 1
    
    return bytes(rom_data)


def main():
    """Test all tracing features."""
    print("=== Testing Complete Tracing Implementation ===\n")
    
    # Create emulator with Perfetto tracing
    emulator = SimplifiedPCE500Emulator(perfetto_trace=True)
    print("✓ Created emulator with Perfetto tracing")
    
    # Load test ROM
    rom_data = create_test_rom()
    emulator.load_rom(rom_data)
    print("✓ Loaded test ROM with control flow instructions")
    
    # Reset emulator
    emulator.reset()
    print("✓ Reset emulator")
    
    # Set stack pointer for CALL/RET
    emulator.cpu.regs.set(RegisterName.S, 0xFFFF)
    print("✓ Set stack pointer to 0xFFFF")
    
    print("\nExecuting test program:")
    print("1. NOP")
    print("2. CALL 0xC0050 (increases call depth)")
    print("3. CALLF 0x0C0080 (far call, increases depth)")
    print("4. JRZ +2 (conditional jump)")
    print("5. RETF (decreases call depth)")
    print("6. RET (returns to main)")
    print("7. JP 0xC0020 (unconditional jump)")
    print("8. JR -2 (relative jump)")
    
    # Execute instructions
    for i in range(12):  # Execute enough to see all control flow
        try:
            if not emulator.step():
                print(f"  Breakpoint hit at step {i+1}")
                break
            pc = emulator.cpu.regs.get(RegisterName.PC)
            print(f"  Step {i+1}: PC=0x{pc:06X}, Depth={emulator.call_depth}, " +
                  f"Reads={emulator.memory_read_count}, Writes={emulator.memory_write_count}")
        except Exception as e:
            print(f"  Error at step {i+1}: {e}")
            break
    
    # Test interrupt
    print("\nTesting interrupt (IR):")
    # Set PC to execute IR instruction
    emulator.cpu.regs.set(RegisterName.PC, 0xC0030)
    # Add IR instruction
    emulator.memory.write_byte(0xC0030, 0xFE)  # IR opcode
    
    # Execute IR
    emulator.step()
    print(f"  After IR: PC=0x{emulator.cpu.regs.get(RegisterName.PC):06X}, Depth={emulator.call_depth}")
    
    # Execute interrupt handler
    emulator.step()  # NOP in handler
    emulator.step()  # RETI
    print(f"  After RETI: PC=0x{emulator.cpu.regs.get(RegisterName.PC):06X}, Depth={emulator.call_depth}")
    
    # Show final statistics
    print(f"\nFinal Statistics:")
    print(f"  Total instructions: {emulator.instruction_count}")
    print(f"  Memory reads: {emulator.memory_read_count}")
    print(f"  Memory writes: {emulator.memory_write_count}")
    print(f"  Cycles: {emulator.cycle_count}")
    print(f"  Call depth: {emulator.call_depth}")
    
    # Stop tracing
    emulator.stop_tracing()
    print("\n✓ Stopped tracing")
    
    # Check trace file
    trace_path = Path("pc-e500.trace")
    if trace_path.exists():
        size_kb = trace_path.stat().st_size / 1024
        print(f"✓ Trace file created: {trace_path} ({size_kb:.1f} KB)")
        print("\nTrace should contain:")
        print("  - Function call slices (CALL/RET, CALLF/RETF)")
        print("  - Jump events (JP, JPF, JR, conditional jumps)")
        print("  - Interrupt flow (IR/RETI)")
        print("  - Performance counters (cycles, call depth, memory ops)")
        print("  - Memory access with PC context")
        print("\nView in Chrome: chrome://tracing")
    

if __name__ == "__main__":
    main()