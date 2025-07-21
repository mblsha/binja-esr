"""Main PC-E500 emulator class."""

import time
from typing import Optional, Dict, Any
from pathlib import Path

# Import the SC62015 CPU from the parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sc62015.pysc62015.cpu import CPU
from sc62015.pysc62015.memory import Memory

from .machine import PCE500Machine


class PCE500Memory(Memory):
    """Memory implementation that bridges SC62015 CPU to PC-E500 memory mapper."""
    
    def __init__(self, machine: PCE500Machine):
        self.machine = machine
    
    def read_byte(self, address: int) -> int:
        return self.machine.memory.read_byte(address)
    
    def write_byte(self, address: int, value: int) -> None:
        self.machine.memory.write_byte(address, value)
    
    def read_word(self, address: int) -> int:
        return self.machine.memory.read_word(address)
    
    def write_word(self, address: int, value: int) -> None:
        self.machine.memory.write_word(address, value)


class PCE500Emulator:
    """Sharp PC-E500 emulator."""
    
    def __init__(self):
        self.machine = PCE500Machine()
        self.memory = PCE500Memory(self.machine)
        self.cpu = CPU(self.memory)
        
        # Emulation state
        self.running = False
        self.breakpoints: set[int] = set()
        self.trace_enabled = False
        
        # Performance tracking
        self.cycle_count = 0
        self.start_time = 0.0
        
    def load_rom(self, rom_path: str, start_address: Optional[int] = None) -> None:
        """Load ROM from file."""
        with open(rom_path, 'rb') as f:
            rom_data = f.read()
        self.machine.load_rom(rom_data, start_address)
    
    def reset(self) -> None:
        """Reset the emulator."""
        self.machine.reset()
        self.cpu.reset()
        self.cycle_count = 0
        
    def step(self) -> int:
        """Execute one instruction and return cycles used."""
        if self.trace_enabled:
            self._trace_state()
        
        # Check breakpoints
        if self.cpu.regs.pc in self.breakpoints:
            self.running = False
            return 0
        
        # Execute instruction
        cycles = self.cpu.step()
        self.cycle_count += cycles
        
        return cycles
    
    def run(self, max_cycles: Optional[int] = None) -> None:
        """Run emulation until stopped or max_cycles reached."""
        self.running = True
        self.start_time = time.time()
        cycles_run = 0
        
        while self.running:
            cycles = self.step()
            if cycles == 0:  # Breakpoint hit
                break
                
            cycles_run += cycles
            if max_cycles and cycles_run >= max_cycles:
                break
            
            # Check for interrupts or other events
            self._check_interrupts()
    
    def stop(self) -> None:
        """Stop emulation."""
        self.running = False
    
    def add_breakpoint(self, address: int) -> None:
        """Add a breakpoint at the specified address."""
        self.breakpoints.add(address & 0xFFFFFF)
    
    def remove_breakpoint(self, address: int) -> None:
        """Remove a breakpoint."""
        self.breakpoints.discard(address & 0xFFFFFF)
    
    def get_cpu_state(self) -> Dict[str, Any]:
        """Get current CPU state."""
        return {
            'pc': self.cpu.regs.pc,
            'a': self.cpu.regs.a,
            'b': self.cpu.regs.b,
            'ba': self.cpu.regs.ba,
            'i': self.cpu.regs.i,
            'x': self.cpu.regs.x,
            'y': self.cpu.regs.y,
            'u': self.cpu.regs.u,
            's': self.cpu.regs.s,
            'flags': {
                'z': self.cpu.regs.f_z,
                'c': self.cpu.regs.f_c
            },
            'cycles': self.cycle_count
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get emulation performance statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        cycles_per_sec = self.cycle_count / elapsed if elapsed > 0 else 0
        
        # SC62015 runs at approximately 2.5 MHz
        emulated_mhz = cycles_per_sec / 1_000_000
        speed_ratio = emulated_mhz / 2.5
        
        return {
            'cycles': self.cycle_count,
            'elapsed_time': elapsed,
            'cycles_per_second': cycles_per_sec,
            'emulated_mhz': emulated_mhz,
            'speed_ratio': speed_ratio
        }
    
    def _check_interrupts(self) -> None:
        """Check and handle interrupts."""
        # TODO: Implement interrupt handling
        pass
    
    def _trace_state(self) -> None:
        """Print current CPU state for debugging."""
        state = self.get_cpu_state()
        print(f"PC:{state['pc']:06X} A:{state['a']:02X} B:{state['b']:02X} "
              f"I:{state['i']:02X} X:{state['x']:06X} Y:{state['y']:06X} "
              f"Z:{int(state['flags']['z'])} C:{int(state['flags']['c'])}")