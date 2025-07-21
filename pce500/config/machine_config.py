"""Machine configuration for PC-E500 emulator."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class MemoryMapConfig:
    """Memory region configuration."""
    name: str
    type: str  # "rom", "ram", "peripheral"
    start: int
    size: int
    file: Optional[str] = None  # For ROM regions
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "start": f"0x{self.start:06X}",
            "size": f"0x{self.size:X}",
            "file": self.file
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryMapConfig':
        return cls(
            name=data["name"],
            type=data["type"],
            start=int(data["start"], 16) if isinstance(data["start"], str) else data["start"],
            size=int(data["size"], 16) if isinstance(data["size"], str) else data["size"],
            file=data.get("file")
        )


@dataclass
class MachineConfig:
    """PC-E500 machine configuration."""
    name: str = "Sharp PC-E500"
    cpu_frequency: int = 2_500_000  # 2.5 MHz
    memory_map: List[MemoryMapConfig] = None
    
    def __post_init__(self):
        if self.memory_map is None:
            self.memory_map = self.get_default_memory_map()
    
    @staticmethod
    def get_default_memory_map() -> List[MemoryMapConfig]:
        """Get default PC-E500 memory map."""
        return [
            MemoryMapConfig("Internal RAM", "ram", 0x000000, 0x2000),
            MemoryMapConfig("Main LCD", "peripheral", 0x007000, 0x800),
            MemoryMapConfig("Sub LCD", "peripheral", 0x007800, 0x100),
            MemoryMapConfig("External RAM", "ram", 0x008000, 0x8000),
            MemoryMapConfig("System ROM", "rom", 0x040000, 0x40000),
        ]
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        data = {
            "name": self.name,
            "cpu_frequency": self.cpu_frequency,
            "memory_map": [region.to_dict() for region in self.memory_map]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'MachineConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data.get("name", "Sharp PC-E500"),
            cpu_frequency=data.get("cpu_frequency", 2_500_000),
            memory_map=[MemoryMapConfig.from_dict(r) for r in data.get("memory_map", [])]
        )
    
    @classmethod
    def for_model(cls, model: str) -> 'MachineConfig':
        """Get configuration for specific PC-E500 model variant."""
        configs = {
            "PC-E500": cls(
                name="Sharp PC-E500",
                memory_map=cls.get_default_memory_map()
            ),
            "PC-E500S": cls(
                name="Sharp PC-E500S",
                memory_map=[
                    MemoryMapConfig("Internal RAM", "ram", 0x000000, 0x2000),
                    MemoryMapConfig("Main LCD", "peripheral", 0x007000, 0x800),
                    MemoryMapConfig("Sub LCD", "peripheral", 0x007800, 0x100),
                    MemoryMapConfig("External RAM", "ram", 0x008000, 0x10000),  # 64KB
                    MemoryMapConfig("System ROM", "rom", 0x040000, 0x40000),
                ]
            ),
        }
        
        return configs.get(model, configs["PC-E500"])