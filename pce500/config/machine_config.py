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
            MemoryMapConfig("Internal ROM", "rom", 0xC0000, 0x40000),
            MemoryMapConfig("Work Area", "ram", 0xBFC00, 0x400),
            MemoryMapConfig("Machine Code Area", "ram", 0xBEC00, 0x1234),
            MemoryMapConfig("Internal RAM", "ram", 0xB8000, 0x8000),
            MemoryMapConfig("LCD Controllers", "peripheral", 0x20000, 0x10000),
            MemoryMapConfig("Extension Area", "ram", 0x50000, 0x38000),
            MemoryMapConfig("64KB Card", "rom", 0x40000, 0x10000),
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
                    MemoryMapConfig("Internal ROM", "rom", 0xC0000, 0x40000),
                    MemoryMapConfig("Work Area", "ram", 0xBFC00, 0x400),
                    MemoryMapConfig("Machine Code Area", "ram", 0xBEC00, 0x1234),
                    MemoryMapConfig("Internal RAM", "ram", 0xB8000, 0x8000),
                    MemoryMapConfig("LCD Controllers", "peripheral", 0x20000, 0x10000),
                    MemoryMapConfig("Extension Area", "ram", 0x50000, 0x38000),
                    MemoryMapConfig("64KB Card", "rom", 0x40000, 0x10000),
                ]
            ),
        }
        
        return configs.get(model, configs["PC-E500"])