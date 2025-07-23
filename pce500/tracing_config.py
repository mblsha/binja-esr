"""Configuration module for Perfetto tracing in PC-E500 emulator."""

import os
from typing import Optional, Union
from pathlib import Path


class TracingConfig:
    """Configuration for Perfetto tracing.
    
    This class provides runtime configuration for enabling/disabling tracing
    and setting trace output paths. Configuration can be set via environment
    variables or programmatically.
    """
    
    # Environment variable names
    ENV_ENABLE_TRACING = "PCE500_ENABLE_TRACING"
    ENV_TRACE_OUTPUT = "PCE500_TRACE_OUTPUT"
    
    # Default values
    _enabled: bool = False
    _output_path: Optional[Path] = None
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if tracing is enabled.
        
        Checks environment variable PCE500_ENABLE_TRACING first,
        then falls back to programmatic setting.
        
        Returns:
            True if tracing is enabled
        """
        env_val = os.environ.get(cls.ENV_ENABLE_TRACING, "").lower()
        if env_val in ("1", "true", "yes", "on"):
            return True
        elif env_val in ("0", "false", "no", "off"):
            return False
        return cls._enabled
    
    @classmethod
    def enable(cls, output_path: Optional[Union[str, Path]] = None) -> None:
        """Enable tracing programmatically.
        
        Args:
            output_path: Optional path for trace output file.
                        If not specified, uses default or environment variable.
        """
        cls._enabled = True
        if output_path:
            cls._output_path = Path(output_path)
    
    @classmethod
    def disable(cls) -> None:
        """Disable tracing programmatically."""
        cls._enabled = False
    
    @classmethod
    def get_output_path(cls) -> Path:
        """Get the output path for trace files.
        
        Checks in order:
        1. Programmatically set path
        2. PCE500_TRACE_OUTPUT environment variable
        3. Default path (./trace.perfetto-trace)
        
        Returns:
            Path object for trace output
        """
        # Check programmatic setting
        if cls._output_path:
            return cls._output_path
            
        # Check environment variable
        env_path = os.environ.get(cls.ENV_TRACE_OUTPUT)
        if env_path:
            return Path(env_path)
            
        # Default path
        return Path("trace.perfetto-trace")
    
    @classmethod
    def set_output_path(cls, path: Union[str, Path]) -> None:
        """Set the output path programmatically.
        
        Args:
            path: Path for trace output files
        """
        cls._output_path = Path(path)
    
    @classmethod
    def reset(cls) -> None:
        """Reset configuration to defaults."""
        cls._enabled = False
        cls._output_path = None


# Convenience functions
def enable_tracing(output_path: Optional[Union[str, Path]] = None) -> None:
    """Enable Perfetto tracing.
    
    Args:
        output_path: Optional path for trace output file
    """
    TracingConfig.enable(output_path)


def disable_tracing() -> None:
    """Disable Perfetto tracing."""
    TracingConfig.disable()


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled.
    
    Returns:
        True if tracing is enabled
    """
    return TracingConfig.is_enabled()