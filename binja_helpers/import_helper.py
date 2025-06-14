"""
Import helper module to handle both standalone and submodule usage.

This module provides a way to import binja_helpers modules that works both when:
1. binja-esr is used as a standalone project (imports like 'binja_helpers.tokens')
2. binja-esr is used as a submodule (imports like 'binja_helpers.binja_helpers.tokens')
"""

import sys
import importlib
from typing import Any


def import_binja_helper(module_name: str) -> Any:
    """
    Import a binja_helpers module, trying both standalone and submodule paths.
    
    Args:
        module_name: The module name relative to binja_helpers (e.g., 'tokens', 'coding')
    
    Returns:
        The imported module
    
    Raises:
        ImportError: If neither import path works
    """
    # First try the submodule path (binja_helpers.binja_helpers.module_name)
    try:
        full_module_name = f"binja_helpers.binja_helpers.{module_name}"
        return importlib.import_module(full_module_name)
    except ImportError:
        pass
    
    # Then try the standalone path (binja_helpers.module_name)
    try:
        full_module_name = f"binja_helpers.{module_name}"
        return importlib.import_module(full_module_name)
    except ImportError:
        pass
    
    # Finally try direct import (for when we're inside binja_helpers directory)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass
    
    # If all fail, raise a helpful error
    raise ImportError(
        f"Could not import binja_helpers.{module_name}. "
        f"Tried 'binja_helpers.binja_helpers.{module_name}' (submodule), "
        f"'binja_helpers.{module_name}' (standalone), and '{module_name}' (direct) import paths."
    )


def import_from_binja_helper(module_name: str, *names: str) -> tuple:
    """
    Import specific names from a binja_helpers module.
    
    Args:
        module_name: The module name relative to binja_helpers (e.g., 'tokens', 'coding')
        *names: The names to import from the module
    
    Returns:
        Tuple of imported objects in the same order as requested
    
    Raises:
        ImportError: If the module or any of the names cannot be imported
    """
    module = import_binja_helper(module_name)
    
    result = []
    for name in names:
        if not hasattr(module, name):
            raise ImportError(f"Cannot import name '{name}' from '{module.__name__}'")
        result.append(getattr(module, name))
    
    return tuple(result) if len(result) > 1 else result[0]
