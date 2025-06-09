import sys
import importlib.util
from pathlib import Path

# Ensure the plugin directory is available on ``sys.path`` so that
# absolute imports like ``binja_helpers`` work when the plugin is loaded
# directly by Binary Ninja.
_plugin_dir = str(Path(__file__).resolve().parent)
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

def module_exists(module_name):
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ValueError, ImportError):
        return False


# we want to only run this on a real Binary Ninja installation,
# and expect __package__ to be set by Binary Ninja.
if module_exists("binaryninja") and __package__:
    from .sc62015.arch import SC62015, SC62015CallingConvention
    from .sc62015.view import SC62015RomView, SC62015FullView

    arch = SC62015.register()
    arch.register_calling_convention(
        default_cc := SC62015CallingConvention(arch, "default")
    )
    arch.default_calling_convention = default_cc

    SC62015RomView.register()
    SC62015FullView.register()


# from .Z80Arch import Z80
# Z80.register()
#
# from .ColecoView import ColecoView
# ColecoView.register()
#
# from .SharpPCG850View import SharpPCG850View, Z80PCG850Arch
#
# Z80PCG850Arch.register()
# SharpPCG850View.register()
#
#
# from .RelView import RelView
# RelView.register()
#
# # built-in view
# EM_Z80 = 220
# binaryninja.BinaryViewType['ELF'].register_arch(EM_Z80, binaryninja.enums.Endianness.LittleEndian, binaryninja.Architecture['Z80'])
#
# class ParametersInRegistersCallingConvention(binaryninja.CallingConvention):
#     name = "ParametersInRegisters"
#     # int_return_reg = 'A'
#
#
# arch = binaryninja.Architecture["Z80"]
# arch.register_calling_convention(
#     ParametersInRegistersCallingConvention(arch, "default")
# )
#
# arch = binaryninja.Architecture["Z80 PC-G850"]
# arch.register_calling_convention(
#     ParametersInRegistersCallingConvention(arch, "default")
# )

