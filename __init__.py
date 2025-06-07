import importlib.util

if importlib.util.find_spec("binaryninja") is not None:
    import binaryninja  # noqa: F401
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
