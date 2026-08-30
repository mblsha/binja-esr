from .opcodes import *  # noqa: F401,F403
from .opcodes import _lift_wrapped_memory_load, _resize_unsigned
from binaryninja.enums import BranchType  # noqa: F401
from .traits import HasWidth
from typing import Callable
from binaryninja import InstructionInfo  # type: ignore

REG3_20BIT_MASK = 0x0FFFFF
PC_PAGE_MASK = PC_MASK & ~0xFFFF
REG3_20BIT_REGS = (
    RegisterName("X"),
    RegisterName("Y"),
    RegisterName("U"),
    RegisterName("S"),
)
CONTROL_20BIT_REGS = REG3_20BIT_REGS + (RegisterName("PC"),)


def _lift_operand_value(
    il: LowLevelILFunction,
    operand: Operand,
    mode: Optional[AddressingMode] = None,
    *,
    side_effects: bool = True,
) -> ExpressionIndex:
    """Lift an operand with the runtime's 20-bit register invariant."""

    value = operand.lift(il, mode, side_effects=side_effects)
    if isinstance(operand, Reg) and operand.reg in CONTROL_20BIT_REGS:
        value = il.and_expr(3, value, il.const(3, REG3_20BIT_MASK))
    return value


def _resize_for_operand(
    il: LowLevelILFunction,
    value: ExpressionIndex,
    source_width: int,
    destination: Operand,
) -> ExpressionIndex:
    assert isinstance(destination, HasWidth)
    resized = _resize_unsigned(il, value, source_width, destination.width())
    if isinstance(destination, Reg) and destination.reg in CONTROL_20BIT_REGS:
        resized = il.and_expr(3, resized, il.const(3, REG3_20BIT_MASK))
    return resized


def _lift_s_push(il: LowLevelILFunction, width: int, value: ExpressionIndex) -> None:
    """Push to S with 20-bit pointer and per-byte external-bus wrapping."""
    old_s = TempReg(TempIncDecHelper, width=3)
    old_s.lift_assign(il, il.reg(3, RegisterName("S")))
    new_s = il.and_expr(
        3,
        il.sub(3, old_s.lift(il), il.const(3, width)),
        il.const(3, REG3_20BIT_MASK),
    )
    il.append(il.set_reg(3, RegisterName("S"), new_s))
    EMemHelper(width, Reg("S")).lift_assign(il, value)


def _lift_s_pop(
    il: LowLevelILFunction,
    width: int,
    *,
    validate_f: bool = False,
) -> ExpressionIndex:
    """Pop from S with 20-bit pointer and per-byte external-bus wrapping."""
    old_s = TempReg(TempIncDecHelper, width=3)
    old_s.lift_assign(il, il.reg(3, RegisterName("S")))
    value = EMemHelper(width, old_s).lift(il)
    if validate_f:
        # Force the lazy memory read into a temporary, then validate the raw
        # byte before S or either modeled flag can change.  CPU transaction
        # rollback used to hide the opposite ordering from direct LLIL users.
        raw_f = TempReg(TempRegF, width=1)
        raw_f.lift_assign(il, value)
        value = raw_f.lift(il)
        il.append(il.intrinsic([], ValidateFIntrinsic, [value]))
    il.append(
        il.set_reg(
            3,
            RegisterName("S"),
            il.and_expr(
                3,
                il.add(3, old_s.lift(il), il.const(3, width)),
                il.const(3, REG3_20BIT_MASK),
            ),
        )
    )
    return value


def _lift_validate_nonzero_count(il: LowLevelILFunction) -> None:
    """Quarantine I=0 before a counted instruction performs any setup."""

    loop_reg = Reg("I")
    il.append(il.intrinsic([], ValidateICountIntrinsic, [loop_reg.lift(il)]))


class NOP(Instruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.nop())


class JumpInstruction(Instruction):
    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        raise NotImplementedError("lift_jump_addr() not implemented")

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        if self._cond:
            # expect TrueBranch to be handled by subclasses as it might require
            # llil logic to calculate the address
            info.add_branch(BranchType.FalseBranch, (addr + self.length()) & PC_MASK)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        if_true = LowLevelILLabel()
        if_false = LowLevelILLabel()

        if self._cond:
            zero = il.const(1, 0)
            one = il.const(1, 1)
            flag = il.flag(ZFlag) if "Z" in self._cond else il.flag(CFlag)
            value = zero if "N" in self._cond else one

            cond = il.compare_equal(1, flag, value)
            il.append(il.if_expr(cond, if_true, if_false))

        il.mark_label(if_true)
        il.append(il.jump(self.lift_jump_addr(il, addr)))
        il.mark_label(if_false)


class JP_Abs(JumpInstruction):
    def name(self) -> str:
        return super().name() + (self._cond if self._cond else "")

    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        if first.width() >= 3:
            if isinstance(first, ImmOperand):
                # Imm20.decode has already discarded the non-address high
                # nibble, so this constant is canonical by construction.
                return first.lift(il)
            return il.and_expr(
                3,
                first.lift(il),
                il.const(3, REG3_20BIT_MASK),
            )
        high_addr = addr & PC_PAGE_MASK
        return il.or_expr(
            3,
            _resize_unsigned(il, first.lift(il), first.width(), 3),
            il.const(3, high_addr),
        )

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)

        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        if isinstance(first, ImmOperand):
            # absolute address
            assert first.value is not None, "Value not set"
            dest = first.value & PC_MASK
            if first.width() < 3:
                dest |= addr & PC_PAGE_MASK
            branch_type = (
                BranchType.TrueBranch if self._cond else BranchType.UnconditionalBranch
            )
            info.add_branch(branch_type, dest)


class JP_Rel(JumpInstruction):
    def name(self) -> str:
        return "JR" + (self._cond if self._cond else "")

    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, ImmOffset), f"Expected ImmOffset, got {type(first)}"
        dest = (addr + self.length() + first.offset_value()) & PC_MASK
        return il.const(3, dest)

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, ImmOffset), f"Expected ImmOffset, got {type(first)}"
        dest = (addr + self.length() + first.offset_value()) & PC_MASK
        branch_type = (
            BranchType.TrueBranch if self._cond else BranchType.UnconditionalBranch
        )
        info.add_branch(branch_type, dest)


class CALL(Instruction):
    def _dest(self) -> ImmOperand:
        dest, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(dest, ImmOperand), "Expected ImmOperand"
        return dest

    def dest_addr(self, addr: int) -> int:
        dest = self._dest()
        result = dest.value
        assert result is not None, "Value not set"
        if dest.width() != 3:
            assert dest.width() == 2
            result = addr & PC_PAGE_MASK | result
        return result & PC_MASK

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.CallDestination, self.dest_addr(addr))

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dest = self._dest()
        return_width = 3 if dest.width() == 3 else 2
        _lift_s_push(
            il,
            return_width,
            il.const(
                return_width,
                (addr + self.length())
                & (REG3_20BIT_MASK if return_width == 3 else 0xFFFF),
            ),
        )
        # Preserve the real analysis operation as a call.  The SC62015 stack
        # frame is explicit above because Binary Ninja's generic LLIL_CALL
        # does not describe the architecture-specific two/three-byte push.
        il.append(il.call(il.const_pointer(3, self.dest_addr(addr))))


class RetInstruction(Instruction):
    def addr_size(self) -> int:
        return 2

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.FunctionReturn)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        pop_val = _lift_s_pop(il, self.addr_size())
        if self.addr_size() == 2:
            high = il.and_expr(
                3,
                il.reg(3, RegisterName("PC")),
                il.const(3, PC_PAGE_MASK),
            )
            pop_val = il.or_expr(
                3,
                _resize_unsigned(il, pop_val, 2, 3),
                high,
            )
        else:
            pop_val = il.and_expr(
                3,
                pop_val,
                il.const(3, REG3_20BIT_MASK),
            )
        il.append(il.ret(pop_val))


class RET(RetInstruction):
    pass


class RETF(RetInstruction):
    def addr_size(self) -> int:
        return 3


class RETI(RetInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        # Snapshot the complete frame before the first architectural write.
        # This both validates F before touching IMR and prevents a later host
        # read failure from being discovered only after part of RETI commits.
        old_s = TempReg(TempIncDecHelper, width=3)
        old_s.lift_assign(il, il.reg(3, RegisterName("S")))

        def frame_addr(offset: int) -> ExpressionIndex:
            return il.and_expr(
                3,
                il.add(3, old_s.lift(il), il.const(3, offset)),
                il.const(3, REG3_20BIT_MASK),
            )

        imr_value = TempReg(TempMultiByte1, width=1)
        imr_value.lift_assign(il, il.load(1, frame_addr(0)))
        f_value = TempReg(TempRegF, width=1)
        f_value.lift_assign(il, il.load(1, frame_addr(1)))
        pc_value = TempReg(TempMultiByte2, width=3)
        pc_value.lift_assign(
            il,
            _lift_wrapped_memory_load(
                il,
                3,
                frame_addr(2),
                address_mask=REG3_20BIT_MASK,
            ),
        )
        il.append(il.intrinsic([], ValidateFIntrinsic, [f_value.lift(il)]))

        imr, *_rest = RegIMR().operands()
        imr.lift_assign(il, imr_value.lift(il))
        RegF().lift_assign(il, f_value.lift(il))
        il.append(
            il.set_reg(
                3,
                RegisterName("S"),
                il.and_expr(
                    3,
                    il.add(3, old_s.lift(il), il.const(3, 5)),
                    il.const(3, REG3_20BIT_MASK),
                ),
            )
        )
        il.append(
            il.ret(
                il.and_expr(
                    3,
                    pc_value.lift(il),
                    il.const(3, REG3_20BIT_MASK),
                )
            )
        )


class MoveInstruction(Instruction):
    pass


class MV(MoveInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode, src_mode = self._addressing_modes()

        operands = tuple(self.operands())
        if len(operands) == 2:
            # For MV instructions, we don't want to "lift" (load from) the destination
            # We only need to get the source value and assign it to the destination
            # This avoids the double-decrement issue for pre-dec destinations
            dst, src = operands
            assert isinstance(dst, HasWidth)
            assert isinstance(src, HasWidth)
            src_value = _lift_operand_value(il, src, src_mode)
            src_value = _resize_for_operand(il, src_value, src.width(), dst)
            dst.lift_assign(il, src_value, dst_mode)
        else:
            # Fall back to default behavior for other cases
            super().lift(il, addr)

    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il_arg2


class MVL(MoveInstruction):
    def modify_addr_il(
        self, il: LowLevelILFunction
    ) -> Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex]:
        return il.add

    def _update_address_with_wrap(
        self,
        il: LowLevelILFunction,
        reg: TempReg,
        update_func: Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex],
        operand: Operand,
    ) -> None:
        """Update address register with wrapping for IMem8 operands."""
        new_addr = update_func(reg.width(), reg.lift(il), il.const(reg.width(), 1))

        if isinstance(operand, IMem8):
            # For IMem8, wrap address within internal memory range (0x00-0xFF)
            # Extract offset by subtracting INTERNAL_MEMORY_START
            offset = il.sub(3, new_addr, il.const(3, INTERNAL_MEMORY_START))
            # Wrap the offset within 0xFF range
            wrapped_offset = il.and_expr(3, offset, il.const(3, 0xFF))
            # Add back the base to get the full address
            wrapped_addr = il.add(3, il.const(3, INTERNAL_MEMORY_START), wrapped_offset)
            reg.lift_assign(il, wrapped_addr)
        else:
            reg.lift_assign(
                il,
                il.and_expr(
                    reg.width(), new_addr, il.const(reg.width(), REG3_20BIT_MASK)
                ),
            )

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        assert isinstance(dst, Pointer), f"Expected Pointer, got {type(dst)}"
        assert isinstance(src, Pointer), f"Expected Pointer, got {type(src)}"
        _lift_validate_nonzero_count(il)

        # 0xCB and 0xCF variants use IMem8, IMem8
        dst_reg = TempReg(TempMvlDst)
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)

        dst_reg.lift_assign(
            il, dst.lift_current_addr(il, pre=dst_mode, side_effects=False)
        )
        src_reg = TempReg(TempMvlSrc)
        initial_src_addr = src.lift_current_addr(il, pre=src_mode, side_effects=False)

        # For pre-decrement sources, check if we need special handling
        is_predec_src = (
            isinstance(src, EMemValueOffsetHelper)
            and isinstance(src.value, RegIncrementDecrementHelper)
            and src.value.mode == EMemRegMode.PRE_DEC
        )
        is_predec_dst = (
            isinstance(dst, EMemValueOffsetHelper)
            and isinstance(dst.value, RegIncrementDecrementHelper)
            and dst.value.mode == EMemRegMode.PRE_DEC
        )

        # Store reference to the actual register for pre-decrement sources
        predec_reg = None
        if (
            is_predec_src
            and isinstance(src, EMemValueOffsetHelper)
            and isinstance(src.value, RegIncrementDecrementHelper)
        ):
            predec_reg = src.value.reg
        # ``initial_src_addr`` already computes the pre-decremented address
        # without mutating the architectural register.  Defer that mutation
        # to the loop body so pointer updates match completed transfers.
        src_reg.lift_assign(il, initial_src_addr)

        # Debug: print initial addresses
        # print(f"MVL: dst_mode={dst_mode}, src_mode={src_mode}")

        with lift_loop(il):
            src_mem = src.memory_helper()(1, src_reg)
            dst_mem = dst.memory_helper()(1, dst_reg)
            # Use AddressingMode.N since src_reg and dst_reg already contain final addresses
            dst_mem.lift_assign(
                il, src_mem.lift(il, pre=AddressingMode.N), pre=AddressingMode.N
            )

            if predec_reg is not None:
                # Commit the pre-decrement only for an iteration that actually
                # transferred a byte.  This also handles I=1, where there is no
                # next-address update below.
                predec_reg.lift_assign(il, src_reg.lift(il))

            # +1 index for normal MVL, -1 for MVLD
            func = self.modify_addr_il(il)
            dst_func = func
            src_func = func

            # Special handling for pre-decrement sources
            if is_predec_src:
                # For pre-decrement sources, continue decrementing in the loop
                src_func = il.sub

            if is_predec_dst:
                dst_func = il.sub

            # Update destination address with wrapping for IMem8
            self._update_address_with_wrap(il, dst_reg, dst_func, dst)

            if is_predec_src:
                # For pre-decrement sources, check if we need to continue
                # Only update if there are more bytes to copy (I > 1)
                loop_reg = Reg("I")
                # I is an unsigned 16-bit byte count.  A signed comparison
                # makes pre-decrement sources stop advancing after the first
                # byte whenever I has bit 15 set.
                continue_cond = il.compare_unsigned_greater_than(
                    loop_reg.width(), loop_reg.lift(il), il.const(loop_reg.width(), 1)
                )

                # Create labels for conditional update
                update_label = LowLevelILLabel()
                skip_label = LowLevelILLabel()

                il.append(il.if_expr(continue_cond, update_label, skip_label))
                il.mark_label(update_label)

                # Update the address using subtraction to continue decrementing
                self._update_address_with_wrap(il, src_reg, src_func, src)

                il.append(il.goto(skip_label))
                il.mark_label(skip_label)
            else:
                # Update source address with wrapping for IMem8
                self._update_address_with_wrap(il, src_reg, src_func, src)
                src.lift_current_addr(il, pre=src_mode)

            # apply any addressing side effects for destination
            dst.lift_current_addr(il, pre=dst_mode)


class MVLD(MVL):
    def modify_addr_il(
        self, il: LowLevelILFunction
    ) -> Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex]:
        return il.sub


class PRE(Instruction):
    def name(self) -> str:
        return f"PRE{self.opcode:02x}"

    def fuse(self, sister: "Instruction") -> Optional["Instruction"]:
        if isinstance(sister, PRE):
            return None

        operands = tuple(sister.operands())
        pre_operand_indexes = [
            index
            for index, operand in enumerate(operands)
            if sister._operand_uses_pre_mode(operand)
        ]
        if not pre_operand_indexes:
            if (self.opcode, sister.opcode) in ROM_PROVEN_IGNORED_PRE_PAIRS:
                sister._pre = self.opcode
                sister.set_length(self.length() + sister.length())
                return sister
            raise InvalidInstruction(
                f"PRE{self.opcode:02X} cannot prefix PRE-insensitive {sister.name()}"
            )

        old_pre = sister._pre
        sister._pre = self.opcode
        try:
            dst_mode, src_mode = sister._addressing_modes()
        finally:
            sister._pre = old_pre

        if len(pre_operand_indexes) == 1:
            operand_index = pre_operand_indexes[0]
            effective_mode = dst_mode if operand_index == 0 else src_mode
            canonical_pre = SINGLE_OPERAND_PRE_LOOKUP.get(effective_mode)
        else:
            canonical_pre = REVERSE_PRE_TABLE.get((dst_mode, src_mode))

        if canonical_pre != self.opcode:
            canonical_text = (
                "no prefix" if canonical_pre is None else f"PRE{canonical_pre:02X}"
            )
            raise InvalidInstruction(
                f"Noncanonical PRE{self.opcode:02X} for {sister.name()}; "
                f"use {canonical_text}"
            )

        # BP+PX/BP+PY addressing ignores the encoded selector byte.  Those
        # aliases are not documented and make byte-level round trips appear
        # meaningful when execution discards the byte, so accept only the
        # canonical zero encoding pending hardware traces.
        modes = (dst_mode, src_mode)
        for operand_index in pre_operand_indexes:
            mode = modes[0 if operand_index == 0 else 1]
            if mode not in (AddressingMode.BP_PX, AddressingMode.BP_PY):
                continue
            operand = operands[operand_index]
            selector: Optional[int] = None
            if isinstance(operand, IMemOperand):
                selector = operand.n_val
            elif isinstance(operand, IMem8):
                selector = operand.value
            elif isinstance(operand, EMemValueOffsetHelper):
                value_operand = operand.value
                if isinstance(value_operand, IMemOperand):
                    selector = value_operand.n_val
                elif isinstance(value_operand, IMem8):
                    selector = value_operand.value
            if selector != 0:
                raise InvalidInstruction(
                    f"Nonzero ignored selector {selector!r} for {mode.value}"
                )

        sister._pre = self.opcode
        sister.set_length(self.length() + sister.length())
        return sister

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        # PRE instructions that couldn't fuse are invalid
        raise InvalidInstruction(f"Unfused PRE instruction at {addr:#x}")

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        # PRE instructions that couldn't fuse are invalid
        raise InvalidInstruction(f"Unfused PRE instruction at {addr:#x}")


class StackInstruction(Instruction):
    def reg(self) -> Operand:
        r, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        return r


class StackPushInstruction(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth), f"Expected HasWidth, got {type(r)}"
        _lift_s_push(il, r.width(), r.lift(il))


class StackPopInstruction(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth), f"Expected HasWidth, got {type(r)}"
        if isinstance(r, RegF):
            value = _lift_s_pop(il, r.width(), validate_f=True)
            r.lift_assign_validated(il, value)
            return
        r.lift_assign(il, _lift_s_pop(il, r.width()))


class PUSHU(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth)
        size = r.width()
        # save the original U so the store uses the pre-decremented value
        old_u = TempReg(TempIncDecHelper, width=3)
        old_u.lift_assign(il, il.reg(3, RegisterName("U")))
        new_u = il.and_expr(
            3,
            il.sub(3, old_u.lift(il), il.const(3, size)),
            il.const(3, REG3_20BIT_MASK),
        )
        il.append(il.set_reg(3, RegisterName("U"), new_u))
        EMemHelper(size, Reg("U")).lift_assign(il, r.lift(il))
        if isinstance(r, RegIMR):
            r.lift_assign(il, il.and_expr(1, r.lift(il), il.const(1, 0x7F)))


class POPU(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth)
        size = r.width()
        # preserve the pointer prior to increment so the load happens at
        # the original U value
        old_u = TempReg(TempIncDecHelper, width=3)
        old_u.lift_assign(il, il.reg(3, RegisterName("U")))
        r.lift_assign(il, EMemHelper(size, old_u).lift(il))
        il.append(
            il.set_reg(
                3,
                RegisterName("U"),
                il.and_expr(
                    3,
                    il.add(3, old_u.lift(il), il.const(3, size)),
                    il.const(3, REG3_20BIT_MASK),
                ),
            )
        )


class PUSHS(StackPushInstruction):
    pass


class POPS(StackPopInstruction):
    pass


class ArithmeticInstruction(Instruction):
    def width(self) -> int:
        first, _second = self.operands()
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        return first.width()

    @staticmethod
    def _lift_unsigned_to_width(
        il: LowLevelILFunction,
        value: ExpressionIndex,
        source_width: int,
        target_width: int,
    ) -> ExpressionIndex:
        """Resize an unsigned arithmetic operand to the destination width.

        Binary Ninja requires both arithmetic inputs to have the operation's
        width.  Mixed register pairs are architectural (for example ROM
        F2B62 uses ADD Y,BA), so relying on the mock evaluator's permissive
        width handling would generate invalid real-BN LLIL.
        """
        if source_width == target_width:
            return value
        if source_width < target_width:
            zero_extend = getattr(il, "zero_extend", None)
            if callable(zero_extend):
                return zero_extend(target_width, value)
            source_mask = (1 << (source_width * 8)) - 1
            return il.and_expr(target_width, value, il.const(target_width, source_mask))

        low_part = getattr(il, "low_part", None)
        if callable(low_part):
            return low_part(target_width, value)
        target_mask = (1 << (target_width * 8)) - 1
        return il.and_expr(target_width, value, il.const(target_width, target_mask))

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        # RegPair encodings may legally have a narrower source register.  Lift
        # these explicitly so the source is zero-extended before ADD/SUB and
        # flag calculation.  Other arithmetic forms retain the common path.
        if len(self._operands) != 1 or not isinstance(self._operands[0], RegPair):
            return super().lift(il, addr)

        lhs_op, rhs_op = tuple(self.operands())
        assert isinstance(lhs_op, HasWidth)
        assert isinstance(rhs_op, HasWidth)
        width = lhs_op.width()
        lhs = lhs_op.lift(il, side_effects=False)
        rhs = self._lift_unsigned_to_width(il, rhs_op.lift(il), rhs_op.width(), width)
        lhs_op.lift_assign(il, self.lift_operation2(il, lhs, rhs))

    def _lift_regpair_20bit_binary(
        self,
        il: LowLevelILFunction,
        subtract: bool,
    ) -> bool:
        if len(self._operands) != 1 or not isinstance(self._operands[0], RegPair):
            return False
        pair = self._operands[0]
        if pair.bit_width() != 20:
            return False

        lhs_op, rhs_op = tuple(self.operands())
        assert isinstance(lhs_op, Reg)
        assert isinstance(rhs_op, Reg)

        mask = il.const(3, REG3_20BIT_MASK)
        lhs = TempReg(TempMultiByte1, width=3)
        lhs_expr = self._lift_unsigned_to_width(il, lhs_op.lift(il), lhs_op.width(), 3)
        lhs.lift_assign(il, il.and_expr(3, lhs_expr, mask))
        rhs = TempReg(TempMultiByte2, width=3)
        rhs_expr = self._lift_unsigned_to_width(il, rhs_op.lift(il), rhs_op.width(), 3)
        rhs.lift_assign(il, il.and_expr(3, rhs_expr, mask))
        raw = TempReg(TempLoopByteResult, width=3)

        if subtract:
            raw.lift_assign(il, il.sub(3, lhs.lift(il), rhs.lift(il)))
            carry_or_borrow = il.compare_unsigned_greater_than(
                3, rhs.lift(il), lhs.lift(il)
            )
        else:
            raw.lift_assign(il, il.add(3, lhs.lift(il), rhs.lift(il)))
            carry_or_borrow = il.compare_unsigned_greater_than(
                3, raw.lift(il), il.const(3, REG3_20BIT_MASK)
            )

        result = TempReg(TempOverallZeroAcc, width=3)
        result.lift_assign(il, il.and_expr(3, raw.lift(il), mask))
        lhs_op.lift_assign(il, result.lift(il))
        il.append(il.set_flag(CFlag, carry_or_borrow))
        il.append(
            il.set_flag(ZFlag, il.compare_equal(3, result.lift(il), il.const(3, 0)))
        )
        return True


class ADD(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        if self._lift_regpair_20bit_binary(il, subtract=False):
            return
        super().lift(il, addr)

    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il.add(self.width(), il_arg1, il_arg2, CZFlag)


def _lift_binary_with_carry(
    il: LowLevelILFunction,
    width: int,
    lhs_expr: ExpressionIndex,
    rhs_expr: ExpressionIndex,
    *,
    subtract: bool,
) -> ExpressionIndex:
    """Lift ADC/SBC without losing carry when ``rhs + C`` wraps.

    A nested byte expression such as ``lhs + (rhs + C)`` cannot represent
    the carry from the inner addition when ``rhs == 0xff`` and ``C == 1``.
    Split the operation into two width-sized steps and combine their carry
    (or borrow) conditions explicitly.
    """
    lhs = TempReg(TempBcdAddEmul, width=width)
    rhs = TempReg(TempBcdSubEmul, width=width)
    carry_in = TempReg(TempBcdDigitCarry, width=1)
    intermediate = TempReg(TempBcdLowNibbleProcessing, width=width)
    result = TempReg(TempLoopByteResult, width=width)

    lhs.lift_assign(il, lhs_expr)
    rhs.lift_assign(il, rhs_expr)
    carry_in.lift_assign(il, il.flag(CFlag))

    if subtract:
        intermediate.lift_assign(il, il.sub(width, lhs.lift(il), rhs.lift(il)))
        first_carry = il.compare_unsigned_greater_than(
            width, rhs.lift(il), lhs.lift(il)
        )
        result.lift_assign(il, il.sub(width, intermediate.lift(il), carry_in.lift(il)))
        second_carry = il.compare_unsigned_greater_than(
            width, carry_in.lift(il), intermediate.lift(il)
        )
    else:
        intermediate.lift_assign(il, il.add(width, lhs.lift(il), rhs.lift(il)))
        first_carry = il.compare_unsigned_greater_than(
            width, lhs.lift(il), intermediate.lift(il)
        )
        result.lift_assign(il, il.add(width, intermediate.lift(il), carry_in.lift(il)))
        second_carry = il.compare_unsigned_greater_than(
            width, intermediate.lift(il), result.lift(il)
        )

    il.append(il.set_flag(CFlag, il.or_expr(1, first_carry, second_carry)))
    il.append(
        il.set_flag(ZFlag, il.compare_equal(width, result.lift(il), il.const(width, 0)))
    )
    return result.lift(il)


class ADC(ArithmeticInstruction):
    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return _lift_binary_with_carry(
            il,
            self.width(),
            il_arg1,
            il_arg2,
            subtract=False,
        )


class SUB(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        if self._lift_regpair_20bit_binary(il, subtract=True):
            return
        super().lift(il, addr)

    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il.sub(self.width(), il_arg1, il_arg2, CZFlag)


class SBC(ArithmeticInstruction):
    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return _lift_binary_with_carry(
            il,
            self.width(),
            il_arg1,
            il_arg2,
            subtract=True,
        )


def _conditional_assign(
    il: LowLevelILFunction,
    temp: TempReg,
    cond: ExpressionIndex,
    true_val: ExpressionIndex,
    false_val: ExpressionIndex,
) -> None:
    """Assign ``true_val`` or ``false_val`` to ``temp`` based on ``cond``."""
    label_true = LowLevelILLabel()
    label_false = LowLevelILLabel()
    label_end = LowLevelILLabel()

    il.append(il.if_expr(cond, label_true, label_false))
    il.mark_label(label_true)
    temp.lift_assign(il, true_val)
    il.append(il.goto(label_end))
    il.mark_label(label_false)
    temp.lift_assign(il, false_val)
    il.mark_label(label_end)


def bcd_add_emul(
    il: LowLevelILFunction, w: int, a: ExpressionIndex, b: ExpressionIndex
) -> Operand:
    assert w == 1, "BCD add currently only supports 1-byte operands"

    # Incoming CFlag is the BCD carry from the previous byte's BCD addition
    incoming_carry = il.flag(CFlag)

    # Low nibble addition: (a & 0xF) + (b & 0xF) + incoming_carry_byte
    a_low = il.and_expr(1, a, il.const(1, 0x0F))
    b_low = il.and_expr(1, b, il.const(1, 0x0F))
    sum_low_nibbles_val = il.add(1, a_low, b_low)
    sum_low_with_carry_val = il.add(
        1, sum_low_nibbles_val, incoming_carry
    )  # Max val 9+9+1 = 19 (0x13)

    # Adjust if low nibble sum > 9
    temp_sum_low_final_reg = TempReg(TempBcdLowNibbleProcessing, width=1)
    adj_low_needed = il.compare_unsigned_greater_than(
        1, sum_low_with_carry_val, il.const(1, 9)
    )
    sum_low_adjusted_val = il.add(1, sum_low_with_carry_val, il.const(1, 0x06))

    _conditional_assign(
        il,
        temp_sum_low_final_reg,
        adj_low_needed,
        sum_low_adjusted_val,
        sum_low_with_carry_val,
    )

    current_sum_low_final = temp_sum_low_final_reg.lift(il)
    result_low_nibble_val = il.and_expr(1, current_sum_low_final, il.const(1, 0x0F))
    carry_to_high_nibble_val = il.logical_shift_right(
        1, current_sum_low_final, il.const(1, 4)
    )  # 0 or 1

    # High nibble addition: (a >> 4) + (b >> 4) + carry_to_high_nibble_val
    a_high = il.logical_shift_right(1, a, il.const(1, 4))
    b_high = il.logical_shift_right(1, b, il.const(1, 4))
    sum_high_nibbles_val = il.add(1, a_high, b_high)
    sum_high_with_carry_val = il.add(
        1, sum_high_nibbles_val, carry_to_high_nibble_val
    )  # Max 9+9+1 = 19 (0x13)

    # Adjust if high nibble sum > 9
    temp_sum_high_final_reg = TempReg(TempBcdHighNibbleProcessing, width=1)
    adj_high_needed = il.compare_unsigned_greater_than(
        1, sum_high_with_carry_val, il.const(1, 9)
    )
    sum_high_adjusted_val = il.add(1, sum_high_with_carry_val, il.const(1, 0x06))

    _conditional_assign(
        il,
        temp_sum_high_final_reg,
        adj_high_needed,
        sum_high_adjusted_val,
        sum_high_with_carry_val,
    )

    current_sum_high_final = temp_sum_high_final_reg.lift(il)
    result_high_nibble_val = il.and_expr(1, current_sum_high_final, il.const(1, 0x0F))
    new_bcd_carry_out_byte_val = il.logical_shift_right(
        1, current_sum_high_final, il.const(1, 4)
    )  # 0 or 1

    result_byte_val = il.or_expr(
        1,
        il.shift_left(1, result_high_nibble_val, il.const(1, 4)),
        result_low_nibble_val,
    )

    output_reg = TempReg(TempBcdAddEmul, width=1)
    output_reg.lift_assign(il, result_byte_val)
    il.append(il.set_flag(CFlag, new_bcd_carry_out_byte_val))
    # Z flag for current byte (overall Z handled by lift_multi_byte)
    il.append(il.set_flag(ZFlag, il.compare_equal(1, result_byte_val, il.const(1, 0))))

    return output_reg


def bcd_sub_emul(
    il: LowLevelILFunction, w: int, a: ExpressionIndex, b: ExpressionIndex
) -> Operand:
    assert w == 1, "BCD sub currently only supports 1-byte operands"

    incoming_borrow = il.flag(CFlag)  # 0 for no borrow, 1 for borrow

    # Low nibble subtraction: (a_low) - (b_low) - incoming_borrow
    a_low = il.and_expr(1, a, il.const(1, 0x0F))
    b_low = il.and_expr(1, b, il.const(1, 0x0F))

    sub_val_low = il.add(1, b_low, incoming_borrow)  # bL + Cin
    temp_sub_low_val = il.sub(1, a_low, sub_val_low)

    # Check for borrow from low nibble
    borrow_from_low_val = il.compare_signed_less_than(
        1, temp_sub_low_val, il.const(1, 0)
    )

    final_low_nibble_reg = TempReg(TempBcdLowNibbleProcessing, width=1)
    adj_val_low = il.sub(1, temp_sub_low_val, il.const(1, 0x06))  # Subtract 6 if borrow

    _conditional_assign(
        il,
        final_low_nibble_reg,
        borrow_from_low_val,
        adj_val_low,
        temp_sub_low_val,
    )

    result_low_nibble_val = il.and_expr(
        1, final_low_nibble_reg.lift(il), il.const(1, 0x0F)
    )

    # High nibble subtraction: (a_high) - (b_high) - borrow_from_low_val
    a_high = il.logical_shift_right(1, a, il.const(1, 4))
    b_high = il.logical_shift_right(1, b, il.const(1, 4))

    sub_val_high = il.add(1, b_high, borrow_from_low_val)  # bH + borrow_low
    temp_sub_high_val = il.sub(1, a_high, sub_val_high)

    new_bcd_borrow_out_byte_val = il.compare_signed_less_than(
        1, temp_sub_high_val, il.const(1, 0)
    )
    final_high_nibble_reg = TempReg(TempBcdHighNibbleProcessing, width=1)
    adj_val_high = il.sub(1, temp_sub_high_val, il.const(1, 0x06))

    _conditional_assign(
        il,
        final_high_nibble_reg,
        new_bcd_borrow_out_byte_val,
        adj_val_high,
        temp_sub_high_val,
    )

    result_high_nibble_val = il.and_expr(
        1, final_high_nibble_reg.lift(il), il.const(1, 0x0F)
    )
    result_byte_val = il.or_expr(
        1,
        il.shift_left(1, result_high_nibble_val, il.const(1, 4)),
        result_low_nibble_val,
    )

    output_reg = TempReg(TempBcdSubEmul, width=1)
    output_reg.lift_assign(il, result_byte_val)
    il.append(il.set_flag(CFlag, new_bcd_borrow_out_byte_val))  # C=1 if borrow
    il.append(il.set_flag(ZFlag, il.compare_equal(1, result_byte_val, il.const(1, 0))))

    return output_reg


def lift_multi_byte(
    il: LowLevelILFunction,
    op1: Operand,
    op2: Operand,
    clear_carry: bool = False,
    reverse: bool = False,
    bcd: bool = False,
    subtract: bool = False,
    pre: Optional[int] = None,
    reg_source_first_byte_only: bool = False,
) -> None:
    assert isinstance(op1, HasWidth), f"Expected HasWidth, got {type(op1)}"
    _lift_validate_nonzero_count(il)

    dst_mode = get_addressing_mode(pre, 1)
    src_mode = get_addressing_mode(pre, 2)

    # Helper to create load/store/advance logic for operands
    def make_handlers(
        op: Operand,
        is_dest_op: bool,
        mode: Optional[AddressingMode],
    ) -> Tuple[
        Callable[[], ExpressionIndex],
        Callable[[ExpressionIndex], None],
        Callable[[], None],
    ]:
        if isinstance(op, Pointer):
            # Temp reg to hold the iterating pointer for memory operands
            ptr_temp_reg_const = TempMultiByte1 if is_dest_op else TempMultiByte2
            ptr = TempReg(
                ptr_temp_reg_const, width=3
            )  # Three-byte LLIL container; address consumers retain 20 bits.

            # Initialize the pointer temp reg with the initial address from the operand
            # side_effects=False for source, potentially True for dest if pre/post inc/dec
            ptr.lift_assign(
                il,
                op.lift_current_addr(il, pre=mode, side_effects=is_dest_op),
            )

            def load() -> ExpressionIndex:
                # Use width 'w' (e.g. 1 for byte) for memory load/store element size
                assert isinstance(op, Pointer)
                # Use AddressingMode.N since ptr already contains the final address
                return op.memory_helper()(w, ptr).lift(il, pre=AddressingMode.N)

            def store(val: ExpressionIndex) -> None:
                assert isinstance(op, Pointer)
                # Use AddressingMode.N since ptr already contains the final address
                op.memory_helper()(w, ptr).lift_assign(il, val, pre=AddressingMode.N)

            def advance() -> None:
                op_il_math = il.sub if reverse else il.add
                # Advance pointer by element width 'w'
                next_addr = op_il_math(3, ptr.lift(il), il.const(3, w))
                if isinstance(op, IMem8):
                    # The encoded internal-memory address is eight bits.  Keep
                    # block iteration inside the 0x00..0xff internal window
                    # instead of spilling a three-byte LLIL temporary into external
                    # memory at either boundary.
                    offset = il.sub(3, next_addr, il.const(3, INTERNAL_MEMORY_START))
                    wrapped_offset = il.and_expr(3, offset, il.const(3, 0xFF))
                    next_addr = il.add(
                        3, il.const(3, INTERNAL_MEMORY_START), wrapped_offset
                    )
                else:
                    next_addr = il.and_expr(3, next_addr, il.const(3, REG3_20BIT_MASK))
                ptr.lift_assign(il, next_addr)  # ptr is 3 bytes
        else:  # Register operand
            if reg_source_first_byte_only and not is_dest_op:
                # DADL/DSBL register source uses the register value for the
                # first byte only; remaining bytes use 0x00.
                src_once = TempReg(TempMultiByte2, width=w)
                src_once.lift_assign(il, op.lift(il))

                def load() -> ExpressionIndex:
                    return src_once.lift(il)

                def store(val: ExpressionIndex) -> None:
                    op.lift_assign(il, val)

                def advance() -> None:
                    src_once.lift_assign(il, il.const(w, 0))
            else:

                def load() -> ExpressionIndex:
                    return op.lift(il)

                def store(val: ExpressionIndex) -> None:
                    op.lift_assign(il, val)

                def advance() -> (
                    None
                ):  # No advancement for direct register operands in a loop
                    pass

        return load, store, advance

    w = op1.width()

    load1, store1, adv1 = make_handlers(op1, True, dst_mode)
    load2, _store2, adv2 = make_handlers(op2, False, src_mode)

    if clear_carry:
        il.append(il.set_flag(CFlag, il.const(1, 0)))

    overall_zero_acc_reg = TempReg(TempOverallZeroAcc, width=w)
    overall_zero_acc_reg.lift_assign(il, il.const(w, 0))

    # TempReg to store the result of the current byte's main arithmetic operation
    byte_op_result_holder = TempReg(TempLoopByteResult, width=w)

    with lift_loop(il):  # loop_reg is 'I', controls number of iterations (bytes)
        a = load1()  # ExpressionIndex for current byte of op1
        b = load2()  # ExpressionIndex for current byte of op2

        # This will hold the evaluated result of the current byte's operation
        # before it's stored or used in overall_zero_acc.
        current_byte_calculated_value_expr: ExpressionIndex

        if bcd:
            # BCD operations are complex; they read il.flag(CFlag) internally for incoming carry,
            # perform BCD arithmetic, set CFlag and ZFlag (for the byte) based on BCD logic,
            # and return an Operand (specifically a TempReg like TempBcdAddEmul or TempBcdSubEmul)
            # which holds the BCD result of the current byte.
            bcd_op_result_operand: Operand
            if subtract:  # DSBL
                bcd_op_result_operand = bcd_sub_emul(il, w, a, b)
            else:  # DADL
                bcd_op_result_operand = bcd_add_emul(il, w, a, b)

            # The expression for the result of this byte's BCD operation
            current_byte_calculated_value_expr = bcd_op_result_operand.lift(il)
            # No need to assign to byte_op_result_holder if flags are fully set by bcd_emul
            # and result is self-contained in its returned TempReg.
            # The flags (C and Z for the byte) are set by set_flag calls within bcd_xxx_emul.
        else:  # Binary: ADCL, SBCL
            byte_op_result_holder.lift_assign(
                il,
                _lift_binary_with_carry(il, w, a, b, subtract=subtract),
            )
            current_byte_calculated_value_expr = byte_op_result_holder.lift(
                il
            )  # = REG(TempLoopByteResult)

        # Store the result for the current byte using the calculated value
        store1(current_byte_calculated_value_expr)

        # Accumulate for overall Zero flag check. This OR must not affect C/Z flags.
        overall_zero_acc_reg.lift_assign(
            il,
            il.or_expr(
                w, overall_zero_acc_reg.lift(il), current_byte_calculated_value_expr
            ),
        )

        adv1()
        adv2()

    # After loop, set the final Zero flag based on the accumulator
    il.append(
        il.set_flag(
            ZFlag, il.compare_equal(w, overall_zero_acc_reg.lift(il), il.const(w, 0))
        )
    )
    # The Carry flag (FC) will hold the carry/borrow from the last byte's operation.


class ADCL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # ADCL uses the incoming carry flag for the first byte.
        lift_multi_byte(il, dst, src, clear_carry=False, pre=self._pre)


class SBCL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # SBCL uses the incoming carry (borrow) flag for the first byte.
        lift_multi_byte(il, dst, src, subtract=True, clear_carry=False, pre=self._pre)


class DADL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # DADL does not use incoming carry for the first byte (implicitly 0).
        lift_multi_byte(
            il,
            dst,
            src,
            clear_carry=True,
            bcd=True,
            reverse=True,
            pre=self._pre,
            reg_source_first_byte_only=not isinstance(src, Pointer),
        )


class DSBL(ArithmeticInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        # DSBL uses the incoming carry (borrow) flag for the first byte.
        lift_multi_byte(
            il,
            dst,
            src,
            bcd=True,
            subtract=True,
            reverse=True,
            clear_carry=False,
            pre=self._pre,
            reg_source_first_byte_only=not isinstance(src, Pointer),
        )


class LogicInstruction(Instruction):
    pass


class AND(LogicInstruction):
    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il.and_expr(1, il_arg1, il_arg2, ZFlag)


class OR(LogicInstruction):
    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il.or_expr(1, il_arg1, il_arg2, ZFlag)


class XOR(LogicInstruction):
    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il.xor_expr(1, il_arg1, il_arg2, ZFlag)


class CompareInstruction(Instruction):
    pass


class TEST(CompareInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)
        first, second = self.operands()
        and_result = il.and_expr(1, first.lift(il, dst_mode), second.lift(il, src_mode))
        il.append(
            il.set_flag(
                ZFlag,
                il.compare_equal(1, and_result, il.const(1, 0)),
            )
        )


class CMP(CompareInstruction):
    def width(self) -> int:
        return 1

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)
        first, second = self.operands()
        il.append(
            il.sub(
                self.width(),
                first.lift(il, dst_mode),
                second.lift(il, src_mode),
                CZFlag,
            )
        )


class CMPW(CMP):
    def width(self) -> int:
        return 2


class CMPP(CMP):
    def width(self) -> int:
        return 3

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        if self.opcode != 0xD7:
            # C7 compares two raw three-byte internal-memory images.
            super().lift(il, addr)
            return

        # D7 compares an internal-memory pointer image with X/Y/U/S. The
        # register class is 20-bit, and the baseline emulator independently
        # models this form with its 20-bit IMEM reader/flag width. Mask both
        # operands explicitly while retaining a three-byte LLIL container.
        # This remains a best-guess pending an upper-nibble hardware probe.
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)
        first, second = self.operands()
        mask = il.const(3, PC_MASK)
        lhs = il.and_expr(3, first.lift(il, dst_mode), mask)
        rhs = il.and_expr(3, second.lift(il, src_mode), mask)
        il.append(il.sub(3, lhs, rhs, CZFlag))


# Shift and rotate instructions operate on one bit
class ShiftRotateInstruction(Instruction):
    def shift_by(self, il: LowLevelILFunction) -> ExpressionIndex:
        return il.const(1, 1)


# bit rotation
class ROR(ShiftRotateInstruction):
    def lift_operation1(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex
    ) -> ExpressionIndex:
        return il.rotate_right(1, il_arg1, self.shift_by(il), CZFlag)


class ROL(ShiftRotateInstruction):
    def lift_operation1(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex
    ) -> ExpressionIndex:
        return il.rotate_left(1, il_arg1, self.shift_by(il), CZFlag)


# bit shift
class SHL(ShiftRotateInstruction):
    def lift_operation1(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex
    ) -> ExpressionIndex:
        return il.rotate_left_carry(
            1, il_arg1, self.shift_by(il), il.flag(CFlag), CZFlag
        )


class SHR(ShiftRotateInstruction):
    def lift_operation1(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex
    ) -> ExpressionIndex:
        return il.rotate_right_carry(
            1, il_arg1, self.shift_by(il), il.flag(CFlag), CZFlag
        )


# digit shift
class DecimalShiftInstruction(Instruction):
    def _lift_decimal_shift(self, il: LowLevelILFunction, is_left_shift: bool) -> None:
        (imem_op,) = self.operands()
        assert isinstance(imem_op, IMem8), (
            f"{self.__class__.__name__} operand should be IMem8, got {type(imem_op)}"
        )
        _lift_validate_nonzero_count(il)

        current_addr_reg = TempReg(TempMultiByte1, width=3)
        mode = get_addressing_mode(self._pre, 1)
        current_addr_reg.lift_assign(
            il, imem_op.lift_current_addr(il, pre=mode, side_effects=False)
        )

        digit_carry_reg = TempReg(TempBcdDigitCarry, width=1)
        digit_carry_reg.lift_assign(il, il.const(1, 0))

        overall_zero_acc_reg = TempReg(TempOverallZeroAcc, width=1)
        overall_zero_acc_reg.lift_assign(il, il.const(1, 0))

        mem_accessor = IMemHelper(width=1, value=current_addr_reg)
        current_byte_reg = TempReg(TempLoopByteResult, width=1)

        with lift_loop(il):
            # Use AddressingMode.N since current_addr_reg already contains the final address
            # Snapshot the source byte before storing the shifted value.  Keeping
            # this as a lazy LOAD expression makes ``next_carry`` reread the
            # just-modified byte and propagate the wrong nibble.
            current_byte_reg.lift_assign(
                il, mem_accessor.lift(il, pre=AddressingMode.N)
            )
            current_byte_T = current_byte_reg.lift(il)

            T_low_nibble = il.and_expr(1, current_byte_T, il.const(1, 0x0F))
            T_high_nibble = il.logical_shift_right(1, current_byte_T, il.const(1, 4))

            shift_part = il.shift_left(1, T_low_nibble, il.const(1, 4))
            carry_part = digit_carry_reg.lift(il)
            next_carry = T_high_nibble
            addr_update = il.sub(3, current_addr_reg.lift(il), il.const(3, 1))

            if not is_left_shift:
                shift_part, T_high_nibble = T_high_nibble, shift_part
                carry_part = il.shift_left(1, carry_part, il.const(1, 4))
                next_carry = T_low_nibble
                addr_update = il.add(3, current_addr_reg.lift(il), il.const(3, 1))

            shifted_byte_S = il.or_expr(1, shift_part, carry_part)
            # Use AddressingMode.N since current_addr_reg already contains the final address
            mem_accessor.lift_assign(il, shifted_byte_S, pre=AddressingMode.N)
            digit_carry_reg.lift_assign(il, next_carry)

            overall_zero_acc_reg.lift_assign(
                il,
                il.or_expr(1, overall_zero_acc_reg.lift(il), shifted_byte_S),
            )

            current_addr_reg.lift_assign(il, addr_update)
            offset = il.sub(
                3, current_addr_reg.lift(il), il.const(3, INTERNAL_MEMORY_START)
            )
            current_addr_reg.lift_assign(
                il,
                il.add(
                    3,
                    il.const(3, INTERNAL_MEMORY_START),
                    il.and_expr(3, offset, il.const(3, 0xFF)),
                ),
            )

        il.append(
            il.set_flag(
                ZFlag,
                il.compare_equal(1, overall_zero_acc_reg.lift(il), il.const(1, 0)),
            )
        )
        # FC is not affected.


class DSLL(DecimalShiftInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        self._lift_decimal_shift(il, is_left_shift=True)


class DSRL(DecimalShiftInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        self._lift_decimal_shift(il, is_left_shift=False)


class IncDecInstruction(Instruction):
    def width(self) -> int:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected only one operand"
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        return first.width()


class INC(IncDecInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected only one operand"
        if isinstance(first, Reg3) and first.reg in REG3_20BIT_REGS:
            current = il.and_expr(3, first.lift(il), il.const(3, REG3_20BIT_MASK))
            result = TempReg(TempLoopByteResult, width=3)
            result.lift_assign(
                il,
                il.and_expr(
                    3,
                    il.add(3, current, il.const(3, 1)),
                    il.const(3, REG3_20BIT_MASK),
                ),
            )
            first.lift_assign(il, result.lift(il))
            il.append(
                il.set_flag(
                    ZFlag,
                    il.compare_equal(3, result.lift(il), il.const(3, 0)),
                )
            )
            return
        super().lift(il, addr)

    def lift_operation1(
        self, il: LowLevelILFunction, il_arg: ExpressionIndex
    ) -> ExpressionIndex:
        return il.add(self.width(), il_arg, il.const(self.width(), 1), ZFlag)


class DEC(IncDecInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected only one operand"
        if isinstance(first, Reg3) and first.reg in REG3_20BIT_REGS:
            current = il.and_expr(3, first.lift(il), il.const(3, REG3_20BIT_MASK))
            result = TempReg(TempLoopByteResult, width=3)
            result.lift_assign(
                il,
                il.and_expr(
                    3,
                    il.sub(3, current, il.const(3, 1)),
                    il.const(3, REG3_20BIT_MASK),
                ),
            )
            first.lift_assign(il, result.lift(il))
            il.append(
                il.set_flag(
                    ZFlag,
                    il.compare_equal(3, result.lift(il), il.const(3, 0)),
                )
            )
            return
        super().lift(il, addr)

    def lift_operation1(
        self, il: LowLevelILFunction, il_arg: ExpressionIndex
    ) -> ExpressionIndex:
        return il.sub(self.width(), il_arg, il.const(self.width(), 1), ZFlag)


class ExchangeInstruction(Instruction):
    def lift_single_exchange(self, il: LowLevelILFunction, addr: int) -> None:
        first, second = self.operands()
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        assert isinstance(second, HasWidth), f"Expected HasWidth, got {type(second)}"
        first_mode, second_mode = self._addressing_modes()

        # Snapshot both values before the first write.  This is required for
        # mixed-width ED forms and also makes an EXP quarantine check atomic.
        first_value = TempReg(TempExchange, width=first.width())
        first_value.lift_assign(
            il,
            _lift_operand_value(
                il,
                first,
                first_mode,
                side_effects=False,
            ),
        )
        second_value = TempReg(TempMultiByte2, width=second.width())
        second_value.lift_assign(
            il,
            _lift_operand_value(
                il,
                second,
                second_mode,
                side_effects=False,
            ),
        )

        if self.name() == "EXP":
            il.append(
                il.intrinsic(
                    [],
                    ValidateExpHighNibbleIntrinsic,
                    [first_value.lift(il), second_value.lift(il)],
                )
            )

        first.lift_assign(
            il,
            _resize_for_operand(
                il,
                second_value.lift(il),
                second.width(),
                first,
            ),
            first_mode,
        )
        second.lift_assign(
            il,
            _resize_for_operand(
                il,
                first_value.lift(il),
                first.width(),
                second,
            ),
            second_mode,
        )

    def encode(self, encoder: Encoder, addr: int) -> None:
        op1, op2 = self.operands()
        if isinstance(op1, IMemOperand) and isinstance(op2, IMemOperand):
            pre_key = (op1.mode, op2.mode)
            if pre_key == (AddressingMode.BP_N, AddressingMode.BP_N):
                # Leave an explicitly supplied prefix intact so the shared
                # encoder rejects it as noncanonical instead of silently
                # repairing a malformed direct Instruction object.
                return super().encode(encoder, addr)
            pre_byte = REVERSE_PRE_TABLE.get(pre_key)
            if pre_byte is None:
                raise ValueError(
                    f"Invalid addressing mode combination for {self.name()}: {op1.mode.value} and {op2.mode.value}"
                )
            if self._pre is None:
                self._pre = pre_byte

        super().encode(encoder, addr)


class EX(ExchangeInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        self.lift_single_exchange(il, addr)


# uses counter
class EXL(ExchangeInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        first, second = self.operands()
        assert isinstance(first, IMem8), f"Expected IMem8, got {type(first)}"
        assert isinstance(second, IMem8), f"Expected IMem8, got {type(second)}"
        _lift_validate_nonzero_count(il)

        first_mode = get_addressing_mode(self._pre, 1)
        second_mode = get_addressing_mode(self._pre, 2)
        first_addr = TempReg(TempMultiByte1, width=3)
        second_addr = TempReg(TempMultiByte2, width=3)
        first_addr.lift_assign(
            il, first.lift_current_addr(il, pre=first_mode, side_effects=False)
        )
        second_addr.lift_assign(
            il, second.lift_current_addr(il, pre=second_mode, side_effects=False)
        )

        def advance_imem_addr(reg: TempReg) -> None:
            next_addr = il.add(3, reg.lift(il), il.const(3, 1))
            offset = il.sub(3, next_addr, il.const(3, INTERNAL_MEMORY_START))
            wrapped_offset = il.and_expr(3, offset, il.const(3, 0xFF))
            reg.lift_assign(
                il,
                il.add(3, il.const(3, INTERNAL_MEMORY_START), wrapped_offset),
            )

        with lift_loop(il):
            first_mem = IMemHelper(width=1, value=first_addr)
            second_mem = IMemHelper(width=1, value=second_addr)
            tmp = TempReg(TempExchange, width=1)
            tmp.lift_assign(il, first_mem.lift(il, pre=AddressingMode.N))
            first_mem.lift_assign(
                il,
                second_mem.lift(il, pre=AddressingMode.N),
                pre=AddressingMode.N,
            )
            second_mem.lift_assign(il, tmp.lift(il), pre=AddressingMode.N)
            advance_imem_addr(first_addr)
            advance_imem_addr(second_addr)


class MiscInstruction(Instruction):
    pass


class WAIT(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        # The intrinsic preserves exact cycle accounting for direct LLIL
        # evaluation; the decoded executor uses an equivalent fast path.
        _lift_validate_nonzero_count(il)
        il.append(il.intrinsic([], WAITIntrinsic, []))


class PMDF(MiscInstruction):
    # The stock ROM uses complementary immediates such as F5/0B and FF/01 on
    # BP/PY to move frame pointers backward/forward.  That strongly supports
    # 8-bit wrapping binary pointer addition and rules out the old "packed
    # BCD" implementation.  Not requesting flag writes remains a
    # manual-derived model contract pending hardware trace.
    def lift_operation2(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex
    ) -> ExpressionIndex:
        return il.add(1, il_arg1, il_arg2)


class SWAP(MiscInstruction):
    def lift_operation1(
        self, il: LowLevelILFunction, il_arg1: ExpressionIndex
    ) -> ExpressionIndex:
        low = il.and_expr(1, il_arg1, il.const(1, 0x0F))
        low = il.shift_left(1, low, il.const(1, 4))
        high = il.and_expr(1, il_arg1, il.const(1, 0xF0))
        high = il.logical_shift_right(1, high, il.const(1, 4))
        return il.or_expr(1, low, high, ZFlag)


class SC(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.set_flag(CFlag, il.const(1, 1)))


class RC(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.set_flag(CFlag, il.const(1, 0)))


# Timer Clear: sub-CG or main-CG timers are reset when STCL / MTCL of LCC are
# set.
# Divider ← D
class TCL(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], TCLIntrinsic, []))


# System Clock Stop: halts main-CG of CPU
# Execution can continue past HALT: ON, IRQ, KI pins
# USR resets bits 0 to 2/5 to 0
# SSR bit 2 and USR 3 and 4 are set to 1
class HALT(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], HALTIntrinsic, []))


# System Clock Stop; Sub Clock Stop: main-CG and sub-CG of CPU are stopped
# Execution can continue past OFF: ON, IRQ, KI pins
class OFF(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], OFFIntrinsic, []))


# AKA `INT / Interrupt`
# 1. Save context to system stack (S-stack), in this strict order:
#      - PS  (Program Status)
#      - PC  (Program Counter, high byte first, then low byte)
#      - FLAG (Status Flags)
#      - IMR (Interrupt Mask Register at FBH)
#    (Total pushed = 5 bytes)
#
# 2. Load new PC and PS from fixed memory locations:
#      - PC high-byte loaded from address FFFFBH
#      - PC low-byte  loaded from address FFFFAH
#      - PS loaded from address FFFFCH
#
# 3. After pushing IMR, bit 7 (IRM) of IMR is forcibly cleared to 0.
class IR(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        # Fetch the architectural vector into a dedicated temporary before
        # any S-stack/IMR mutation. The validation intrinsic compares this
        # value with a side-effect-free peek and statically checks its target;
        # the final jump reuses it rather than reading a volatile bus twice.
        mem = EMemAddr(width=3)
        mem.value = INTERRUPT_VECTOR_ADDR
        vector_target = TempReg(TempVectorTarget, width=3)
        vector_target.lift_assign(il, mem.lift(il))
        il.append(
            il.intrinsic(
                [],
                ValidateVectorTransferIntrinsic,
                [
                    il.const(3, INTERRUPT_VECTOR_ADDR),
                    il.const(3, addr & PC_MASK),
                    vector_target.lift(il),
                ],
            )
        )
        imr, *_rest = RegIMR().operands()
        imr_value = imr.lift(il)
        # Software IR saves the address of the IR opcode itself.  The ROM
        # dispatcher identifies 0xFE at that saved PC and advances the frame
        # by one before RETI; hardware interrupt delivery instead saves the
        # already-current resume PC in the runtime's separate entry path.
        _lift_s_push(il, 3, il.const(3, addr & PC_MASK))
        _lift_s_push(il, 1, RegF().lift(il))
        _lift_s_push(il, 1, imr_value)
        imr.lift_assign(il, il.and_expr(1, imr.lift(il), il.const(1, 0x7F)))

        il.append(il.jump(vector_target.lift(il)))


# RESET vector selection is ROM-grounded.  The register mutations performed by
# the emulator intrinsic (including preserving IMR and the arithmetic flags)
# remain an explicit model/manual-derived contract pending hardware tracing.
class RESET(MiscInstruction):
    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        # RESET transfers control to the reset vector (destination comes from memory), so there is no
        # fallthrough.
        info.add_branch(BranchType.UnresolvedBranch)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], RESETIntrinsic, []))


class UnknownInstruction(Instruction):
    def name(self) -> str:
        return f"??? ({self.opcode:02X})"
