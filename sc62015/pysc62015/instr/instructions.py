from .opcodes import *  # noqa: F401,F403
from binaryninja.enums import BranchType  # noqa: F401
from .traits import HasWidth
from typing import Callable
from binaryninja import InstructionInfo  # type: ignore
class NOP(Instruction):
     def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.nop())

class JumpInstruction(Instruction):
    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        raise NotImplementedError("lift_jump_addr() not implemented")

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        # expect TrueBranch to be handled by subclasses as it might require
        # llil logic to calculate the address
        info.add_branch(BranchType.FalseBranch, addr + self.length())


    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        if_true  = LowLevelILLabel()
        if_false = LowLevelILLabel()

        if self._cond:
            zero = il.const(1, 0)
            one  = il.const(1, 1)
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
            return first.lift(il)
        high_addr = addr & 0xFF0000
        return il.or_expr(3, first.lift(il), il.const(3, high_addr))

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)

        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        if isinstance(first, ImmOperand):
            # absolute address
            assert first.value is not None, "Value not set"
            dest = first.value
            info.add_branch(BranchType.TrueBranch, dest)

class JP_Rel(JumpInstruction):
    def name(self) -> str:
        return "JR" + (self._cond if self._cond else "")

    def lift_jump_addr(self, il: LowLevelILFunction, addr: int) -> ExpressionIndex:
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, ImmOffset), f"Expected ImmOffset, got {type(first)}"
        return il.const(3, addr + self.length() + first.offset_value())

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        first, *rest = self.operands()
        assert len(rest) == 0, "Expected no extra operands"
        assert isinstance(first, ImmOffset), f"Expected ImmOffset, got {type(first)}"
        dest = addr + self.length() + first.offset_value()
        info.add_branch(BranchType.TrueBranch, dest)

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
            result = addr & 0xFF0000 | result
        return result

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.CallDestination, self.dest_addr(addr))

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dest = self._dest()
        if dest.width() == 3:
            il.append(il.call(il.const_pointer(3, self.dest_addr(addr))))
            return
        # manually push 2 bytes of address + self.length()
        il.append(il.push(2, il.const(2, addr + self.length())))
        il.append(il.jump(il.const_pointer(3, self.dest_addr(addr))))


class RetInstruction(Instruction):
    def addr_size(self) -> int:
        return 2

    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.FunctionReturn)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        pop_val = il.pop(self.addr_size())
        if self.addr_size() == 2:
            high = il.and_expr(
                3, il.reg(3, RegisterName("PC")), il.const(3, 0xFF0000)
            )
            pop_val = il.or_expr(3, pop_val, high)
        il.append(il.ret(pop_val))

class RET(RetInstruction): pass
class RETF(RetInstruction):
    def addr_size(self) -> int:
        return 3
class RETI(RetInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        imr, *_rest = RegIMR().operands()
        imr.lift_assign(il, il.pop(1))
        RegF().lift_assign(il, il.pop(1))
        il.append(il.ret(il.pop(3)))


class MoveInstruction(Instruction):
    pass

class MV(MoveInstruction):
    pass

    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il_arg2

class MVL(MoveInstruction):
    def modify_addr_il(self, il: LowLevelILFunction) -> Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex]:
        return il.add
    
    def _update_address_with_wrap(
        self,
        il: LowLevelILFunction,
        reg: TempReg,
        update_func: Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex],
        operand: Operand
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
            reg.lift_assign(il, new_addr)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst, src = self.operands()
        assert isinstance(dst, Pointer), f"Expected Pointer, got {type(dst)}"
        assert isinstance(src, Pointer), f"Expected Pointer, got {type(src)}"
        # 0xCB and 0xCF variants use IMem8, IMem8
        dst_reg = TempReg(TempMvlDst)
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)

        dst_reg.lift_assign(
            il, dst.lift_current_addr(il, pre=dst_mode, side_effects=False)
        )
        src_reg = TempReg(TempMvlSrc)
        src_reg.lift_assign(
            il, src.lift_current_addr(il, pre=src_mode, side_effects=False)
        )
        
        # Debug: print initial addresses
        # print(f"MVL: dst_mode={dst_mode}, src_mode={src_mode}")

        with lift_loop(il):
            src_mem = src.memory_helper()(1, src_reg)
            dst_mem = dst.memory_helper()(1, dst_reg)
            # Use AddressingMode.N since src_reg and dst_reg already contain final addresses
            dst_mem.lift_assign(il, src_mem.lift(il, pre=AddressingMode.N), pre=AddressingMode.N)

            # +1 index
            func = self.modify_addr_il(il)
            dst_func = func
            if (
                isinstance(dst, IMem8)
                and isinstance(src, EMemValueOffsetHelper)
                and isinstance(src.value, RegIncrementDecrementHelper)
                and src.value.mode == EMemRegMode.PRE_DEC
            ):
                dst_func = il.sub

            # Update destination address with wrapping for IMem8
            self._update_address_with_wrap(il, dst_reg, dst_func, dst)

            if (
                isinstance(src, EMemValueOffsetHelper)
                and isinstance(src.value, RegIncrementDecrementHelper)
                and src.value.mode == EMemRegMode.PRE_DEC
            ):
                updated_src = src.lift_current_addr(il, pre=src_mode)
                src_reg.lift_assign(il, updated_src)
            else:
                # Update source address with wrapping for IMem8
                self._update_address_with_wrap(il, src_reg, func, src)
                src.lift_current_addr(il, pre=src_mode)

            # apply any addressing side effects for destination
            dst.lift_current_addr(il, pre=dst_mode)

class MVLD(MVL):
    def modify_addr_il(self, il: LowLevelILFunction) -> Callable[[int, ExpressionIndex, ExpressionIndex], ExpressionIndex]:
        return il.sub

class PRE(Instruction):
    def name(self) -> str:
        return f"PRE{self.opcode:02x}"

    def fuse(self, sister: 'Instruction') -> Optional['Instruction']:
        if isinstance(sister, PRE):
            return None
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
        il.append(il.push(r.width(), r.lift(il)))
class StackPopInstruction(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth), f"Expected HasWidth, got {type(r)}"
        r.lift_assign(il, il.pop(r.width()))

class PUSHU(StackInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        r = self.reg()
        assert isinstance(r, HasWidth)
        size = r.width()
        # save the original U so the store uses the pre-decremented value
        old_u = TempReg(TempIncDecHelper, width=3)
        old_u.lift_assign(il, il.reg(3, RegisterName("U")))
        new_u = il.sub(3, old_u.lift(il), il.const(3, size))
        il.append(il.set_reg(3, RegisterName("U"), new_u))
        il.append(il.store(size, new_u, r.lift(il)))
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
        r.lift_assign(il, il.load(size, old_u.lift(il)))
        il.append(
            il.set_reg(3, RegisterName("U"), il.add(3, old_u.lift(il), il.const(3, size)))
        )

class PUSHS(StackPushInstruction): pass
class POPS(StackPopInstruction): pass

class ArithmeticInstruction(Instruction):
    def width(self) -> int:
        first, _second = self.operands()
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        return first.width()
class ADD(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.add(self.width(), il_arg1, il_arg2, CZFlag)
class ADC(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.add(self.width(), il_arg1, il.add(self.width(), il_arg2,
                                                    il.flag(CFlag)), CZFlag)
class SUB(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.sub(self.width(), il_arg1, il_arg2, CZFlag)
class SBC(ArithmeticInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.sub(self.width(), il_arg1, il.add(self.width(), il_arg2,
                                                    il.flag(CFlag)), CZFlag)


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


def bcd_add_emul(il: LowLevelILFunction, w: int, a: ExpressionIndex, b: ExpressionIndex) -> Operand:
    assert w == 1, "BCD add currently only supports 1-byte operands"

    # Incoming CFlag is the BCD carry from the previous byte's BCD addition
    incoming_carry = il.flag(CFlag)

    # Low nibble addition: (a & 0xF) + (b & 0xF) + incoming_carry_byte
    a_low = il.and_expr(1, a, il.const(1, 0x0F))
    b_low = il.and_expr(1, b, il.const(1, 0x0F))
    sum_low_nibbles_val = il.add(1, a_low, b_low)
    sum_low_with_carry_val = il.add(1, sum_low_nibbles_val, incoming_carry) # Max val 9+9+1 = 19 (0x13)

    # Adjust if low nibble sum > 9
    temp_sum_low_final_reg = TempReg(TempBcdLowNibbleProcessing, width=1)
    adj_low_needed = il.compare_unsigned_greater_than(1, sum_low_with_carry_val, il.const(1, 9))
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
    carry_to_high_nibble_val = il.logical_shift_right(1, current_sum_low_final, il.const(1, 4)) # 0 or 1

    # High nibble addition: (a >> 4) + (b >> 4) + carry_to_high_nibble_val
    a_high = il.logical_shift_right(1, a, il.const(1, 4))
    b_high = il.logical_shift_right(1, b, il.const(1, 4))
    sum_high_nibbles_val = il.add(1, a_high, b_high)
    sum_high_with_carry_val = il.add(1, sum_high_nibbles_val, carry_to_high_nibble_val) # Max 9+9+1 = 19 (0x13)

    # Adjust if high nibble sum > 9
    temp_sum_high_final_reg = TempReg(TempBcdHighNibbleProcessing, width=1)
    adj_high_needed = il.compare_unsigned_greater_than(1, sum_high_with_carry_val, il.const(1, 9))
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
    new_bcd_carry_out_byte_val = il.logical_shift_right(1, current_sum_high_final, il.const(1, 4)) # 0 or 1

    result_byte_val = il.or_expr(1, il.shift_left(1, result_high_nibble_val, il.const(1, 4)), result_low_nibble_val)

    output_reg = TempReg(TempBcdAddEmul, width=1)
    output_reg.lift_assign(il, result_byte_val)
    il.append(il.set_flag(CFlag, new_bcd_carry_out_byte_val))
    # Z flag for current byte (overall Z handled by lift_multi_byte)
    il.append(il.set_flag(ZFlag, il.compare_equal(1, result_byte_val, il.const(1,0))))

    return output_reg

def bcd_sub_emul(il: LowLevelILFunction, w: int, a: ExpressionIndex, b: ExpressionIndex) -> Operand:
    assert w == 1, "BCD sub currently only supports 1-byte operands"

    incoming_borrow = il.flag(CFlag) # 0 for no borrow, 1 for borrow

    # Low nibble subtraction: (a_low) - (b_low) - incoming_borrow
    a_low = il.and_expr(1, a, il.const(1, 0x0F))
    b_low = il.and_expr(1, b, il.const(1, 0x0F))

    sub_val_low = il.add(1, b_low, incoming_borrow) # bL + Cin
    temp_sub_low_val = il.sub(1, a_low, sub_val_low)

    # Check for borrow from low nibble
    borrow_from_low_val = il.compare_signed_less_than(1, temp_sub_low_val, il.const(1, 0))

    final_low_nibble_reg = TempReg(TempBcdLowNibbleProcessing, width=1)
    adj_val_low = il.sub(1, temp_sub_low_val, il.const(1, 0x06)) # Subtract 6 if borrow

    _conditional_assign(
        il,
        final_low_nibble_reg,
        borrow_from_low_val,
        adj_val_low,
        temp_sub_low_val,
    )

    result_low_nibble_val = il.and_expr(1, final_low_nibble_reg.lift(il), il.const(1, 0x0F))

    # High nibble subtraction: (a_high) - (b_high) - borrow_from_low_val
    a_high = il.logical_shift_right(1, a, il.const(1, 4))
    b_high = il.logical_shift_right(1, b, il.const(1, 4))

    sub_val_high = il.add(1, b_high, borrow_from_low_val) # bH + borrow_low
    temp_sub_high_val = il.sub(1, a_high, sub_val_high)

    new_bcd_borrow_out_byte_val = il.compare_signed_less_than(1, temp_sub_high_val, il.const(1, 0))
    final_high_nibble_reg = TempReg(TempBcdHighNibbleProcessing, width=1)
    adj_val_high = il.sub(1, temp_sub_high_val, il.const(1, 0x06))

    _conditional_assign(
        il,
        final_high_nibble_reg,
        new_bcd_borrow_out_byte_val,
        adj_val_high,
        temp_sub_high_val,
    )

    result_high_nibble_val = il.and_expr(1, final_high_nibble_reg.lift(il), il.const(1, 0x0F))
    result_byte_val = il.or_expr(1, il.shift_left(1, result_high_nibble_val, il.const(1, 4)), result_low_nibble_val)

    output_reg = TempReg(TempBcdSubEmul, width=1)
    output_reg.lift_assign(il, result_byte_val)
    il.append(il.set_flag(CFlag, new_bcd_borrow_out_byte_val)) # C=1 if borrow
    il.append(il.set_flag(ZFlag, il.compare_equal(1, result_byte_val, il.const(1,0))))

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
) -> None:
    assert isinstance(op1, HasWidth), f"Expected HasWidth, got {type(op1)}"

    dst_mode = get_addressing_mode(pre, 1)
    src_mode = get_addressing_mode(pre, 2)

    # Helper to create load/store/advance logic for operands
    def make_handlers(
        op: Operand,
        is_dest_op: bool,
        mode: Optional[AddressingMode],
    ) -> Tuple[Callable[[], ExpressionIndex],
                     Callable[[ExpressionIndex], None],
                     Callable[[], None]]:
        if isinstance(op, Pointer):
            # Temp reg to hold the iterating pointer for memory operands
            ptr_temp_reg_const = TempMultiByte1 if is_dest_op else TempMultiByte2
            ptr = TempReg(ptr_temp_reg_const, width=3) # Addresses are 3 bytes (20/24 bit)

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
                ptr.lift_assign(il, op_il_math(3, ptr.lift(il), il.const(3, w))) # ptr is 3 bytes
        else: # Register operand
            def load() -> ExpressionIndex:
                return op.lift(il)
            def store(val: ExpressionIndex) -> None:
                op.lift_assign(il, val)
            def advance() -> None: # No advancement for direct register operands in a loop
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

    with lift_loop(il): # loop_reg is 'I', controls number of iterations (bytes)
        a = load1() # ExpressionIndex for current byte of op1
        b = load2() # ExpressionIndex for current byte of op2

        # This will hold the evaluated result of the current byte's operation
        # before it's stored or used in overall_zero_acc.
        current_byte_calculated_value_expr: ExpressionIndex

        if bcd:
            # BCD operations are complex; they read il.flag(CFlag) internally for incoming carry,
            # perform BCD arithmetic, set CFlag and ZFlag (for the byte) based on BCD logic,
            # and return an Operand (specifically a TempReg like TempBcdAddEmul or TempBcdSubEmul)
            # which holds the BCD result of the current byte.
            bcd_op_result_operand: Operand
            if subtract: # DSBL
                bcd_op_result_operand = bcd_sub_emul(il, w, a, b)
            else: # DADL
                bcd_op_result_operand = bcd_add_emul(il, w, a, b)

            # The expression for the result of this byte's BCD operation
            current_byte_calculated_value_expr = bcd_op_result_operand.lift(il)
            # No need to assign to byte_op_result_holder if flags are fully set by bcd_emul
            # and result is self-contained in its returned TempReg.
            # The flags (C and Z for the byte) are set by set_flag calls within bcd_xxx_emul.
        else: # Binary: ADCL, SBCL
            # These operations use il.flag(CFlag) for incoming carry and set CZFlag for outgoing.
            main_op_llil: ExpressionIndex

            # Capture the incoming carry flag *before* this byte's main operation
            initial_c_flag_expr = il.flag(CFlag)

            if subtract: # SBCL: m = m - n - C_in. Implemented as m - (n + C_in)
                         # The inner add (n + C_in) must NOT alter flags.
                term_to_subtract = il.add(w, b, initial_c_flag_expr)
                main_op_llil = il.sub(w, a, term_to_subtract, CZFlag) # This SUB sets C and Z flags
            else: # ADCL: m = m + n + C_in. Implemented as m + (n + C_in)
                  # The inner add (n + C_in) must NOT alter flags.
                term_to_add = il.add(w, b, initial_c_flag_expr)
                main_op_llil = il.add(w, a, term_to_add, CZFlag) # This ADD sets C and Z flags

            # Execute the main operation and store its result in byte_op_result_holder.
            # The flags (C and Z) are set when main_op_llil is evaluated as part of this set_reg.
            byte_op_result_holder.lift_assign(il, main_op_llil)
            current_byte_calculated_value_expr = byte_op_result_holder.lift(il) # = REG(TempLoopByteResult)

        # Store the result for the current byte using the calculated value
        store1(current_byte_calculated_value_expr)

        # Accumulate for overall Zero flag check. This OR must not affect C/Z flags.
        overall_zero_acc_reg.lift_assign(il, il.or_expr(w, overall_zero_acc_reg.lift(il), current_byte_calculated_value_expr))

        adv1()
        adv2()

    # After loop, set the final Zero flag based on the accumulator
    il.append(il.set_flag(ZFlag, il.compare_equal(w, overall_zero_acc_reg.lift(il), il.const(w, 0))))
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
        lift_multi_byte(il, dst, src, clear_carry=True, bcd=True, reverse=True, pre=self._pre)

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
        )


class LogicInstruction(Instruction): pass
class AND(LogicInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.and_expr(1, il_arg1, il_arg2, ZFlag)
class OR(LogicInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.or_expr(1, il_arg1, il_arg2, ZFlag)
class XOR(LogicInstruction):
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.xor_expr(1, il_arg1, il_arg2, ZFlag)

class CompareInstruction(Instruction): pass
class TEST(CompareInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        dst_mode = get_addressing_mode(self._pre, 1)
        src_mode = get_addressing_mode(self._pre, 2)
        first, second = self.operands()
        il.append(
            il.set_flag(
                ZFlag,
                il.and_expr(3, first.lift(il, dst_mode), second.lift(il, src_mode)),
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

# Shift and rotate instructions operate on one bit
class ShiftRotateInstruction(Instruction):
    def shift_by(self, il: LowLevelILFunction) -> ExpressionIndex:
        return il.const(1, 1)
# bit rotation
class ROR(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_right(1, il_arg1, self.shift_by(il), CZFlag)
class ROL(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_left(1, il_arg1, self.shift_by(il), CZFlag)
# bit shift
class SHL(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_left_carry(1, il_arg1, self.shift_by(il),
                                    il.flag(CFlag), CZFlag)
class SHR(ShiftRotateInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
        return il.rotate_right_carry(1, il_arg1, self.shift_by(il),
                                     il.flag(CFlag), CZFlag)

# digit shift
class DecimalShiftInstruction(Instruction):
    def _lift_decimal_shift(self, il: LowLevelILFunction, is_left_shift: bool) -> None:
        imem_op, = self.operands()
        assert isinstance(imem_op, IMem8), f"{self.__class__.__name__} operand should be IMem8, got {type(imem_op)}"

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

        with lift_loop(il):
            # Use AddressingMode.N since current_addr_reg already contains the final address
            current_byte_T = mem_accessor.lift(il, pre=AddressingMode.N)

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


class IncDecInstruction(Instruction): pass
class INC(IncDecInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg: ExpressionIndex) -> ExpressionIndex:
        return il.add(1, il_arg, il.const(1, 1), ZFlag)
class DEC(IncDecInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg: ExpressionIndex) -> ExpressionIndex:
        return il.sub(1, il_arg, il.const(1, 1), ZFlag)

class ExchangeInstruction(Instruction):
    def lift_single_exchange(self, il: LowLevelILFunction, addr: int) -> None:
        first, second = self.operands()
        assert isinstance(first, HasWidth), f"Expected HasWidth, got {type(first)}"
        width = first.width()
        tmp = TempReg(TempExchange, width=width)
        tmp.lift_assign(il, first.lift(il))
        first.lift_assign(il, second.lift(il))
        second.lift_assign(il, tmp.lift(il))

    def encode(self, encoder: Encoder, addr: int) -> None:
        op1, op2 = self.operands()
        if isinstance(op1, IMemOperand) and isinstance(op2, IMemOperand):
            pre_key = (op1.mode, op2.mode)
            pre_byte = REVERSE_PRE_TABLE.get(pre_key)
            if pre_byte is None:
                raise ValueError(
                    f"Invalid addressing mode combination for {self.name()}: {op1.mode.value} and {op2.mode.value}"
                )
            self._pre = pre_byte

        super().encode(encoder, addr)
class EX(ExchangeInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        self.lift_single_exchange(il, addr)
# uses counter
class EXL(ExchangeInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        with lift_loop(il):
            self.lift_single_exchange(il, addr)

class MiscInstruction(Instruction): pass
class WAIT(MiscInstruction):
    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        with lift_loop(il):
            # Wait is just an idle loop
            pass

class PMDF(MiscInstruction):
    # FIXME: verify
    def lift_operation2(self, il: LowLevelILFunction, il_arg1: ExpressionIndex, il_arg2: ExpressionIndex) -> ExpressionIndex:
        return il.add(1, il_arg1, il_arg2)

class SWAP(MiscInstruction):
    def lift_operation1(self, il: LowLevelILFunction, il_arg1: ExpressionIndex) -> ExpressionIndex:
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
        il.append(il.push(3, RegPC().lift(il)))
        il.append(il.push(1, RegF().lift(il)))
        imr, *_rest = RegIMR().operands()
        il.append(il.push(1, imr.lift(il)))
        imr.lift_assign(il, il.and_expr(1, imr.lift(il), il.const(1, 0x7F)))

        mem = EMemAddr(width=3)
        mem.value = INTERRUPT_VECTOR_ADDR
        il.append(il.jump(mem.lift(il)))

# ACM bit 7, UCR + USR bits 0 to 2/5, IMR, SCR, SSR bit 2 are all reset to 0
# USR bits 3 and 4 are set to 1
class RESET(MiscInstruction):
    def analyze(self, info: InstructionInfo, addr: int) -> None:
        super().analyze(info, addr)
        info.add_branch(BranchType.FunctionReturn)

    def lift(self, il: LowLevelILFunction, addr: int) -> None:
        il.append(il.intrinsic([], RESETIntrinsic, []))

class UnknownInstruction(Instruction):
    def name(self) -> str:
        return f"??? ({self.opcode:02X})"


