use pyo3::{PyErr, PyResult, Python};

use crate::constants::{INTERNAL_MEMORY_START, PC_MASK};
use crate::lowering::OpFlags;
use crate::memory::MemoryBus;
use crate::state::{Flag, Register, RegisterError, Registers};
use crate::{LlilExpr, LlilNode, LlilOperand, LlilProgram};

#[derive(Debug)]
pub enum ExecutionError {
    UnsupportedOpcode,
    Unimplemented(&'static str),
    InvalidOperand(&'static str),
    MissingValue(&'static str),
    Register(RegisterError),
    Python(PyErr),
}

#[derive(Clone, Copy, Debug, Default)]
struct EvalResult {
    value: Option<i64>,
    carry: Option<u8>,
    zero: Option<u8>,
}

impl EvalResult {
    fn none() -> Self {
        Self::default()
    }

    fn with_value(value: i64) -> Self {
        Self {
            value: Some(value),
            carry: None,
            zero: None,
        }
    }

    fn with_flags(mut self, carry: Option<u8>, zero: Option<u8>) -> Self {
        self.carry = carry;
        self.zero = zero;
        self
    }

    fn expect_value(self, context: &'static str) -> Result<i64, ExecutionError> {
        self.value.ok_or(ExecutionError::MissingValue(context))
    }

    fn flag(&self, name: char) -> Option<u8> {
        match name {
            'C' => self.carry,
            'Z' => self.zero,
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RuntimeState {
    halted: bool,
}

impl RuntimeState {
    pub fn halted(&self) -> bool {
        self.halted
    }

    pub fn set_halted(&mut self, value: bool) {
        self.halted = value;
    }
}

pub type ExecutionResult = Result<(), ExecutionError>;

const IMEM_LCC: u32 = INTERNAL_MEMORY_START + 0xFE;
const IMEM_UCR: u32 = INTERNAL_MEMORY_START + 0xF7;
const IMEM_USR: u32 = INTERNAL_MEMORY_START + 0xF8;
const IMEM_ISR: u32 = INTERNAL_MEMORY_START + 0xFC;
const IMEM_SCR: u32 = INTERNAL_MEMORY_START + 0xFD;
const IMEM_SSR: u32 = INTERNAL_MEMORY_START + 0xFF;
const RESET_VECTOR_BASE: u32 = 0x00FFFFA;

pub struct LlilRuntime {
    registers: Registers,
    memory: MemoryBus,
    state: RuntimeState,
}

impl LlilRuntime {
    pub fn new(registers: Registers, memory: MemoryBus) -> Self {
        Self {
            registers,
            memory,
            state: RuntimeState::default(),
        }
    }

    pub fn write_named_register(
        &mut self,
        name: &str,
        value: i64,
        width: Option<u8>,
    ) -> ExecutionResult {
        self.write_register(name, value, width)?;
        Ok(())
    }

    pub fn read_named_register(&self, name: &str) -> Result<i64, ExecutionError> {
        self.read_register(name)
    }

    pub fn read_memory_value(&mut self, address: i64, width: u8) -> Result<i64, ExecutionError> {
        let size = width.max(1) as usize;
        self.read_memory_bytes(address, size)
    }

    pub fn write_memory_value(
        &mut self,
        address: i64,
        width: u8,
        value: i64,
    ) -> Result<(), ExecutionError> {
        let size = width.max(1) as usize;
        self.write_memory_bytes(address, size, value)
    }

    pub fn apply_op_flags(&mut self, flags: OpFlags) -> ExecutionResult {
        if let Some(value) = flags.carry {
            self.write_flag_char('C', value)?;
        }
        if let Some(value) = flags.zero {
            self.write_flag_char('Z', value)?;
        }
        Ok(())
    }

    pub fn into_parts(self) -> (Registers, MemoryBus, RuntimeState) {
        (self.registers, self.memory, self.state)
    }

    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut RuntimeState {
        &mut self.state
    }

    pub fn registers(&self) -> &Registers {
        &self.registers
    }

    pub fn registers_mut(&mut self) -> &mut Registers {
        &mut self.registers
    }

    pub fn prepare_for_opcode(&mut self, opcode: u8, length: u8) {
        let current_pc = self.registers.get(Register::PC);
        let next_pc = current_pc.wrapping_add(length as u32) & PC_MASK;
        self.registers.set(Register::PC, next_pc);

        let effect = crate::CALL_STACK_EFFECTS[opcode as usize];
        if effect > 0 {
            self.registers.increment_call_sub_level();
        } else if effect < 0 {
            self.registers.decrement_call_sub_level();
        }
    }

    pub fn execute_program(&mut self, program: &LlilProgram) -> ExecutionResult {
        let mut cache: Vec<Option<EvalResult>> = vec![None; program.expressions.len()];
        let mut labels: Vec<usize> = Vec::new();

        for (idx, node) in program.nodes.iter().enumerate() {
            if let LlilNode::Label { label } = node {
                let label_idx = *label as usize;
                if labels.len() <= label_idx {
                    labels.resize(label_idx + 1, 0);
                }
                labels[label_idx] = idx;
            }
        }

        let mut pc: usize = 0;
        while pc < program.nodes.len() {
            match program.nodes[pc] {
                LlilNode::Expr { expr } => {
                    self.eval_expr(program, expr, &mut cache)?;
                    pc += 1;
                }
                LlilNode::If {
                    cond,
                    true_label,
                    false_label,
                } => {
                    let cond_val = self
                        .eval_expr(program, cond, &mut cache)?
                        .value
                        .unwrap_or(0);
                    let target = if cond_val != 0 {
                        true_label
                    } else {
                        false_label
                    };
                    pc = labels
                        .get(target as usize)
                        .copied()
                        .unwrap_or_else(|| pc + 1);
                }
                LlilNode::Goto { label } => {
                    pc = labels
                        .get(label as usize)
                        .copied()
                        .unwrap_or_else(|| pc + 1);
                }
                LlilNode::Label { .. } => pc += 1,
            }
        }
        Ok(())
    }

    fn eval_expr(
        &mut self,
        program: &LlilProgram,
        index: u16,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let idx = index as usize;
        if let Some(result) = cache[idx] {
            return Ok(result);
        }

        let expr = &program.expressions[idx];
        let result = match expr.op {
            "NOP" => EvalResult::with_value(0),
            "CONST" | "CONST_PTR" => {
                let value = self.operand_value(program, expr.operands, 0, cache)?;
                EvalResult::with_value(value)
            }
            "REG" => {
                let name = self.operand_reg(expr.operands, 0)?;
                EvalResult::with_value(self.read_register(name)?)
            }
            "FLAG" => {
                let name = self.operand_flag(expr.operands, 0)?;
                EvalResult::with_value(self.read_flag(name)?)
            }
            "SET_REG" => {
                let name = self.operand_reg(expr.operands, 0)?;
                let value = self.operand_value(program, expr.operands, 1, cache)?;
                self.write_register(name, value, expr.width)?;
                EvalResult::none()
            }
            "SET_FLAG" => {
                let name = self.operand_flag(expr.operands, 0)?;
                let value = self.operand_value(program, expr.operands, 1, cache)?;
                self.write_flag(name, value)?;
                EvalResult::none()
            }
            "ADD" => self.eval_add(program, expr, cache)?,
            "SUB" => self.eval_sub(program, expr, cache)?,
            "AND" | "OR" | "XOR" => self.eval_logical(program, expr, cache)?,
            "CMP_E" | "CMP_SLT" | "CMP_UGT" => self.eval_compare(program, expr, cache)?,
            "LOAD" => {
                let size = expr
                    .width
                    .ok_or(ExecutionError::Unimplemented("LOAD width"))?;
                let address = self.operand_value(program, expr.operands, 0, cache)?;
                let value = self.read_memory_bytes(address, size as usize)?;
                EvalResult::with_value(value)
            }
            "STORE" => {
                let size = expr
                    .width
                    .ok_or(ExecutionError::Unimplemented("STORE width"))?;
                let address = self.operand_value(program, expr.operands, 0, cache)?;
                let value = self.operand_value(program, expr.operands, 1, cache)?;
                self.write_memory_bytes(address, size as usize, value)?;
                EvalResult::none()
            }
            "PUSH" => {
                let size = expr
                    .width
                    .ok_or(ExecutionError::Unimplemented("PUSH width"))?;
                let value = self.operand_value(program, expr.operands, 0, cache)?;
                self.push_value(size as u8, value)?;
                EvalResult::none()
            }
            "POP" => {
                let size = expr
                    .width
                    .ok_or(ExecutionError::Unimplemented("POP width"))?;
                let value = self.pop_value(size as u8)?;
                EvalResult::with_value(value)
            }
            "JUMP" => {
                let target = self.operand_value(program, expr.operands, 0, cache)?;
                self.write_register("PC", target, Some(3))?;
                EvalResult::none()
            }
            "RET" => {
                let target = self.operand_value(program, expr.operands, 0, cache)?;
                self.write_register("PC", target, Some(3))?;
                EvalResult::none()
            }
            "CALL" => {
                let target_idx = self.operand_expr_index(expr.operands, 0)?;
                let target_value = self
                    .eval_expr(program, target_idx, cache)?
                    .expect_value("CALL target")?;
                let push_size = self.call_push_size(program, expr);
                let ret_addr = self.read_register("PC")?;
                let sp = self.read_register("S")?;
                let new_sp = sp - push_size as i64;
                self.write_memory_bytes(new_sp, push_size, ret_addr)?;
                self.write_register("S", new_sp, Some(3))?;
                self.write_register("PC", target_value, Some(3))?;
                EvalResult::none()
            }
            "LSL" | "LSR" => self.eval_shift(program, expr, cache)?,
            "ROL" | "ROR" => self.eval_rotate(program, expr, cache)?,
            "RLC" | "RRC" => self.eval_rotate_through_carry(program, expr, cache)?,
            "INTRINSIC" => {
                let name = expr
                    .intrinsic
                    .ok_or(ExecutionError::Unimplemented("intrinsic name"))?;
                self.invoke_intrinsic(name)?;
                EvalResult::none()
            }
            "UNIMPL" => return Err(ExecutionError::Unimplemented("UNIMPL")),
            other => return Err(ExecutionError::Unimplemented(other)),
        };

        self.apply_flags(expr, &result)?;
        cache[idx] = Some(result);
        Ok(result)
    }

    fn eval_add(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let lhs = self.operand_value(program, expr.operands, 0, cache)?;
        let rhs = self.operand_value(program, expr.operands, 1, cache)?;
        let width = expr
            .width
            .ok_or(ExecutionError::Unimplemented("ADD width"))?;
        let mask = mask_for_bytes_i128(width);
        let full = lhs as i128 + rhs as i128;
        let result = (full & mask) as i64;
        let carry = bool_to_u8(full > mask);
        let zero = bool_to_u8(result == 0);
        Ok(EvalResult::with_value(result).with_flags(Some(carry), Some(zero)))
    }

    fn eval_sub(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let lhs = self.operand_value(program, expr.operands, 0, cache)?;
        let rhs = self.operand_value(program, expr.operands, 1, cache)?;
        let width = expr
            .width
            .ok_or(ExecutionError::Unimplemented("SUB width"))?;
        let mask = mask_for_bytes_i128(width);
        let full = lhs as i128 - rhs as i128;
        let result = (full & mask) as i64;
        let carry = bool_to_u8(full < 0);
        let zero = bool_to_u8(result == 0);
        Ok(EvalResult::with_value(result).with_flags(Some(carry), Some(zero)))
    }

    fn eval_logical(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let lhs = self.operand_value(program, expr.operands, 0, cache)?;
        let rhs = self.operand_value(program, expr.operands, 1, cache)?;
        let value = match expr.op {
            "AND" => lhs & rhs,
            "OR" => lhs | rhs,
            "XOR" => lhs ^ rhs,
            _ => unreachable!(),
        };
        Ok(EvalResult::with_value(value).with_flags(Some(0), Some(bool_to_u8(value == 0))))
    }

    fn eval_compare(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let lhs = self.operand_value(program, expr.operands, 0, cache)?;
        let rhs = self.operand_value(program, expr.operands, 1, cache)?;
        let width = expr
            .width
            .ok_or(ExecutionError::Unimplemented("CMP width"))?;
        let result = match expr.op {
            "CMP_E" => bool_to_i64(lhs == rhs),
            "CMP_SLT" => {
                let a = to_signed(lhs, width);
                let b = to_signed(rhs, width);
                bool_to_i64(a < b)
            }
            "CMP_UGT" => {
                let mask = mask_for_bytes_u64(width);
                let a = (lhs as u64) & mask;
                let b = (rhs as u64) & mask;
                bool_to_i64(a > b)
            }
            _ => unreachable!(),
        };
        Ok(EvalResult::with_value(result))
    }

    fn eval_shift(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let size = expr
            .width
            .ok_or(ExecutionError::Unimplemented("SHIFT width"))?;
        let value = self.operand_value(program, expr.operands, 0, cache)?;
        let count = self.operand_value(program, expr.operands, 1, cache)?;
        let left = expr.op == "LSL";
        let (result, carry, zero) = shift_impl(size, value, count, left);
        Ok(EvalResult::with_value(result).with_flags(Some(carry), Some(zero)))
    }

    fn eval_rotate(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let size = expr
            .width
            .ok_or(ExecutionError::Unimplemented("ROT width"))?;
        let value = self.operand_value(program, expr.operands, 0, cache)?;
        let count = self.operand_value(program, expr.operands, 1, cache)?;
        let left = expr.op == "ROL";
        let (result, carry, zero) = rotate_impl(size, value, count, left);
        Ok(EvalResult::with_value(result).with_flags(Some(carry), Some(zero)))
    }

    fn eval_rotate_through_carry(
        &mut self,
        program: &LlilProgram,
        expr: &LlilExpr,
        cache: &mut [Option<EvalResult>],
    ) -> Result<EvalResult, ExecutionError> {
        let size = expr
            .width
            .ok_or(ExecutionError::Unimplemented("RTC width"))?;
        let value = self.operand_value(program, expr.operands, 0, cache)?;
        let count = self.operand_value(program, expr.operands, 1, cache)?;
        let carry_in = self.operand_value(program, expr.operands, 2, cache)?;
        let left = expr.op == "RLC";
        let (result, carry, zero) =
            rotate_through_carry_impl(size, value, count, carry_in != 0, left);
        Ok(EvalResult::with_value(result).with_flags(Some(carry), Some(zero)))
    }

    pub fn invoke_intrinsic(&mut self, name: &str) -> Result<(), ExecutionError> {
        match name {
            "TCL" => Ok(()),
            "HALT" | "OFF" => {
                self.enter_low_power_state()?;
                Ok(())
            }
            "RESET" => self.eval_reset(),
            _ => Err(ExecutionError::Unimplemented("intrinsic")),
        }
    }

    fn enter_low_power_state(&mut self) -> Result<(), ExecutionError> {
        let mut usr = self.read_memory_byte(IMEM_USR)? as u8;
        usr &= !0x3F;
        usr |= 0x18;
        self.write_memory_byte(IMEM_USR, usr)?;

        let mut ssr = self.read_memory_byte(IMEM_SSR)? as u8;
        ssr |= 0x04;
        self.write_memory_byte(IMEM_SSR, ssr)?;

        self.state.set_halted(true);
        Ok(())
    }

    fn eval_reset(&mut self) -> Result<(), ExecutionError> {
        let mut lcc = self.read_memory_byte(IMEM_LCC)? as u8;
        lcc &= !0x80;
        self.write_memory_byte(IMEM_LCC, lcc)?;

        self.write_memory_byte(IMEM_UCR, 0x00)?;
        self.write_memory_byte(IMEM_ISR, 0x00)?;
        self.write_memory_byte(IMEM_SCR, 0x00)?;

        let mut usr = self.read_memory_byte(IMEM_USR)? as u8;
        usr &= !0x3F;
        usr |= 0x18;
        self.write_memory_byte(IMEM_USR, usr)?;

        let mut ssr = self.read_memory_byte(IMEM_SSR)? as u8;
        ssr &= !0x04;
        self.write_memory_byte(IMEM_SSR, ssr)?;

        let mut reset_vector = self.read_memory_byte(RESET_VECTOR_BASE)? as u32;
        reset_vector |= (self.read_memory_byte(RESET_VECTOR_BASE + 1)? as u32) << 8;
        reset_vector |= (self.read_memory_byte(RESET_VECTOR_BASE + 2)? as u32) << 16;
        self.registers
            .set_by_name("PC", reset_vector & PC_MASK)
            .map_err(ExecutionError::Register)?;
        Ok(())
    }

    fn operand_value(
        &mut self,
        program: &LlilProgram,
        operands: &[LlilOperand],
        slot: usize,
        cache: &mut [Option<EvalResult>],
    ) -> Result<i64, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Expr(idx)) => self
                .eval_expr(program, *idx, cache)?
                .expect_value("operand"),
            Some(LlilOperand::Imm(value)) => Ok(*value),
            Some(LlilOperand::Reg(name)) => self.read_register(name),
            Some(LlilOperand::Flag(name)) => self.read_flag(name),
            Some(LlilOperand::None) => Ok(0),
            _ => Err(ExecutionError::InvalidOperand("operand")),
        }
    }

    fn operand_expr_index(
        &self,
        operands: &[LlilOperand],
        slot: usize,
    ) -> Result<u16, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Expr(idx)) => Ok(*idx),
            _ => Err(ExecutionError::InvalidOperand("expr index")),
        }
    }

    fn operand_reg<'a>(
        &self,
        operands: &'a [LlilOperand],
        slot: usize,
    ) -> Result<&'a str, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Reg(name)) => Ok(name),
            _ => Err(ExecutionError::InvalidOperand("register operand")),
        }
    }

    fn operand_flag<'a>(
        &self,
        operands: &'a [LlilOperand],
        slot: usize,
    ) -> Result<&'a str, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Flag(name)) => Ok(name),
            _ => Err(ExecutionError::InvalidOperand("flag operand")),
        }
    }

    fn read_register(&self, name: &str) -> Result<i64, ExecutionError> {
        let value = self
            .registers
            .get_by_name(name)
            .map_err(ExecutionError::Register)?;
        Ok(value as i64)
    }

    fn write_register(
        &mut self,
        name: &str,
        value: i64,
        width: Option<u8>,
    ) -> Result<(), ExecutionError> {
        let masked = mask_value(value, width);
        self.registers
            .set_by_name(name, masked as u32)
            .map_err(ExecutionError::Register)
    }

    pub fn read_flag(&self, name: &str) -> Result<i64, ExecutionError> {
        let ch = name
            .chars()
            .next()
            .ok_or(ExecutionError::InvalidOperand("flag name"))?;
        let flag = Flag::from_char(ch).map_err(|_| ExecutionError::InvalidOperand("flag"))?;
        Ok(self.registers.get_flag(flag) as i64)
    }

    pub fn write_flag(&mut self, name: &str, value: i64) -> Result<(), ExecutionError> {
        let ch = name
            .chars()
            .next()
            .ok_or(ExecutionError::InvalidOperand("flag name"))?;
        self.write_flag_char(ch, bool_to_u8(value != 0))
    }

    fn write_flag_char(&mut self, flag: char, value: u8) -> Result<(), ExecutionError> {
        let flag = Flag::from_char(flag).map_err(|_| ExecutionError::InvalidOperand("flag"))?;
        self.registers.set_flag(flag, bool_to_u8(value != 0));
        Ok(())
    }

    fn read_memory_bytes(&self, address: i64, size: usize) -> Result<i64, ExecutionError> {
        if !(1..=3).contains(&size) {
            return Err(ExecutionError::InvalidOperand("memory size"));
        }
        let address = address
            .try_into()
            .map_err(|_| ExecutionError::InvalidOperand("address"))?;
        let value = self.with_gil(|py| self.memory.read_bytes(py, address, size))?;
        Ok(value as i64)
    }

    fn write_memory_bytes(
        &mut self,
        address: i64,
        size: usize,
        value: i64,
    ) -> Result<(), ExecutionError> {
        if !(1..=3).contains(&size) {
            return Err(ExecutionError::InvalidOperand("memory size"));
        }
        let address = address
            .try_into()
            .map_err(|_| ExecutionError::InvalidOperand("address"))?;
        let masked = mask_value(value, Some(size as u8)) as u32;
        self.with_gil(|py| self.memory.write_bytes(py, address, size, masked))?;
        Ok(())
    }

    fn read_memory_byte(&self, address: u32) -> Result<u8, ExecutionError> {
        self.with_gil(|py| self.memory.read_byte(py, address))
    }

    fn write_memory_byte(&mut self, address: u32, value: u8) -> Result<(), ExecutionError> {
        self.with_gil(|py| self.memory.write_byte(py, address, value))
    }

    pub fn call_absolute(&mut self, return_width: u8, target: i64) -> ExecutionResult {
        if !(1..=3).contains(&return_width) {
            return Err(ExecutionError::InvalidOperand("CALL width"));
        }
        let return_addr = self.read_named_register("PC")?;
        self.push_value(return_width, return_addr)?;
        self.write_named_register("PC", target, Some(3))?;
        Ok(())
    }

    pub fn push_value(&mut self, width: u8, value: i64) -> ExecutionResult {
        if width == 0 {
            return Err(ExecutionError::InvalidOperand("PUSH width"));
        }
        let size = width as i64;
        let sp = self.read_register("S")?;
        let new_sp = sp - size;
        self.write_memory_bytes(new_sp, width as usize, value)?;
        self.write_register("S", new_sp, Some(3))?;
        Ok(())
    }

    pub fn pop_value(&mut self, width: u8) -> Result<i64, ExecutionError> {
        if width == 0 {
            return Err(ExecutionError::InvalidOperand("POP width"));
        }
        let size = width as i64;
        let sp = self.read_register("S")?;
        let value = self.read_memory_bytes(sp, width as usize)?;
        self.write_register("S", sp + size, Some(3))?;
        Ok(value)
    }

    fn with_gil<R>(&self, f: impl FnOnce(Python<'_>) -> PyResult<R>) -> Result<R, ExecutionError> {
        Python::with_gil(|py| f(py)).map_err(ExecutionError::Python)
    }

    fn apply_flags(&mut self, expr: &LlilExpr, result: &EvalResult) -> Result<(), ExecutionError> {
        let Some(spec) = expr.flags else {
            return Ok(());
        };

        if spec == "0" {
            return Ok(());
        }

        for flag in spec.chars() {
            let mut value = result.flag(flag);
            if value.is_none() {
                if let (Some(width), Some(result_value)) = (expr.width, result.value) {
                    match flag {
                        'Z' => {
                            let masked = mask_value(result_value, Some(width));
                            value = Some(bool_to_u8(masked == 0));
                        }
                        'C' => {
                            let mask = mask_for_bytes_i128(width);
                            value = Some(bool_to_u8((result_value as i128) > mask));
                        }
                        _ => {}
                    }
                }
            }

            if let Some(flag_value) = value {
                self.write_flag_char(flag, flag_value)?;
            }
        }
        Ok(())
    }

    fn call_push_size(&self, program: &LlilProgram, expr: &LlilExpr) -> usize {
        if let Some(LlilOperand::Expr(target_idx)) = expr.operands.get(0) {
            let target = &program.expressions[*target_idx as usize];
            if target.width == Some(2) || target.suffix == Some("w") {
                return 2;
            }

            if target.op == "OR" && target.suffix == Some("l") {
                if let Some(LlilOperand::Expr(inner_idx)) = target.operands.get(0) {
                    let inner = &program.expressions[*inner_idx as usize];
                    if inner.width == Some(2) || inner.suffix == Some("w") {
                        return 2;
                    }
                }
            }
        }
        3
    }
}

fn shift_impl(size: u8, val: i64, count: i64, left: bool) -> (i64, u8, u8) {
    let width = (size as i64) * 8;
    let mask = if width <= 0 {
        0
    } else if width >= 64 {
        -1_i64
    } else {
        (1_i64 << width) - 1
    };

    if count == 0 {
        let result = val & mask;
        return (result, 0, bool_to_u8(result == 0));
    }

    let mut carry_out = 0;
    let result = if left {
        if width > 0 && count <= width {
            carry_out = bool_to_u8(((val >> (width - count)) & 1) != 0);
        }
        (val << count) & mask
    } else {
        if width > 0 && count > 0 && count <= width {
            carry_out = bool_to_u8(((val >> (count - 1)) & 1) != 0);
        }
        (val >> count) & mask
    };

    (result, carry_out, bool_to_u8(result == 0))
}

fn rotate_impl(size: u8, val: i64, count: i64, left: bool) -> (i64, u8, u8) {
    let width = (size as i64) * 8;
    if width <= 0 {
        let masked = val & 0;
        return (masked, 0, bool_to_u8(masked == 0));
    }
    let mask = if width >= 64 {
        -1_i64
    } else {
        (1_i64 << width) - 1
    };

    let mut count = count % width;
    if count < 0 {
        count += width;
    }
    if count == 0 {
        let result = val & mask;
        let carry = if left {
            bool_to_u8(((val >> (width - 1)) & 1) != 0)
        } else {
            bool_to_u8((val & 1) != 0)
        };
        return (result, carry, bool_to_u8(result == 0));
    }

    let result = if left {
        ((val << count) | ((val & mask) >> (width - count))) & mask
    } else {
        ((val >> count) | ((val & mask) << (width - count))) & mask
    };
    let carry = if left {
        bool_to_u8(((val >> (width - count)) & 1) != 0)
    } else {
        bool_to_u8(((val >> (count - 1)) & 1) != 0)
    };
    (result, carry, bool_to_u8(result == 0))
}

fn rotate_through_carry_impl(
    size: u8,
    val: i64,
    count: i64,
    carry_in: bool,
    left: bool,
) -> (i64, u8, u8) {
    let width = (size as i64) * 8;
    if width <= 0 {
        let masked = val & 0;
        return (masked, 0, bool_to_u8(masked == 0));
    }
    let mask = if width >= 64 {
        -1_i64
    } else {
        (1_i64 << width) - 1
    };

    let count = if count <= 0 { 0 } else { 1 };
    if count == 0 {
        let result = val & mask;
        return (
            result,
            if left {
                bool_to_u8(((val >> (width - 1)) & 1) != 0)
            } else {
                bool_to_u8((val & 1) != 0)
            },
            bool_to_u8(result == 0),
        );
    }

    let carry_in_bit = if carry_in { 1_i64 } else { 0 };
    if left {
        let new_carry_out = bool_to_u8(((val >> (width - 1)) & 1) != 0);
        let result = ((val << count) | carry_in_bit) & mask;
        (result, new_carry_out, bool_to_u8(result == 0))
    } else {
        let new_carry_out = bool_to_u8((val & 1) != 0);
        let shift = if width - count <= 0 { 0 } else { width - count };
        let carry_part = (carry_in_bit << shift) & mask;
        let result = ((val >> count) | carry_part) & mask;
        (result, new_carry_out, bool_to_u8(result == 0))
    }
}

fn mask_for_bytes_i128(bytes: u8) -> i128 {
    if bytes == 0 {
        -1
    } else {
        (1_i128 << (bytes as i32 * 8)) - 1
    }
}

fn mask_for_bytes_u64(bytes: u8) -> u64 {
    if bytes == 0 {
        u64::MAX
    } else {
        (1_u64 << (bytes as u32 * 8)) - 1
    }
}

fn to_signed(value: i64, size: u8) -> i64 {
    let bits = (size as u32) * 8;
    if bits == 0 || bits >= 63 {
        return value;
    }
    let mask = (1_i64 << bits) - 1;
    let sign_bit = 1_i64 << (bits - 1);
    let masked = value & mask;
    if masked & sign_bit != 0 {
        masked | !mask
    } else {
        masked
    }
}

fn bool_to_u8(value: bool) -> u8 {
    if value {
        1
    } else {
        0
    }
}

fn bool_to_i64(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

fn mask_value(value: i64, width: Option<u8>) -> i64 {
    match width {
        Some(0) | None => value,
        Some(bytes) => {
            let mask = if bytes >= 8 {
                -1_i64
            } else {
                (1_i64 << (bytes as i64 * 8)) - 1
            };
            value & mask
        }
    }
}

impl From<RegisterError> for ExecutionError {
    fn from(err: RegisterError) -> Self {
        ExecutionError::Register(err)
    }
}

impl From<PyErr> for ExecutionError {
    fn from(err: PyErr) -> Self {
        ExecutionError::Python(err)
    }
}
