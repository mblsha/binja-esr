use crate::memory::MemoryBus;
use crate::state::{Flag, RegisterError, Registers};
use crate::{LlilNode, LlilOperand, LlilProgram};

#[derive(Debug)]
pub enum ExecutionError {
    UnsupportedOpcode,
    Unimplemented(&'static str),
    Register(RegisterError),
}

pub type ExecutionResult = Result<(), ExecutionError>;

pub struct LlilRuntime {
    registers: Registers,
    memory: MemoryBus,
}

impl LlilRuntime {
    pub fn new(registers: Registers, memory: MemoryBus) -> Self {
        Self { registers, memory }
    }

    pub fn into_parts(self) -> (Registers, MemoryBus) {
        (self.registers, self.memory)
    }

    pub fn execute_program(&mut self, program: &LlilProgram) -> ExecutionResult {
        let mut cache: Vec<Option<i64>> = vec![None; program.expressions.len()];
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
                    let cond_val = self.eval_expr(program, cond, &mut cache)?;
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
                LlilNode::Label { .. } => {
                    pc += 1;
                }
            }
        }
        Ok(())
    }

    fn eval_expr(
        &mut self,
        program: &LlilProgram,
        index: u16,
        cache: &mut Vec<Option<i64>>,
    ) -> Result<i64, ExecutionError> {
        let idx = index as usize;
        if let Some(value) = cache[idx] {
            return Ok(value);
        }
        let expr = &program.expressions[idx];
        let value = match expr.op {
            "NOP" => 0,
            "CONST" | "CONST_PTR" => self.operand_value(program, expr.operands, 0, cache)?,
            "REG" => {
                let name = self.operand_reg(expr.operands, 0)?;
                self.read_register(name)?
            }
            "FLAG" => {
                let name = self.operand_flag(expr.operands, 0)?;
                self.read_flag(name)?
            }
            "SET_REG" => {
                let name = self.operand_reg(expr.operands, 0)?;
                let value = self.operand_value(program, expr.operands, 1, cache)?;
                self.write_register(name, value, expr.width)?;
                value
            }
            "SET_FLAG" => {
                let name = self.operand_flag(expr.operands, 0)?;
                let value = self.operand_value(program, expr.operands, 1, cache)?;
                self.write_flag(name, value)?;
                value
            }
            "UNIMPL" => return Err(ExecutionError::Unimplemented("UNIMPL")),
            other => return Err(ExecutionError::Unimplemented(other)),
        };
        cache[idx] = Some(value);
        Ok(value)
    }

    fn operand_value(
        &mut self,
        program: &LlilProgram,
        operands: &[LlilOperand],
        slot: usize,
        cache: &mut Vec<Option<i64>>,
    ) -> Result<i64, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Expr(idx)) => self.eval_expr(program, *idx, cache),
            Some(LlilOperand::Imm(value)) => Ok(*value),
            Some(LlilOperand::Reg(name)) => self.read_register(name),
            Some(LlilOperand::Flag(name)) => self.read_flag(name),
            Some(LlilOperand::None) => Ok(0),
            _ => Err(ExecutionError::Unimplemented("operand")),
        }
    }

    fn operand_reg<'a>(
        &self,
        operands: &'a [LlilOperand],
        slot: usize,
    ) -> Result<&'a str, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Reg(name)) => Ok(name),
            _ => Err(ExecutionError::Unimplemented("reg_operand")),
        }
    }

    fn operand_flag<'a>(
        &self,
        operands: &'a [LlilOperand],
        slot: usize,
    ) -> Result<&'a str, ExecutionError> {
        match operands.get(slot) {
            Some(LlilOperand::Flag(name)) => Ok(name),
            _ => Err(ExecutionError::Unimplemented("flag_operand")),
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

    fn read_flag(&self, name: &str) -> Result<i64, ExecutionError> {
        let ch = name
            .chars()
            .next()
            .ok_or(ExecutionError::Unimplemented("flag"))?;
        let flag = Flag::from_char(ch).map_err(|_| ExecutionError::Unimplemented("flag"))?;
        Ok(self.registers.get_flag(flag) as i64)
    }

    fn write_flag(&mut self, name: &str, value: i64) -> Result<(), ExecutionError> {
        let ch = name
            .chars()
            .next()
            .ok_or(ExecutionError::Unimplemented("flag"))?;
        let flag = Flag::from_char(ch).map_err(|_| ExecutionError::Unimplemented("flag"))?;
        self.registers.set_flag(flag, (value != 0) as u8);
        Ok(())
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
