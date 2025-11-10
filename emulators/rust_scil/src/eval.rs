use crate::ast::{Binder, Expr, Instr, PreLatch, Space as AstSpace, Stmt};
use crate::bus::{Bus, Space};
use crate::state::State;
use std::collections::HashMap;
use thiserror::Error;

const BP_ADDR: u32 = 0xEC;
const PX_ADDR: u32 = 0xED;
const PY_ADDR: u32 = 0xEE;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("missing binder value for {0}")]
    MissingBinder(String),
    #[error("temporary {0} not populated")]
    MissingTemp(String),
    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),
}

struct Env<'a, B: Bus> {
    state: &'a mut State,
    bus: &'a mut B,
    binder: HashMap<String, (u32, u8)>,
    tmps: HashMap<String, (u32, u8)>,
    pre_latch: Option<PreLatch>,
    imem_index: usize,
}

impl<'a, B: Bus> Env<'a, B> {
    fn bind_from(binder: &Binder) -> Result<HashMap<String, (u32, u8)>> {
        binder
            .iter()
            .map(|(name, expr)| match expr {
                Expr::Const { value, size } => Ok((name.clone(), (*value, *size))),
                _ => Err(Error::Unsupported("non-const binder value")),
            })
            .collect()
    }

    fn set_tmp(&mut self, name: &str, value: u32, size: u8) {
        self.tmps.insert(name.to_string(), (value & mask(size), size));
    }

    fn get_tmp(&self, name: &str) -> Result<(u32, u8)> {
        self.tmps
            .get(name)
            .copied()
            .ok_or_else(|| Error::MissingTemp(name.to_string()))
    }

    fn next_imem_mode(&mut self) -> String {
        let default = "(BP+n)".to_string();
        if let Some(pre) = &self.pre_latch {
            let mode = match self.imem_index {
                0 => Some(pre.first.as_str()),
                1 => Some(pre.second.as_str()),
                _ => None,
            };
            self.imem_index += 1;
            if let Some(value) = mode {
                return value.to_string();
            }
        } else {
            self.imem_index += 1;
        }
        default
    }
}

pub fn step<B: Bus>(
    state: &mut State,
    bus: &mut B,
    instr: &Instr,
    binder: &Binder,
    pre_latch: Option<PreLatch>,
) -> Result<()> {
    let map = Env::<B>::bind_from(binder)?;
    let mut env = Env {
        state,
        bus,
        binder: map,
        tmps: HashMap::new(),
        pre_latch,
        imem_index: 0,
    };
    for stmt in &instr.semantics {
        exec_stmt(stmt, &mut env)?;
    }
    Ok(())
}

fn exec_stmt<B: Bus>(stmt: &Stmt, env: &mut Env<B>) -> Result<()> {
    match stmt {
        Stmt::Fetch { kind: _, dst } => {
            let (value, size) = env
                .binder
                .get(&dst.name)
                .copied()
                .ok_or_else(|| Error::MissingBinder(dst.name.clone()))?;
            env.set_tmp(&dst.name, value, size);
        }
        Stmt::SetReg { reg, value, flags } => {
            let (val, bits) = eval_expr(value, env)?;
            env.state.set_reg(&reg.name, val, reg.size);
            if let Some(flag_list) = flags {
                if flag_list.iter().any(|f| f == "Z") {
                    env.state.set_flag("Z", (val == 0) as u32);
                }
                if flag_list.iter().any(|f| f == "C") {
                    let c = (val >> (bits.saturating_sub(1))) & 1;
                    env.state.set_flag("C", c);
                }
            }
        }
        Stmt::Store { dst, value } => {
            let (addr, _) = eval_expr(&dst.addr, env)?;
            let (val, _) = eval_expr(value, env)?;
            let target = match dst.space {
                AstSpace::Int => {
                    let offset = resolve_imem_addr(env, addr as u32);
                    (Space::Int, offset)
                }
                AstSpace::Ext => (Space::Ext, addr),
                AstSpace::Code => (Space::Ext, addr),
            };
            env.bus.store(target.0, target.1, dst.size, val);
        }
        Stmt::SetFlag { flag, value } => {
            let (val, _) = eval_expr(value, env)?;
            env.state.set_flag(flag, val);
        }
        Stmt::If { cond, then, r#else } => {
            if eval_cond(cond, env)? {
                for inner in then {
                    exec_stmt(inner, env)?;
                }
            } else {
                for inner in r#else {
                    exec_stmt(inner, env)?;
                }
            }
        }
        Stmt::Goto { target } => {
            let (val, _) = eval_expr(target, env)?;
            env.state.set_reg("PC", val, 20);
        }
        Stmt::Call { target, .. } => {
            let (val, _) = eval_expr(target, env)?;
            env.state.set_reg("PC", val, 20);
        }
        Stmt::Ret { .. } => {}
        Stmt::ExtRegLoad { dst, ptr, mode, disp } => {
            let disp_val = disp.as_ref().map_or(0, |expr| eval_expr(expr, env).unwrap().0);
            let value = ext_ptr_read(env, &ptr.name, mode, disp_val as i32, dst.size)?;
            env.state.set_reg(&dst.name, value, dst.size);
        }
        Stmt::ExtRegStore { src, ptr, mode, disp } => {
            let disp_val = disp.as_ref().map_or(0, |expr| eval_expr(expr, env).unwrap().0);
            let value = env.state.get_reg(&src.name, src.size);
            ext_ptr_store(env, &ptr.name, mode, disp_val as i32, src.size, value)?;
        }
        Stmt::IntMemSwap { left, right, width } => {
            let (l, _) = eval_expr(left, env)?;
            let (r, _) = eval_expr(right, env)?;
            let l_addr = resolve_imem_addr(env, l);
            let r_addr = resolve_imem_addr(env, r);
            let l_val = env.bus.load(Space::Int, l_addr, *width);
            let r_val = env.bus.load(Space::Int, r_addr, *width);
            env.bus.store(Space::Int, l_addr, *width, r_val);
            env.bus.store(Space::Int, r_addr, *width, l_val);
        }
        Stmt::ExtRegToInt { ptr, mode, dst, disp } => {
            let disp_val = disp.as_ref().map_or(0, |expr| eval_expr(expr, env).unwrap().0);
            let value = ext_ptr_read(env, &ptr.name, mode, disp_val as i32, dst.size)?;
            let (addr, _) = eval_expr(&dst.addr, env)?;

            let offset = match dst.space {
                AstSpace::Int => resolve_imem_addr(env, addr),
                _ => addr,
            };
            let space = match dst.space {
                AstSpace::Int => Space::Int,
                AstSpace::Ext | AstSpace::Code => Space::Ext,
            };
            env.bus.store(space, offset, dst.size, value);
        }
        Stmt::IntToExtReg { ptr, mode, src, disp } => {
            let disp_val = disp.as_ref().map_or(0, |expr| eval_expr(expr, env).unwrap().0);
            let (addr, _) = eval_expr(&src.addr, env)?;
            let offset = resolve_imem_addr(env, addr);
            let value = env.bus.load(Space::Int, offset, src.size);
            ext_ptr_store(env, &ptr.name, mode, disp_val as i32, src.size, value)?;
        }
        Stmt::Label { .. } | Stmt::Comment { .. } => {}
    }
    Ok(())
}

fn eval_expr<B: Bus>(expr: &Expr, env: &mut Env<B>) -> Result<(u32, u8)> {
    Ok(match expr {
        Expr::Const { value, size } => (*value & mask(*size), *size),
        Expr::Tmp { name, .. } => env.get_tmp(name)?,
        Expr::Reg { name, size, .. } => (env.state.get_reg(name, *size), *size),
        Expr::Flag { name } => (env.state.get_flag(name), 1),
        Expr::Mem { space, size, addr } => {
            let (ptr, _) = eval_expr(addr, env)?;
            let (bus_space, bus_addr) = match space {
                AstSpace::Int => (Space::Int, resolve_imem_addr(env, ptr)),
                AstSpace::Ext | AstSpace::Code => (Space::Ext, ptr),
            };
            (env.bus.load(bus_space, bus_addr, *size) & mask(*size), *size)
        }
        Expr::UnOp { op, a, out_size, .. } => {
            let (val, bits) = eval_expr(a, env)?;
            match op.as_str() {
                "neg" => ((-(val as i32)) as u32 & mask(*out_size), *out_size),
                "not" => ((!val) & mask(*out_size), *out_size),
                "sext" => (sign_extend(val, bits, *out_size), *out_size),
                "zext" => (val & mask(*out_size), *out_size),
                "low_part" => (val & mask(*out_size), *out_size),
                "high_part" => {
                    let shift = bits.saturating_sub(*out_size);
                    ((val >> shift) & mask(*out_size), *out_size)
                }
                _ => return Err(Error::Unsupported("unop")),
            }
        }
        Expr::BinOp { op, a, b, out_size } => {
            let (lhs, _) = eval_expr(a, env)?;
            let (rhs, _) = eval_expr(b, env)?;
            let res = match op.as_str() {
                "add" => lhs.wrapping_add(rhs),
                "sub" => lhs.wrapping_sub(rhs),
                "and" => lhs & rhs,
                "or" => lhs | rhs,
                "xor" => lhs ^ rhs,
                "shl" => lhs << rhs,
                "shr" => lhs >> rhs,
                "sar" => ((lhs as i32) >> rhs) as u32,
                _ => return Err(Error::Unsupported("binop")),
            };
            (res & mask(*out_size), *out_size)
        }
        Expr::TernOp { op, cond, t, f, .. } => {
            if op != "select" {
                return Err(Error::Unsupported("ternop"));
            }
            if eval_cond(cond, env)? {
                eval_expr(t, env)?
            } else {
                eval_expr(f, env)?
            }
        }
        Expr::PcRel { base, out_size, disp } => {
            let mut target = (env.state.get_reg("PC", *out_size) + *base as u32) & mask(*out_size);
            if let Some(delta) = disp {
                let (value, bits) = eval_expr(delta, env)?;
                let signed = sign_extend(value, bits, *out_size);
                target = (target.wrapping_add(signed)) & mask(*out_size);
            }
            (target, *out_size)
        }
        Expr::Join24 { hi, mid, lo } => {
            let (h, _) = eval_expr(hi, env)?;
            let (m, _) = eval_expr(mid, env)?;
            let (l, _) = eval_expr(lo, env)?;
            (((h & 0xFF) << 16) | ((m & 0xFF) << 8) | (l & 0xFF), 24)
        }
    })
}

fn eval_cond<B: Bus>(cond: &crate::ast::Cond, env: &mut Env<B>) -> Result<bool> {
    match cond {
        crate::ast::Cond::Prim { kind, flag, .. } if kind == "flag" => {
            let name = flag.as_deref().ok_or(Error::Unsupported("flag missing"))?;
            Ok(env.state.get_flag(name) != 0)
        }
        crate::ast::Cond::Prim { kind, a, b, .. } => {
            let (lhs, la) = eval_expr(a.as_ref().ok_or(Error::Unsupported("cond lhs"))?, env)?;
            let (rhs, lb) = eval_expr(b.as_ref().ok_or(Error::Unsupported("cond rhs"))?, env)?;
            Ok(match kind.as_str() {
                "eq" => lhs == rhs,
                "ne" => lhs != rhs,
                "ltu" => lhs < rhs,
                "geu" => lhs >= rhs,
                "lts" => sign_extend(lhs, la, 32) < sign_extend(rhs, lb, 32),
                "ges" => sign_extend(lhs, la, 32) >= sign_extend(rhs, lb, 32),
                _ => return Err(Error::Unsupported("cond kind")),
            })
        }
    }
}

fn resolve_imem_addr<B: Bus>(env: &mut Env<B>, offset: u32) -> u32 {
    let mode = env.next_imem_mode();
    let base_bp = env.bus.load(Space::Int, BP_ADDR, 8) as u32;
    match mode.as_str() {
        "(n)" => offset & 0xFF,
        "(BP+n)" => (base_bp + offset) & 0xFF,
        "(PX+n)" => {
            let px = env.bus.load(Space::Int, PX_ADDR, 8) as u32;
            (px + offset) & 0xFF
        }
        "(PY+n)" => {
            let py = env.bus.load(Space::Int, PY_ADDR, 8) as u32;
            (py + offset) & 0xFF
        }
        "(BP+PX)" => {
            let px = env.bus.load(Space::Int, PX_ADDR, 8) as u32;
            (base_bp + px) & 0xFF
        }
        "(BP+PY)" => {
            let py = env.bus.load(Space::Int, PY_ADDR, 8) as u32;
            (base_bp + py) & 0xFF
        }
        _ => (base_bp + offset) & 0xFF,
    }
}

fn ext_ptr_read<B: Bus>(
    env: &mut Env<B>,
    ptr: &str,
    mode: &str,
    disp: i32,
    bits: u8,
) -> Result<u32> {
    let addr = resolve_ext_ptr(env, ptr, mode, disp, bits, true);
    Ok(env.bus.load(Space::Ext, addr, bits))
}

fn ext_ptr_store<B: Bus>(
    env: &mut Env<B>,
    ptr: &str,
    mode: &str,
    disp: i32,
    bits: u8,
    value: u32,
) -> Result<()> {
    let addr = resolve_ext_ptr(env, ptr, mode, disp, bits, false);
    env.bus.store(Space::Ext, addr, bits, value);
    Ok(())
}

fn resolve_ext_ptr<B: Bus>(
    env: &mut Env<B>,
    ptr: &str,
    mode: &str,
    disp: i32,
    bits: u8,
    is_read: bool,
) -> u32 {
    let width_bytes = (bits / 8).max(1) as i32;
    let current = env.state.get_reg(ptr, 24) as i32;
    let mut base = current;
    match mode {
        "offset" => base = current.wrapping_add(disp),
        "pre_dec" => {
            base = current.wrapping_sub(width_bytes);
            env.state
                .set_reg(ptr, base as u32 & mask(24), 24);
        }
        _ => {}
    }

    if mode == "post_inc" && is_read {
        env.state
            .set_reg(ptr, current.wrapping_add(width_bytes) as u32 & mask(24), 24);
    } else if mode == "post_inc" && !is_read {
        env.state
            .set_reg(ptr, current.wrapping_add(width_bytes) as u32 & mask(24), 24);
    }
    (base as u32) & mask(24)
}

fn mask(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

fn sign_extend(value: u32, from_bits: u8, to_bits: u8) -> u32 {
    if from_bits == 0 {
        return 0;
    }
    let sign_bit = 1u32 << (from_bits - 1);
    let mut val = value & mask(from_bits);
    if val & sign_bit != 0 {
        let extend_mask = mask(to_bits) ^ mask(from_bits);
        val |= extend_mask;
    }
    val & mask(to_bits)
}
