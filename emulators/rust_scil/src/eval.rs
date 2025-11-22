use crate::ast::{Binder, Expr, Instr, PreLatch, Space as AstSpace, Stmt};
use crate::bus::{Bus, Space};
use crate::state::State;
use std::collections::HashMap;
use std::env;
use thiserror::Error;

const INTERNAL_MEMORY_START: u32 = 0x100000;
const BP_ADDR: u32 = 0xEC;
const PX_ADDR: u32 = 0xED;
const PY_ADDR: u32 = 0xEE;
const UCR_ADDR: u32 = 0xF7;
const USR_ADDR: u32 = 0xF8;
const IMR_ADDR: u32 = 0xFB;
const ISR_ADDR: u32 = 0xFC;
const SCR_ADDR: u32 = 0xFD;
const LCC_ADDR: u32 = 0xFE;
const SSR_ADDR: u32 = 0xFF;
const STACK_MASK: u32 = 0xFF_FFFF;
const PC_MASK: u32 = 0x0F_FFFF;
const INTERRUPT_VECTOR_ADDR: u32 = 0xFF_FFA;
const PTR_MODE_SIMPLE: u32 = 0;
const PTR_MODE_POST_INC: u32 = 1;
const PTR_MODE_PRE_DEC: u32 = 2;
const PTR_MODE_OFFSET: u32 = 3;
const LOOP_EXT_MODE_IMM: u32 = 0xFF;

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
    binder: HashMap<String, Expr>,
    tmps: HashMap<String, (u32, u8)>,
    pre_latch: Option<PreLatch>,
    imem_index: usize,
    imem_cache: HashMap<String, u32>,
    imem_last_write: HashMap<u32, u32>,
}

impl<'a, B: Bus> Env<'a, B> {
    fn bind_from(binder: &Binder) -> Result<HashMap<String, Expr>> {
        binder
            .iter()
            .map(|(name, value)| Ok((name.clone(), value.clone())))
            .collect()
    }

    fn set_tmp(&mut self, name: &str, value: u32, size: u8) {
        self.imem_cache.remove(name);
        self.tmps
            .insert(name.to_string(), (value & mask(size), size));
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
        imem_cache: HashMap::new(),
        imem_last_write: HashMap::new(),
    };
    for stmt in &instr.semantics {
        exec_stmt(stmt, &mut env)?;
    }
    Ok(())
}

fn exec_stmt<B: Bus>(stmt: &Stmt, env: &mut Env<B>) -> Result<()> {
    match stmt {
        Stmt::Fetch { kind: _, dst } => {
            let expr = env
                .binder
                .get(&dst.name)
                .ok_or_else(|| Error::MissingBinder(dst.name.clone()))?
                .clone();
            let (value, size) = eval_expr(&expr, env)?;
            env.set_tmp(&dst.name, value, size);
        }
        Stmt::SetReg { reg, value, flags } => {
            let (val, bits) = eval_expr(value, env)?;
            env.state.set_reg(&reg.name, val, reg.size);
            trace_reg_write(env, &reg.name, val & mask(reg.size), bits);
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
            let (addr, addr_bits) = eval_expr(&dst.addr, env)?;
            let key = match &dst.addr {
                Expr::Tmp { name, .. } => Some(name.as_str()),
                _ => None,
            };
            let target = match dst.space {
                AstSpace::Int => {
                    let offset = if addr_bits > 8 {
                        let absolute =
                            addr.wrapping_sub(INTERNAL_MEMORY_START as u32) & 0xFF;
                        absolute
                    } else {
                        resolve_imem_addr(env, addr as u32, key)
                    };
                    (Space::Int, offset)
                }
                AstSpace::Ext => (Space::Ext, addr),
                AstSpace::Code => (Space::Ext, addr),
            };
            let (val, _) = eval_expr(value, env)?;
            env.bus.store(target.0, target.1, dst.size, val);
            if matches!(dst.space, AstSpace::Int) {
                env.imem_last_write
                    .insert(target.1 & 0xFF, val & mask(dst.size));
            }
            trace_mem_store(
                target.0,
                target.1,
                dst.size,
                val,
                env.state.get_reg("PC", 20),
            );
        }
        Stmt::SetFlag { flag, value } => {
            let (val, _) = eval_expr(value, env)?;
            env.state.set_flag(flag, val);
            if std::env::var("RUST_FLAG_TRACE").is_ok() {
                let pc = env.state.get_reg("PC", 20);
                println!(
                    "[flag-trace] pc=0x{pc:06X} flag={flag} value=0x{val:X}",
                    pc = pc,
                    flag = flag,
                    val = val
                );
            }
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
        Stmt::ExtRegLoad {
            dst,
            ptr,
            mode,
            disp,
        } => {
            let (ptr_name, mode_val, disp_val) =
                resolve_ext_ptr_params(env, "ptr", &ptr.name, mode, disp.as_ref())?;
            let ptr_before = env.state.get_reg(&ptr_name, 24);
            let (value, addr) = ext_ptr_read(env, &ptr_name, &mode_val, disp_val, dst.size)?;
            trace_ext_access(
                env,
                "load",
                &ptr_name,
                ptr_before,
                &mode_val,
                disp_val,
                addr,
                dst.size,
                value,
            );
            env.state.set_reg(&dst.name, value, dst.size);
        }
        Stmt::ExtRegStore {
            src,
            ptr,
            mode,
            disp,
        } => {
            let (ptr_name, mode_val, disp_val) =
                resolve_ext_ptr_params(env, "ptr", &ptr.name, mode, disp.as_ref())?;
            let value = env.state.get_reg(&src.name, src.size);
            let ptr_before = env.state.get_reg(&ptr_name, 24);
            let addr = ext_ptr_store(env, &ptr_name, &mode_val, disp_val, src.size, value)?;
            trace_ext_access(
                env,
                "store",
                &ptr_name,
                ptr_before,
                &mode_val,
                disp_val,
                addr,
                src.size,
                value,
            );
        }
        Stmt::IntMemSwap { left, right, width } => {
            let (l, _) = eval_expr(left, env)?;
            let (r, _) = eval_expr(right, env)?;
            let l_addr = resolve_imem_addr(env, l, None);
            let r_addr = resolve_imem_addr(env, r, None);
            let l_val = env.bus.load(Space::Int, l_addr, *width);
            let r_val = env.bus.load(Space::Int, r_addr, *width);
            env.bus.store(Space::Int, l_addr, *width, r_val);
            env.bus.store(Space::Int, r_addr, *width, l_val);
        }
        Stmt::ExtRegToInt {
            ptr,
            mode,
            dst,
            disp,
        } => {
            let (ptr_name, mode_val, disp_val) =
                resolve_ext_ptr_params(env, "ptr", &ptr.name, mode, disp.as_ref())?;
            let ptr_before = env.state.get_reg(&ptr_name, 24);
            let (value, addr_src) = ext_ptr_read(env, &ptr_name, &mode_val, disp_val, dst.size)?;
            let (addr, _) = eval_expr(&dst.addr, env)?;
            let key = match &dst.addr {
                Expr::Tmp { name, .. } => Some(name.as_str()),
                _ => None,
            };

            let offset = match dst.space {
                AstSpace::Int => resolve_imem_addr(env, addr, key),
                _ => addr,
            };
            let space = match dst.space {
                AstSpace::Int => Space::Int,
                AstSpace::Ext | AstSpace::Code => Space::Ext,
            };
            env.bus.store(space, offset, dst.size, value);
            trace_ext_access(
                env,
                "to_int",
                &ptr_name,
                ptr_before,
                &mode_val,
                disp_val,
                addr_src,
                dst.size,
                value,
            );
        }
        Stmt::IntToExtReg {
            ptr,
            mode,
            src,
            disp,
        } => {
            let (ptr_name, mode_val, disp_val) =
                resolve_ext_ptr_params(env, "ptr", &ptr.name, mode, disp.as_ref())?;
            if let Some(expr) = env.binder.get("src_pre_slot") {
                if let Expr::Const { value, .. } = expr {
                    let target = (*value as usize).saturating_sub(1);
                    while env.imem_index < target {
                        let _ = env.next_imem_mode();
                    }
                }
            }
            let (addr, _) = eval_expr(&src.addr, env)?;
            let key = match &src.addr {
                Expr::Tmp { name, .. } => Some(name.as_str()),
                _ => None,
            };
            let offset = resolve_imem_addr(env, addr, key);
            let value = env.bus.load(Space::Int, offset, src.size);
            let ptr_before = env.state.get_reg(&ptr_name, 24);
            let addr_written =
                ext_ptr_store(env, &ptr_name, &mode_val, disp_val, src.size, value)?;
            trace_ext_access(
                env,
                "int_to_ext",
                &ptr_name,
                ptr_before,
                &mode_val,
                disp_val,
                addr_written,
                src.size,
                value,
            );
        }
        Stmt::Effect { kind, args } => {
            exec_effect(kind, args, env)?;
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
            let (ptr, ptr_bits) = eval_expr(addr, env)?;
            let key = match addr.as_ref() {
                Expr::Tmp { name, .. } => Some(name.clone()),
                _ => None,
            };
            let (bus_space, bus_addr) = match space {
                AstSpace::Int => {
                    if ptr_bits > 8 {
                        let absolute =
                            ptr.wrapping_sub(INTERNAL_MEMORY_START as u32) & 0xFF;
                        (Space::Int, absolute)
                    } else {
                        (
                            Space::Int,
                            resolve_imem_addr(env, ptr, key.as_deref()),
                        )
                    }
                }
                AstSpace::Ext | AstSpace::Code => (Space::Ext, ptr),
            };
            let value = if bus_space == Space::Int {
                env.imem_last_write
                    .get(&(bus_addr & 0xFF))
                    .copied()
                    .unwrap_or_else(|| env.bus.load(bus_space, bus_addr, *size))
            } else {
                env.bus.load(bus_space, bus_addr, *size)
            } & mask(*size);
            if std::env::var("RUST_MEM_TRACE").is_ok() {
                let pc = env.state.get_reg("PC", 20);
                let space_name = match space {
                    AstSpace::Int => "int",
                    AstSpace::Ext => "ext",
                    AstSpace::Code => "code",
                };
                println!(
                    "[mem-load] pc=0x{pc:06X} space={space_name} addr=0x{addr:06X} bits={bits} value=0x{value:06X}",
                    pc = pc,
                    addr = bus_addr & STACK_MASK,
                    bits = size,
                    value = value & mask(*size),
                    space_name = space_name,
                );
            }
            (value, *size)
        }
        Expr::UnOp {
            op, a, out_size, ..
        } => {
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
                "eq" => {
                    if std::env::var("RUST_EQ_TRACE").is_ok() {
                        let pc = env.state.get_reg("PC", 20);
                        println!(
                            "[eq-trace] pc=0x{pc:06X} lhs=0x{lhs:X} rhs=0x{rhs:X} result={} ",
                            if lhs == rhs { 1 } else { 0 }
                        );
                    }
                    u32::from(lhs == rhs)
                }
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
        Expr::PcRel {
            base,
            out_size,
            disp,
        } => {
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
        Expr::LoopPtr { .. } => {
            return Err(Error::Unsupported("loop ptr cannot be evaluated directly"))
        }
        Expr::ExtRegPtr { ptr, .. } => eval_expr(ptr, env)?,
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

fn resolve_imem_addr<B: Bus>(env: &mut Env<B>, offset: u32, key: Option<&str>) -> u32 {
    if let Some(name) = key {
        if let Some(cached) = env.imem_cache.get(name) {
            trace_imem_addr(env, offset, "(cached)", *cached);
            return *cached;
        }
    }
    let mode = env.next_imem_mode();
    let base_bp = env.bus.load(Space::Int, BP_ADDR, 8) as u32;
    let addr = match mode.as_str() {
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
    };
    trace_imem_addr(env, offset, &mode, addr);
    if let Some(name) = key {
        env.imem_cache.insert(name.to_string(), addr);
    }
    addr
}

fn ext_ptr_read<B: Bus>(
    env: &mut Env<B>,
    ptr: &str,
    mode: &str,
    disp: i32,
    bits: u8,
) -> Result<(u32, u32)> {
    let addr = resolve_ext_ptr(env, ptr, mode, disp, bits, true);
    let value = env.bus.load(Space::Ext, addr, bits);
    Ok((value, addr))
}

fn ext_ptr_store<B: Bus>(
    env: &mut Env<B>,
    ptr: &str,
    mode: &str,
    disp: i32,
    bits: u8,
    value: u32,
) -> Result<u32> {
    let addr = resolve_ext_ptr(env, ptr, mode, disp, bits, false);
    env.bus.store(Space::Ext, addr, bits, value);
    Ok(addr)
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
            env.state.set_reg(ptr, base as u32 & mask(24), 24);
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

fn trace_ext_access<B: Bus>(
    env: &Env<B>,
    kind: &str,
    ptr: &str,
    ptr_before: u32,
    mode: &str,
    disp: i32,
    addr: u32,
    bits: u8,
    value: u32,
) {
    if !ext_trace_enabled() {
        return;
    }
    let pc = env.state.get_reg("PC", 20);
    println!(
        "[ext-reg-{kind}] pc=0x{pc:06X} ptr={ptr} ptr_val=0x{ptr_before:06X} mode={mode} disp={disp:+} addr=0x{addr:06X} bits={bits} value=0x{value:06X}",
        kind = kind,
        pc = pc,
        ptr = ptr,
        ptr_before = ptr_before,
        mode = mode,
        disp = disp,
        addr = addr,
        bits = bits,
        value = value & mask(bits),
    );
}

fn trace_imem_addr<B: Bus>(env: &Env<B>, offset: u32, mode: &str, addr: u32) {
    let pc = env.state.get_reg("PC", 20);
    if !should_trace_imem(pc) {
        return;
    }
    eprintln!(
        "[imem-trace] pc=0x{pc:06X} mode={mode} offset=0x{off:02X} addr=0x{addr:02X}",
        pc = pc,
        mode = mode,
        off = offset & 0xFF,
        addr = addr & 0xFF
    );
}

fn resolve_ext_ptr_params<B: Bus>(
    env: &mut Env<B>,
    binder_key: &str,
    default_ptr: &str,
    default_mode: &str,
    disp_expr: Option<&Expr>,
) -> Result<(String, String, i32)> {
    let mut ptr_name = default_ptr.to_string();
    let mut mode = default_mode.to_string();
    let mut disp_val = if let Some(expr) = disp_expr {
        let (value, bits) = eval_expr(expr, env)?;
        sign_extend(value, bits, 32) as i32
    } else {
        0
    };
    if let Some(expr) = env.binder.get(binder_key).cloned() {
        if let Expr::ExtRegPtr {
            ptr,
            mode: binder_mode,
            disp,
        } = expr
        {
            mode = binder_mode;
            if let Expr::Reg { name, .. } = *ptr {
                ptr_name = name;
            }
            if let Some(dexpr) = disp {
                let (value, bits) = eval_expr(&dexpr, env)?;
                disp_val = sign_extend(value, bits, 32) as i32;
            } else {
                disp_val = 0;
            }
        }
    }
    Ok((ptr_name, mode, disp_val))
}

fn should_trace_imem(pc: u32) -> bool {
    if env::var("IMEM_TRACE").is_ok() {
        return true;
    }
    if let Ok(raw) = env::var("IMEM_TRACE_RANGE") {
        if let Some((start, end)) = parse_range(&raw) {
            return pc >= start && pc <= end;
        }
    }
    false
}

fn trace_reg_write<B: Bus>(env: &Env<B>, name: &str, value: u32, bits: u8) {
    if !should_trace_lcd(env.state.get_reg("PC", 20)) {
        return;
    }
    println!(
        "[reg-write] pc=0x{pc:06X} reg={name} value=0x{value:06X} bits={bits}",
        pc = env.state.get_reg("PC", 20),
        name = name,
        value = value & mask(bits),
        bits = bits
    );
}

fn mem_trace_enabled() -> bool {
    env::var("RUST_MEM_TRACE").is_ok()
}

fn ext_trace_enabled() -> bool {
    env::var("RUST_EXT_TRACE").is_ok()
}

fn trace_mem_store(space: Space, addr: u32, size: u8, value: u32, pc: u32) {
    if !mem_trace_enabled() {
        return;
    }
    match space {
        Space::Int => println!(
            "[mem-store] pc=0x{pc:06X} space=int addr=0x{addr:02X} size={size} value=0x{val:02X}",
            pc = pc,
            addr = addr & 0xFF,
            size = size,
            val = value & 0xFF
        ),
        Space::Ext | Space::Code => {}
    }
}

fn should_trace_lcd(pc: u32) -> bool {
    if env::var("LCD_LOOP_TRACE").is_err() {
        return false;
    }
    let (start, end) = lcd_trace_range();
    pc >= start && pc <= end
}

fn loop_trace_enabled() -> bool {
    env::var("RUST_LOOP_TRACE").is_ok()
}

fn log_loop_int_store(kind: &str, pc: u32, addr: u32, bits: u8, value: u32) {
    if !loop_trace_enabled() {
        return;
    }
    println!(
        "[loop-store] kind={kind} pc=0x{pc:06X} addr=0x{addr:02X} bits={bits} value=0x{val:06X}",
        kind = kind,
        pc = pc,
        addr = addr & 0xFF,
        bits = bits,
        val = value & mask(bits),
    );
}

fn lcd_trace_range() -> (u32, u32) {
    if let Ok(raw) = env::var("LCD_LOOP_RANGE") {
        if let Some((start, end)) = parse_range(&raw) {
            return (start, end);
        }
    }
    (0x0F1180, 0x0F3200)
}

fn parse_range(raw: &str) -> Option<(u32, u32)> {
    let mut parts = raw.split('-');
    let first = parts.next()?.trim();
    let second = parts.next().unwrap_or(first).trim();
    let start = parse_u32(first)?;
    let end = parse_u32(second)?;
    Some((start.min(end), start.max(end)))
}

fn parse_u32(raw: &str) -> Option<u32> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(hex) = trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        u32::from_str_radix(trimmed, 16).ok().or_else(|| trimmed.parse().ok())
    }
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

fn exec_effect<B: Bus>(kind: &str, args: &[Expr], env: &mut Env<B>) -> Result<()> {
    match kind {
        "nop" => {}
        "push_ret16" => {
            let (value, _) = eval_expr(expect_arg(args, 0)?, env)?;
            stack_push(env, "S", value, 2);
            log_pushenv(env, value);
        }
        "push_ret24" => {
            let (value, _) = eval_expr(expect_arg(args, 0)?, env)?;
            stack_push(env, "S", value, 3);
            log_pushenv(env, value);
        }
        "goto_page_join" => {
            let (lo, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let (page, _) = eval_expr(expect_arg(args, 1)?, env)?;
            let target = ((page & 0xF0000) | (lo & 0xFFFF)) & PC_MASK;
            env.state.set_reg("PC", target, 20);
        }
        "goto_far24" => {
            let (addr, _) = eval_expr(expect_arg(args, 0)?, env)?;
            env.state.set_reg("PC", addr & PC_MASK, 20);
        }
        "ret_near" => {
            let value = stack_pop(env, "S", 2);
            let page = env.state.get_reg("PC", 20) & 0xF0000;
            let target = (page | (value & 0xFFFF)) & PC_MASK;
            env.state.set_reg("PC", target, 20);
            log_ret(env, target);
        }
        "ret_far" => {
            let value = stack_pop(env, "S", 3) & PC_MASK;
            env.state.set_reg("PC", value, 20);
            log_ret(env, value);
        }
        "reti" => {
            reti(env)?;
        }
        "push_bytes" => {
            let (stack_name, _) =
                expr_as_reg(expect_arg(args, 0)?).ok_or(Error::Unsupported("stack reg"))?;
            let (value, _) = eval_expr(expect_arg(args, 1)?, env)?;
            let (width_bits, _) = expect_const_arg(expect_arg(args, 2)?)?;
            let bits = width_bits as u8;
            let bytes = ((bits / 8).max(1)) as u8;
            let masked = value & mask(bits);
            stack_push(env, stack_name, masked, bytes);
        }
        "pop_bytes" => {
            let (stack_name, _) =
                expr_as_reg(expect_arg(args, 0)?).ok_or(Error::Unsupported("stack reg"))?;
            let dest = expect_arg(args, 1)?;
            let (width_bits, _) = expect_const_arg(expect_arg(args, 2)?)?;
            let bits = width_bits as u8;
            let bytes = ((bits / 8).max(1)) as u8;
            let value = stack_pop(env, stack_name, bytes);
            match dest {
                Expr::Reg { name, size, .. } => {
                    env.state.set_reg(name, value, *size);
                }
                Expr::Mem { space, size, addr } => {
                    let (addr_val, addr_bits) = eval_expr(addr, env)?;
                    let key = match addr.as_ref() {
                        Expr::Tmp { name, .. } => Some(name.as_str()),
                        _ => None,
                    };
                    let (bus_space, bus_addr) = match space {
                        AstSpace::Int => {
                            if addr_bits > 8 {
                                (
                                    Space::Int,
                                    addr_val
                                        .wrapping_sub(INTERNAL_MEMORY_START as u32)
                                        & 0xFF,
                                )
                            } else {
                                (
                                    Space::Int,
                                    resolve_imem_addr(env, addr_val, key),
                                )
                            }
                        }
                        AstSpace::Ext | AstSpace::Code => (Space::Ext, addr_val),
                    };
                    if std::env::var("RUST_IMR_TRACE").map(|v| v == "1").unwrap_or(false)
                        && matches!(bus_space, Space::Int)
                        && bus_addr == IMR_ADDR
                    {
                        let pc = env.state.get_reg("PC", 20);
                        eprintln!(
                            "[rust-imr-trace] pc=0x{pc:06X} action=pop_bytes value=0x{value:02X}"
                        );
                    }
                    env.bus.store(bus_space, bus_addr, *size, value);
                }
                _ => return Err(Error::Unsupported("pop_bytes dst")),
            }
        }
        "loop_move" => {
            let (count, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let dst_ptr = expect_arg(args, 1)?;
            let src_ptr = expect_arg(args, 2)?;
            let (step_val, step_bits) = expect_const_arg(expect_arg(args, 3)?)?;
            let (width_bits, _) = expect_const_arg(expect_arg(args, 4)?)?;
            let width_bytes = ((width_bits as u8) / 8).max(1) as u32;
            let mut remaining = count & 0xFFFF;
            let mut dst_offset = loop_ptr_addr(dst_ptr, env)?;
            let mut src_offset = loop_ptr_addr(src_ptr, env)?;
            let step = sign_extend(step_val, step_bits as u8, 32) as i32;
            while remaining != 0 {
                for byte in 0..width_bytes {
                    let src_addr = wrap_add_u8(src_offset, byte as i32);
                    let dst_addr = wrap_add_u8(dst_offset, byte as i32);
                    let value = env.bus.load(Space::Int, src_addr, 8);
                    env.bus.store(Space::Int, dst_addr, 8, value);
                }
                dst_offset = wrap_add_u8(dst_offset, step);
                src_offset = wrap_add_u8(src_offset, step);
                remaining = (remaining - 1) & 0xFFFF;
            }
            env.state.set_reg("I", remaining, 16);
        }
        "loop_ext_to_int" => {
            let (count, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let dst_ptr = expect_arg(args, 1)?;
            let dst_offset = loop_ptr_addr(dst_ptr, env)?;
            let src_operand = expect_arg(args, 2)?;
            let (mode_tag, _) = expect_const_arg(expect_arg(args, 3)?)?;
            let (disp_raw, disp_bits) = expect_const_arg(expect_arg(args, 4)?)?;
            let disp = sign_extend(disp_raw, disp_bits as u8, 32) as i32;
            let (width_bits, _) = expect_const_arg(expect_arg(args, 5)?)?;
            let width_bytes = ((width_bits as u32) / 8).max(1);
            let (step_raw, step_bits) = expect_const_arg(expect_arg(args, 6)?)?;
            let step = sign_extend(step_raw, step_bits as u8, 32) as i32;
            let mut remaining = count & 0xFFFF;
            let mut dst_offset = dst_offset;
            let pc = env.state.get_reg("PC", 20);
            if mode_tag == LOOP_EXT_MODE_IMM {
                let (mut src_addr, _) = eval_expr(src_operand, env)?;
                src_addr &= STACK_MASK;
                while remaining != 0 {
                    let value = env.bus.load(Space::Ext, src_addr, width_bits as u8);
                    env.bus
                        .store(Space::Int, dst_offset, width_bits as u8, value);
                    log_loop_int_store("ext_to_int", pc, dst_offset, width_bits as u8, value);
                    src_addr = (src_addr + width_bytes) & STACK_MASK;
                    dst_offset = wrap_add_u8(dst_offset, step);
                    remaining = (remaining - 1) & 0xFFFF;
                }
            } else {
                let (ptr_name, _) =
                    expr_as_reg(src_operand).ok_or(Error::Unsupported("loop ext src"))?;
                let mode = loop_mode_from_tag(mode_tag as u32)?;
                while remaining != 0 {
                    let (value, _) = ext_ptr_read(env, ptr_name, mode, disp, width_bits as u8)?;
                    env.bus
                        .store(Space::Int, dst_offset, width_bits as u8, value);
                    log_loop_int_store("ext_to_int", pc, dst_offset, width_bits as u8, value);
                    dst_offset = wrap_add_u8(dst_offset, step);
                    remaining = (remaining - 1) & 0xFFFF;
                }
            }
            env.state.set_reg("I", remaining, 16);
        }
        "loop_int_to_ext" => {
            let (count, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let src_ptr = expect_arg(args, 1)?;
            let mut src_offset = loop_ptr_addr(src_ptr, env)?;
            let dest_operand = expect_arg(args, 2)?;
            let (mode_tag, _) = expect_const_arg(expect_arg(args, 3)?)?;
            let (disp_raw, disp_bits) = expect_const_arg(expect_arg(args, 4)?)?;
            let disp = sign_extend(disp_raw, disp_bits as u8, 32) as i32;
            let (width_bits, _) = expect_const_arg(expect_arg(args, 5)?)?;
            let width_bytes = ((width_bits as u32) / 8).max(1);
            let (step_raw, step_bits) = expect_const_arg(expect_arg(args, 6)?)?;
            let step = sign_extend(step_raw, step_bits as u8, 32) as i32;
            let mut remaining = count & 0xFFFF;
            if mode_tag == LOOP_EXT_MODE_IMM {
                let (mut dest_addr, _) = eval_expr(dest_operand, env)?;
                dest_addr &= STACK_MASK;
                let pc = env.state.get_reg("PC", 20);
                while remaining != 0 {
                    let value = env.bus.load(Space::Int, src_offset, width_bits as u8);
                    env.bus
                        .store(Space::Ext, dest_addr, width_bits as u8, value);
                    if loop_trace_enabled() {
                        println!(
                            "[loop-store] kind=int_to_ext pc=0x{pc:06X} dst=0x{dest:06X} bits={bits} value=0x{val:06X}",
                            pc = pc,
                            dest = dest_addr & STACK_MASK,
                            bits = width_bits,
                            val = value & mask(width_bits as u8),
                        );
                    }
                    src_offset = wrap_add_u8(src_offset, step);
                    dest_addr = (dest_addr + width_bytes) & STACK_MASK;
                    remaining = (remaining - 1) & 0xFFFF;
                }
            } else {
                let (ptr_name, _) =
                    expr_as_reg(dest_operand).ok_or(Error::Unsupported("loop ext dst"))?;
                let mode = loop_mode_from_tag(mode_tag as u32)?;
                while remaining != 0 {
                    let value = env.bus.load(Space::Int, src_offset, width_bits as u8);
                    ext_ptr_store(env, ptr_name, mode, disp, width_bits as u8, value)?;
                    src_offset = wrap_add_u8(src_offset, step);
                    remaining = (remaining - 1) & 0xFFFF;
                }
            }
            env.state.set_reg("I", remaining, 16);
        }
        "loop_add_carry" | "loop_sub_borrow" => {
            let (count, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let dst_ptr = expect_arg(args, 1)?;
            let src_operand = expect_arg(args, 2)?;
            let carry_flag = expect_arg(args, 3)?;
            let mut remaining = count & 0xFFFF;
            let mut dst_offset = loop_ptr_addr(dst_ptr, env)?;
            let mut src_offset = None;
            let mut src_reg: Option<(&str, u8)> = None;
            if let Some(reg) = expr_as_reg(src_operand) {
                src_reg = Some(reg);
            } else if matches!(src_operand, Expr::LoopPtr { .. }) {
                src_offset = Some(loop_ptr_addr(src_operand, env)?);
            } else {
                return Err(Error::Unsupported("loop carry src"));
            }
            let mut carry = if let Expr::Flag { name } = carry_flag {
                env.state.get_flag(name)
            } else {
                env.state.get_flag("C")
            };
            let mut overall_zero = 0u32;
            while remaining != 0 {
                let dst_byte = env.bus.load(Space::Int, dst_offset, 8) & 0xFF;
                let src_byte = if let Some((name, bits)) = src_reg {
                    env.state.get_reg(name, bits) & 0xFF
                } else {
                    let addr = src_offset.expect("loop src addr");
                    env.bus.load(Space::Int, addr, 8) & 0xFF
                };
                let result = if kind == "loop_sub_borrow" {
                    let diff = dst_byte as i32 - src_byte as i32 - carry as i32;
                    if diff < 0 {
                        carry = 1;
                        (diff + 0x100) as u32 & 0xFF
                    } else {
                        carry = 0;
                        diff as u32 & 0xFF
                    }
                } else {
                    let total = dst_byte + src_byte + carry;
                    carry = if total > 0xFF { 1 } else { 0 };
                    total & 0xFF
                };
                env.bus.store(Space::Int, dst_offset, 8, result);
                overall_zero |= result;
                dst_offset = wrap_add_u8(dst_offset, 1);
                if let Some(offset) = src_offset.as_mut() {
                    *offset = wrap_add_u8(*offset, 1);
                }
                remaining = (remaining - 1) & 0xFFFF;
            }
            env.state.set_reg("I", remaining, 16);
            env.state.set_flag("C", carry);
            env.state
                .set_flag("Z", if (overall_zero & 0xFF) == 0 { 1 } else { 0 });
        }
        "loop_bcd_add" | "loop_bcd_sub" => {
            let (count, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let dst_ptr = expect_arg(args, 1)?;
            let src_operand = expect_arg(args, 2)?;
            let direction = {
                let (value, bits) = expect_const_arg(expect_arg(args, 5)?)?;
                sign_extend(value, bits as u8, 32) as i32
            };
            let clear_carry = {
                let (value, _) = expect_const_arg(expect_arg(args, 6)?)?;
                value != 0
            };
            let mut remaining = count & 0xFFFF;
            let mut dst_offset = loop_ptr_addr(dst_ptr, env)?;
            let mut src_offset = None;
            let mut src_reg: Option<(&str, u8)> = None;
            if let Some(reg) = expr_as_reg(src_operand) {
                src_reg = Some(reg);
            } else if matches!(src_operand, Expr::LoopPtr { .. }) {
                src_offset = Some(loop_ptr_addr(src_operand, env)?);
            } else {
                return Err(Error::Unsupported("loop_bcd src"));
            }
            let mut carry = if clear_carry {
                0
            } else {
                env.state.get_flag("C")
            };
            let mut overall_zero = 0u32;
            while remaining != 0 {
                let dst_byte = env.bus.load(Space::Int, dst_offset, 8) as u8;
                let src_byte = if let Some((name, bits)) = src_reg {
                    env.state.get_reg(name, bits) as u8
                } else {
                    let addr = src_offset.expect("loop src addr");
                    env.bus.load(Space::Int, addr, 8) as u8
                };
                let (result, next_carry) = if kind == "loop_bcd_sub" {
                    bcd_sub_byte(dst_byte, src_byte, carry)
                } else {
                    bcd_add_byte(dst_byte, src_byte, carry)
                };
                carry = next_carry;
                env.bus.store(Space::Int, dst_offset, 8, result as u32);
                overall_zero |= result as u32;
                dst_offset = wrap_add_u8(dst_offset, direction);
                if let Some(offset) = src_offset.as_mut() {
                    *offset = wrap_add_u8(*offset, direction);
                }
                remaining = (remaining - 1) & 0xFFFF;
            }
            env.state.set_reg("I", remaining, 16);
            env.state.set_flag("C", carry);
            env.state
                .set_flag("Z", if (overall_zero & 0xFF) == 0 { 1 } else { 0 });
        }
        "decimal_shift" => {
            let (count, _) = eval_expr(expect_arg(args, 0)?, env)?;
            let ptr = expect_arg(args, 1)?;
            let is_left = {
                let (value, _) = expect_const_arg(expect_arg(args, 3)?)?;
                value != 0
            };
            let mut remaining = count & 0xFFFF;
            let mut current_addr = loop_ptr_addr(ptr, env)?;
            let mut digit_carry = 0u8;
            let mut overall_zero = 0u32;
            while remaining != 0 {
                let byte = env.bus.load(Space::Int, current_addr, 8) as u8;
                let low = byte & 0x0F;
                let high = (byte >> 4) & 0x0F;
                let (shifted, next_addr, new_carry) = if is_left {
                    let shifted = ((low << 4) & 0xF0) | (digit_carry & 0x0F);
                    (shifted, wrap_add_u8(current_addr, -1), low)
                } else {
                    let shifted = (high & 0x0F) | ((digit_carry & 0x0F) << 4);
                    (shifted, wrap_add_u8(current_addr, 1), high)
                };
                env.bus.store(Space::Int, current_addr, 8, shifted as u32);
                overall_zero |= shifted as u32;
                digit_carry = new_carry;
                current_addr = next_addr;
                remaining = (remaining - 1) & 0xFFFF;
            }
            env.state.set_reg("I", remaining, 16);
            env.state
                .set_flag("Z", if (overall_zero & 0xFF) == 0 { 1 } else { 0 });
        }
        "pmdf" => {
            let addr_ptr = expect_arg(args, 0)?;
            let value_expr = expect_arg(args, 1)?;
            let mut dst_literal: Option<u32> = None;
            if let Some(expr) = env
                .binder
                .get("dst_off")
                .or_else(|| env.binder.get("dst"))
            {
                if let Expr::Const { value, .. } = expr {
                    dst_literal = Some(*value & 0xFF);
                }
            }
            let addr = if let Some(const_addr) = dst_literal {
                const_addr
            } else {
                loop_ptr_addr(addr_ptr, env)?
            };
            let (value, _) = eval_expr(value_expr, env)?;
            let current = env.bus.load(Space::Int, addr, 8) & 0xFF;
            let updated = (current + (value & 0xFF)) & 0xFF;
            env.bus.store(Space::Int, addr, 8, updated);
            if env::var("RUST_PMDF_TRACE").is_ok() {
                let pc = env.state.get_reg("PC", 20);
                let pre = env
                    .pre_latch
                    .as_ref()
                    .map(|l| format!("({}, {})", l.first, l.second))
                    .unwrap_or_else(|| "none".to_string());
                println!(
                    "[pmdf] pc=0x{pc:06X} pre={pre} dst=0x{dst:02X} addr=0x{addr:02X} current=0x{cur:02X} value=0x{val:02X} updated=0x{upd:02X}",
                    pc = pc,
                    pre = pre,
                    dst = dst_literal.unwrap_or(0xFF),
                    addr = addr & 0xFF,
                    cur = current,
                    val = value & 0xFF,
                    upd = updated
                );
            }
        }
        "swap_reg" => {
            let (lhs_name, lhs_size) = expr_as_reg(expect_arg(args, 0)?)
                .ok_or(Error::Unsupported("swap_reg lhs"))?;
            let (rhs_name, rhs_size) = expr_as_reg(expect_arg(args, 1)?)
                .ok_or(Error::Unsupported("swap_reg rhs"))?;
            if lhs_size != rhs_size {
                return Err(Error::Unsupported("swap_reg size mismatch"));
            }
            let lhs_val = env.state.get_reg(lhs_name, lhs_size);
            let rhs_val = env.state.get_reg(rhs_name, rhs_size);
            env.state.set_reg(lhs_name, rhs_val, lhs_size);
            env.state.set_reg(rhs_name, lhs_val, rhs_size);
        }
        "halt" | "off" => {
            enter_low_power_state(env);
            env.state.halted = true;
        }
        "reset" => {
            perform_reset(env);
            env.state.halted = false;
        }
        "wait" => {
            // Python backend models WAIT as decrementing I until it reaches zero.
            // Mirror that fast-path so both backends agree on the post-state.
            env.state.set_reg("I", 0, 16);
        }
        "interrupt_enter" => {
            interrupt_enter(env);
            env.state.halted = false;
        }
        _ => return Err(Error::Unsupported("effect kind")),
    }
    Ok(())
}

fn expect_arg<'a>(args: &'a [Expr], idx: usize) -> Result<&'a Expr> {
    args.get(idx)
        .ok_or(Error::Unsupported("missing effect arg"))
}

fn expect_const_arg(expr: &Expr) -> Result<(u32, u8)> {
    if let Expr::Const { value, size } = expr {
        Ok((*value, *size))
    } else {
        Err(Error::Unsupported("const expected"))
    }
}

fn expr_as_reg(expr: &Expr) -> Option<(&str, u8)> {
    if let Expr::Reg { name, size, .. } = expr {
        Some((name.as_str(), *size))
    } else {
        None
    }
}

fn loop_ptr_addr<B: Bus>(expr: &Expr, env: &mut Env<B>) -> Result<u32> {
    if let Expr::LoopPtr { offset } = expr {
        let (value, _) = eval_expr(offset, env)?;
        Ok(resolve_imem_addr(env, value & 0xFF, None))
    } else {
        Err(Error::Unsupported("loop pointer required"))
    }
}

fn loop_mode_from_tag(tag: u32) -> Result<&'static str> {
    match tag {
        PTR_MODE_SIMPLE => Ok("simple"),
        PTR_MODE_POST_INC => Ok("post_inc"),
        PTR_MODE_PRE_DEC => Ok("pre_dec"),
        PTR_MODE_OFFSET => Ok("offset"),
        _ => Err(Error::Unsupported("loop pointer mode")),
    }
}

fn stack_push<B: Bus>(env: &mut Env<B>, reg: &str, value: u32, bytes: u8) {
    let sp = env.state.get_reg(reg, 24);
    let count = bytes as u32;
    let new_sp = (sp.wrapping_sub(count)) & STACK_MASK;
    env.state.set_reg(reg, new_sp, 24);
    for i in 0..count {
        let byte = (value >> (i * 8)) & 0xFF;
        env.bus
            .store(Space::Ext, (new_sp + i) & STACK_MASK, 8, byte);
        log_stack_write(env, (new_sp + i) & STACK_MASK, byte as u32);
    }
    if env::var("RUST_STACK_PUSH_TRACE").map(|v| v == "1").unwrap_or(false)
        && value == 0x0F46C1
    {
        let pc = env.state.get_reg("PC", 20);
        eprintln!(
            "[stack-capture] pc=0x{pc:06X} pushing return=0x{value:06X} on {reg}"
        );
    }
    if stack_trace_enabled() {
        let pc = env.state.get_reg("PC", 20);
        eprintln!(
            "[stack-push] rust pc=0x{pc:06X} reg={reg} value=0x{value:06X} bytes={} new_sp=0x{sp:06X}",
            bytes,
            sp = new_sp
        );
    }
}

fn log_pushenv<B: Bus>(env: &Env<B>, value: u32) {
    if !lcd_trace_enabled() {
        return;
    }
    let stack = env.state.get_reg("S", 24);
    let pc = env.state.get_reg("PC", 20);
    eprintln!(
        "[lcd-loop] rust push_ret24 pc=0x{pc:06X} stack=0x{stack:06X} value=0x{value:06X}"
    );
}

fn log_interrupt_enter<B: Bus>(env: &Env<B>) {
    if !lcd_trace_enabled() {
        return;
    }
    let pc = env.state.get_reg("PC", 20);
    let stack = env.state.get_reg("S", 24);
    eprintln!(
        "[lcd-loop] rust interrupt_enter pc=0x{pc:06X} stack=0x{stack:06X}"
    );
}

fn log_interrupt_push<B: Bus>(env: &Env<B>, label: &str, value: u32, bytes: u8) {
    if !lcd_trace_enabled() {
        return;
    }
    let stack = env.state.get_reg("S", 24);
    eprintln!(
        "[lcd-loop] rust {label} stack=0x{stack:06X} value=0x{value:06X} bytes={bytes}",
        label = label,
        stack = stack,
        value = value & 0x00FF_FFFF,
        bytes = bytes,
    );
}

fn log_ret<B: Bus>(env: &Env<B>, value: u32) {
    if !lcd_trace_enabled() {
        return;
    }
    let stack = env.state.get_reg("S", 24);
    let pc = env.state.get_reg("PC", 20);
    let a = env.state.get_reg("A", 8);
    let ba = env.state.get_reg("BA", 16);
    eprintln!(
        "[lcd-loop] rust ret_far pc=0x{pc:06X} stack=0x{stack:06X} pop=0x{value:06X} A=0x{a:06X} BA=0x{ba:06X}"
    );
}

fn log_reti<B: Bus>(env: &Env<B>, imr: u32, flags: u32, target: u32) {
    if !lcd_trace_enabled() {
        return;
    }
    let stack = env.state.get_reg("S", 24);
    let pc = env.state.get_reg("PC", 20);
    eprintln!(
        "[lcd-loop] rust reti pc=0x{pc:06X} stack=0x{stack:06X} pop_pc=0x{target:06X} imr=0x{imr:02X} flags=0x{flags:02X}"
    );
}

fn lcd_trace_enabled() -> bool {
    std::env::var("LCD_LOOP_TRACE").map(|v| v == "1").unwrap_or(false)
}

fn stack_pop<B: Bus>(env: &mut Env<B>, reg: &str, bytes: u8) -> u32 {
    let sp = env.state.get_reg(reg, 24);
    let count = bytes as u32;
    let mut value = 0u32;
    for i in 0..count {
        let byte = env.bus.load(Space::Ext, (sp + i) & STACK_MASK, 8) & 0xFF;
        value |= byte << (i * 8);
    }
    env.state
        .set_reg(reg, (sp.wrapping_add(count)) & STACK_MASK, 24);
    if stack_trace_enabled() {
        let pc = env.state.get_reg("PC", 20);
        let new_sp = (sp.wrapping_add(count)) & STACK_MASK;
        eprintln!(
            "[stack-pop] rust pc=0x{pc:06X} reg={reg} value=0x{value:06X} bytes={} new_sp=0x{new_sp:06X}",
            bytes
        );
    }
    if env::var("RUST_RETI_TRACE").map(|v| v == "1").unwrap_or(false) {
        let pc = env.state.get_reg("PC", 20);
        if pc == 0x0F2060 || pc == 0x0F204B {
            eprintln!(
                "[reti-pop] pc=0x{pc:06X} der=0x{value:06X} sp=0x{sp:06X} reg={reg} bytes={bytes}"
            );
        }
    }
    value
}

fn log_stack_write<B: Bus>(env: &Env<B>, addr: u32, value: u32) {
    if !stack_trace_enabled() {
        return;
    }
    let pc = env.state.get_reg("PC", 20);
    eprintln!(
        "[stack-write] pc=0x{pc:06X} addr=0x{addr:06X} val=0x{value:02X}",
        pc = pc,
        addr = addr & STACK_MASK,
        value = value & 0xFF
    );
}

fn stack_trace_enabled() -> bool {
    env::var("STACK_TRACE").is_ok()
}

fn wrap_add_u8(base: u32, delta: i32) -> u32 {
    let base_i = base as i32;
    (((base_i + delta).rem_euclid(256)) as u32) & 0xFF
}

fn enter_low_power_state<B: Bus>(env: &mut Env<B>) {
    let mut usr = env.bus.load(Space::Int, USR_ADDR, 8) & 0xFF;
    usr = (usr & !0x3F) | 0x18;
    env.bus.store(Space::Int, USR_ADDR, 8, usr);

    let mut ssr = env.bus.load(Space::Int, SSR_ADDR, 8) & 0xFF;
    ssr |= 0x04;
    env.bus.store(Space::Int, SSR_ADDR, 8, ssr);
}

fn perform_reset<B: Bus>(env: &mut Env<B>) {
    for &addr in &[UCR_ADDR, ISR_ADDR, SCR_ADDR] {
        env.bus.store(Space::Int, addr, 8, 0);
    }
    let mut lcc = env.bus.load(Space::Int, LCC_ADDR, 8) & 0xFF;
    lcc &= !0x80;
    env.bus.store(Space::Int, LCC_ADDR, 8, lcc);

    let mut usr = env.bus.load(Space::Int, USR_ADDR, 8) & 0xFF;
    usr = (usr & !0x3F) | 0x18;
    env.bus.store(Space::Int, USR_ADDR, 8, usr);

    let mut ssr = env.bus.load(Space::Int, SSR_ADDR, 8) & 0xFF;
    ssr &= !0x04;
    env.bus.store(Space::Int, SSR_ADDR, 8, ssr);

    let vector = read_interrupt_vector(env);
    env.state.set_reg("PC", vector, 20);
}

fn interrupt_enter<B: Bus>(env: &mut Env<B>) {
    log_interrupt_enter(env);
    let pc = env.state.get_reg("PC", 24);
    stack_push(env, "S", pc, 3);
    log_interrupt_push(env, "interrupt_push_pc", pc, 3);
    let flag = env.state.get_reg("F", 8) & 0xFF;
    stack_push(env, "S", flag, 1);
    log_interrupt_push(env, "interrupt_push_f", flag, 1);
    let imr = env.bus.load(Space::Int, IMR_ADDR, 8) & 0xFF;
    stack_push(env, "S", imr, 1);
    log_interrupt_push(env, "interrupt_push_imr", imr, 1);
    env.bus.store(Space::Int, IMR_ADDR, 8, imr & 0x7F);
    let vector = read_interrupt_vector(env);
    env.state.set_reg("PC", vector, 20);
}

fn read_interrupt_vector<B: Bus>(env: &mut Env<B>) -> u32 {
    let lo = env.bus.load(Space::Code, INTERRUPT_VECTOR_ADDR, 8) & 0xFF;
    let mid = env.bus.load(Space::Code, INTERRUPT_VECTOR_ADDR + 1, 8) & 0xFF;
    let hi = env.bus.load(Space::Code, INTERRUPT_VECTOR_ADDR + 2, 8) & 0xFF;
    ((hi << 16) | (mid << 8) | lo) & PC_MASK
}

fn reti<B: Bus>(env: &mut Env<B>) -> Result<()> {
    let sp = env.state.get_reg("S", 24);
    let imr = env.bus.load(Space::Ext, sp & STACK_MASK, 8) & 0xFF;
    let f_val = env.bus.load(Space::Ext, (sp + 1) & STACK_MASK, 8) & 0xFF;
    let lo = env.bus.load(Space::Ext, (sp + 2) & STACK_MASK, 8) & 0xFF;
    let mid = env.bus.load(Space::Ext, (sp + 3) & STACK_MASK, 8) & 0xFF;
    let hi = env.bus.load(Space::Ext, (sp + 4) & STACK_MASK, 8) & 0xFF;
    env.state.set_reg("S", (sp + 5) & STACK_MASK, 24);
    env.bus.store(Space::Int, IMR_ADDR, 8, imr);
    env.state.set_reg("F", f_val, 8);
    env.state.set_flag("C", (f_val & 1) as u32);
    env.state.set_flag("Z", ((f_val >> 1) & 1) as u32);
    let target = (((hi as u32) << 16) | ((mid as u32) << 8) | lo as u32) & PC_MASK;
    env.state.set_reg("PC", target, 20);
    log_reti(env, imr, f_val, target);
    Ok(())
}

fn bcd_add_byte(a: u8, b: u8, carry_in: u32) -> (u8, u32) {
    let mut low = (a & 0x0F) as i32 + (b & 0x0F) as i32 + (carry_in & 1) as i32;
    let mut carry_high = 0;
    if low >= 10 {
        low -= 10;
        carry_high = 1;
    }
    let mut high = ((a >> 4) & 0x0F) as i32 + ((b >> 4) & 0x0F) as i32 + carry_high;
    let mut carry_out = 0;
    if high >= 10 {
        high -= 10;
        carry_out = 1;
    }
    (
        (((high as u8) & 0x0F) << 4) | ((low as u8) & 0x0F),
        carry_out as u32,
    )
}

fn bcd_sub_byte(a: u8, b: u8, borrow_in: u32) -> (u8, u32) {
    let mut low = (a & 0x0F) as i32 - (b & 0x0F) as i32 - (borrow_in & 1) as i32;
    let mut borrow_high = 0;
    if low < 0 {
        low += 10;
        borrow_high = 1;
    }
    let mut high = ((a >> 4) & 0x0F) as i32 - ((b >> 4) & 0x0F) as i32 - borrow_high;
    let mut borrow_out = 0;
    if high < 0 {
        high += 10;
        borrow_out = 1;
    }
    (
        (((high as u8) & 0x0F) << 4) | ((low as u8) & 0x0F),
        borrow_out as u32,
    )
}
