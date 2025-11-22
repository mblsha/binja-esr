use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Space {
    #[serde(alias = "int")]
    Int,
    #[serde(alias = "ext")]
    Ext,
    #[serde(alias = "code")]
    Code,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Expr {
    #[serde(rename = "const")]
    Const { value: u32, size: u8 },
    #[serde(rename = "tmp")]
    Tmp { name: String, size: u8 },
    #[serde(rename = "reg")]
    Reg {
        name: String,
        size: u8,
        bank: Option<String>,
    },
    #[serde(rename = "flag")]
    Flag { name: String },
    #[serde(rename = "mem")]
    Mem {
        space: Space,
        size: u8,
        addr: Box<Expr>,
    },
    #[serde(rename = "unop")]
    UnOp {
        op: String,
        a: Box<Expr>,
        out_size: u8,
        param: Option<i32>,
    },
    #[serde(rename = "binop")]
    BinOp {
        op: String,
        a: Box<Expr>,
        b: Box<Expr>,
        out_size: u8,
    },
    #[serde(rename = "ternop")]
    TernOp {
        op: String,
        cond: Cond,
        t: Box<Expr>,
        f: Box<Expr>,
        out_size: u8,
    },
    #[serde(rename = "pcrel")]
    PcRel {
        base: i32,
        out_size: u8,
        disp: Option<Box<Expr>>,
    },
    #[serde(rename = "join24")]
    Join24 {
        hi: Box<Expr>,
        mid: Box<Expr>,
        lo: Box<Expr>,
    },
    #[serde(rename = "loop_ptr")]
    LoopPtr { offset: Box<Expr> },
    #[serde(rename = "ext_reg_ptr")]
    ExtRegPtr {
        ptr: Box<Expr>,
        mode: String,
        #[serde(default)]
        disp: Option<Box<Expr>>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Cond {
    #[serde(rename = "cond")]
    Prim {
        kind: String,
        a: Option<Box<Expr>>,
        b: Option<Box<Expr>>,
        flag: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tmp {
    pub name: String,
    pub size: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Reg {
    pub name: String,
    pub size: u8,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Stmt {
    #[serde(rename = "fetch")]
    Fetch { kind: String, dst: Tmp },
    #[serde(rename = "set_reg")]
    SetReg {
        reg: Reg,
        value: Expr,
        flags: Option<Vec<String>>,
    },
    #[serde(rename = "store")]
    Store { dst: Mem, value: Expr },
    #[serde(rename = "set_flag")]
    SetFlag { flag: String, value: Expr },
    #[serde(rename = "if")]
    If {
        cond: Cond,
        #[serde(default)]
        then: Vec<Stmt>,
        #[serde(default)]
        r#else: Vec<Stmt>,
    },
    #[serde(rename = "goto")]
    Goto { target: Expr },
    #[serde(rename = "call")]
    Call { target: Expr, far: bool },
    #[serde(rename = "ret")]
    Ret { far: bool, reti: bool },
    #[serde(rename = "ext_reg_load")]
    ExtRegLoad {
        dst: Reg,
        ptr: Reg,
        mode: String,
        #[serde(default)]
        disp: Option<Expr>,
    },
    #[serde(rename = "ext_reg_store")]
    ExtRegStore {
        src: Reg,
        ptr: Reg,
        mode: String,
        #[serde(default)]
        disp: Option<Expr>,
    },
    #[serde(rename = "int_mem_swap")]
    IntMemSwap { left: Expr, right: Expr, width: u8 },
    #[serde(rename = "ext_reg_to_int")]
    ExtRegToInt {
        ptr: Reg,
        mode: String,
        dst: Mem,
        #[serde(default)]
        disp: Option<Expr>,
    },
    #[serde(rename = "int_to_ext_reg")]
    IntToExtReg {
        ptr: Reg,
        mode: String,
        src: Mem,
        #[serde(default)]
        disp: Option<Expr>,
    },
    #[serde(rename = "label")]
    Label { name: String },
    #[serde(rename = "comment")]
    Comment { text: String },
    #[serde(rename = "effect")]
    Effect { kind: String, args: Vec<Expr> },
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mem {
    pub space: Space,
    pub size: u8,
    pub addr: Expr,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Instr {
    pub name: String,
    pub length: u8,
    pub semantics: Vec<Stmt>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreLatch {
    pub first: String,
    pub second: String,
}

pub type Binder = HashMap<String, Expr>;
