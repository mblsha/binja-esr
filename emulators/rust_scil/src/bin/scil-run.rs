use rust_scil::{
    ast::{Binder, Instr, PreLatch},
    bus::MemoryBus,
    eval::step,
    state::State,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Deserialize)]
struct Input {
    state: State,
    #[serde(default)]
    int_mem: Vec<(u32, u8)>,
    #[serde(default)]
    ext_mem: Vec<(u32, u8)>,
    instr: Instr,
    #[serde(default)]
    binder: Binder,
    #[serde(default)]
    pre_applied: Option<PreLatch>,
}

#[derive(Serialize)]
struct Output {
    state: State,
    int_mem: Vec<(u32, u8)>,
    ext_mem: Vec<(u32, u8)>,
}

fn main() -> anyhow::Result<()> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    let mut input: Input = serde_json::from_str(&buffer)?;

    let mut bus = MemoryBus::default();
    bus.preload_int(input.int_mem.clone());
    bus.preload_ext(input.ext_mem.clone());

    let start_pc = input.state.pc & 0xF_FFFF;
    step(
        &mut input.state,
        &mut bus,
        &input.instr,
        &input.binder,
        input.pre_applied,
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    if input.state.pc == start_pc {
        let adv = start_pc.wrapping_add(input.instr.length as u32) & 0xF_FFFF;
        input.state.pc = adv;
    }

    let output = Output {
        state: input.state,
        int_mem: bus.dump_int(),
        ext_mem: bus.dump_ext(),
    };
    serde_json::to_writer(io::stdout(), &output)?;
    Ok(())
}
