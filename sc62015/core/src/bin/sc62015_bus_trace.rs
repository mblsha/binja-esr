// PY_SOURCE: pce500/emulator.py:PCE500Emulator
//! Offline fixed-record bus trace to legacy JSONL conversion.
use sc62015_core::bus_trace::{BusTraceEvent, HEADER};
use std::{
    error::Error,
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: sc62015-bus-trace INPUT > output.jsonl")?;
    if args.next().is_some() {
        return Err("unexpected arguments".into());
    }
    let mut input = BufReader::new(File::open(path)?);
    let mut header = [0; 8];
    input.read_exact(&mut header)?;
    if &header != HEADER {
        return Err("unsupported bus trace header/version".into());
    }
    let mut output = BufWriter::new(io::stdout().lock());
    let mut index = 0u64;
    while let Some(event) = BusTraceEvent::read_binary(&mut input)? {
        if event.index != index {
            return Err("bus trace has missing, duplicate, or reordered events".into());
        }
        serde_json::to_writer(&mut output, &event)?;
        output.write_all(b"\n")?;
        index = index.checked_add(1).ok_or("bus trace index overflow")?;
    }
    output.flush()?;
    Ok(())
}
