use sc62015_core::perfetto::PerfettoTracer;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> sc62015_core::Result<()> {
    let mut tracer = PerfettoTracer::new_call_ui(PathBuf::from("thread-names.perfetto-trace"));
    let regs: HashMap<String, u32> = [
        ("A".to_string(), 0x41),
        ("B".to_string(), 0x01),
        ("P".to_string(), 0x10),
        ("F".to_string(), 0x00),
        ("X".to_string(), 0x00),
        ("Y".to_string(), 0x00),
        ("SP".to_string(), 0x1000),
    ]
    .into_iter()
    .collect();
    tracer.record_regs(0, 0x00FFFE8, 0x00FFFE8, 0x00, Some("NOP"), &regs, 0, 0);
    tracer.finish()
}

