#![cfg(feature = "perfetto")]

use perfetto_protos::trace::Trace;
use protobuf::Message;
use sc62015_core::perfetto::PerfettoTracer;
use std::collections::HashMap;

fn parse_trace(bytes: &[u8]) -> Trace {
    Trace::parse_from_bytes(bytes).expect("valid trace")
}

#[test]
fn callui_emits_root_function_slice_from_zero_to_end() {
    let mut tracer = PerfettoTracer::new_call_ui("callui-root.perfetto-trace".into());
    tracer.call_ui_begin_root(0x00F2A87);

    let regs: HashMap<String, u32> = [
        ("A".to_string(), 0x41),
        ("B".to_string(), 0x00),
        ("P".to_string(), 0x10),
        ("F".to_string(), 0x00),
        ("X".to_string(), 0x00),
        ("Y".to_string(), 0x00),
        ("SP".to_string(), 0x1000),
    ]
    .into_iter()
    .collect();
    tracer.record_regs(0, 0x00F2A87, 0x00F2A87, 0x00, Some("NOP"), &regs, 0, 0);
    tracer.record_regs(1, 0x00F2A88, 0x00F2A88, 0x00, Some("NOP"), &regs, 0, 0);
    let bytes = tracer.serialize().expect("serialize");

    let trace = parse_trace(&bytes);

    let functions_uuid = trace
        .packet
        .iter()
        .filter(|p| p.has_track_descriptor())
        .find(|p| p.track_descriptor().name() == "Functions")
        .map(|p| p.track_descriptor().uuid())
        .expect("Functions track descriptor");

    let mut events = trace
        .packet
        .iter()
        .filter(|p| p.has_track_event())
        .filter(|p| p.track_event().track_uuid() == functions_uuid)
        .map(|p| {
            let ev = p.track_event();
            (
                ev.type_(),
                ev.timestamp_absolute_us(),
                ev.name_iid(),
            )
        })
        .collect::<Vec<_>>();

    assert!(
        events.iter().any(|(ty, ts, _)| {
            *ty == perfetto_protos::track_event::track_event::Type::TYPE_SLICE_BEGIN && *ts == 0
        }),
        "expected a Functions slice begin at t=0us"
    );

    // The root slice is ended on serialize(); ensure Functions has a slice end at the end
    // of the trace timeline (>= max timestamp observed on any track event).
    let max_ts = trace
        .packet
        .iter()
        .filter(|p| p.has_track_event())
        .map(|p| p.track_event().timestamp_absolute_us())
        .max()
        .unwrap_or(0);
    let max_fn_end = events
        .iter()
        .filter(|(ty, _, _)| {
            *ty == perfetto_protos::track_event::track_event::Type::TYPE_SLICE_END
        })
        .map(|(_, ts, _)| *ts)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_fn_end, max_ts,
        "expected root slice to end at trace end time"
    );

    // Keep `events` used to avoid clippy warning about unused mut if assertions above are compiled out.
    events.sort_by_key(|(_, ts, _)| *ts);
}
