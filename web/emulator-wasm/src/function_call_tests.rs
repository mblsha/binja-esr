// PY_SOURCE: pce500/emulator.py:PCE500Emulator
use super::*;
use wasm_bindgen_test::wasm_bindgen_test;

const ENTRY: u32 = 0xB8000;
const STACK: u32 = 0xB9003;

#[wasm_bindgen_test]
fn structured_results_match_json_and_release_call_ownership() {
    for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
        for cancel in [false, true] {
            let mut legacy = machine(model);
            let mut structured = machine(model);
            let a = legacy
                .call_function_begin(ENTRY, 4, JsValue::UNDEFINED)
                .unwrap();
            let b = structured
                .call_function_begin(ENTRY, 4, JsValue::UNDEFINED)
                .unwrap();
            legacy.call_function_slice(a, 4, 16.0).unwrap();
            structured.call_function_slice(b, 4, 16.0).unwrap();
            let text = if cancel {
                legacy.call_function_cancel(a)
            } else {
                legacy.call_function_finish(a)
            }
            .unwrap();
            let object = if cancel {
                structured.call_function_cancel_object(b)
            } else {
                structured.call_function_finish_object(b)
            }
            .unwrap();
            assert!(!object.is_string());
            let registers =
                js_sys::Reflect::get(&object, &JsValue::from_str("before_regs")).unwrap();
            assert!(!registers.is_instance_of::<js_sys::Map>());
            let actual: serde_json::Value = serde_wasm_bindgen::from_value(object).unwrap();
            let expected: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(actual, expected);
            assert_machine_matches(&legacy, &structured);
            let next = structured
                .call_function_begin(ENTRY, 1, JsValue::UNDEFINED)
                .unwrap();
            structured.call_function_cancel_object(next).unwrap();
        }
    }
}

fn json_js(value: serde_json::Value) -> JsValue {
    // JSON objects match the JavaScript call API. serde_wasm_bindgen's default
    // generic-map serializer produces a JS Map, not an options object.
    js_sys::JSON::parse(&value.to_string()).unwrap()
}

fn machine(model: DeviceModel) -> Sc62015Emulator {
    let mut emulator = Sc62015Emulator {
        runtime: CoreRuntime::for_model(model, &[]).unwrap(),
        model,
        ..Sc62015Emulator::new()
    };
    // Plain RAM NOPs; no fabricated app/display output. Exercise actual timers
    // with interrupt delivery masked, and an independent deterministic IQ RTC.
    emulator.set_reg("PC", ENTRY).unwrap();
    emulator.set_reg("S", STACK).unwrap();
    emulator
        .runtime
        .memory
        .write_internal_byte(IMEM_IMR_OFFSET, 0);
    emulator
        .runtime
        .timer
        .configure_scr_periods(3, 3, 7, 7, 0, 0);
    emulator.runtime.timer.next_mti = 3;
    emulator.runtime.timer.next_sti = 7;
    if model == DeviceModel::Iq7000 {
        emulator
            .set_iq7000_rtc_yyyymmddhhmm("202609060000")
            .unwrap();
    }
    for (offset, byte) in [0xA1, 0xB2, 0xC3].into_iter().enumerate() {
        emulator.write_u8(STACK - 3 + offset as u32, byte).unwrap();
    }
    emulator
}

fn slice(emulator: &mut Sc62015Emulator, id: u32, budget: u32) -> serde_json::Value {
    serde_wasm_bindgen::from_value(emulator.call_function_slice(id, budget, 16.0).unwrap()).unwrap()
}

fn assert_machine_matches(a: &Sc62015Emulator, b: &Sc62015Emulator) {
    assert_eq!(
        sc62015_core::collect_registers(&a.runtime.state),
        sc62015_core::collect_registers(&b.runtime.state)
    );
    assert_eq!(
        a.runtime.memory.internal_slice(),
        b.runtime.memory.internal_slice()
    );
    assert_eq!(
        a.runtime.memory.external_slice(),
        b.runtime.memory.external_slice()
    );
    assert_eq!(
        a.runtime.memory.memory_read_count(),
        b.runtime.memory.memory_read_count()
    );
    assert_eq!(
        a.runtime.memory.memory_write_count(),
        b.runtime.memory.memory_write_count()
    );
    assert_eq!(a.cycle_count(), b.cycle_count());
    assert_eq!(a.instruction_count(), b.instruction_count());
    assert_eq!(a.runtime.timer.next_mti, b.runtime.timer.next_mti);
    assert_eq!(a.runtime.timer.next_sti, b.runtime.timer.next_sti);
    assert_eq!(
        serde_json::to_value(a.runtime.iq7000_rtc_state()).unwrap(),
        serde_json::to_value(b.runtime.iq7000_rtc_state()).unwrap()
    );
    assert_eq!(
        format!("{:?}", a.runtime.state.snapshot_call_metrics()),
        format!("{:?}", b.runtime.state.snapshot_call_metrics())
    );
}

#[wasm_bindgen_test]
fn resumable_calls_match_sync_calls_for_both_models_and_chunk_sizes() {
    for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
        for return_opcode in [None, Some(0x06), Some(0x07)] {
            for budget in [1, 17, 64, 1000] {
                let mut direct = machine(model);
                let mut sliced = machine(model);
                if let Some(opcode) = return_opcode {
                    direct.write_u8(ENTRY + 70, opcode).unwrap();
                    sliced.write_u8(ENTRY + 70, opcode).unwrap();
                }
                let expected: serde_json::Value = serde_json::from_str(
                    &direct
                        .call_function(ENTRY, 257)
                        .unwrap()
                        .as_string()
                        .unwrap(),
                )
                .unwrap();
                let id = sliced
                    .call_function_begin(ENTRY, 257, JsValue::UNDEFINED)
                    .unwrap();
                for attempt in 0..1000 {
                    if slice(&mut sliced, id, budget)["state"] == "complete" {
                        break;
                    }
                    assert!(attempt < 999, "call must finish its finite budget");
                }
                let actual: serde_json::Value =
                    serde_json::from_str(&sliced.call_function_finish(id).unwrap()).unwrap();
                assert_eq!(actual, expected);
                assert_eq!(
                    actual["report"]["reason"],
                    if return_opcode.is_some() {
                        "returned"
                    } else {
                        "timeout"
                    }
                );
                assert_machine_matches(&sliced, &direct);
            }
        }
    }
}

#[wasm_bindgen_test]
fn zero_deadline_and_invalid_slices_preserve_the_owned_call_without_bus_reads() {
    let mut emulator = machine(DeviceModel::PcE500);
    let id = emulator
        .call_function_begin(ENTRY, 1000, JsValue::UNDEFINED)
        .unwrap();
    let reads = emulator.runtime.memory.memory_read_count();
    let writes = emulator.runtime.memory.memory_write_count();
    assert_eq!(slice(&mut emulator, id, 0)["steps"], 0);
    let immediate: serde_json::Value =
        serde_wasm_bindgen::from_value(emulator.call_function_slice(id, u32::MAX, 0.0).unwrap())
            .unwrap();
    assert_eq!(immediate["steps"], 0);
    for bad in [-1.0, 17.0, f64::NAN, f64::INFINITY] {
        assert!(emulator.call_function_slice(id, 100, bad).is_err());
    }
    assert!(emulator.call_function_finish(id).is_err());
    assert_eq!(emulator.runtime.memory.memory_read_count(), reads);
    assert_eq!(emulator.runtime.memory.memory_write_count(), writes);
    emulator.call_function_cancel(id).unwrap();
}

#[wasm_bindgen_test]
fn cancel_restores_scaffolding_retains_progress_and_releases_trace_and_ownership() {
    for model in [DeviceModel::PcE500, DeviceModel::Iq7000] {
        let mut emulator = machine(model);
        let metrics = format!("{:?}", emulator.runtime.state.snapshot_call_metrics());
        let id = emulator
            .call_function_begin(ENTRY, 1000, json_js(serde_json::json!({ "trace": true })))
            .unwrap();
        assert!(emulator.step(1).is_err());
        assert!(emulator.run_slice(1, 4.0).is_err());
        assert!(emulator.reset().is_err());
        assert!(emulator.load_rom_with_model(&[0], "iq-7000").is_err());
        assert_eq!(emulator.model, model);
        assert!(emulator.set_reg("A", 42).is_err());
        assert!(emulator.write_u8(0x12000, 42).is_err());
        assert!(emulator.configure_timer(false, 1, 1).is_err());
        assert!(emulator.perfetto_stop_b64().is_err());
        assert!(emulator
            .call_function_begin(ENTRY, 1, JsValue::UNDEFINED)
            .is_err());
        // The public slice may yield at its host deadline before consuming the
        // requested boundary budget. Resume it to the same deterministic guest
        // endpoint rather than assuming CI completes 65 traced steps in 16 ms.
        let mut completed = 0;
        for attempt in 0..1000 {
            let status = slice(&mut emulator, id, 65 - completed);
            assert_eq!(status["state"], "running");
            completed = status["steps"].as_u64().unwrap() as u32;
            assert!(completed <= 65);
            if completed == 65 {
                break;
            }
            assert!(attempt < 999, "call must reach its finite boundary target");
        }
        assert_eq!(completed, 65);
        assert_eq!(emulator.get_reg("S"), STACK - 3);
        let report: serde_json::Value =
            serde_json::from_str(&emulator.call_function_cancel(id).unwrap()).unwrap();
        assert_eq!(report["report"]["reason"], "cancelled");
        assert_eq!(report["report"]["scheduler_boundaries"], 65);
        assert!(!report["perfetto_trace_b64"].as_str().unwrap().is_empty());
        assert_eq!(emulator.get_reg("PC"), ENTRY);
        assert_eq!(emulator.get_reg("S"), STACK);
        assert_eq!(emulator.instruction_count(), 65);
        assert!(emulator.cycle_count() > 0);
        assert!(
            emulator
                .runtime
                .memory
                .read_internal_byte_silent(IMEM_ISR_OFFSET)
                .unwrap()
                != 0
        );
        assert_eq!(
            format!("{:?}", emulator.runtime.state.snapshot_call_metrics()),
            metrics
        );
        for (offset, byte) in [0xA1, 0xB2, 0xC3].into_iter().enumerate() {
            assert_eq!(emulator.read_u8(STACK - 3 + offset as u32), byte);
        }
        emulator.perfetto_start("after-cancel").unwrap();
        emulator.perfetto_stop_b64().unwrap();
        emulator.step(1).unwrap();
        let next = emulator
            .call_function_begin(ENTRY, 1, JsValue::UNDEFINED)
            .unwrap();
        assert_ne!(next, id);
        assert!(emulator.call_function_cancel(id).is_err());
        emulator.call_function_cancel(next).unwrap();
    }
}

#[wasm_bindgen_test]
fn pending_stub_is_a_stable_host_handoff_without_execution_or_duplicate_probe_hits() {
    let mut emulator = machine(DeviceModel::Iq7000);
    let options =
        json_js(serde_json::json!({ "stubs": [{"id": 7, "pc": ENTRY}], "probe_pc": ENTRY }));
    let id = emulator.call_function_begin(ENTRY, 3, options).unwrap();
    let request = slice(&mut emulator, id, 10);
    assert_eq!(request["state"], "stub");
    assert_eq!(request["steps"], 0);
    assert_eq!(slice(&mut emulator, id, 10), request);
    assert_eq!(emulator.instruction_count(), 0);
    // Current-machine reads are now legal outside a mutable WASM borrow.
    assert_eq!(emulator.read_u8(STACK - 3), 0x0D);
    assert!(emulator
        .call_function_apply_stub(id, 9, JsValue::NULL)
        .is_err());
    let patch = json_js(serde_json::json!({
        "regs": [{"name": "A", "value": 42}],
        "mem_writes": [{"addr": 0x12000, "value": 0xA5, "size": 1}],
        "ret": {"kind": "retf"}
    }));
    emulator.call_function_apply_stub(id, 1, patch).unwrap();
    assert!(emulator
        .call_function_apply_stub(id, 1, JsValue::NULL)
        .is_err());
    assert_eq!(slice(&mut emulator, id, 10)["state"], "complete");
    let result: serde_json::Value =
        serde_json::from_str(&emulator.call_function_finish(id).unwrap()).unwrap();
    assert_eq!(result["report"]["reason"], "returned");
    assert_eq!(result["report"]["steps"], 1);
    assert_eq!(result["report"]["scheduler_boundaries"], 0);
    assert_eq!(result["probe_samples"].as_array().unwrap().len(), 1);
    assert_eq!(result["stubs_used"][0]["hits"], 1);
    assert_eq!(emulator.get_reg("A"), 42);
    assert_eq!(emulator.read_u8(0x12000), 0xA5);
    assert_eq!(emulator.get_reg("S"), STACK);
}

#[wasm_bindgen_test]
fn faults_and_halt_remain_distinct_from_cancellation() {
    for halted in [false, true] {
        let mut emulator = machine(DeviceModel::PcE500);
        emulator.write_u8(ENTRY, 0x20).unwrap(); // Reserved opcode must fail closed.
        emulator.runtime.state.set_halted(halted);
        let id = emulator
            .call_function_begin(ENTRY, 100, JsValue::UNDEFINED)
            .unwrap();
        assert_eq!(slice(&mut emulator, id, 10)["state"], "complete");
        let result: serde_json::Value =
            serde_json::from_str(&emulator.call_function_cancel(id).unwrap()).unwrap();
        assert_eq!(
            result["report"]["reason"],
            if halted { "halted" } else { "fault" }
        );
        assert_eq!(result["report"]["steps"], 0);
        assert_eq!(emulator.get_reg("S"), STACK);
    }
}

#[wasm_bindgen_test]
fn malformed_options_do_not_silently_start_an_untraced_unstubbed_call() {
    let mut emulator = machine(DeviceModel::PcE500);
    let writes = emulator.runtime.memory.memory_write_count();
    assert!(emulator
        .call_function_begin(ENTRY, 10, JsValue::from_str("invalid"))
        .is_err());
    let map = serde_wasm_bindgen::to_value(&serde_json::json!({"trace": true})).unwrap();
    assert!(emulator.call_function_begin(ENTRY, 10, map).is_err());
    assert_eq!(emulator.get_reg("S"), STACK);
    assert_eq!(emulator.runtime.memory.memory_write_count(), writes);
    assert!(emulator.call_session.is_none());
}

#[wasm_bindgen_test]
fn a_stale_stub_response_cannot_be_applied_to_the_next_hit_of_the_same_stub() {
    let mut emulator = machine(DeviceModel::PcE500);
    let id = emulator
        .call_function_begin(
            ENTRY,
            3,
            json_js(serde_json::json!({
                "stubs": [{"id": 7, "pc": ENTRY}]
            })),
        )
        .unwrap();
    assert_eq!(slice(&mut emulator, id, 10)["stub_request"]["sequence"], 1);
    emulator
        .call_function_apply_stub(id, 1, json_js(serde_json::json!({"ret": {"kind": "stay"}})))
        .unwrap();
    assert_eq!(slice(&mut emulator, id, 10)["stub_request"]["sequence"], 2);
    assert!(emulator
        .call_function_apply_stub(id, 1, JsValue::NULL)
        .is_err());
    assert!(emulator
        .call_function_stub_failed(id, 1, "stale failure")
        .is_err());
    emulator
        .call_function_stub_failed(id, 2, "current failure")
        .unwrap();
    let result: serde_json::Value =
        serde_json::from_str(&emulator.call_function_finish(id).unwrap()).unwrap();
    assert_eq!(result["report"]["reason"], "fault");
    assert_eq!(result["report"]["fault"]["message"], "current failure");
    assert_eq!(emulator.get_reg("S"), STACK);
}
