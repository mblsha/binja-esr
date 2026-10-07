// PY_SOURCE: pce500/tests/test_uart.py
//! Minimal UART device repro, independent of private firmware.
#[cfg(feature = "oz9600")]
#[test]
fn ordinary_guest_txd_write_reaches_host_without_an_iocs_shortcut() {
    use sc62015_core::oz9600::{configure_hardware, Hardware};
    let mut fixed = vec![0; 0x20000];
    // MV (UCR),48 ; MV A,55 ; MV (TXD),A ; JR -2
    fixed[..11].copy_from_slice(&[
        0x30, 0xcc, 0xf7, 0x48, 0x08, 0x55, 0x30, 0xa0, 0xfa, 0x13, 2,
    ]);
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 0x0e]);
    let mut rt = configure_hardware(&fixed, Hardware::default()).unwrap();
    rt.step_scheduler_boundaries(50_000).unwrap();
    let mut bytes = Vec::new();
    if let Some(sio) = rt.sio.as_mut() {
        while let Some(byte) = sio.complete_transmit(&mut rt.memory) {
            bytes.push(byte);
        }
    }
    assert_eq!(bytes, [0x55]);
}

#[cfg(feature = "oz9600")]
#[test]
fn runtime_receive_irq_latches_on_arrival_without_reasserting_on_every_instruction() {
    use sc62015_core::oz9600::{configure_hardware, Hardware};
    let mut fixed = vec![0; 0x20000];
    fixed[..6].copy_from_slice(&[0x30, 0xcc, 0xf7, 0x48, 0x13, 2]);
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 0x0e]);
    let mut rt = configure_hardware(&fixed, Hardware::default()).unwrap();
    rt.step_scheduler_boundaries(10).unwrap();
    rt.queue_sio_receive_byte(0x41);
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 0x20, 0);
    rt.step_scheduler_boundaries(50_000).unwrap();
    assert_eq!(
        rt.memory.read_internal_byte_silent(0xfc).unwrap() & 0x20,
        0x20
    );
    assert_eq!(
        rt.memory.read_internal_byte_silent(0xf8).unwrap() & 0x20,
        0x20
    );
    // Component fixture acknowledges ISR while deliberately leaving RXD unread.
    // RXR alone must not create a second completion event.
    rt.memory.write_internal_byte(0xfc, 0);
    rt.step_scheduler_boundaries(50_000).unwrap();
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 0x20, 0);
    rt.queue_sio_receive_byte(0x42);
    rt.step_scheduler_boundaries(50_000).unwrap();
    assert_eq!(
        rt.memory.read_internal_byte_silent(0xfc).unwrap() & 0x20,
        0x20
    );
    assert_eq!(rt.memory.read_internal_byte_silent(0xf8).unwrap() & 2, 2);
    rt.power_on_reset().unwrap();
    assert_eq!(rt.sio.as_ref().unwrap().uart().unwrap().control, 0);
    assert_eq!(rt.memory.read_internal_byte_silent(0xf8), Some(0x18));
    assert!(rt.sio.as_ref().unwrap().pending_receive().is_empty());
}

#[cfg(feature = "oz9600")]
#[test]
fn runtime_tx_irq_occurs_at_holding_transfer_and_not_whole_frame_completion() {
    use sc62015_core::oz9600::{configure_hardware, Hardware};
    let mut fixed = vec![0; 0x20000];
    fixed[..11].copy_from_slice(&[
        0x30, 0xcc, 0xf7, 0x48, 0x08, 0x55, 0x30, 0xa0, 0xfa, 0x13, 2,
    ]);
    fixed[0x1fffd..].copy_from_slice(&[0, 0, 0x0e]);
    let mut rt = configure_hardware(&fixed, Hardware::default()).unwrap();
    rt.step_scheduler_boundaries(1000).unwrap();
    assert_eq!(
        rt.memory.read_internal_byte_silent(0xfc).unwrap() & 0x10,
        0x10
    );
    assert_eq!(rt.memory.read_internal_byte_silent(0xf8), Some(8));
    assert_eq!(rt.sio.as_ref().unwrap().completed_transmit_len(), 0);
    rt.memory.write_internal_byte(0xfc, 0);
    rt.step_scheduler_boundaries(20_000).unwrap();
    assert_eq!(rt.sio.as_ref().unwrap().completed_transmit_len(), 1);
    assert_eq!(rt.memory.read_internal_byte_silent(0xfc).unwrap() & 0x10, 0);
}

use sc62015_core::{
    memory::MemoryImage,
    sio::{SioQueuedByte, SioStub},
    uart::{Uart, UartEvent},
};

#[test]
fn separate_holding_and_shift_registers_drive_txe_txr_and_ready_edges() {
    let mut u = Uart::new(1_024_000, 2);
    u.write_control(0x48);
    assert_eq!(u.baud(), 1200);
    assert_eq!(u.status(), 0x18);
    assert!(u.write_tx(0x55));
    assert_eq!(u.status(), 0x10);
    assert!(u.advance(2 * u.bit_units() - 1).is_empty());
    assert_eq!(u.advance(1), [UartEvent::TxReady(0x55)]);
    assert_eq!(u.status(), 8);
    assert!(u.write_tx(0xaa));
    assert_eq!(u.status(), 0);
    assert!(!u.write_tx(0xcc));
    assert_eq!(
        u.advance(u.frame_units()),
        [UartEvent::TxComplete(0x55), UartEvent::TxReady(0xaa)]
    );
    assert_eq!(u.status(), 8);
    assert_eq!(u.advance(u.frame_units()), [UartEvent::TxComplete(0xaa)]);
    assert_eq!(u.status(), 0x18);
    assert_eq!(
        (u.take_tx(), u.take_tx(), u.take_tx()),
        (Some(0x55), Some(0xaa), None)
    );
}

#[test]
fn rx_has_one_hardware_latch_errors_persist_until_next_arrival_and_seven_bit_masking() {
    let mut u = Uart::new(1_024_000, 2);
    u.write_control(0x4a);
    let mut byte = SioQueuedByte::new(0xff);
    byte.parity_error = true;
    byte.framing_error = true;
    assert!(u.queue_rx(byte));
    assert_eq!(u.advance(u.frame_units()), [UartEvent::RxReady(0x7f)]);
    assert_eq!(u.status(), 0x3d);
    assert_eq!(u.read_rx(), 0x7f);
    assert_eq!(u.status(), 0x1d);
    u.queue_rx(SioQueuedByte::new(0x31));
    u.queue_rx(SioQueuedByte::new(0x32));
    u.advance(2 * u.frame_units());
    assert_eq!(u.status(), 0x3a);
    assert_eq!(u.read_rx(), 0x32);
    assert_eq!(u.status(), 0x1a);
    u.queue_rx(SioQueuedByte::new(0x33));
    u.advance(u.frame_units());
    assert_eq!(u.status(), 0x38);
}

#[test]
fn elapsed_time_slicing_and_mid_frame_clone_preserve_every_event_and_byte() {
    let mut whole = Uart::new(1_024_000, 2);
    whole.write_control(0x74);
    whole.write_tx(0x5a);
    for b in 0..10 {
        whole.queue_rx(SioQueuedByte::new(b));
    }
    whole.advance(53);
    let mut sliced = whole.clone();
    let events = whole.advance(50000);
    let mut split_events = Vec::new();
    for _ in 0..500 {
        split_events.extend(sliced.advance(100));
    }
    assert_eq!(split_events, events);
    assert_eq!(whole, sliced);
}

#[test]
fn register_owner_ignores_read_only_writes_and_never_edits_software_state() {
    let mut mem = MemoryImage::new();
    mem.write_internal_byte(0xd5, 0xab);
    mem.write_internal_byte(0xf5, 0xcd);
    for address in [0xbfe46, 0x1fe6a, 0x1fede] {
        mem.store(address, 8, 0x77).unwrap();
    }
    let mut sio = SioStub::register_only(1_024_000, 2);
    sio.init(&mut mem);
    sio.handle_write(0xf7, 0x48, &mut mem);
    sio.handle_write(0xf8, 0xff, &mut mem);
    sio.handle_write(0xf9, 0xff, &mut mem);
    assert_eq!(sio.handle_read(0xf8, &mut mem), Some(0x18));
    assert_eq!(sio.handle_read(0xf9, &mut mem), Some(0));
    sio.queue_receive_byte(0x42, &mut mem);
    let units = sio.uart().unwrap().frame_units();
    sio.tick_cycles(units / 2, &mut mem);
    let snap = sio.snapshot(&mem);
    assert!(snap.workspace.is_empty());
    sio.tick_cycles(units - units / 2, &mut mem);
    assert_eq!(sio.handle_read(0xf9, &mut mem), Some(0x42));
    sio.restore(snap, &mut mem);
    sio.tick_cycles(units - units / 2, &mut mem);
    assert_eq!(sio.handle_read(0xf9, &mut mem), Some(0x42));
    sio.set_auto_response(1);
    sio.enable_rom_shortcuts_for_diagnostics();
    sio.set_handshake(&mut mem, 0);
    sio.set_input_lines(&mut mem, Some(false), Some(false));
    assert_eq!(mem.read_internal_byte_silent(0xd5), Some(0xab));
    assert_eq!(mem.read_internal_byte_silent(0xf5), Some(0xcd));
    for address in [0xbfe46, 0x1fe6a, 0x1fede] {
        assert_eq!(mem.load(address, 8), Some(0x77));
    }
    assert!(!sio.snapshot(&mem).rom_shortcuts_enabled);
}

#[test]
fn reset_aborts_work_break_suppresses_frames_and_backlog_is_bounded() {
    let mut u = Uart::new(1_024_000, 2);
    assert!(!u.write_tx(1));
    assert!(!u.queue_rx(SioQueuedByte::new(1)));
    u.write_control(0xc8);
    u.write_tx(0x44);
    u.advance(2 * u.bit_units() + u.frame_units());
    assert_eq!(u.suppressed_break_frames, 1);
    assert_eq!(u.take_tx(), None);
    u.write_control(0x48);
    u.write_tx(1);
    u.queue_rx(SioQueuedByte::new(2));
    u.write_control(0);
    assert!(u.advance(1_000_000).is_empty());
    assert_eq!(u.status(), 0x18);
    u.write_control(0x48);
    for i in 0..4097 {
        u.write_tx(i as u8);
        u.advance(2 * u.bit_units() + u.frame_units());
    }
    assert_eq!(u.completed_tx.len(), 4096);
    assert_eq!(u.dropped_tx, 1);
    for _ in 0..4096 {
        assert!(u.queue_rx(SioQueuedByte::new(3)));
    }
    assert!(!u.queue_rx(SioQueuedByte::new(4)));
}

#[cfg(feature = "json-compat")]
#[test]
fn python_reference_corpus_matches_every_operation_and_deadline() {
    use serde_json::{json, Value};
    let corpus: Value = serde_json::from_str(include_str!("../data/uart_reference.json")).unwrap();
    let mut u = Uart::new(1_024_000, 2);
    let mut clone = None;
    for (index, op) in corpus["operations"].as_array().unwrap().iter().enumerate() {
        let mut result = Value::Null;
        match op[0].as_str().unwrap() {
            "control" => u.write_control(op[1].as_u64().unwrap() as u8),
            "tx" => result = json!(u.write_tx(op[1].as_u64().unwrap() as u8)),
            "rx" => {
                result = json!(u.queue_rx(SioQueuedByte {
                    value: op[1].as_u64().unwrap() as u8,
                    parity_error: op[2].as_bool().unwrap(),
                    overrun_error: op[3].as_bool().unwrap(),
                    framing_error: op[4].as_bool().unwrap()
                }))
            }
            "advance" => {
                result = json!(u
                    .advance(op[1].as_u64().unwrap())
                    .into_iter()
                    .map(|event| match event {
                        UartEvent::RxReady(b) => json!(["RxReady", b]),
                        UartEvent::TxReady(b) => json!(["TxReady", b]),
                        UartEvent::TxComplete(b) => json!(["TxComplete", b]),
                    })
                    .collect::<Vec<_>>())
            }
            "read" => result = json!(u.read_rx()),
            "take" => result = json!(u.take_tx()),
            "clone" => clone = Some(u.clone()),
            "restore" => u = clone.clone().unwrap(),
            _ => panic!("unknown corpus operation"),
        }
        assert_eq!(
            json!({"result":result,"state":u.report()}),
            corpus["replies"][index],
            "operation {index}: {op}"
        );
    }
}
