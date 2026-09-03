use sc62015_core::memory::{IMEM_EIL_OFFSET, IMEM_RXD_OFFSET, IMEM_TXD_OFFSET, IMEM_USR_OFFSET};
use sc62015_core::{MemoryImage, SioInputLines, SioStub, SioTimedEvent, SioTimingConfig};

const USR_RX_READY: u8 = 0x20;
const USR_TX_EMPTY: u8 = 0x10;
const USR_TX_READY: u8 = 0x08;
const USR_ERROR_MASK: u8 = 0x07;
const USR_FRAMING_ERROR: u8 = 0x04;
const USR_PARITY_ERROR: u8 = 0x01;

#[test]
fn receive_queue_sets_status_and_error_bits() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();

    stub.queue_receive(0x41, true, false, true, &mut memory);

    assert_eq!(memory.read_internal_byte(IMEM_RXD_OFFSET), Some(0x41));
    let usr = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
    assert!(usr & USR_RX_READY != 0);
    assert!(usr & USR_FRAMING_ERROR != 0);
    assert!(usr & USR_PARITY_ERROR != 0);

    let consumed = stub.consume_received(&mut memory).unwrap();
    assert_eq!(consumed.value, 0x41);
    assert!(consumed.framing_error);
    assert!(consumed.parity_error);

    let usr_after = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
    assert_eq!(usr_after & USR_RX_READY, 0);
    assert_eq!(usr_after & USR_ERROR_MASK, 0);
}

#[test]
fn timing_profile_delays_rx_ready_tx_complete_handshake_and_timeout() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();
    stub.set_timing_config(SioTimingConfig {
        rx_ready_delay_cycles: 3,
        tx_complete_cycles: 2,
        handshake_delay_cycles: 4,
        direct_input_timeout_cycles: 5,
        xoff_threshold: 8,
        xon_threshold: 2,
    });

    stub.queue_receive_byte(0x42, &mut memory);
    assert_eq!(
        memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0) & USR_RX_READY,
        0
    );
    assert_eq!(stub.tick_cycles(2, &mut memory), vec![]);
    assert_eq!(
        stub.tick_cycles(1, &mut memory),
        vec![SioTimedEvent::RxReady(0x42)]
    );
    assert_ne!(
        memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0) & USR_RX_READY,
        0
    );
    assert_eq!(stub.consume_received(&mut memory).unwrap().value, 0x42);

    stub.queue_transmit(0x55, &mut memory);
    assert_eq!(stub.tick_cycles(1, &mut memory), vec![]);
    assert_eq!(
        stub.tick_cycles(1, &mut memory),
        vec![SioTimedEvent::TxComplete(0x55)]
    );

    stub.set_input_lines_delayed(true, false);
    assert_eq!(stub.tick_cycles(3, &mut memory), vec![]);
    assert_eq!(
        stub.tick_cycles(1, &mut memory),
        vec![SioTimedEvent::HandshakeSettled(SioInputLines {
            cs: true,
            cd: false,
        })]
    );
    assert_eq!(
        stub.input_lines(&memory),
        SioInputLines {
            cs: true,
            cd: false,
        }
    );

    stub.set_direct_input_timeout(true);
    assert_eq!(stub.tick_cycles(4, &mut memory), vec![]);
    assert_eq!(
        stub.tick_cycles(1, &mut memory),
        vec![SioTimedEvent::DirectInputTimeout]
    );
}

#[test]
fn xon_xoff_events_follow_receive_queue_watermarks() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();
    stub.set_timing_config(SioTimingConfig {
        rx_ready_delay_cycles: 0,
        tx_complete_cycles: 1,
        handshake_delay_cycles: 1,
        direct_input_timeout_cycles: 1,
        xoff_threshold: 3,
        xon_threshold: 1,
    });

    stub.queue_receive_byte(0x10, &mut memory);
    stub.queue_receive_byte(0x11, &mut memory);
    assert_eq!(stub.tick_cycles(1, &mut memory), vec![]);
    stub.queue_receive_byte(0x12, &mut memory);
    assert_eq!(stub.tick_cycles(1, &mut memory), vec![SioTimedEvent::Xoff]);

    assert_eq!(stub.consume_received(&mut memory).unwrap().value, 0x10);
    assert_eq!(stub.tick_cycles(1, &mut memory), vec![]);
    assert_eq!(stub.consume_received(&mut memory).unwrap().value, 0x11);
    assert_eq!(stub.tick_cycles(1, &mut memory), vec![SioTimedEvent::Xon]);
}

#[test]
fn transmit_queue_tracks_writes_until_completed() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();
    stub.init(&mut memory);

    assert!(stub.handle_write(IMEM_TXD_OFFSET, 0x55, &mut memory));
    assert_eq!(stub.pending_transmit(), vec![0x55]);

    let usr = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
    assert_eq!(usr & USR_TX_READY, 0);
    assert_eq!(usr & USR_TX_EMPTY, 0);

    assert_eq!(stub.complete_transmit(&mut memory), None);
    assert_eq!(
        stub.tick_cycles(1, &mut memory),
        vec![SioTimedEvent::TxComplete(0x55)]
    );
    assert_eq!(stub.complete_transmit(&mut memory), Some(0x55));
    let usr_after = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
    assert!(usr_after & USR_TX_READY != 0);
    assert!(usr_after & USR_TX_EMPTY != 0);
}

#[test]
fn large_cycle_advances_jump_between_serial_events() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();
    stub.set_timing_config(SioTimingConfig {
        tx_complete_cycles: 100_000,
        ..SioTimingConfig::default()
    });
    for byte in [0x11, 0x22, 0x33] {
        stub.queue_transmit(byte, &mut memory);
    }

    assert_eq!(
        stub.tick_cycles(300_000, &mut memory),
        vec![
            SioTimedEvent::TxComplete(0x11),
            SioTimedEvent::TxComplete(0x22),
            SioTimedEvent::TxComplete(0x33),
        ]
    );
    assert!(stub.pending_transmit().is_empty());
    assert_eq!(stub.completed_transmit_len(), 3);
}

#[test]
fn idle_serial_time_does_not_manufacture_bus_transactions() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();
    stub.init(&mut memory);
    let reads = memory.memory_read_count();
    let writes = memory.memory_write_count();

    assert!(stub.tick_cycles(1_000_000, &mut memory).is_empty());
    assert_eq!(memory.memory_read_count(), reads);
    assert_eq!(memory.memory_write_count(), writes);
}

#[test]
fn input_lines_and_snapshot_restore_roundtrip() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();

    stub.queue_receive_byte(0x41, &mut memory);
    stub.queue_transmit(0x55, &mut memory);
    stub.set_input_lines(&mut memory, Some(true), Some(true));
    stub.set_handshake(&mut memory, 0xA5);
    memory.store(0x00BFE40, 8, 0x12).unwrap();
    memory.store(0x00BFE47, 8, 0x99).unwrap();
    let snap = stub.snapshot(&memory);
    let expected = snap.clone();

    assert_eq!(stub.consume_received(&mut memory).unwrap().value, 0x41);
    assert_eq!(
        stub.tick_cycles(1, &mut memory),
        vec![SioTimedEvent::TxComplete(0x55)]
    );
    assert_eq!(stub.complete_transmit(&mut memory), Some(0x55));
    stub.set_input_lines(&mut memory, Some(false), Some(false));
    stub.set_handshake(&mut memory, 0x00);
    memory.store(0x00BFE40, 8, 0x00).unwrap();
    memory.store(0x00BFE47, 8, 0x00).unwrap();

    stub.restore(snap, &mut memory);
    assert_eq!(stub.snapshot(&memory), expected);

    assert_eq!(
        stub.pending_receive()
            .iter()
            .map(|entry| entry.value)
            .collect::<Vec<_>>(),
        vec![0x41]
    );
    assert_eq!(stub.pending_transmit(), vec![0x55]);
    assert_eq!(
        stub.input_lines(&memory),
        SioInputLines { cs: true, cd: true }
    );
    assert_eq!(
        memory.read_internal_byte(IMEM_EIL_OFFSET).unwrap_or(0) & 0x06,
        0x06
    );
    assert_eq!(stub.get_handshake(&memory), 0xA5);
    assert_eq!(memory.load(0x00BFE40, 8), Some(0x12));
    assert_eq!(memory.load(0x00BFE47, 8), Some(0x99));
}
