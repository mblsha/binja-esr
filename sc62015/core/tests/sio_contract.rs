use sc62015_core::memory::{IMEM_EIL_OFFSET, IMEM_RXD_OFFSET, IMEM_TXD_OFFSET, IMEM_USR_OFFSET};
use sc62015_core::{MemoryImage, SioInputLines, SioStub};

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
fn transmit_queue_tracks_writes_until_completed() {
    let mut memory = MemoryImage::new();
    let mut stub = SioStub::new();
    stub.init(&mut memory);

    assert!(stub.handle_write(IMEM_TXD_OFFSET, 0x55, &mut memory));
    assert_eq!(stub.pending_transmit(), vec![0x55]);

    let usr = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
    assert_eq!(usr & USR_TX_READY, 0);
    assert_eq!(usr & USR_TX_EMPTY, 0);

    assert_eq!(stub.complete_transmit(&mut memory), Some(0x55));
    let usr_after = memory.read_internal_byte(IMEM_USR_OFFSET).unwrap_or(0);
    assert!(usr_after & USR_TX_READY != 0);
    assert!(usr_after & USR_TX_EMPTY != 0);
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

    assert_eq!(stub.consume_received(&mut memory).unwrap().value, 0x41);
    assert_eq!(stub.complete_transmit(&mut memory), Some(0x55));
    stub.set_input_lines(&mut memory, Some(false), Some(false));
    stub.set_handshake(&mut memory, 0x00);
    memory.store(0x00BFE40, 8, 0x00).unwrap();
    memory.store(0x00BFE47, 8, 0x00).unwrap();

    stub.restore(snap, &mut memory);

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
