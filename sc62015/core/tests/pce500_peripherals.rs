use sc62015_core::{
    CassetteBlock, CassetteBlockKind, CassetteError, CassetteTapeImage, MemoryCardImage,
    RamDiskImage, StorageError,
};

#[test]
fn cassette_tape_image_reads_and_verifies_deterministically() {
    let mut tape = CassetteTapeImage::new();
    let header = (0..0x30).collect::<Vec<u8>>();
    let payload = b"HELLO";

    let header_checksum = tape.append_header(&header).unwrap().checksum;
    let data_checksum = tape.append_data(payload).unwrap().checksum;

    assert_eq!(
        header_checksum,
        header.iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte))
    );
    assert_eq!(
        data_checksum,
        payload
            .iter()
            .fold(0u8, |sum, byte| sum.wrapping_add(*byte))
    );
    assert_eq!(
        tape.read_next(Some(CassetteBlockKind::Header))
            .unwrap()
            .payload,
        header
    );
    assert!(tape.verify_next(payload).unwrap());

    tape.rewind();
    assert_eq!(
        tape.read_next(Some(CassetteBlockKind::Header))
            .unwrap()
            .kind,
        CassetteBlockKind::Header
    );
    assert_eq!(
        tape.read_next(Some(CassetteBlockKind::Data))
            .unwrap()
            .payload,
        payload
    );
}

#[test]
fn cassette_tape_image_rejects_wrong_block_shape_and_checksum() {
    let mut tape = CassetteTapeImage::new();

    assert_eq!(
        tape.append_header(b"short").unwrap_err(),
        CassetteError::InvalidHeaderLength
    );

    tape.blocks.push(CassetteBlock {
        kind: CassetteBlockKind::Data,
        payload: b"BAD".to_vec(),
        checksum: 0,
    });
    assert_eq!(
        tape.read_next(None).unwrap_err(),
        CassetteError::ChecksumMismatch
    );
}

#[test]
fn memory_card_edge_cases_and_ramfile_backing() {
    let mut card = MemoryCardImage::new(16);
    let block = card.create("foo", 4, false).unwrap();
    block.data.copy_from_slice(b"ABCD");

    assert_eq!(card.used(), 4);
    assert_eq!(card.free(), 12);
    assert_eq!(card.find("FOO").unwrap().data, b"ABCD");

    assert_eq!(
        card.create("foo", 1, false).unwrap_err(),
        StorageError::DuplicateBlock("FOO".to_string())
    );
    assert_eq!(
        card.create("too-big", 32, false).unwrap_err(),
        StorageError::InsufficientSpace("TOO-BIG".to_string())
    );

    card.resize("foo", 2).unwrap();
    assert_eq!(card.find("foo").unwrap().data, b"AB");
    card.rename("foo", "bar").unwrap();
    assert_eq!(card.find("bar").unwrap().size(), 2);

    let mut ramdisk = RamDiskImage::new(card);
    ramdisk.format(8).unwrap();
    assert_eq!(ramdisk.size().unwrap(), 8);
    assert_eq!(ramdisk.card.blocks[0].name, "RAMFILE");

    ramdisk.card.delete("bar").unwrap();
    assert_eq!(
        ramdisk.card.find("bar").unwrap_err(),
        StorageError::BlockNotFound("BAR".to_string())
    );
}

#[test]
fn memory_card_names_are_normalised_and_condense_removes_empty_blocks() {
    let mut card = MemoryCardImage::new(16);

    assert_eq!(
        card.create("   ", 1, false).unwrap_err(),
        StorageError::EmptyName
    );
    assert_eq!(
        card.create("123456789012", 1, false).unwrap_err(),
        StorageError::NameTooLong
    );

    card.create("a", 0, false).unwrap();
    card.create("b", 1, false).unwrap();
    card.condense();

    assert_eq!(card.blocks.len(), 1);
    assert_eq!(card.blocks[0].name, "B");
}
