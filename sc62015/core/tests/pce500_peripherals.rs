use sc62015_core::{
    CassetteBlock, CassetteBlockKind, CassetteError, CassettePulse, CassettePulseError,
    CassettePulseStream, CassettePulseTiming, CassetteRetryPolicy, CassetteTapeImage,
    MemoryCardImage, RamDiskImage, StorageError,
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
fn cassette_pulse_stream_roundtrips_bytes_and_rejects_bad_shapes() {
    let timing = CassettePulseTiming {
        p01: 4,
        p00: 6,
        p11: 16,
        p10: 18,
        threshold: 10,
    };
    let stream = CassettePulseStream::encode_bytes(timing, &[0b1010_0101, 0x00]);

    assert_eq!(stream.pulses.len(), 32);
    assert_eq!(stream.decode_bytes().unwrap(), vec![0b1010_0101, 0x00]);

    let mut odd = stream.clone();
    odd.pulses.push(CassettePulse {
        high: true,
        cycles: timing.p11,
    });
    assert_eq!(
        odd.decode_bytes().unwrap_err(),
        CassettePulseError::OddPhaseCount
    );

    let short = CassettePulseStream {
        timing,
        pulses: stream.pulses[..2].to_vec(),
    };
    assert_eq!(
        short.decode_bytes().unwrap_err(),
        CassettePulseError::NotByteAligned
    );
}

#[test]
fn cassette_retry_policy_produces_deterministic_attempt_deadlines() {
    let policy = CassetteRetryPolicy {
        max_retries: 2,
        motor_settle_cycles: 100,
        retry_spacing_cycles: 25,
    };

    assert_eq!(policy.attempt_deadlines(), vec![100, 125, 150]);
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

#[test]
fn memory_card_media_bytes_roundtrip_rom_block_chain_layout() {
    let mut card = MemoryCardImage::new(64);
    card.create("foo", 3, false)
        .unwrap()
        .data
        .copy_from_slice(b"ABC");
    card.create("bar", 2, false)
        .unwrap()
        .data
        .copy_from_slice(b"DE");

    let media = card.to_media_bytes();

    assert_eq!(media[0x12], 0x18);
    assert_eq!(media[0x18], 0xFB);
    assert_eq!(&media[0x19..0x24], b"FOO        ");
    assert_eq!(&media[0x30..0x33], b"ABC");
    let second = 0x18 + 0x18 + 3;
    assert_eq!(media[second], 0xFB);
    assert_eq!(&media[second + 1..second + 12], b"BAR        ");

    let parsed = MemoryCardImage::from_media_bytes(&media).unwrap();
    assert_eq!(parsed.find("foo").unwrap().data, b"ABC");
    assert_eq!(parsed.find("bar").unwrap().data, b"DE");
}

#[test]
fn memory_card_media_parser_rejects_truncated_or_invalid_images() {
    assert_eq!(
        MemoryCardImage::from_media_bytes(&[0; 4]).unwrap_err(),
        StorageError::InvalidMediaHeader
    );

    let mut invalid = vec![0; 0x20];
    invalid[0x12] = 0x18;
    invalid[0x18] = 0xAA;
    assert_eq!(
        MemoryCardImage::from_media_bytes(&invalid).unwrap_err(),
        StorageError::InvalidMediaBlockMarker
    );

    let mut truncated = vec![0; 0x30];
    truncated[0x12] = 0x18;
    truncated[0x18] = 0xFB;
    truncated[0x29] = 0x40;
    assert_eq!(
        MemoryCardImage::from_media_bytes(&truncated).unwrap_err(),
        StorageError::TruncatedMediaBlock
    );
}
