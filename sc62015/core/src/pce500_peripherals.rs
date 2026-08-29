use thiserror::Error;

use crate::llama::opcodes::RegName;
use crate::llama::state::{mask_for, LlamaState};
use crate::memory::MemoryImage;

const CAS_WRITE_DATA_BLOCK_PC: u32 = 0x00E9E0A;
const CAS_READ_DATA_BLOCK_PC: u32 = 0x00E9E52;
const CAS_VERIFY_DATA_BLOCK_PC: u32 = 0x00E9EB0;
const CAS_WRITE_HEADER_BLOCK_PC: u32 = 0x00E9DA8;
const CAS_READ_HEADER_BLOCK_PC: u32 = 0x00E9DD8;

const CARD_SEARCH_BLOCK_PC: u32 = 0x00F0063;
const CARD_RESIZE_BLOCK_PC: u32 = 0x00F00C3;
const CARD_CREATE_TOP_BLOCK_PC: u32 = 0x00F034B;
const CARD_CREATE_BLOCK_PC: u32 = 0x00F039E;
const CARD_DELETE_BLOCK_PC: u32 = 0x00F043E;
const CARD_CONDENSE_PC: u32 = 0x00F0A19;
const CARD_FILE_CREATE_PC: u32 = 0x00F0470;
const CARD_FILE_OPEN_PC: u32 = 0x00F0517;
const CARD_FILE_CLOSE_PC: u32 = 0x00F0570;
const CARD_FILE_READ_BLOCK_PC: u32 = 0x00F0633;
const CARD_FILE_WRITE_BLOCK_PC: u32 = 0x00F0752;
const CARD_FILE_READ_BYTE_PC: u32 = 0x00F05F9;
const CARD_FILE_WRITE_BYTE_PC: u32 = 0x00F073A;
const CARD_FILE_VERIFY_PC: u32 = 0x00F067C;
const CARD_FILE_PEEK_PC: u32 = 0x00F061C;
const CARD_FILE_SEEK_PC: u32 = 0x00F07AD;
const CARD_FILE_INFO_PC: u32 = 0x00F0821;
const CARD_FILE_CHANGE_DIR_PC: u32 = 0x00F0859;
const CARD_FILE_SEARCH_PC: u32 = 0x00F08DC;
const CARD_FILE_RENAME_DELETE_PC: u32 = 0x00F09D8;
const CARD_FILE_FREE_PC: u32 = 0x00F0948;
const RAMDISK_READ_BLOCK_PC: u32 = 0x00E4A70;
const RAMDISK_WRITE_BLOCK_PC: u32 = 0x00E4A94;
const RAMDISK_FORMAT_PC: u32 = 0x00E4C6E;
const RAMDISK_BACKING_NAME: &str = "RAMFILE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CassetteBlockKind {
    Header,
    Data,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CassetteError {
    #[error("cassette header blocks must be exactly 0x30 bytes")]
    InvalidHeaderLength,
    #[error("cassette image is at end of tape")]
    EndOfTape,
    #[error("expected {expected:?} block, got {actual:?}")]
    UnexpectedBlockKind {
        expected: CassetteBlockKind,
        actual: CassetteBlockKind,
    },
    #[error("cassette block checksum mismatch")]
    ChecksumMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassetteBlock {
    pub kind: CassetteBlockKind,
    pub payload: Vec<u8>,
    pub checksum: u8,
}

impl CassetteBlock {
    pub fn from_payload(
        kind: CassetteBlockKind,
        payload: impl AsRef<[u8]>,
    ) -> Result<Self, CassetteError> {
        let payload = payload.as_ref();
        if kind == CassetteBlockKind::Header && payload.len() != 0x30 {
            return Err(CassetteError::InvalidHeaderLength);
        }
        Ok(Self {
            kind,
            payload: payload.to_vec(),
            checksum: checksum(payload),
        })
    }

    pub fn verify(&self) -> bool {
        self.checksum == checksum(&self.payload)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CassetteTapeImage {
    pub blocks: Vec<CassetteBlock>,
    pub cursor: usize,
}

impl CassetteTapeImage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn append_header(
        &mut self,
        payload: impl AsRef<[u8]>,
    ) -> Result<&CassetteBlock, CassetteError> {
        let block = CassetteBlock::from_payload(CassetteBlockKind::Header, payload)?;
        self.blocks.push(block);
        Ok(self.blocks.last().expect("cassette block was just pushed"))
    }

    pub fn append_data(
        &mut self,
        payload: impl AsRef<[u8]>,
    ) -> Result<&CassetteBlock, CassetteError> {
        let block = CassetteBlock::from_payload(CassetteBlockKind::Data, payload)?;
        self.blocks.push(block);
        Ok(self.blocks.last().expect("cassette block was just pushed"))
    }

    pub fn rewind(&mut self) {
        self.cursor = 0;
    }

    pub fn read_next(
        &mut self,
        expected_kind: Option<CassetteBlockKind>,
    ) -> Result<&CassetteBlock, CassetteError> {
        let block = self
            .blocks
            .get(self.cursor)
            .ok_or(CassetteError::EndOfTape)?;
        if let Some(expected) = expected_kind {
            if block.kind != expected {
                return Err(CassetteError::UnexpectedBlockKind {
                    expected,
                    actual: block.kind,
                });
            }
        }
        if !block.verify() {
            return Err(CassetteError::ChecksumMismatch);
        }
        self.cursor += 1;
        Ok(block)
    }

    pub fn verify_next(&mut self, payload: impl AsRef<[u8]>) -> Result<bool, CassetteError> {
        let block = self.read_next(None)?;
        Ok(block.payload == payload.as_ref())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassettePulseTiming {
    pub p01: u16,
    pub p00: u16,
    pub p11: u16,
    pub p10: u16,
    pub threshold: u16,
}

impl Default for CassettePulseTiming {
    fn default() -> Self {
        Self {
            p01: 12,
            p00: 12,
            p11: 24,
            p10: 24,
            threshold: 18,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassettePulse {
    pub high: bool,
    pub cycles: u16,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CassettePulseError {
    #[error("cassette pulse stream has odd phase count")]
    OddPhaseCount,
    #[error("cassette pulse stream is not byte aligned")]
    NotByteAligned,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CassettePulseStream {
    pub timing: CassettePulseTiming,
    pub pulses: Vec<CassettePulse>,
}

impl CassettePulseStream {
    pub fn encode_bytes(timing: CassettePulseTiming, bytes: &[u8]) -> Self {
        let mut pulses = Vec::with_capacity(bytes.len() * 16);
        for byte in bytes {
            for bit in 0..8 {
                let one = (byte >> (7 - bit)) & 1 != 0;
                if one {
                    pulses.push(CassettePulse {
                        high: true,
                        cycles: timing.p11,
                    });
                    pulses.push(CassettePulse {
                        high: false,
                        cycles: timing.p10,
                    });
                } else {
                    pulses.push(CassettePulse {
                        high: true,
                        cycles: timing.p01,
                    });
                    pulses.push(CassettePulse {
                        high: false,
                        cycles: timing.p00,
                    });
                }
            }
        }
        Self { timing, pulses }
    }

    pub fn decode_bytes(&self) -> Result<Vec<u8>, CassettePulseError> {
        if !self.pulses.len().is_multiple_of(2) {
            return Err(CassettePulseError::OddPhaseCount);
        }
        let bit_count = self.pulses.len() / 2;
        if !bit_count.is_multiple_of(8) {
            return Err(CassettePulseError::NotByteAligned);
        }

        let mut bytes = Vec::with_capacity(bit_count / 8);
        let mut current = 0u8;
        for (idx, pair) in self.pulses.as_chunks::<2>().0.iter().enumerate() {
            let average = (u32::from(pair[0].cycles) + u32::from(pair[1].cycles)) / 2;
            let one = average >= u32::from(self.timing.threshold);
            current = (current << 1) | u8::from(one);
            if idx % 8 == 7 {
                bytes.push(current);
                current = 0;
            }
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CassetteRetryPolicy {
    pub max_retries: u8,
    pub motor_settle_cycles: u32,
    pub retry_spacing_cycles: u32,
}

impl Default for CassetteRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            motor_settle_cycles: 2_000,
            retry_spacing_cycles: 1_000,
        }
    }
}

impl CassetteRetryPolicy {
    pub fn attempt_deadlines(&self) -> Vec<u32> {
        let mut deadlines = Vec::with_capacity(usize::from(self.max_retries) + 1);
        let mut cycle = self.motor_settle_cycles;
        for _ in 0..=self.max_retries {
            deadlines.push(cycle);
            cycle = cycle.saturating_add(self.retry_spacing_cycles);
        }
        deadlines
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StorageError {
    #[error("block name must not be empty")]
    EmptyName,
    #[error("block names are limited to 11 characters")]
    NameTooLong,
    #[error("block size must be non-negative")]
    InvalidSize,
    #[error("duplicate block: {0}")]
    DuplicateBlock(String),
    #[error("block not found: {0}")]
    BlockNotFound(String),
    #[error("insufficient card space for: {0}")]
    InsufficientSpace(String),
    #[error("media image is too small for the memory card header")]
    InvalidMediaHeader,
    #[error("media block chain is truncated")]
    TruncatedMediaBlock,
    #[error("media block header marker is invalid")]
    InvalidMediaBlockMarker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCardBlock {
    pub name: String,
    pub data: Vec<u8>,
    pub protected: bool,
}

impl MemoryCardBlock {
    pub fn new(name: String, size: usize) -> Self {
        Self {
            name,
            data: vec![0; size],
            protected: false,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCardImage {
    pub capacity: usize,
    pub blocks: Vec<MemoryCardBlock>,
}

impl MemoryCardImage {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            blocks: Vec::new(),
        }
    }

    pub fn used(&self) -> usize {
        self.blocks.iter().map(MemoryCardBlock::size).sum()
    }

    pub fn free(&self) -> usize {
        self.capacity.saturating_sub(self.used())
    }

    pub fn create(
        &mut self,
        name: &str,
        size: usize,
        at_top: bool,
    ) -> Result<&mut MemoryCardBlock, StorageError> {
        let name = normalise_name(name)?;
        if self.find_index(&name).is_some() {
            return Err(StorageError::DuplicateBlock(name));
        }
        if size > self.free() {
            return Err(StorageError::InsufficientSpace(name));
        }
        let block = MemoryCardBlock::new(name, size);
        if at_top {
            self.blocks.insert(0, block);
            Ok(&mut self.blocks[0])
        } else {
            self.blocks.push(block);
            Ok(self
                .blocks
                .last_mut()
                .expect("memory-card block was just pushed"))
        }
    }

    pub fn find(&self, name: &str) -> Result<&MemoryCardBlock, StorageError> {
        let name = normalise_name(name)?;
        self.find_by_normalised_name(&name)
            .ok_or(StorageError::BlockNotFound(name))
    }

    pub fn find_optional(&self, name: &str) -> Result<Option<&MemoryCardBlock>, StorageError> {
        let name = normalise_name(name)?;
        Ok(self.find_by_normalised_name(&name))
    }

    pub fn resize(
        &mut self,
        name: &str,
        size: usize,
    ) -> Result<&mut MemoryCardBlock, StorageError> {
        let name = normalise_name(name)?;
        let index = self
            .find_index(&name)
            .ok_or_else(|| StorageError::BlockNotFound(name.clone()))?;
        let current_size = self.blocks[index].size();
        if size > current_size && size - current_size > self.free() {
            return Err(StorageError::InsufficientSpace(name));
        }
        self.blocks[index].data.resize(size, 0);
        Ok(&mut self.blocks[index])
    }

    pub fn rename(
        &mut self,
        old_name: &str,
        new_name: &str,
    ) -> Result<&mut MemoryCardBlock, StorageError> {
        let old_name = normalise_name(old_name)?;
        let new_name = normalise_name(new_name)?;
        let index = self
            .find_index(&old_name)
            .ok_or_else(|| StorageError::BlockNotFound(old_name.clone()))?;
        if let Some(existing) = self.find_index(&new_name) {
            if existing != index {
                return Err(StorageError::DuplicateBlock(new_name));
            }
        }
        self.blocks[index].name = new_name;
        Ok(&mut self.blocks[index])
    }

    pub fn delete(&mut self, name: &str) -> Result<MemoryCardBlock, StorageError> {
        let name = normalise_name(name)?;
        let index = self
            .find_index(&name)
            .ok_or_else(|| StorageError::BlockNotFound(name.clone()))?;
        Ok(self.blocks.remove(index))
    }

    pub fn condense(&mut self) {
        self.blocks.retain(|block| block.size() > 0);
    }

    pub fn to_media_bytes(&self) -> Vec<u8> {
        const SLOT_HEADER_LEN: usize = 0x18;
        const BLOCK_HEADER_LEN: usize = 0x18;
        let mut image = vec![0u8; self.capacity.max(SLOT_HEADER_LEN + 1)];
        image[0x12] = SLOT_HEADER_LEN as u8;
        let mut cursor = SLOT_HEADER_LEN;
        for block in &self.blocks {
            let advance = BLOCK_HEADER_LEN + block.data.len();
            if cursor + advance + 1 > image.len() {
                image.resize(cursor + advance + 1, 0);
            }
            image[cursor] = 0xFB;
            let mut name = [b' '; 11];
            for (idx, byte) in block.name.as_bytes().iter().take(11).enumerate() {
                name[idx] = *byte;
            }
            image[cursor + 1..cursor + 12].copy_from_slice(&name);
            write_u24_slice(&mut image[cursor + 0x11..cursor + 0x14], advance as u32);
            image[cursor + BLOCK_HEADER_LEN..cursor + BLOCK_HEADER_LEN + block.data.len()]
                .copy_from_slice(&block.data);
            cursor += advance;
        }
        if cursor >= image.len() {
            image.resize(cursor + 1, 0);
        }
        image[cursor] = 0xFF;
        image
    }

    pub fn from_media_bytes(bytes: &[u8]) -> Result<Self, StorageError> {
        const SLOT_HEADER_LEN: usize = 0x18;
        const BLOCK_HEADER_LEN: usize = 0x18;
        if bytes.len() < SLOT_HEADER_LEN {
            return Err(StorageError::InvalidMediaHeader);
        }
        if read_u24_slice(&bytes[0x12..0x15]) != SLOT_HEADER_LEN as u32 {
            return Err(StorageError::InvalidMediaHeader);
        }

        let mut cursor = SLOT_HEADER_LEN;
        let mut blocks = Vec::new();
        while cursor < bytes.len() {
            match bytes[cursor] {
                0xFF => {
                    return Ok(Self {
                        capacity: bytes.len(),
                        blocks,
                    });
                }
                0xFB => {}
                _ => return Err(StorageError::InvalidMediaBlockMarker),
            }
            if cursor + BLOCK_HEADER_LEN > bytes.len() {
                return Err(StorageError::TruncatedMediaBlock);
            }
            let name = std::str::from_utf8(&bytes[cursor + 1..cursor + 12])
                .map_err(|_| StorageError::InvalidMediaBlockMarker)?
                .trim()
                .to_string();
            let advance = read_u24_slice(&bytes[cursor + 0x11..cursor + 0x14]) as usize;
            if advance < BLOCK_HEADER_LEN || cursor + advance > bytes.len() {
                return Err(StorageError::TruncatedMediaBlock);
            }
            let data = bytes[cursor + BLOCK_HEADER_LEN..cursor + advance].to_vec();
            blocks.push(MemoryCardBlock {
                name: normalise_name(&name)?,
                data,
                protected: false,
            });
            cursor += advance;
        }
        Err(StorageError::TruncatedMediaBlock)
    }

    fn find_by_normalised_name(&self, name: &str) -> Option<&MemoryCardBlock> {
        self.blocks.iter().find(|block| block.name == name)
    }

    fn find_index(&self, name: &str) -> Option<usize> {
        self.blocks.iter().position(|block| block.name == name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RamDiskImage {
    pub card: MemoryCardImage,
    pub backing_name: String,
}

impl RamDiskImage {
    pub fn new(card: MemoryCardImage) -> Self {
        Self {
            card,
            backing_name: "RAMFILE".to_string(),
        }
    }

    pub fn format(&mut self, size: usize) -> Result<&mut MemoryCardBlock, StorageError> {
        if self.card.find_optional(&self.backing_name)?.is_none() {
            let name = self.backing_name.clone();
            return self.card.create(&name, size, true);
        }
        let name = self.backing_name.clone();
        self.card.resize(&name, size)
    }

    pub fn size(&self) -> Result<usize, StorageError> {
        Ok(self
            .card
            .find_optional(&self.backing_name)?
            .map(MemoryCardBlock::size)
            .unwrap_or(0))
    }
}

fn checksum(payload: &[u8]) -> u8 {
    payload
        .iter()
        .fold(0u8, |sum, byte| sum.wrapping_add(*byte))
}

fn normalise_name(name: &str) -> Result<String, StorageError> {
    let value = name.trim().to_ascii_uppercase();
    if value.is_empty() {
        return Err(StorageError::EmptyName);
    }
    if value.len() > 11 {
        return Err(StorageError::NameTooLong);
    }
    Ok(value)
}

#[derive(Debug, Clone)]
pub struct Pce500PeripheralBridge {
    pub cassette: CassetteTapeImage,
    pub card: MemoryCardImage,
    open_file: Option<String>,
    open_file_pos: usize,
}

impl Pce500PeripheralBridge {
    pub fn new(card_capacity: usize) -> Self {
        Self {
            cassette: CassetteTapeImage::new(),
            card: MemoryCardImage::new(card_capacity),
            open_file: None,
            open_file_pos: 0,
        }
    }

    pub fn maybe_short_circuit(
        &mut self,
        pc: u32,
        state: &mut LlamaState,
        memory: &mut MemoryImage,
    ) -> bool {
        let pc = pc & 0x000f_ffff;
        let handled = match pc {
            CAS_WRITE_DATA_BLOCK_PC => self.cas_write_data(state, memory),
            CAS_READ_DATA_BLOCK_PC => self.cas_read_data(state, memory),
            CAS_VERIFY_DATA_BLOCK_PC => self.cas_verify_data(state, memory),
            CAS_WRITE_HEADER_BLOCK_PC => self.cas_write_header(state, memory),
            CAS_READ_HEADER_BLOCK_PC => self.cas_read_header(state, memory),
            CARD_SEARCH_BLOCK_PC => self.card_search(state, memory),
            CARD_RESIZE_BLOCK_PC => self.card_resize(state, memory),
            CARD_CREATE_BLOCK_PC => self.card_create(state, memory, false),
            CARD_CREATE_TOP_BLOCK_PC => self.card_create(state, memory, true),
            CARD_DELETE_BLOCK_PC => self.card_delete(state, memory),
            CARD_CONDENSE_PC => self.card_condense(state, memory),
            CARD_FILE_CREATE_PC => self.card_file_create(state, memory),
            CARD_FILE_OPEN_PC => self.card_file_open(state, memory),
            CARD_FILE_CLOSE_PC => self.card_file_close(state, memory),
            CARD_FILE_READ_BLOCK_PC => self.card_file_read_block(state, memory, true),
            CARD_FILE_WRITE_BLOCK_PC => self.card_file_write_block(state, memory),
            CARD_FILE_READ_BYTE_PC => self.card_file_read_byte(state),
            CARD_FILE_WRITE_BYTE_PC => self.card_file_write_byte(state),
            CARD_FILE_VERIFY_PC => self.card_file_verify(state, memory),
            CARD_FILE_PEEK_PC => self.card_file_read_block(state, memory, false),
            CARD_FILE_SEEK_PC => self.card_file_seek(state),
            CARD_FILE_INFO_PC => self.card_file_info(state, memory),
            CARD_FILE_CHANGE_DIR_PC => Ok(()),
            CARD_FILE_SEARCH_PC => self.card_search(state, memory),
            CARD_FILE_RENAME_DELETE_PC => self.card_file_rename_or_delete(state, memory),
            CARD_FILE_FREE_PC => self.card_file_free(state),
            RAMDISK_READ_BLOCK_PC => self.ramdisk_read(state, memory),
            RAMDISK_WRITE_BLOCK_PC => self.ramdisk_write(state, memory),
            RAMDISK_FORMAT_PC => self.ramdisk_format(state, memory),
            _ => return false,
        };
        set_call_result(state, handled);
        force_return_auto(state, memory);
        true
    }

    fn cas_write_header(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let payload = read_bytes(memory, state.get_reg(RegName::X), 0x30);
        self.cassette
            .append_header(payload)
            .map(|_| ())
            .map_err(|_| 0x80)
    }

    fn cas_read_header(&mut self, state: &LlamaState, memory: &mut MemoryImage) -> Result<(), u8> {
        let block = self
            .cassette
            .read_next(Some(CassetteBlockKind::Header))
            .map_err(|_| 0x80)?;
        write_bytes(memory, state.get_reg(RegName::X), &block.payload);
        Ok(())
    }

    fn cas_write_data(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let payload = read_bytes(
            memory,
            state.get_reg(RegName::X),
            state.get_reg(RegName::Y) as usize,
        );
        self.cassette
            .append_data(payload)
            .map(|_| ())
            .map_err(|_| 0x80)
    }

    fn cas_read_data(&mut self, state: &LlamaState, memory: &mut MemoryImage) -> Result<(), u8> {
        let block = self
            .cassette
            .read_next(Some(CassetteBlockKind::Data))
            .map_err(|_| 0x80)?;
        let len = (state.get_reg(RegName::Y) as usize).min(block.payload.len());
        write_bytes(memory, state.get_reg(RegName::X), &block.payload[..len]);
        Ok(())
    }

    fn cas_verify_data(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let expected = read_bytes(
            memory,
            state.get_reg(RegName::X),
            state.get_reg(RegName::Y) as usize,
        );
        match self.cassette.verify_next(expected) {
            Ok(true) => Ok(()),
            _ => Err(0x80),
        }
    }

    fn card_search(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        self.card
            .find(&name)
            .map(|_| ())
            .map_err(storage_error_code)
    }

    fn card_resize(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        self.card
            .resize(&name, state.get_reg(RegName::Y) as usize)
            .map(|_| ())
            .map_err(storage_error_code)
    }

    fn card_create(
        &mut self,
        state: &LlamaState,
        memory: &MemoryImage,
        at_top: bool,
    ) -> Result<(), u8> {
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        self.card
            .create(&name, state.get_reg(RegName::Y) as usize, at_top)
            .map(|_| ())
            .map_err(storage_error_code)
    }

    fn card_delete(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        self.card
            .delete(&name)
            .map(|_| ())
            .map_err(storage_error_code)
    }

    fn card_condense(&mut self, _state: &LlamaState, _memory: &MemoryImage) -> Result<(), u8> {
        self.card.condense();
        Ok(())
    }

    fn card_file_create(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        self.card
            .create(&name, state.get_reg(RegName::Y) as usize, false)
            .map(|_| {
                self.open_file = Some(name);
                self.open_file_pos = 0;
            })
            .map_err(storage_error_code)
    }

    fn card_file_open(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        self.card.find(&name).map_err(storage_error_code)?;
        self.open_file = Some(normalise_name(&name).map_err(storage_error_code)?);
        self.open_file_pos = 0;
        Ok(())
    }

    fn card_file_close(&mut self, _state: &LlamaState, _memory: &MemoryImage) -> Result<(), u8> {
        self.open_file = None;
        self.open_file_pos = 0;
        Ok(())
    }

    fn card_file_read_block(
        &mut self,
        state: &mut LlamaState,
        memory: &mut MemoryImage,
        advance: bool,
    ) -> Result<(), u8> {
        let len = state.get_reg(RegName::Y) as usize;
        let block = self.open_block()?;
        let available = block.data.len().saturating_sub(self.open_file_pos);
        let len = len.min(available);
        write_bytes(
            memory,
            state.get_reg(RegName::X),
            &block.data[self.open_file_pos..self.open_file_pos + len],
        );
        if advance {
            self.open_file_pos = self.open_file_pos.saturating_add(len);
        }
        state.set_reg(RegName::Y, len as u32);
        Ok(())
    }

    fn card_file_write_block(
        &mut self,
        state: &LlamaState,
        memory: &MemoryImage,
    ) -> Result<(), u8> {
        let len = state.get_reg(RegName::Y) as usize;
        let payload = read_bytes(memory, state.get_reg(RegName::X), len);
        let name = self.open_file_name()?;
        let block = self
            .card
            .resize(&name, self.open_file_pos + len)
            .map_err(storage_error_code)?;
        block.data[self.open_file_pos..self.open_file_pos + len].copy_from_slice(&payload);
        self.open_file_pos += len;
        Ok(())
    }

    fn card_file_read_byte(&mut self, state: &mut LlamaState) -> Result<(), u8> {
        let byte = {
            let block = self.open_block()?;
            *block.data.get(self.open_file_pos).ok_or(0x01)?
        };
        self.open_file_pos += 1;
        state.set_reg(RegName::A, u32::from(byte));
        Ok(())
    }

    fn card_file_write_byte(&mut self, state: &LlamaState) -> Result<(), u8> {
        let name = self.open_file_name()?;
        let block = self
            .card
            .resize(&name, self.open_file_pos + 1)
            .map_err(storage_error_code)?;
        block.data[self.open_file_pos] = (state.get_reg(RegName::A) & 0xff) as u8;
        self.open_file_pos += 1;
        Ok(())
    }

    fn card_file_verify(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let len = state.get_reg(RegName::Y) as usize;
        let expected = read_bytes(memory, state.get_reg(RegName::X), len);
        let block = self.open_block()?;
        if block
            .data
            .get(self.open_file_pos..self.open_file_pos + len)
            .is_some_and(|actual| actual == expected.as_slice())
        {
            self.open_file_pos += len;
            Ok(())
        } else {
            Err(0x0B)
        }
    }

    fn card_file_seek(&mut self, state: &LlamaState) -> Result<(), u8> {
        let pos = state.get_reg(RegName::Y) as usize;
        let block = self.open_block()?;
        if pos <= block.data.len() {
            self.open_file_pos = pos;
            Ok(())
        } else {
            Err(0x01)
        }
    }

    fn card_file_info(
        &mut self,
        state: &mut LlamaState,
        memory: &mut MemoryImage,
    ) -> Result<(), u8> {
        let block = self.open_block()?;
        state.set_reg(RegName::Y, block.data.len() as u32);
        write_bytes(memory, state.get_reg(RegName::X), block.name.as_bytes());
        Ok(())
    }

    fn card_file_rename_or_delete(
        &mut self,
        state: &LlamaState,
        memory: &MemoryImage,
    ) -> Result<(), u8> {
        let command = state.get_reg(RegName::I) & 0xff;
        let name = read_card_name(memory, state.get_reg(RegName::X))?;
        if command == 0x2D {
            let new_name = read_card_name(memory, state.get_reg(RegName::Y))?;
            self.card
                .rename(&name, &new_name)
                .map(|_| {
                    self.open_file = Some(normalise_name(&new_name).unwrap_or(new_name));
                    self.open_file_pos = 0;
                })
                .map_err(storage_error_code)
        } else {
            self.card
                .delete(&name)
                .map(|_| {
                    if self.open_file.as_deref() == normalise_name(&name).ok().as_deref() {
                        self.open_file = None;
                        self.open_file_pos = 0;
                    }
                })
                .map_err(storage_error_code)
        }
    }

    fn card_file_free(&mut self, state: &mut LlamaState) -> Result<(), u8> {
        state.set_reg(RegName::Y, self.card.free() as u32);
        Ok(())
    }

    fn ramdisk_format(&mut self, state: &LlamaState, _memory: &MemoryImage) -> Result<(), u8> {
        let size = state.get_reg(RegName::Y) as usize;
        if self
            .card
            .find_optional(RAMDISK_BACKING_NAME)
            .map_err(storage_error_code)?
            .is_none()
        {
            self.card
                .create(RAMDISK_BACKING_NAME, size, true)
                .map(|_| ())
                .map_err(storage_error_code)
        } else {
            self.card
                .resize(RAMDISK_BACKING_NAME, size)
                .map(|_| ())
                .map_err(storage_error_code)
        }
    }

    fn ramdisk_read(&mut self, state: &LlamaState, memory: &mut MemoryImage) -> Result<(), u8> {
        let len = state.get_reg(RegName::Y) as usize;
        let block = self
            .card
            .find(RAMDISK_BACKING_NAME)
            .map_err(storage_error_code)?;
        let len = len.min(block.data.len());
        write_bytes(memory, state.get_reg(RegName::X), &block.data[..len]);
        Ok(())
    }

    fn ramdisk_write(&mut self, state: &LlamaState, memory: &MemoryImage) -> Result<(), u8> {
        let len = state.get_reg(RegName::Y) as usize;
        let payload = read_bytes(memory, state.get_reg(RegName::X), len);
        let block = self
            .card
            .resize(RAMDISK_BACKING_NAME, len)
            .map_err(storage_error_code)?;
        block.data.copy_from_slice(&payload);
        Ok(())
    }

    fn open_file_name(&self) -> Result<String, u8> {
        self.open_file.clone().ok_or(0x01)
    }

    fn open_block(&self) -> Result<&MemoryCardBlock, u8> {
        let name = self.open_file.as_deref().ok_or(0x01)?;
        self.card.find(name).map_err(storage_error_code)
    }
}

fn set_call_result(state: &mut LlamaState, result: Result<(), u8>) {
    match result {
        Ok(()) => {
            state.set_reg(RegName::FC, 0);
        }
        Err(code) => {
            state.set_reg(RegName::A, u32::from(code));
            state.set_reg(RegName::FC, 1);
        }
    }
}

fn storage_error_code(error: StorageError) -> u8 {
    match error {
        StorageError::BlockNotFound(_) => 0x02,
        StorageError::DuplicateBlock(_) => 0x05,
        StorageError::InsufficientSpace(_) => 0x3C,
        StorageError::EmptyName
        | StorageError::NameTooLong
        | StorageError::InvalidSize
        | StorageError::InvalidMediaHeader
        | StorageError::TruncatedMediaBlock
        | StorageError::InvalidMediaBlockMarker => 0x0A,
    }
}

fn read_card_name(memory: &MemoryImage, addr: u32) -> Result<String, u8> {
    let bytes = read_bytes(memory, addr, 11);
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end])
        .map(|name| name.trim().to_string())
        .map_err(|_| 0x0A)
}

fn read_bytes(memory: &MemoryImage, addr: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|offset| {
            memory
                .load(addr.wrapping_add(offset as u32), 8)
                .map(|value| value as u8)
                .unwrap_or(0)
        })
        .collect()
}

fn write_bytes(memory: &mut MemoryImage, addr: u32, bytes: &[u8]) {
    for (offset, byte) in bytes.iter().enumerate() {
        let _ = memory.store(addr.wrapping_add(offset as u32), 8, u32::from(*byte));
    }
}

fn read_u24_slice(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

fn write_u24_slice(bytes: &mut [u8], value: u32) {
    bytes[0] = (value & 0xff) as u8;
    bytes[1] = ((value >> 8) & 0xff) as u8;
    bytes[2] = ((value >> 16) & 0xff) as u8;
}

fn force_return_auto(state: &mut LlamaState, memory: &mut MemoryImage) {
    let ret_bits = match state.peek_call_return_width() {
        Some(16) => 16,
        Some(24) => 24,
        _ => {
            let call_depth = state.call_stack().len();
            let page_depth = state.call_page_depth();
            if page_depth < call_depth {
                24
            } else {
                16
            }
        }
    };
    if ret_bits == 24 {
        let ret = pop_stack_value(state, memory, 24);
        state.set_pc(ret & 0xFFFFF);
    } else {
        let pc_before = state.pc();
        let ret = pop_stack_value(state, memory, 16);
        let current_page = pc_before & 0xFF0000;
        let dest = (current_page | (ret & 0xFFFF)) & 0xFFFFF;
        let _ = state.pop_call_page();
        state.set_pc(dest);
    }
    state.call_depth_dec();
    let _ = state.pop_call_frame();
}

fn pop_stack_value(state: &mut LlamaState, memory: &mut MemoryImage, bits: u8) -> u32 {
    let bytes = bits.div_ceil(8);
    let mask = mask_for(RegName::S);
    let mut value = 0u32;
    let mut sp = state.get_reg(RegName::S);
    for i in 0..bytes {
        let byte = memory.load_with_pc(sp, 8, Some(state.pc())).unwrap_or(0) & 0xFF;
        value |= byte << (8 * i);
        sp = sp.wrapping_add(1) & mask;
    }
    state.set_reg(RegName::S, sp);
    value
}
