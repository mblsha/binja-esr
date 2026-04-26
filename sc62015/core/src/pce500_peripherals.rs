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
}

impl Pce500PeripheralBridge {
    pub fn new(card_capacity: usize) -> Self {
        Self {
            cassette: CassetteTapeImage::new(),
            card: MemoryCardImage::new(card_capacity),
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
}

fn set_call_result(state: &mut LlamaState, result: Result<(), u8>) {
    match result {
        Ok(()) => {
            state.set_reg(RegName::A, 0);
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
        StorageError::EmptyName | StorageError::NameTooLong | StorageError::InvalidSize => 0x0A,
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
