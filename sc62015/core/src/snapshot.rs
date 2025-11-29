// PY_SOURCE: pce500/emulator.py:save_snapshot
// PY_SOURCE: pce500/emulator.py:load_snapshot

use crate::memory::{
    MemoryImage, INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START, INTERNAL_SPACE,
};
use crate::{CoreError, Result, SnapshotMetadata};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use zip::read::ZipArchive;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipWriter};

pub const SNAPSHOT_MAGIC: &str = "pc-e500.snapshot";
pub const SNAPSHOT_VERSION: u32 = 1;
pub const SNAPSHOT_REGISTER_LAYOUT: [(&str, usize); 8] = [
    ("PC", 3),
    ("BA", 2),
    ("I", 2),
    ("X", 3),
    ("Y", 3),
    ("U", 3),
    ("S", 3),
    ("F", 1),
];

#[derive(Debug)]
pub struct SnapshotLoad {
    pub metadata: SnapshotMetadata,
    pub registers: HashMap<String, u32>,
    pub lcd_payload: Option<Vec<u8>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RangeSerde {
    Tuple((u32, u32)),
    Array(Vec<u32>),
    Object { start: u32, size: u32 },
}

impl RangeSerde {
    fn into_tuple(self) -> (u32, u32) {
        match self {
            RangeSerde::Tuple(pair) => pair,
            RangeSerde::Array(items) => {
                if items.len() >= 2 {
                    (items[0], items[1])
                } else {
                    (0, 0)
                }
            }
            RangeSerde::Object { start, size } => (start, size),
        }
    }
}

pub(crate) fn deserialize_range<'de, D>(
    deserializer: D,
) -> std::result::Result<(u32, u32), D::Error>
where
    D: serde::Deserializer<'de>,
{
    let helper = RangeSerde::deserialize(deserializer)?;
    Ok(helper.into_tuple())
}

pub fn pack_registers(regs: &HashMap<String, u32>) -> Vec<u8> {
    let mut buf = Vec::with_capacity(18);
    for (name, width_bytes) in SNAPSHOT_REGISTER_LAYOUT.iter() {
        let mut chunk = vec![0u8; *width_bytes];
        let value = regs.get(*name).copied().unwrap_or(0);
        for (idx, byte) in chunk.iter_mut().enumerate() {
            *byte = ((value >> (idx * 8)) & 0xFF) as u8;
        }
        buf.extend_from_slice(&chunk);
    }
    buf
}

pub fn unpack_registers(payload: &[u8]) -> Result<HashMap<String, u32>> {
    let expected: usize = SNAPSHOT_REGISTER_LAYOUT.iter().map(|(_, w)| *w).sum();
    if payload.len() != expected {
        return Err(CoreError::InvalidSnapshot(format!(
            "registers.bin length mismatch (expected {expected}, got {})",
            payload.len()
        )));
    }
    let mut offset = 0usize;
    let mut regs = HashMap::new();
    for (name, width_bytes) in SNAPSHOT_REGISTER_LAYOUT.iter() {
        let mut value = 0u32;
        for idx in 0..*width_bytes {
            value |= (payload[offset + idx] as u32) << (idx * 8);
        }
        regs.insert((*name).to_string(), value);
        offset += *width_bytes;
    }
    Ok(regs)
}

pub fn save_snapshot(
    path: &Path,
    metadata: &SnapshotMetadata,
    registers: &HashMap<String, u32>,
    memory: &MemoryImage,
    lcd_payload: Option<&[u8]>,
) -> Result<()> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);
    let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

    let lcd_len = lcd_payload.map(|buf| buf.len()).unwrap_or(0);
    let mut meta = metadata.clone();
    meta.memory_image_size = memory.external_len();
    meta.lcd_payload_size = lcd_len;
    if meta.internal_ram == (0, 0) {
        meta.internal_ram = (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32);
    }
    if meta.imem == (0, 0) {
        meta.imem = (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32);
    }
    if meta.fallback_ranges.is_empty() {
        meta.fallback_ranges = memory.python_ranges().to_vec();
    }
    if meta.readonly_ranges.is_empty() {
        meta.readonly_ranges = memory.readonly_ranges().to_vec();
    }

    zip.start_file("snapshot.json", options)?;
    let meta_bytes = serde_json::to_vec_pretty(&meta)?;
    zip.write_all(&meta_bytes)?;

    zip.start_file("registers.bin", options)?;
    let registers_blob = pack_registers(registers);
    zip.write_all(&registers_blob)?;

    zip.start_file("external_ram.bin", options)?;
    zip.write_all(memory.external_slice())?;

    zip.start_file("internal_ram.bin", options)?;
    zip.write_all(memory.internal_ram_slice())?;

    zip.start_file("imem.bin", options)?;
    zip.write_all(memory.internal_slice())?;

    if lcd_len > 0 {
        zip.start_file("lcd_vram.bin", options)?;
        if let Some(buf) = lcd_payload {
            zip.write_all(buf)?;
        }
    }

    zip.finish()?;
    Ok(())
}

pub fn load_snapshot(path: &Path, memory: &mut MemoryImage) -> Result<SnapshotLoad> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;

    let metadata = {
        let mut meta_buf = Vec::new();
        {
            let mut meta_file = archive
                .by_name("snapshot.json")
                .map_err(|e| CoreError::InvalidSnapshot(format!("snapshot.json missing: {e}")))?;
            meta_file.read_to_end(&mut meta_buf)?;
        }
        let metadata: SnapshotMetadata = serde_json::from_slice(&meta_buf)?;
        if metadata.magic != SNAPSHOT_MAGIC || metadata.version != SNAPSHOT_VERSION {
            return Err(CoreError::InvalidSnapshot(
                "snapshot magic/version mismatch".to_string(),
            ));
        }
        metadata
    };

    let registers = {
        let mut reg_buf = Vec::new();
        let mut reg_file = archive
            .by_name("registers.bin")
            .map_err(|e| CoreError::InvalidSnapshot(format!("registers.bin missing: {e}")))?;
        reg_file.read_to_end(&mut reg_buf)?;
        unpack_registers(&reg_buf)?
    };

    if let Ok(mut ext_file) = archive.by_name("external_ram.bin") {
        let mut ext_buf = Vec::new();
        ext_file.read_to_end(&mut ext_buf)?;
        memory.copy_external_from(&ext_buf)?;
    }

    if let Ok(mut int_file) = archive.by_name("internal_ram.bin") {
        let mut int_buf = Vec::new();
        int_file.read_to_end(&mut int_buf)?;
        let start = metadata.internal_ram.0;
        memory.write_internal_ram(start, &int_buf);
    }

    if let Ok(mut imem_file) = archive.by_name("imem.bin") {
        let mut imem_buf = Vec::new();
        imem_file.read_to_end(&mut imem_buf)?;
        memory.write_imem(&imem_buf);
    }

    memory.clear_dirty();
    memory.set_python_ranges(metadata.fallback_ranges.clone());
    memory.set_readonly_ranges(metadata.readonly_ranges.clone());

    let mut lcd_payload: Option<Vec<u8>> = None;
    if metadata.lcd.is_some() {
        if let Ok(mut lcd_file) = archive.by_name("lcd_vram.bin") {
            let mut lcd_buf = Vec::new();
            lcd_file.read_to_end(&mut lcd_buf)?;
            lcd_payload = Some(lcd_buf);
        }
    }

    let mut metadata = metadata;
    metadata.memory_image_size = memory.external_len();
    metadata.lcd_payload_size = lcd_payload.as_ref().map(|buf| buf.len()).unwrap_or(0);

    Ok(SnapshotLoad {
        metadata,
        registers,
        lcd_payload,
    })
}
