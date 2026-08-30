// PY_SOURCE: pce500/emulator.py:save_snapshot
// PY_SOURCE: pce500/emulator.py:load_snapshot

use crate::memory::{MemoryCardMode, MemoryCardSnapshot};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use crate::memory::{
    MemoryImage, INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START, INTERNAL_SPACE,
};
use crate::{CoreError, Result, SnapshotMetadata};
use serde::de::Error as _;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use std::fs::{self, File, OpenOptions};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use std::io::{Read, Write};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use std::path::{Path, PathBuf};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use zip::read::ZipArchive;
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use zip::write::FileOptions;
#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use zip::{CompressionMethod, ZipWriter};

pub const SNAPSHOT_MAGIC: &str = "pc-e500.snapshot";
pub const SNAPSHOT_VERSION: u32 = 4;
const SNAPSHOT_JSON_MAX_BYTES: usize = 4 * 1024 * 1024;
const SNAPSHOT_LCD_MAX_BYTES: usize = 0x10_0000;
const SNAPSHOT_MEMORY_CARD_CAPACITIES: [usize; 4] = [8192, 16384, 32768, 65536];
const SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX: u32 = crate::memory::EXTERNAL_SPACE as u32 - 1;
const SNAPSHOT_ARCHIVE_BASE_ENTRIES: [&str; 5] = [
    "snapshot.json",
    "registers.bin",
    "external_ram.bin",
    "internal_ram.bin",
    "imem.bin",
];
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
    pub external_memory: Vec<u8>,
    pub internal_ram: Vec<u8>,
    pub imem: Vec<u8>,
    pub lcd_payload: Option<Vec<u8>>,
    pub memory_card: Option<MemoryCardSnapshot>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum SnapshotMemoryCardMode {
    Absent,
    Present,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SnapshotMemoryCardMetadata {
    mode: SnapshotMemoryCardMode,
    capacity: usize,
    writable: bool,
    payload_size: usize,
}

impl SnapshotMemoryCardMetadata {
    fn from_snapshot(snapshot: &MemoryCardSnapshot) -> Result<Self> {
        let metadata = Self {
            mode: match snapshot.mode {
                MemoryCardMode::Absent => SnapshotMemoryCardMode::Absent,
                MemoryCardMode::Present => SnapshotMemoryCardMode::Present,
            },
            capacity: snapshot.capacity,
            writable: snapshot.writable,
            payload_size: snapshot.payload.len(),
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<()> {
        if !SNAPSHOT_MEMORY_CARD_CAPACITIES.contains(&self.capacity) {
            return Err(CoreError::InvalidSnapshot(format!(
                "unsupported snapshot memory-card capacity: {} bytes",
                self.capacity
            )));
        }
        if self.payload_size != self.capacity {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot memory-card payload length mismatch (capacity {}, payload {})",
                self.capacity, self.payload_size
            )));
        }
        Ok(())
    }

    fn with_payload(self, payload: Vec<u8>) -> Result<MemoryCardSnapshot> {
        self.validate()?;
        if payload.len() != self.payload_size {
            return Err(CoreError::InvalidSnapshot(format!(
                "memory_card.bin length mismatch (expected {}, got {})",
                self.payload_size,
                payload.len()
            )));
        }
        Ok(MemoryCardSnapshot {
            mode: match self.mode {
                SnapshotMemoryCardMode::Absent => MemoryCardMode::Absent,
                SnapshotMemoryCardMode::Present => MemoryCardMode::Present,
            },
            capacity: self.capacity,
            writable: self.writable,
            payload,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, untagged)]
enum RangeSerde {
    Tuple((u32, u32)),
    Array(Vec<u32>),
    Object { start: u32, size: u32 },
}

/// Deserialize JSON without allowing a later object member to silently replace
/// an earlier one. `serde_json::Value` normally keeps only the last duplicate,
/// which is inappropriate for an exact checkpoint format.
#[derive(Debug)]
struct DuplicateFreeJson(serde_json::Value);

impl<'de> Deserialize<'de> for DuplicateFreeJson {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DuplicateFreeJsonVisitor)
    }
}

struct DuplicateFreeJsonVisitor;

impl<'de> Visitor<'de> for DuplicateFreeJsonVisitor {
    type Value = DuplicateFreeJson;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("duplicate-free JSON")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("snapshot JSON contains a non-finite number"))?;
        Ok(DuplicateFreeJson(serde_json::Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::String(
            value.to_string(),
        )))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(DuplicateFreeJson(serde_json::Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        DuplicateFreeJson::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<DuplicateFreeJson>()? {
            values.push(value.0);
        }
        Ok(DuplicateFreeJson(serde_json::Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!(
                    "snapshot JSON contains duplicate object member {key:?}"
                )));
            }
            let value = map.next_value::<DuplicateFreeJson>()?;
            values.insert(key, value.0);
        }
        Ok(DuplicateFreeJson(serde_json::Value::Object(values)))
    }
}

fn parse_duplicate_free_json(bytes: &[u8]) -> Result<serde_json::Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = DuplicateFreeJson::deserialize(&mut deserializer)
        .map_err(|error| CoreError::InvalidSnapshot(format!("invalid snapshot.json: {error}")))?;
    deserializer
        .end()
        .map_err(|error| CoreError::InvalidSnapshot(format!("invalid snapshot.json: {error}")))?;
    Ok(value.0)
}

impl RangeSerde {
    fn into_tuple<E: serde::de::Error>(self) -> std::result::Result<(u32, u32), E> {
        match self {
            RangeSerde::Tuple(pair) => Ok(pair),
            RangeSerde::Array(items) if items.len() == 2 => Ok((items[0], items[1])),
            RangeSerde::Array(items) => Err(E::custom(format!(
                "snapshot range array must contain exactly two values, got {}",
                items.len()
            ))),
            RangeSerde::Object { start, size } => Ok((start, size)),
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
    helper.into_tuple()
}

pub fn canonical_snapshot_ranges(label: &str, ranges: &[(u32, u32)]) -> Result<Vec<(u32, u32)>> {
    let mut canonical = ranges.to_vec();
    canonical.sort_unstable();
    let external_end = crate::memory::EXTERNAL_SPACE as u32 - 1;
    let imem_end = crate::memory::INTERNAL_MEMORY_START
        .checked_add(crate::memory::INTERNAL_SPACE as u32 - 1)
        .expect("fixed IMEM range must fit in u32");
    let mut previous_end = None;
    for (start, end) in &canonical {
        let in_external = *start <= *end && *end <= external_end;
        let in_imem =
            *start <= *end && *start >= crate::memory::INTERNAL_MEMORY_START && *end <= imem_end;
        if !in_external && !in_imem {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot {label} range 0x{start:X}..0x{end:X} is outside external memory or IMEM"
            )));
        }
        if previous_end.is_some_and(|previous| previous >= *start) {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot {label} ranges overlap at 0x{start:X}"
            )));
        }
        previous_end = Some(*end);
    }
    Ok(canonical)
}

fn require_canonical_snapshot_ranges(label: &str, ranges: &[(u32, u32)]) -> Result<()> {
    if canonical_snapshot_ranges(label, ranges)? != ranges {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot {label} ranges are not in canonical address order"
        )));
    }
    Ok(())
}

fn encode_snapshot_metadata(
    metadata: &SnapshotMetadata,
    memory_card: Option<&SnapshotMemoryCardMetadata>,
) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(metadata)?;
    let object = value.as_object_mut().ok_or_else(|| {
        CoreError::InvalidSnapshot("snapshot metadata must serialize as an object".to_string())
    })?;
    object.insert(
        "memory_card".to_string(),
        serde_json::to_value(memory_card)?,
    );
    Ok(serde_json::to_vec_pretty(&value)?)
}

fn decode_snapshot_metadata(
    bytes: &[u8],
) -> Result<(SnapshotMetadata, Option<SnapshotMemoryCardMetadata>)> {
    let mut value = parse_duplicate_free_json(bytes)?;
    let object = value.as_object_mut().ok_or_else(|| {
        CoreError::InvalidSnapshot("snapshot.json must contain an object".to_string())
    })?;

    let magic = object
        .get("magic")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            CoreError::InvalidSnapshot("snapshot magic is missing or invalid".to_string())
        })?;
    if magic != SNAPSHOT_MAGIC {
        return Err(CoreError::InvalidSnapshot(
            "snapshot magic mismatch".to_string(),
        ));
    }
    let version = object
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            CoreError::InvalidSnapshot("snapshot version is missing or invalid".to_string())
        })?;
    if version == 3 {
        return Err(CoreError::InvalidSnapshot(
            "snapshot version 3 is not accepted; version 4 is required".to_string(),
        ));
    }
    if version != u64::from(SNAPSHOT_VERSION) {
        return Err(CoreError::InvalidSnapshot(format!(
            "unsupported snapshot version {version}; version {SNAPSHOT_VERSION} is required"
        )));
    }

    let required_keys: HashSet<&str> = [
        "magic",
        "version",
        "backend",
        "created",
        "instruction_count",
        "cycle_count",
        "memory_reads",
        "memory_writes",
        "pc",
        "power_state",
        "external_interrupt_level",
        "onk_level",
        "call_depth",
        "call_sub_level",
        "call_stack",
        "call_page_stack",
        "call_return_widths",
        "temps",
        "timer",
        "interrupts",
        "keyboard",
        "kb_metrics",
        "fallback_ranges",
        "readonly_ranges",
        "internal_ram",
        "imem",
        "memory_dump_pc",
        "fast_mode",
        "memory_image_size",
        "lcd_payload_size",
        "lcd",
        "memory_card",
    ]
    .into_iter()
    .collect();
    if required_keys.iter().any(|key| !object.contains_key(*key))
        || object
            .keys()
            .any(|key| key != "device_model" && !required_keys.contains(key.as_str()))
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot metadata has missing or unexpected top-level fields".to_string(),
        ));
    }

    let card_value = object
        .remove("memory_card")
        .expect("required memory_card member was checked");
    let memory_card: Option<SnapshotMemoryCardMetadata> = serde_json::from_value(card_value)
        .map_err(|error| {
            CoreError::InvalidSnapshot(format!("invalid snapshot memory_card metadata: {error}"))
        })?;
    if let Some(card) = memory_card.as_ref() {
        card.validate()?;
    }
    let metadata: SnapshotMetadata = serde_json::from_value(value).map_err(|error| {
        CoreError::InvalidSnapshot(format!("invalid snapshot metadata: {error}"))
    })?;
    Ok((metadata, memory_card))
}

fn validate_snapshot_architectural_addresses(
    metadata: &SnapshotMetadata,
    registers: &HashMap<String, u32>,
) -> Result<()> {
    for name in ["PC", "X", "Y", "U", "S"] {
        if registers.get(name).copied().unwrap_or(0) > SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot register {name} exceeds the modeled 20-bit address space"
            )));
        }
    }
    if metadata
        .call_stack
        .iter()
        .any(|address| *address > SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX)
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot call stack contains an address above 0xFFFFF".to_string(),
        ));
    }
    if metadata
        .call_page_stack
        .iter()
        .any(|page| *page > SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX)
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot call-page stack contains an address above 0xFFFFF".to_string(),
        ));
    }
    if metadata
        .interrupts
        .last_irq
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|last_irq| {
            ["pc", "vector"].iter().any(|name| {
                last_irq
                    .get(*name)
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|address| address > u64::from(SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX))
            })
        })
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot last_irq contains an address above 0xFFFFF".to_string(),
        ));
    }
    if metadata
        .interrupts
        .irq_bit_watch
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .is_some_and(|watch| {
            watch.values().any(|bits| {
                bits.as_object().is_some_and(|bits| {
                    bits.values().any(|actions| {
                        actions.as_object().is_some_and(|actions| {
                            actions.values().any(|pcs| {
                                pcs.as_array().is_some_and(|pcs| {
                                    pcs.iter().any(|pc| {
                                        pc.as_u64().is_some_and(|pc| {
                                            pc > u64::from(SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX)
                                        })
                                    })
                                })
                            })
                        })
                    })
                })
            })
        })
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot irq_bit_watch contains an address above 0xFFFFF".to_string(),
        ));
    }
    Ok(())
}

fn validate_snapshot_register_file(registers: &HashMap<String, u32>) -> Result<()> {
    let expected: HashSet<&str> = SNAPSHOT_REGISTER_LAYOUT
        .iter()
        .map(|(name, _)| *name)
        .collect();
    if registers.len() != expected.len()
        || registers
            .keys()
            .any(|name| !expected.contains(name.as_str()))
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot register file must contain exactly PC, BA, I, X, Y, U, S, and F".to_string(),
        ));
    }
    for (name, width_bytes) in SNAPSHOT_REGISTER_LAYOUT {
        let value = registers.get(name).copied().ok_or_else(|| {
            CoreError::InvalidSnapshot(format!("snapshot register {name} is missing"))
        })?;
        let width_bits = width_bytes * 8;
        let encoded_max = if width_bits >= u32::BITS as usize {
            u32::MAX
        } else {
            (1u32 << width_bits) - 1
        };
        if value > encoded_max {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot register {name} does not fit its {width_bits}-bit container"
            )));
        }
    }
    for name in ["PC", "X", "Y", "U", "S"] {
        if registers[name] > SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot register {name} exceeds the modeled 20-bit address space"
            )));
        }
    }
    if registers.get("F").copied().unwrap_or(0) > 3 {
        return Err(CoreError::InvalidSnapshot(
            "snapshot register F contains unmodeled bits 2-7".to_string(),
        ));
    }
    Ok(())
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

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
fn fixed_snapshot_entry_size(name: &str) -> Option<usize> {
    match name {
        "registers.bin" => Some(
            SNAPSHOT_REGISTER_LAYOUT
                .iter()
                .map(|(_, width)| *width)
                .sum(),
        ),
        "external_ram.bin" => Some(crate::memory::EXTERNAL_SPACE),
        "internal_ram.bin" => Some(INTERNAL_RAM_SIZE),
        "imem.bin" => Some(INTERNAL_SPACE),
        _ => None,
    }
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
fn snapshot_entry_maximum_size(name: &str) -> Option<usize> {
    match name {
        "snapshot.json" => Some(SNAPSHOT_JSON_MAX_BYTES),
        "lcd_vram.bin" => Some(SNAPSHOT_LCD_MAX_BYTES),
        "memory_card.bin" => SNAPSHOT_MEMORY_CARD_CAPACITIES.iter().copied().max(),
        _ => fixed_snapshot_entry_size(name),
    }
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
fn inventory_snapshot_archive(file: &mut ZipArchive<File>) -> Result<HashMap<String, u64>> {
    let mut entries = HashMap::new();
    for index in 0..file.len() {
        let entry = file.by_index(index)?;
        let name = entry.name().to_string();
        if entry.is_dir() || name.ends_with('/') {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot contains unexpected directory {name:?}"
            )));
        }
        let maximum = snapshot_entry_maximum_size(&name).ok_or_else(|| {
            CoreError::InvalidSnapshot(format!(
                "snapshot archive contains unexpected entry {name:?}"
            ))
        })?;
        if entries.insert(name.clone(), entry.size()).is_some() {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot contains duplicate entry {name:?}"
            )));
        }
        if let Some(exact) = fixed_snapshot_entry_size(&name) {
            if entry.size() != exact as u64 {
                return Err(CoreError::InvalidSnapshot(format!(
                    "snapshot entry {name:?} must contain exactly {exact} bytes (declared {})",
                    entry.size()
                )));
            }
        } else if entry.size() > maximum as u64 {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot entry {name:?} exceeds the {maximum}-byte limit"
            )));
        }
    }
    Ok(entries)
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
fn read_snapshot_entry(
    archive: &mut ZipArchive<File>,
    name: &str,
    declared_size: u64,
    exact_size: Option<usize>,
    maximum_size: usize,
) -> Result<Vec<u8>> {
    if let Some(exact) = exact_size {
        if declared_size != exact as u64 {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot entry {name:?} must contain exactly {exact} bytes (declared {declared_size})"
            )));
        }
    } else if declared_size > maximum_size as u64 {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot entry {name:?} exceeds the {maximum_size}-byte limit"
        )));
    }

    let limit = exact_size.unwrap_or(maximum_size);
    let entry = archive.by_name(name)?;
    if entry.size() != declared_size {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot entry {name:?} changed size while reading"
        )));
    }
    let mut payload = Vec::with_capacity(limit.min(declared_size as usize));
    entry.take(limit as u64 + 1).read_to_end(&mut payload)?;
    if payload.len() > limit {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot entry {name:?} exceeds its decompressed-size limit"
        )));
    }
    if payload.len() as u64 != declared_size {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot entry {name:?} declared {declared_size} bytes but produced {}",
            payload.len()
        )));
    }
    if exact_size.is_some_and(|exact| payload.len() != exact) {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot entry {name:?} has an invalid decompressed length"
        )));
    }
    Ok(payload)
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
pub fn save_snapshot(
    path: &Path,
    metadata: &SnapshotMetadata,
    registers: &HashMap<String, u32>,
    memory: &MemoryImage,
    lcd_payload: Option<&[u8]>,
) -> Result<()> {
    validate_snapshot_register_file(registers)?;
    memory.validate_snapshot_overlay_contract()?;
    let memory_card = memory.memory_card_snapshot()?;
    let memory_card_metadata = memory_card
        .as_ref()
        .map(SnapshotMemoryCardMetadata::from_snapshot)
        .transpose()?;
    let lcd_len = lcd_payload.map_or(0, <[u8]>::len);
    if lcd_len > SNAPSHOT_LCD_MAX_BYTES {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot LCD payload exceeds the {SNAPSHOT_LCD_MAX_BYTES}-byte limit"
        )));
    }
    if metadata.lcd.is_none() && lcd_payload.is_some() {
        return Err(CoreError::InvalidSnapshot(
            "snapshot LCD payload requires LCD metadata".to_string(),
        ));
    }

    let mut meta = metadata.clone();
    meta.magic = SNAPSHOT_MAGIC.to_string();
    meta.version = SNAPSHOT_VERSION;
    meta.memory_image_size = memory.external_len();
    meta.lcd_payload_size = lcd_len;
    // These layouts are fixed properties of the Rust memory implementation,
    // not caller-owned metadata shadows.
    meta.internal_ram = (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32);
    meta.imem = (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32);
    // Range configuration belongs to the live memory image. Never preserve a
    // stale metadata shadow merely because it is non-empty. Sorting produces
    // one deterministic archive representation; overlap and unsupported
    // address spaces fail before a temporary file is created.
    meta.fallback_ranges = canonical_snapshot_ranges("fallback", memory.python_ranges())?;
    meta.readonly_ranges = canonical_snapshot_ranges("readonly", memory.readonly_ranges())?;
    validate_snapshot_architectural_addresses(&meta, registers)?;
    let meta_bytes = encode_snapshot_metadata(&meta, memory_card_metadata.as_ref())?;
    if meta_bytes.len() > SNAPSHOT_JSON_MAX_BYTES {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot.json exceeds the {SNAPSHOT_JSON_MAX_BYTES}-byte limit"
        )));
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("snapshot.pcsnap");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut temp: Option<(PathBuf, File)> = None;
    for nonce in 0..32u8 {
        let candidate = parent.join(format!(
            ".{file_name}.{}.{}.tmp",
            std::process::id(),
            stamp + u128::from(nonce)
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temp = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let (temp_path, file) = temp.ok_or_else(|| {
        CoreError::Other("unable to allocate a unique snapshot temporary file".to_string())
    })?;

    let write_result = (|| -> Result<()> {
        let mut zip = ZipWriter::new(file);
        let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

        zip.start_file("snapshot.json", options)?;
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

        if meta.lcd.is_some() {
            zip.start_file("lcd_vram.bin", options)?;
            if let Some(buf) = lcd_payload {
                zip.write_all(buf)?;
            }
        }

        if let Some(card) = memory_card.as_ref() {
            zip.start_file("memory_card.bin", options)?;
            zip.write_all(&card.payload)?;
        }

        let file = zip.finish()?;
        file.sync_all()?;
        Ok(())
    })();
    let finish_result = write_result.and_then(|()| {
        // Re-open through the same strict reader before replacing an existing
        // checkpoint. This keeps serialization bugs or inconsistent live
        // metadata from producing an archive the runtime itself cannot load.
        let _ = load_snapshot(&temp_path)?;
        fs::rename(&temp_path, path)?;
        Ok(())
    });
    if finish_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    finish_result
}

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
pub fn load_snapshot(path: &Path) -> Result<SnapshotLoad> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    let entry_sizes = inventory_snapshot_archive(&mut archive)?;
    let metadata_size = *entry_sizes.get("snapshot.json").ok_or_else(|| {
        CoreError::InvalidSnapshot("snapshot entry \"snapshot.json\" is missing".to_string())
    })?;
    let metadata_payload = read_snapshot_entry(
        &mut archive,
        "snapshot.json",
        metadata_size,
        None,
        SNAPSHOT_JSON_MAX_BYTES,
    )?;
    let (metadata, memory_card_metadata) = decode_snapshot_metadata(&metadata_payload)?;
    if !matches!(
        metadata.backend.as_str(),
        "core" | "rust" | "llama" | "python"
    ) {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot backend {:?} is unsupported",
            metadata.backend
        )));
    }
    if metadata.memory_image_size != crate::memory::EXTERNAL_SPACE {
        return Err(CoreError::InvalidSnapshot(format!(
            "external memory must contain exactly {} bytes",
            crate::memory::EXTERNAL_SPACE
        )));
    }
    if metadata.internal_ram != (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32) {
        return Err(CoreError::InvalidSnapshot(format!(
            "internal RAM must be ({:#X}, {}) with exactly {} bytes",
            INTERNAL_RAM_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_SIZE
        )));
    }
    if metadata.imem != (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32) {
        return Err(CoreError::InvalidSnapshot(format!(
            "IMEM must be ({INTERNAL_MEMORY_START:#X}, {INTERNAL_SPACE}) with exactly {INTERNAL_SPACE} bytes"
        )));
    }
    require_canonical_snapshot_ranges("fallback", &metadata.fallback_ranges)?;
    require_canonical_snapshot_ranges("readonly", &metadata.readonly_ranges)?;
    if metadata.lcd_payload_size > SNAPSHOT_LCD_MAX_BYTES {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot LCD payload exceeds the {SNAPSHOT_LCD_MAX_BYTES}-byte limit"
        )));
    }
    if metadata.lcd.is_none() && metadata.lcd_payload_size != 0 {
        return Err(CoreError::InvalidSnapshot(
            "snapshot has an LCD payload size without LCD metadata".to_string(),
        ));
    }

    let mut expected_names: HashSet<&str> = SNAPSHOT_ARCHIVE_BASE_ENTRIES.into_iter().collect();
    if metadata.lcd.is_some() {
        expected_names.insert("lcd_vram.bin");
    }
    if memory_card_metadata.is_some() {
        expected_names.insert("memory_card.bin");
    }
    let actual_names: HashSet<&str> = entry_sizes.keys().map(String::as_str).collect();
    if actual_names != expected_names {
        let mut missing: Vec<_> = expected_names.difference(&actual_names).copied().collect();
        let mut unexpected: Vec<_> = actual_names.difference(&expected_names).copied().collect();
        missing.sort_unstable();
        unexpected.sort_unstable();
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot archive entry set mismatch (missing {missing:?}, unexpected {unexpected:?})"
        )));
    }

    let entry_size = |name: &str| -> Result<u64> {
        entry_sizes.get(name).copied().ok_or_else(|| {
            CoreError::InvalidSnapshot(format!("snapshot entry {name:?} is missing"))
        })
    };
    let registers_payload = read_snapshot_entry(
        &mut archive,
        "registers.bin",
        entry_size("registers.bin")?,
        fixed_snapshot_entry_size("registers.bin"),
        fixed_snapshot_entry_size("registers.bin").expect("fixed register size"),
    )?;
    let external_memory = read_snapshot_entry(
        &mut archive,
        "external_ram.bin",
        entry_size("external_ram.bin")?,
        Some(crate::memory::EXTERNAL_SPACE),
        crate::memory::EXTERNAL_SPACE,
    )?;
    let internal_ram = read_snapshot_entry(
        &mut archive,
        "internal_ram.bin",
        entry_size("internal_ram.bin")?,
        Some(INTERNAL_RAM_SIZE),
        INTERNAL_RAM_SIZE,
    )?;
    let imem = read_snapshot_entry(
        &mut archive,
        "imem.bin",
        entry_size("imem.bin")?,
        Some(INTERNAL_SPACE),
        INTERNAL_SPACE,
    )?;
    let lcd_payload = if metadata.lcd.is_some() {
        Some(read_snapshot_entry(
            &mut archive,
            "lcd_vram.bin",
            entry_size("lcd_vram.bin")?,
            Some(metadata.lcd_payload_size),
            metadata.lcd_payload_size,
        )?)
    } else {
        None
    };
    let memory_card = match memory_card_metadata {
        Some(card_metadata) => {
            let capacity = card_metadata.capacity;
            let payload = read_snapshot_entry(
                &mut archive,
                "memory_card.bin",
                entry_size("memory_card.bin")?,
                Some(capacity),
                capacity,
            )?;
            Some(card_metadata.with_payload(payload)?)
        }
        None => None,
    };
    // Reuse the opaque MemoryImage validator so archive parsing cannot drift
    // from the restore API's supported mode/capacity/payload invariants.
    let validation_memory = MemoryImage::new();
    let _ = validation_memory.prepare_memory_card_restore(memory_card.clone())?;

    let registers = unpack_registers(&registers_payload)?;
    validate_snapshot_register_file(&registers)?;
    validate_snapshot_architectural_addresses(&metadata, &registers)?;
    let internal_start = INTERNAL_RAM_START;
    let internal_end = internal_start + INTERNAL_RAM_SIZE;
    if external_memory[internal_start..internal_end] != internal_ram {
        return Err(CoreError::InvalidSnapshot(
            "internal_ram.bin disagrees with external_ram.bin".to_string(),
        ));
    }
    if metadata.pc != registers.get("PC").copied().unwrap_or(0) {
        return Err(CoreError::InvalidSnapshot(
            "snapshot metadata PC disagrees with registers.bin".to_string(),
        ));
    }
    if metadata.temps.len() != crate::NUM_TEMP_REGISTERS as usize {
        return Err(CoreError::InvalidSnapshot(format!(
            "snapshot must contain exactly {} temporary registers",
            crate::NUM_TEMP_REGISTERS
        )));
    }
    for index in 0..crate::NUM_TEMP_REGISTERS {
        let key = index.to_string();
        match metadata.temps.get(&key) {
            Some(value) if *value <= 0xFF_FFFF => {}
            Some(_) => {
                return Err(CoreError::InvalidSnapshot(format!(
                    "snapshot temporary register TEMP{index} exceeds 24 bits"
                )));
            }
            None => {
                return Err(CoreError::InvalidSnapshot(format!(
                    "snapshot is missing temporary register TEMP{index}"
                )));
            }
        }
    }
    if metadata.call_stack.len() != metadata.call_return_widths.len() {
        return Err(CoreError::InvalidSnapshot(
            "snapshot call stack and return-width stack have different lengths".to_string(),
        ));
    }
    if metadata
        .call_stack
        .iter()
        .any(|address| *address > SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX)
        || metadata
            .call_page_stack
            .iter()
            .any(|page| *page > SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX || page & 0xFFFF != 0)
        || metadata
            .call_return_widths
            .iter()
            .any(|width| !matches!(*width, 0 | 16 | 24))
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot contains an invalid call-metrics stack".to_string(),
        ));
    }
    for (name, period) in [
        ("MTI", metadata.timer.mti_period),
        ("STI", metadata.timer.sti_period),
    ] {
        if metadata.timer.enabled && period == 0 {
            return Err(CoreError::InvalidSnapshot(format!(
                "enabled snapshot timer has zero {name} period"
            )));
        }
        if period >= (1u64 << 63) {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot {name} period is ambiguous under wrapping deadline order"
            )));
        }
    }
    for (name, fired, cycle) in [
        (
            "MTI",
            metadata.timer.fired_mti_since_boundary,
            metadata.timer.last_mti_fire_cycle,
        ),
        (
            "STI",
            metadata.timer.fired_sti_since_boundary,
            metadata.timer.last_sti_fire_cycle,
        ),
    ] {
        if fired != cycle.is_some() {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot {name} phase flag/fire cycle disagree"
            )));
        }
    }
    let valid_irq_source = |source: &str| {
        matches!(
            source,
            "RX" | "EX" | "TX" | "ONK" | "KEY" | "STI" | "MTI" | "IR" | "IRQ"
        )
    };
    if metadata
        .interrupts
        .source
        .as_deref()
        .is_some_and(|source| !valid_irq_source(source))
        || metadata
            .interrupts
            .last_fired
            .as_deref()
            .is_some_and(|source| !valid_irq_source(source))
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot contains an unknown interrupt source".to_string(),
        ));
    }
    if metadata.interrupts.pending
        && (metadata.interrupts.source.is_none() || metadata.interrupts.isr == 0)
    {
        return Err(CoreError::InvalidSnapshot(
            "pending snapshot interrupt lacks a source or asserted ISR bit".to_string(),
        ));
    }
    if metadata
        .interrupts
        .stack
        .iter()
        .max()
        .is_some_and(|maximum| metadata.interrupts.next_id <= *maximum)
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot next interrupt id must exceed every active flow id".to_string(),
        ));
    }
    if metadata.interrupts.imr != imem[crate::memory::IMEM_IMR_OFFSET as usize]
        || metadata.interrupts.isr != imem[crate::memory::IMEM_ISR_OFFSET as usize]
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot interrupt mirrors disagree with imem.bin".to_string(),
        ));
    }
    if metadata.interrupts.pending && !metadata.interrupts.in_interrupt {
        let source_mask = match metadata.interrupts.source.as_deref() {
            Some("RX") => Some(0x20),
            Some("EX") => Some(0x40),
            Some("TX") => Some(0x10),
            Some("ONK") => Some(0x08),
            Some("KEY") => Some(0x04),
            Some("STI") => Some(0x02),
            Some("MTI") => Some(0x01),
            _ => None,
        };
        if source_mask.is_some_and(|mask| metadata.interrupts.isr & mask == 0) {
            return Err(CoreError::InvalidSnapshot(
                "snapshot pending interrupt source disagrees with ISR".to_string(),
            ));
        }
    }
    let counts = metadata
        .interrupts
        .irq_counts
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CoreError::InvalidSnapshot("snapshot irq_counts must be an object".to_string())
        })?;
    if counts.len() != 4
        || ["total", "KEY", "MTI", "STI"].iter().any(|name| {
            counts
                .get(*name)
                .and_then(serde_json::Value::as_u64)
                .is_none_or(|value| value > u64::from(u32::MAX))
        })
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot irq_counts must contain exactly bounded total/KEY/MTI/STI values".to_string(),
        ));
    }
    let last_irq = metadata
        .interrupts
        .last_irq
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CoreError::InvalidSnapshot("snapshot last_irq must be an object".to_string())
        })?;
    if last_irq.len() != 3
        || ["src", "pc", "vector"]
            .iter()
            .any(|name| !last_irq.contains_key(*name))
        || ["pc", "vector"].iter().any(|name| {
            last_irq
                .get(*name)
                .filter(|value| !value.is_null())
                .is_some_and(|value| {
                    value.as_u64().is_none_or(|address| {
                        address > u64::from(SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX)
                    })
                })
        })
        || last_irq
            .get("src")
            .filter(|value| !value.is_null())
            .is_some_and(|value| {
                value
                    .as_str()
                    .is_none_or(|source| !valid_irq_source(source))
            })
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot last_irq has an invalid shape or value".to_string(),
        ));
    }
    let watch = metadata
        .interrupts
        .irq_bit_watch
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CoreError::InvalidSnapshot("snapshot irq_bit_watch must be an object".to_string())
        })?;
    if watch.len() != 2 || ["IMR", "ISR"].iter().any(|name| !watch.contains_key(*name)) {
        return Err(CoreError::InvalidSnapshot(
            "snapshot irq_bit_watch must contain exactly IMR and ISR".to_string(),
        ));
    }
    for register in ["IMR", "ISR"] {
        let bits = watch[register].as_object().ok_or_else(|| {
            CoreError::InvalidSnapshot(format!(
                "snapshot irq_bit_watch.{register} must be an object"
            ))
        })?;
        if bits.len() != 8 {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot irq_bit_watch.{register} must contain eight bits"
            )));
        }
        for bit in 0..8u8 {
            let actions = bits
                .get(&bit.to_string())
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| {
                    CoreError::InvalidSnapshot(format!(
                        "snapshot irq_bit_watch.{register} is missing bit {bit}"
                    ))
                })?;
            if actions.len() != 2
                || ["set", "clear"].iter().any(|action| {
                    actions
                        .get(*action)
                        .and_then(serde_json::Value::as_array)
                        .is_none_or(|pcs| {
                            pcs.iter().any(|pc| {
                                pc.as_u64().is_none_or(|pc| {
                                    pc > u64::from(SNAPSHOT_ARCHITECTURAL_ADDRESS_MAX)
                                })
                            })
                        })
                })
            {
                return Err(CoreError::InvalidSnapshot(format!(
                    "snapshot irq_bit_watch.{register}.{bit} has invalid set/clear history"
                )));
            }
        }
    }

    match metadata.keyboard.as_ref() {
        Some(value) => {
            let keyboard_snapshot: crate::keyboard::KeyboardSnapshot =
                serde_json::from_value(value.clone()).map_err(|error| {
                    CoreError::InvalidSnapshot(format!(
                        "invalid keyboard snapshot metadata: {error}"
                    ))
                })?;
            let mut keyboard = crate::keyboard::KeyboardMatrix::new();
            keyboard
                .load_snapshot_state(&keyboard_snapshot)
                .map_err(CoreError::InvalidSnapshot)?;
            if serde_json::to_value(keyboard.snapshot_state())? != *value {
                return Err(CoreError::InvalidSnapshot(
                    "keyboard snapshot is not exactly representable".to_string(),
                ));
            }
            let metrics = metadata
                .kb_metrics
                .as_ref()
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| {
                    CoreError::InvalidSnapshot(
                        "snapshot keyboard metrics must be an object".to_string(),
                    )
                })?;
            let expected_metric_keys: HashSet<&str> = [
                "irq_count",
                "strobe_count",
                "column_hist",
                "last_cols",
                "last_kol",
                "last_koh",
                "kil_reads",
                "kb_irq_enabled",
            ]
            .into_iter()
            .collect();
            if metrics.len() != expected_metric_keys.len()
                || metrics
                    .keys()
                    .any(|name| !expected_metric_keys.contains(name.as_str()))
                || metrics.get("irq_count").and_then(serde_json::Value::as_u64)
                    != Some(u64::from(keyboard_snapshot.irq_count))
                || metrics
                    .get("strobe_count")
                    .and_then(serde_json::Value::as_u64)
                    != Some(u64::from(keyboard_snapshot.strobe_count))
                || metrics.get("last_kol").and_then(serde_json::Value::as_u64)
                    != Some(u64::from(keyboard_snapshot.kol))
                || metrics.get("last_koh").and_then(serde_json::Value::as_u64)
                    != Some(u64::from(keyboard_snapshot.koh))
                || metrics.get("kil_reads").and_then(serde_json::Value::as_u64)
                    != Some(u64::from(keyboard_snapshot.kil_read_count))
                || metrics
                    .get("kb_irq_enabled")
                    .and_then(serde_json::Value::as_bool)
                    != Some(metadata.timer.kb_irq_enabled)
                || metrics.get("column_hist")
                    != Some(&serde_json::json!(keyboard_snapshot.column_histogram))
                || metrics.get("last_cols")
                    != Some(&serde_json::json!(keyboard_snapshot.active_columns))
            {
                return Err(CoreError::InvalidSnapshot(
                    "snapshot keyboard metrics disagree with keyboard state".to_string(),
                ));
            }
        }
        None if metadata.kb_metrics.is_none() => {}
        None => {
            return Err(CoreError::InvalidSnapshot(
                "snapshot has keyboard metrics without keyboard state".to_string(),
            ));
        }
    }
    Ok(SnapshotLoad {
        metadata,
        registers,
        external_memory,
        internal_ram,
        imem,
        lcd_payload,
        memory_card,
    })
}

#[cfg(test)]
mod strict_json_tests {
    use super::*;

    fn valid_registers() -> HashMap<String, u32> {
        SNAPSHOT_REGISTER_LAYOUT
            .iter()
            .map(|(name, _)| ((*name).to_string(), 0))
            .collect()
    }

    #[test]
    fn duplicate_json_members_are_rejected_at_any_depth() {
        let error = parse_duplicate_free_json(br#"{"timer":{"next":1,"next":2}}"#)
            .expect_err("duplicate member must fail");
        assert!(error
            .to_string()
            .contains("duplicate object member \"next\""));
    }

    #[test]
    fn range_objects_reject_unknown_members() {
        serde_json::from_str::<RangeSerde>(r#"{"start":1040384,"size":4096,"ignored":true}"#)
            .expect_err("unknown range member must fail");
    }

    #[test]
    fn v3_metadata_is_rejected_explicitly() {
        let metadata = SnapshotMetadata::default();
        let encoded = encode_snapshot_metadata(&metadata, None).expect("encode v4 metadata");
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        value["version"] = serde_json::json!(3);
        let error = decode_snapshot_metadata(&serde_json::to_vec(&value).unwrap())
            .expect_err("v3 must not be reinterpreted as v4");
        assert!(error
            .to_string()
            .contains("version 3 is not accepted; version 4 is required"));
    }

    #[test]
    fn onk_level_is_a_required_strict_boolean() {
        let encoded =
            encode_snapshot_metadata(&SnapshotMetadata::default(), None).expect("encode metadata");

        let mut missing: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        missing.as_object_mut().unwrap().remove("onk_level");
        let error = decode_snapshot_metadata(&serde_json::to_vec(&missing).unwrap())
            .expect_err("missing held ON-key level must fail closed");
        assert!(error
            .to_string()
            .contains("missing or unexpected top-level fields"));

        let mut wrong_type: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        wrong_type["onk_level"] = serde_json::json!(1);
        let error = decode_snapshot_metadata(&serde_json::to_vec(&wrong_type).unwrap())
            .expect_err("non-boolean held ON-key level must fail closed");
        assert!(error.to_string().contains("invalid snapshot metadata"));
    }

    #[test]
    fn memory_card_metadata_is_typed_and_exact() {
        let snapshot = MemoryCardSnapshot {
            mode: MemoryCardMode::Absent,
            capacity: 8192,
            writable: false,
            payload: vec![0xA5; 8192],
        };
        let card = SnapshotMemoryCardMetadata::from_snapshot(&snapshot).unwrap();
        let encoded = encode_snapshot_metadata(&SnapshotMetadata::default(), Some(&card)).unwrap();
        let (_, decoded) = decode_snapshot_metadata(&encoded).unwrap();
        let restored = decoded
            .expect("typed card metadata")
            .with_payload(snapshot.payload.clone())
            .unwrap();
        assert_eq!(restored, snapshot);

        let mut malformed: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        malformed["memory_card"]["payload_size"] = serde_json::json!(8191);
        decode_snapshot_metadata(&serde_json::to_vec(&malformed).unwrap())
            .expect_err("payload size must equal capacity");
        malformed["memory_card"]["payload_size"] = serde_json::json!(8192);
        malformed["memory_card"]["extra"] = serde_json::json!(true);
        decode_snapshot_metadata(&serde_json::to_vec(&malformed).unwrap())
            .expect_err("unknown card metadata must fail closed");
    }

    #[test]
    fn snapshot_ranges_are_canonical_and_restricted_to_real_spaces() {
        assert_eq!(
            canonical_snapshot_ranges("test", &[(0x100010, 0x10001F), (0x2000, 0x2FFF)]).unwrap(),
            vec![(0x2000, 0x2FFF), (0x100010, 0x10001F)]
        );
        for ranges in [
            vec![(0x2000, 0x2FFF), (0x2F00, 0x3000)],
            vec![(0x0FFFFF, 0x100000)],
            vec![(0x100100, 0x100100)],
            vec![(0x1000FF, 0x100100)],
        ] {
            canonical_snapshot_ranges("test", &ranges)
                .expect_err("overlap, spanning, and unsupported spaces must fail");
        }
        require_canonical_snapshot_ranges("test", &[(0x100000, 0x1000FF), (0x000000, 0x0FFFFF)])
            .expect_err("load requires canonical order");
    }

    #[test]
    fn register_files_fail_closed_before_packing() {
        let registers = valid_registers();
        validate_snapshot_register_file(&registers).unwrap();

        for register in ["PC", "X", "Y", "U", "S"] {
            let mut invalid = registers.clone();
            invalid.insert(register.to_string(), 0x100000);
            validate_snapshot_register_file(&invalid)
                .expect_err("architectural register must be 20-bit");
        }
        let mut invalid = registers.clone();
        invalid.insert("BA".to_string(), 0x10000);
        validate_snapshot_register_file(&invalid).expect_err("BA must fit 16 bits");
        let mut invalid = registers.clone();
        invalid.insert("F".to_string(), 4);
        validate_snapshot_register_file(&invalid).expect_err("F bits 2-7 are quarantined");
        let mut invalid = registers.clone();
        invalid.remove("I");
        validate_snapshot_register_file(&invalid).expect_err("missing register must fail");
        let mut invalid = registers;
        invalid.insert("TEMP0".to_string(), 0);
        validate_snapshot_register_file(&invalid).expect_err("extra register must fail");
    }

    #[test]
    fn every_snapshot_address_class_rejects_0x100000() {
        let registers = valid_registers();

        let metadata = SnapshotMetadata {
            call_stack: vec![0x100000],
            ..SnapshotMetadata::default()
        };
        validate_snapshot_architectural_addresses(&metadata, &registers)
            .expect_err("call stack address must be 20-bit");

        let metadata = SnapshotMetadata {
            call_page_stack: vec![0x100000],
            ..SnapshotMetadata::default()
        };
        validate_snapshot_architectural_addresses(&metadata, &registers)
            .expect_err("call-page address must be 20-bit");

        for field in ["pc", "vector"] {
            let mut metadata = SnapshotMetadata::default();
            metadata.interrupts.last_irq = Some(serde_json::json!({
                "src": null,
                "pc": null,
                "vector": null,
            }));
            metadata.interrupts.last_irq.as_mut().unwrap()[field] = serde_json::json!(0x100000);
            validate_snapshot_architectural_addresses(&metadata, &registers)
                .expect_err("last IRQ address must be 20-bit");
        }

        let mut metadata = SnapshotMetadata::default();
        metadata.interrupts.irq_bit_watch = Some(serde_json::json!({
            "IMR": {"0": {"set": [0x100000], "clear": []}}
        }));
        validate_snapshot_architectural_addresses(&metadata, &registers)
            .expect_err("IRQ history PC must be 20-bit");
    }
}

#[cfg(all(test, feature = "snapshot", not(target_arch = "wasm32")))]
mod archive_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_PATH: AtomicU64 = AtomicU64::new(0);

    struct TestSnapshotPath(PathBuf);

    impl TestSnapshotPath {
        fn new(label: &str) -> Self {
            let nonce = NEXT_TEST_PATH.fetch_add(1, Ordering::Relaxed);
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            Self(std::env::temp_dir().join(format!(
                "sc62015-snapshot-{label}-{}-{stamp}-{nonce}.pcsnap",
                std::process::id()
            )))
        }

        fn as_path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestSnapshotPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn valid_registers() -> HashMap<String, u32> {
        SNAPSHOT_REGISTER_LAYOUT
            .iter()
            .map(|(name, _)| ((*name).to_string(), 0))
            .collect()
    }

    fn empty_irq_watch_bits() -> serde_json::Value {
        serde_json::Value::Object(
            (0..8)
                .map(|bit| (bit.to_string(), serde_json::json!({"set": [], "clear": []})))
                .collect(),
        )
    }

    fn valid_metadata() -> SnapshotMetadata {
        let mut metadata = SnapshotMetadata {
            temps: (0..crate::NUM_TEMP_REGISTERS)
                .map(|index| (index.to_string(), 0))
                .collect(),
            ..SnapshotMetadata::default()
        };
        metadata.interrupts.irq_counts = Some(serde_json::json!({
            "total": 0,
            "KEY": 0,
            "MTI": 0,
            "STI": 0,
        }));
        metadata.interrupts.last_irq = Some(serde_json::json!({
            "src": null,
            "pc": null,
            "vector": null,
        }));
        metadata.interrupts.irq_bit_watch = Some(serde_json::json!({
            "IMR": empty_irq_watch_bits(),
            "ISR": empty_irq_watch_bits(),
        }));
        metadata
    }

    fn archive_entries(path: &Path) -> Vec<(String, Vec<u8>)> {
        let file = File::open(path).expect("open snapshot archive");
        let mut archive = ZipArchive::new(file).expect("parse snapshot archive");
        (0..archive.len())
            .map(|index| {
                let mut entry = archive.by_index(index).expect("open snapshot member");
                let name = entry.name().to_string();
                let mut payload = Vec::new();
                entry
                    .read_to_end(&mut payload)
                    .expect("read snapshot member");
                (name, payload)
            })
            .collect()
    }

    fn write_archive(path: &Path, entries: &[(String, Vec<u8>)]) {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create mutated snapshot archive");
        let mut archive = ZipWriter::new(file);
        for (name, payload) in entries {
            archive
                .start_file(
                    name,
                    FileOptions::default().compression_method(CompressionMethod::Stored),
                )
                .expect("start mutated snapshot member");
            archive
                .write_all(payload)
                .expect("write mutated snapshot member");
        }
        archive.finish().expect("finish mutated snapshot archive");
    }

    fn mutated_archive<F>(source: &Path, label: &str, mutate: F) -> TestSnapshotPath
    where
        F: FnOnce(&mut Vec<(String, Vec<u8>)>),
    {
        let mut entries = archive_entries(source);
        mutate(&mut entries);
        let destination = TestSnapshotPath::new(label);
        write_archive(destination.as_path(), &entries);
        destination
    }

    fn metadata_entry_mut(entries: &mut [(String, Vec<u8>)]) -> &mut Vec<u8> {
        &mut entries
            .iter_mut()
            .find(|(name, _)| name == "snapshot.json")
            .expect("snapshot.json member")
            .1
    }

    #[test]
    fn unconfigured_card_roundtrips_without_a_card_member() {
        let path = TestSnapshotPath::new("unconfigured");
        let mut metadata = valid_metadata();
        metadata.internal_ram = (1, 2);
        metadata.imem = (3, 4);
        let registers = valid_registers();
        let memory = MemoryImage::new();

        save_snapshot(path.as_path(), &metadata, &registers, &memory, None)
            .expect("save unconfigured-card snapshot");
        let loaded = load_snapshot(path.as_path()).expect("load unconfigured-card snapshot");

        assert_eq!(loaded.memory_card, None);
        assert_eq!(
            loaded.metadata.internal_ram,
            (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32)
        );
        assert_eq!(
            loaded.metadata.imem,
            (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32)
        );
        assert!(!archive_entries(path.as_path())
            .iter()
            .any(|(name, _)| name == "memory_card.bin"));
    }

    #[test]
    fn typed_cards_roundtrip_every_capacity_mode_and_write_policy() {
        for (capacity_index, capacity) in SNAPSHOT_MEMORY_CARD_CAPACITIES.into_iter().enumerate() {
            for mode in [MemoryCardMode::Present, MemoryCardMode::Absent] {
                for writable in [false, true] {
                    let path = TestSnapshotPath::new("card-matrix");
                    let payload: Vec<u8> = (0..capacity)
                        .map(|offset| {
                            (offset as u8)
                                .wrapping_mul(31)
                                .wrapping_add(capacity_index as u8)
                        })
                        .collect();
                    let expected = MemoryCardSnapshot {
                        mode,
                        capacity,
                        writable,
                        payload: payload.clone(),
                    };
                    let mut memory = MemoryImage::new();
                    memory
                        .load_memory_card_with_writable(&payload, writable)
                        .expect("install source card");
                    if mode == MemoryCardMode::Absent {
                        memory.set_memory_card_slot_present(false);
                    }

                    save_snapshot(
                        path.as_path(),
                        &valid_metadata(),
                        &valid_registers(),
                        &memory,
                        None,
                    )
                    .expect("save typed-card snapshot");
                    let loaded = load_snapshot(path.as_path()).expect("load typed-card snapshot");
                    assert_eq!(loaded.memory_card, Some(expected.clone()));
                    assert!(archive_entries(path.as_path())
                        .iter()
                        .any(|(name, bytes)| name == "memory_card.bin" && bytes == &payload));

                    let mut restored = MemoryImage::new();
                    let candidate = restored
                        .prepare_memory_card_restore(loaded.memory_card)
                        .expect("prepare typed-card restore");
                    restored
                        .commit_memory_card_restore(candidate)
                        .expect("commit typed-card restore");
                    assert_eq!(restored.memory_card_snapshot().unwrap(), Some(expected));
                }
            }
        }
    }

    #[test]
    fn missing_or_tampered_card_members_and_metadata_are_rejected() {
        let source = TestSnapshotPath::new("card-tamper-source");
        let mut memory = MemoryImage::new();
        memory
            .load_memory_card_with_writable(&vec![0xA5; 8192], false)
            .unwrap();
        save_snapshot(
            source.as_path(),
            &valid_metadata(),
            &valid_registers(),
            &memory,
            None,
        )
        .unwrap();

        let missing = mutated_archive(source.as_path(), "card-missing", |entries| {
            entries.retain(|(name, _)| name != "memory_card.bin");
        });
        let error = load_snapshot(missing.as_path()).expect_err("missing card member must fail");
        assert!(error.to_string().contains("entry set mismatch"));

        let wrong_member_size = mutated_archive(source.as_path(), "card-wrong-size", |entries| {
            entries
                .iter_mut()
                .find(|(name, _)| name == "memory_card.bin")
                .unwrap()
                .1
                .push(0xFF);
        });
        let error = load_snapshot(wrong_member_size.as_path())
            .expect_err("card member with a wrong declared size must fail");
        assert!(error.to_string().contains("exactly 8192 bytes"));

        let metadata_mismatch =
            mutated_archive(source.as_path(), "card-metadata-size", |entries| {
                let payload = metadata_entry_mut(entries);
                let mut metadata: serde_json::Value = serde_json::from_slice(payload).unwrap();
                metadata["memory_card"]["capacity"] = serde_json::json!(16384);
                metadata["memory_card"]["payload_size"] = serde_json::json!(16384);
                *payload = serde_json::to_vec(&metadata).unwrap();
            });
        let error = load_snapshot(metadata_mismatch.as_path())
            .expect_err("card metadata/member length mismatch must fail");
        assert!(error.to_string().contains("exactly 16384 bytes"));

        let removed_metadata = mutated_archive(source.as_path(), "card-metadata-null", |entries| {
            let payload = metadata_entry_mut(entries);
            let mut metadata: serde_json::Value = serde_json::from_slice(payload).unwrap();
            metadata["memory_card"] = serde_json::Value::Null;
            *payload = serde_json::to_vec(&metadata).unwrap();
        });
        let error = load_snapshot(removed_metadata.as_path())
            .expect_err("card member without typed metadata must fail");
        assert!(error.to_string().contains("entry set mismatch"));
    }

    #[test]
    fn genuine_v3_archive_is_rejected_with_an_explicit_version_error() {
        let source = TestSnapshotPath::new("v3-source");
        save_snapshot(
            source.as_path(),
            &valid_metadata(),
            &valid_registers(),
            &MemoryImage::new(),
            None,
        )
        .unwrap();
        let v3 = mutated_archive(source.as_path(), "v3", |entries| {
            let payload = metadata_entry_mut(entries);
            let mut metadata: serde_json::Value = serde_json::from_slice(payload).unwrap();
            metadata["version"] = serde_json::json!(3);
            metadata.as_object_mut().unwrap().remove("memory_card");
            *payload = serde_json::to_vec(&metadata).unwrap();
        });

        let error = load_snapshot(v3.as_path()).expect_err("v3 archive must fail explicitly");
        assert!(error
            .to_string()
            .contains("version 3 is not accepted; version 4 is required"));
    }

    #[test]
    fn invalid_live_addresses_fail_without_replacing_an_existing_snapshot() {
        let destination = TestSnapshotPath::new("preserve-existing");
        let sentinel = b"previous-valid-destination";
        fs::write(destination.as_path(), sentinel).unwrap();
        let mut metadata = valid_metadata();
        metadata.call_stack = vec![0x100000];
        metadata.call_return_widths = vec![24];

        let error = save_snapshot(
            destination.as_path(),
            &metadata,
            &valid_registers(),
            &MemoryImage::new(),
            None,
        )
        .expect_err("invalid live address must fail before replacement");
        assert!(error.to_string().contains("call stack"));
        assert_eq!(fs::read(destination.as_path()).unwrap(), sentinel);
    }
}
