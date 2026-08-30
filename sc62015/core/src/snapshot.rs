// PY_SOURCE: pce500/emulator.py:save_snapshot
// PY_SOURCE: pce500/emulator.py:load_snapshot

#[cfg(all(feature = "snapshot", not(target_arch = "wasm32")))]
use crate::memory::{
    MemoryImage, INTERNAL_MEMORY_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_START, INTERNAL_SPACE,
};
use crate::{CoreError, Result, SnapshotMetadata};
use serde::de::Error as _;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
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
pub const SNAPSHOT_VERSION: u32 = 3;
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
pub fn save_snapshot(
    path: &Path,
    metadata: &SnapshotMetadata,
    registers: &HashMap<String, u32>,
    memory: &MemoryImage,
    lcd_payload: Option<&[u8]>,
) -> Result<()> {
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

    let write_result = (|| -> Result<()> {
        let mut zip = ZipWriter::new(file);
        let options = FileOptions::default().compression_method(CompressionMethod::Deflated);

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

        if meta.lcd.is_some() {
            zip.start_file("lcd_vram.bin", options)?;
            if let Some(buf) = lcd_payload {
                zip.write_all(buf)?;
            }
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

    let mut entries: HashMap<String, Vec<u8>> = HashMap::new();
    let mut names = HashSet::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_string();
        if name.ends_with('/') {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot contains unexpected directory {name:?}"
            )));
        }
        if !names.insert(name.clone()) {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot contains duplicate entry {name:?}"
            )));
        }
        if entry.size() > (crate::memory::EXTERNAL_SPACE as u64 + 0x10_0000) {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot entry {name:?} is unreasonably large"
            )));
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        entries.insert(name, bytes);
    }

    let required_entry = |name: &str| -> Result<&[u8]> {
        entries.get(name).map(Vec::as_slice).ok_or_else(|| {
            CoreError::InvalidSnapshot(format!("snapshot entry {name:?} is missing"))
        })
    };

    let metadata = {
        let raw = parse_duplicate_free_json(required_entry("snapshot.json")?)?;
        let object = raw.as_object().ok_or_else(|| {
            CoreError::InvalidSnapshot("snapshot.json must contain an object".to_string())
        })?;
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
        let metadata: SnapshotMetadata = serde_json::from_value(raw)?;
        if metadata.magic != SNAPSHOT_MAGIC || metadata.version != SNAPSHOT_VERSION {
            return Err(CoreError::InvalidSnapshot(
                "snapshot magic/version mismatch".to_string(),
            ));
        }
        if !matches!(
            metadata.backend.as_str(),
            "core" | "rust" | "llama" | "python"
        ) {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot backend {:?} is unsupported",
                metadata.backend
            )));
        }
        metadata
    };

    let registers = unpack_registers(required_entry("registers.bin")?)?;
    let external_memory = required_entry("external_ram.bin")?.to_vec();
    let internal_ram = required_entry("internal_ram.bin")?.to_vec();
    let imem = required_entry("imem.bin")?.to_vec();

    if metadata.memory_image_size != crate::memory::EXTERNAL_SPACE
        || external_memory.len() != crate::memory::EXTERNAL_SPACE
    {
        return Err(CoreError::InvalidSnapshot(format!(
            "external memory must contain exactly {} bytes",
            crate::memory::EXTERNAL_SPACE
        )));
    }
    if metadata.internal_ram != (INTERNAL_RAM_START as u32, INTERNAL_RAM_SIZE as u32)
        || internal_ram.len() != INTERNAL_RAM_SIZE
    {
        return Err(CoreError::InvalidSnapshot(format!(
            "internal RAM must be ({:#X}, {}) with exactly {} bytes",
            INTERNAL_RAM_START, INTERNAL_RAM_SIZE, INTERNAL_RAM_SIZE
        )));
    }
    if metadata.imem != (INTERNAL_MEMORY_START, INTERNAL_SPACE as u32)
        || imem.len() != INTERNAL_SPACE
    {
        return Err(CoreError::InvalidSnapshot(format!(
            "IMEM must be ({INTERNAL_MEMORY_START:#X}, {INTERNAL_SPACE}) with exactly {INTERNAL_SPACE} bytes"
        )));
    }
    let internal_start = INTERNAL_RAM_START;
    let internal_end = internal_start + INTERNAL_RAM_SIZE;
    if external_memory[internal_start..internal_end] != internal_ram {
        return Err(CoreError::InvalidSnapshot(
            "internal_ram.bin disagrees with external_ram.bin".to_string(),
        ));
    }
    for name in ["PC", "X", "Y", "U", "S"] {
        if registers.get(name).copied().unwrap_or(0) > crate::memory::ADDRESS_MASK {
            return Err(CoreError::InvalidSnapshot(format!(
                "snapshot register {name} exceeds the modeled 20-bit address space"
            )));
        }
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
        .any(|address| *address > crate::memory::ADDRESS_MASK)
        || metadata
            .call_page_stack
            .iter()
            .any(|page| *page > crate::memory::ADDRESS_MASK || page & 0xFFFF != 0)
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
                    value
                        .as_u64()
                        .is_none_or(|address| address > u64::from(crate::memory::ADDRESS_MASK))
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
                                pc.as_u64()
                                    .is_none_or(|pc| pc > u64::from(crate::memory::ADDRESS_MASK))
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
    for (label, ranges) in [
        ("fallback", &metadata.fallback_ranges),
        ("readonly", &metadata.readonly_ranges),
    ] {
        for (start, end) in ranges {
            if start > end || *end > crate::memory::ADDRESS_MASK {
                return Err(CoreError::InvalidSnapshot(format!(
                    "snapshot {label} range 0x{start:X}..0x{end:X} is invalid"
                )));
            }
        }
    }

    let lcd_payload = if metadata.lcd.is_some() {
        let payload = required_entry("lcd_vram.bin")?.to_vec();
        if payload.len() != metadata.lcd_payload_size {
            return Err(CoreError::InvalidSnapshot(
                "LCD payload length disagrees with snapshot metadata".to_string(),
            ));
        }
        Some(payload)
    } else {
        if metadata.lcd_payload_size != 0 || entries.contains_key("lcd_vram.bin") {
            return Err(CoreError::InvalidSnapshot(
                "snapshot has an LCD payload without LCD metadata".to_string(),
            ));
        }
        None
    };

    let expected_names: HashSet<&str> = [
        "snapshot.json",
        "registers.bin",
        "external_ram.bin",
        "internal_ram.bin",
        "imem.bin",
    ]
    .into_iter()
    .chain(metadata.lcd.is_some().then_some("lcd_vram.bin"))
    .collect();
    if names.len() != expected_names.len()
        || names
            .iter()
            .any(|name| !expected_names.contains(name.as_str()))
    {
        return Err(CoreError::InvalidSnapshot(
            "snapshot archive contains unexpected entries".to_string(),
        ));
    }

    Ok(SnapshotLoad {
        metadata,
        registers,
        external_memory,
        internal_ram,
        imem,
        lcd_payload,
    })
}

#[cfg(test)]
mod strict_json_tests {
    use super::*;

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
}
