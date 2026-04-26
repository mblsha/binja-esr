// PY_SOURCE: iq7000/emulator.py:IQ7000Emulator (placeholder)

use crate::memory::MemoryImage;
use crate::{CoreRuntime, Result};

pub const ROM_WINDOW_START: usize = 0x0C0000;
pub const ROM_WINDOW_LEN: usize = 0x40000;
pub const ROM_READONLY_START: u32 = ROM_WINDOW_START as u32;
pub const ROM_READONLY_END: u32 = (ROM_WINDOW_START + ROM_WINDOW_LEN - 1) as u32;
pub const CLOCK_WORKSPACE_START: u32 = 0x01FD20;
pub const CLOCK_WORKSPACE_LEN: usize = 13;
pub const CLOCK_INITIALIZED_FLAG: u32 = 0x01FE72;
pub const IMEM_EOL_OFFSET: u32 = 0xF3;
pub const IMEM_EIL_OFFSET: u32 = 0xF5;
const EOL_STROBE: u8 = 0x01;
const EOL_OUT_DATA: u8 = 0x02;
const EIL_IN_DATA: u8 = 0x08;
const EIL_READY: u8 = 0x10;
const RTC_COMMAND_CURRENT_DATETIME: u8 = 0xF4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Iq7000ContractStatus {
    Confirmed,
    RuntimeCovered,
    StructurallyMapped,
    RomAuthoritative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000IocsDeviceRecord {
    pub device: u8,
    pub attr: u8,
    pub handler: u32,
    pub name: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_IOCS_DEVICE_RECORDS: &[Iq7000IocsDeviceRecord] = &[
    Iq7000IocsDeviceRecord {
        device: 0x00,
        attr: 0xA2,
        handler: 0x00F04A5,
        name: "STDO:SCRN:",
        status: Iq7000ContractStatus::RuntimeCovered,
    },
    Iq7000IocsDeviceRecord {
        device: 0x01,
        attr: 0xA1,
        handler: 0x00F360A,
        name: "STDI:KYBD:",
        status: Iq7000ContractStatus::RuntimeCovered,
    },
    Iq7000IocsDeviceRecord {
        device: 0x02,
        attr: 0xD3,
        handler: 0x00FA5FE,
        name: "COM:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x03,
        attr: 0xA2,
        handler: 0x00FB439,
        name: "STDL:PRN:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x04,
        attr: 0xD7,
        handler: 0x00FAEC0,
        name: "CAS:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x06,
        attr: 0xC3,
        handler: 0x00F7C83,
        name: "S1:S2:S3:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x08,
        attr: 0x00,
        handler: 0x00F5702,
        name: "SYSTM:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x0A,
        attr: 0x00,
        handler: 0x00F31EF,
        name: "RTC:",
        status: Iq7000ContractStatus::RuntimeCovered,
    },
    Iq7000IocsDeviceRecord {
        device: 0x0B,
        attr: 0xC0,
        handler: 0x00FBC3D,
        name: "PANET:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x0C,
        attr: 0x53,
        handler: 0x00FBC00,
        name: "PACOM:",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x0D,
        attr: 0x00,
        handler: 0x00F6777,
        name: "dev0D:System/UserDic/Alarm",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000IocsDeviceRecord {
        device: 0x0E,
        attr: 0x00,
        handler: 0x00F5C88,
        name: "dev0E:SCR handshake",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000HeaderRecord {
    pub address: u32,
    pub name: &'static str,
    pub attr: Option<u8>,
    pub entry_or_walker: Option<u32>,
    pub trailer_ptr: Option<u32>,
    pub trailer_index: Option<u8>,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_HEADER_RECORDS: &[Iq7000HeaderRecord] = &[
    Iq7000HeaderRecord {
        address: 0x051720,
        name: "STDO:SCRN:",
        attr: Some(0xA2),
        entry_or_walker: Some(0x0602EC),
        trailer_ptr: Some(0x05174F),
        trailer_index: Some(0x01),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x051740,
        name: "STDI:KYBD:",
        attr: Some(0xA1),
        entry_or_walker: None,
        trailer_ptr: Some(0x05175C),
        trailer_index: Some(0x02),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x051750,
        name: "COM:",
        attr: Some(0xD3),
        entry_or_walker: None,
        trailer_ptr: Some(0x05176E),
        trailer_index: Some(0x03),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x051760,
        name: "STDL:PRN:",
        attr: Some(0xA2),
        entry_or_walker: None,
        trailer_ptr: Some(0x05177B),
        trailer_index: Some(0x04),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x051770,
        name: "CAS:",
        attr: Some(0xD7),
        entry_or_walker: None,
        trailer_ptr: Some(0x051790),
        trailer_index: Some(0x06),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x051780,
        name: "S1:S2:S3:S4:",
        attr: Some(0xC3),
        entry_or_walker: None,
        trailer_ptr: Some(0x05179F),
        trailer_index: Some(0x05),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x051790,
        name: "E:F:G:",
        attr: Some(0x83),
        entry_or_walker: None,
        trailer_ptr: Some(0x0517AE),
        trailer_index: Some(0x0C),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x0517A0,
        name: "PACOM:",
        attr: Some(0xD7),
        entry_or_walker: None,
        trailer_ptr: Some(0x0517B7),
        trailer_index: Some(0x09),
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000HeaderRecord {
        address: 0x0517B0,
        name: "SYSTEM block",
        attr: None,
        entry_or_walker: Some(0x048C31),
        trailer_ptr: None,
        trailer_index: None,
        status: Iq7000ContractStatus::StructurallyMapped,
    },
];

pub const IQ7000_STDO_WALKER_BLOB_START: u32 = 0x060200;
pub const IQ7000_STDO_WALKER_ENTRY: u32 = 0x0602EC;
pub const IQ7000_STDO_WALKER_DEVICE_STRING_ADDR: u32 = 0x060980;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000RomField {
    pub address: u32,
    pub width: u8,
    pub value: u64,
    pub name: &'static str,
    pub role: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_STDO_HEADER_FIELDS: &[Iq7000RomField] = &[
    Iq7000RomField {
        address: 0x051720,
        width: 6,
        value: 0x47303A3B3D42,
        name: "stdo_header_preamble_tag_seed",
        role: "raw STDO preamble bytes consumed by the colocated bootstrap/walker path",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000RomField {
        address: 0x051726,
        width: 3,
        value: IQ7000_STDO_WALKER_ENTRY as u64,
        name: "stdo_header_walker_entry",
        role: "24-bit pointer into the 0x060200 walker blob",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000RomField {
        address: 0x051729,
        width: 3,
        value: 0x05173C,
        name: "stdo_header_next_record_ptr",
        role: "24-bit pointer to the next header record boundary",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000RomField {
        address: 0x05172C,
        width: 1,
        value: 0x00,
        name: "stdo_header_zero_flag",
        role: "flag byte; no nonzero use identified in decoded walker paths",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000RomField {
        address: 0x05172D,
        width: 1,
        value: 0xA2,
        name: "stdo_header_iocs_attr",
        role: "matches the STDO IOCS blob attribute",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000RomField {
        address: 0x05172E,
        width: 1,
        value: 0xDF,
        name: "stdo_header_mode_mask",
        role: "mode/mask byte adjacent to attr; exact bit semantics pending",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000RomField {
        address: 0x05172F,
        width: 1,
        value: 0x0D,
        name: "stdo_header_terminator_or_selector",
        role: "trailing selector/terminator byte in long STDO preamble",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000RomField {
        address: 0x051730,
        width: 12,
        value: 0,
        name: "stdo_header_name_chunk",
        role: "length/name bytes for STDO:SCRN: ending at the NUL before trailer",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000RomField {
        address: 0x05173C,
        width: 4,
        value: 0x0105174F,
        name: "stdo_header_trailer",
        role: "24-bit intra-header pointer 0x05174F plus trailer index 0x01",
        status: Iq7000ContractStatus::Confirmed,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000WalkerStep {
    pub address: u32,
    pub role: &'static str,
    pub consumed_state: &'static str,
    pub produced_state: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_STDO_WALKER_STEPS: &[Iq7000WalkerStep] = &[
    Iq7000WalkerStep {
        address: 0x060204,
        role: "copy up to six bytes from a device tag, stopping at ':' and padding with space",
        consumed_state: "Y source tag, X destination tag buffer",
        produced_state: "six-byte display/device tag field",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000WalkerStep {
        address: 0x060225,
        role: "walk linked records through [X] until [X+3] matches D6",
        consumed_state: "record list pointer in X, selector in D6",
        produced_state: "carry-coded match result",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000WalkerStep {
        address: 0x06026A,
        role: "walk records rooted at 0x3FD4A and emit IL=0x40 through the walker callback",
        consumed_state: "3FD4A linked-list root, record tag at X+5",
        produced_state: "callback-visible copied tag and advanced X=[X]",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000WalkerStep {
        address: 0x0602B1,
        role: "scan 0x1F-byte records behind (PX+0A) until an 0xFF marker",
        consumed_state: "(PX+0A) base and +0x0C length",
        produced_state: "D8 record index/count",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000WalkerStep {
        address: 0x0604DA,
        role: "store a record index into the 0x3FD2F table",
        consumed_state: "A table index, D8 record index",
        produced_state: "3FD2F[A]=D8",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000WalkerStep {
        address: 0x060892,
        role: "initialize walker header tables and emit stdo:stdi:stdl: tags",
        consumed_state: "walker scratch at 0x3FD2F..0x3FD45 and string at 0x06098D",
        produced_state: "cleared table, FFFF terminator, initialized tag set",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000WalkerStep {
        address: 0x0609A5,
        role: "compare 8.3-style names with ? wildcard and dot separator handling",
        consumed_state: "X/Y name buffers",
        produced_state: "carry-coded compare result",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000WalkerStep {
        address: 0x060A2B,
        role: "format a four-byte name, '.', then two-byte suffix",
        consumed_state: "source bytes and output pointer",
        produced_state: "formatted dotted label",
        status: Iq7000ContractStatus::Confirmed,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000KeyboardSelector {
    pub raw_code: u8,
    pub translated_code: u8,
    pub label: &'static str,
}

pub const IQ7000_APP_SELECTOR_MAP: &[Iq7000KeyboardSelector] = &[
    Iq7000KeyboardSelector {
        raw_code: 0x18,
        translated_code: 0x51,
        label: "CALENDAR",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x19,
        translated_code: 0x50,
        label: "SCHEDULE",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x10,
        translated_code: 0x52,
        label: "TEL",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x08,
        translated_code: 0x53,
        label: "MEMO",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x00,
        translated_code: 0x54,
        label: "CALC",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x1A,
        translated_code: 0x55,
        label: "CARD/SAMPLES",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x11,
        translated_code: 0x56,
        label: "WORLD",
    },
    Iq7000KeyboardSelector {
        raw_code: 0x09,
        translated_code: 0x57,
        label: "HOME",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000KeyboardActionTarget {
    pub address: u32,
    pub role: &'static str,
}

pub const IQ7000_KEYBOARD_ACTION_TARGETS: &[Iq7000KeyboardActionTarget] = &[
    Iq7000KeyboardActionTarget {
        address: 0x00F523E,
        role: "clear keycode bit 0x04",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F5247,
        role: "ack alarm IRQ then run RTC alarm worker when pending",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F525D,
        role: "RETF stub",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F563C,
        role: "serial RX interrupt handler",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F537C,
        role: "RETF stub before reset/rescan helper at 0xF537D",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F525E,
        role: "timer scan tick: increments 1FD67 and launches the matrix scanner",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F527F,
        role: "serial TX interrupt action: services COM IL=0x46 or defers with 1FDC5 bit 0x40",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F55CF,
        role: "queue translated key event into the 16-byte IOCS ring buffer",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F5623,
        role: "test keyboard ring-full flag at IOCS workspace offset +4 bit 0x80",
    },
    Iq7000KeyboardActionTarget {
        address: 0x00F562F,
        role: "compute keyboard ring head-tail delta from IOCS workspace offsets +4/+5",
    },
];

pub const IQ7000_KEYBOARD_SCAN_CHAIN: &[u32] = &[
    0x00F5394, 0x00F54D9, 0x00F537D, 0x00F5510, 0x00F5588, 0x00F55AE,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000KeyboardMatrixContract {
    pub address: u32,
    pub name: &'static str,
    pub role: &'static str,
    pub f0_effect: &'static str,
    pub f1_effect: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_KEYBOARD_MATRIX_CONTRACTS: &[Iq7000KeyboardMatrixContract] = &[
    Iq7000KeyboardMatrixContract {
        address: 0x00F3713,
        name: "empty_ring_halt_wait",
        role:
            "when no queued key is available, drives F0/F1 high and HALTs for the next scan event",
        f0_effect: "F0=0xFF",
        f1_effect: "F1|=0x3F",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000KeyboardMatrixContract {
        address: 0x00F537D,
        name: "matrix_line_reset",
        role: "clears keyboard output lines and pulses LCC until KIL settles",
        f0_effect: "F0=0x00",
        f1_effect: "F1&=0xC0",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000KeyboardMatrixContract {
        address: 0x00F54D9,
        name: "scan_prime",
        role: "primes SCR/LCC and raises the high output-line enable bit before a sweep",
        f0_effect: "unchanged",
        f1_effect: "F1|=0x20",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000KeyboardMatrixContract {
        address: 0x00F5510,
        name: "sweep_seed",
        role: "seeds the sweep mask/counter before falling into the bit scanner",
        f0_effect: "mask seeded indirectly through 0x100001",
        f1_effect: "column mask seeded indirectly through 0x100002",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000KeyboardMatrixContract {
        address: 0x00F5548,
        name: "row_column_probe",
        role: "sets KOL from caller arg, ORs a 5-bit mask into KOH, reads KIL through 1FDAB[row]",
        f0_effect: "F0 receives the active output-line mask",
        f1_effect: "F1 receives the high column mask ORed into KOH",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000KeyboardMatrixContract {
        address: 0x00F55AE,
        name: "row_column_translate",
        role: "computes idx=col*8+row and translates via 1FDC5/1FDCD tables",
        f0_effect: "not modified",
        f1_effect: "not modified",
        status: Iq7000ContractStatus::Confirmed,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000TranslationTableAnchor {
    pub address: u32,
    pub role: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_KEYBOARD_TRANSLATION_TABLES: &[Iq7000TranslationTableAnchor] = &[
    Iq7000TranslationTableAnchor {
        address: 0x00F4019,
        role: "8x8 unshifted/runtime keycode grid used through 1FDC5",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000TranslationTableAnchor {
        address: 0x0050DB7,
        role: "8x8 shifted/ASCII-like keycode grid used through 1FDCD",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000TranslationTableAnchor {
        address: 0x001FDAB,
        role: "row mask table consumed by the physical KIL sampler",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000TranslationTableAnchor {
        address: 0x001FDC5,
        role: "primary translation-table pointer/flag byte; bit 0x40 gates scan dequeue",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000TranslationTableAnchor {
        address: 0x001FDCD,
        role: "secondary translation table used when primary byte is >=0x60",
        status: Iq7000ContractStatus::Confirmed,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000CommandTarget {
    pub index_or_command: u8,
    pub target: u32,
    pub role: &'static str,
}

pub const IQ7000_COM_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x20,
        target: 0x00FA7E3,
        role: "rearm line and flow",
    },
    Iq7000CommandTarget {
        index_or_command: 0x21,
        target: 0x00FA7E3,
        role: "rearm line and flow",
    },
    Iq7000CommandTarget {
        index_or_command: 0x22,
        target: 0x00FA81A,
        role: "line close worker",
    },
    Iq7000CommandTarget {
        index_or_command: 0x23,
        target: 0x00FA74D,
        role: "read bytes",
    },
    Iq7000CommandTarget {
        index_or_command: 0x24,
        target: 0x00FA6A5,
        role: "load line/config block",
    },
    Iq7000CommandTarget {
        index_or_command: 0x25,
        target: 0x00FA6EC,
        role: "sync countdown and service",
    },
    Iq7000CommandTarget {
        index_or_command: 0x26,
        target: 0x00FA67A,
        role: "send byte with line config",
    },
    Iq7000CommandTarget {
        index_or_command: 0x27,
        target: 0x00FA670,
        role: "read/compare bytes frontend",
    },
    Iq7000CommandTarget {
        index_or_command: 0x28,
        target: 0x00FA7A2,
        role: "TX/RX gate",
    },
    Iq7000CommandTarget {
        index_or_command: 0x29,
        target: 0x00FA861,
        role: "stub error A=0x0A",
    },
    Iq7000CommandTarget {
        index_or_command: 0x2A,
        target: 0x00FA845,
        role: "prepare DD/DB/DA pointers",
    },
    Iq7000CommandTarget {
        index_or_command: 0x2F,
        target: 0x00FAB65,
        role: "reset COM state and rings",
    },
    Iq7000CommandTarget {
        index_or_command: 0x31,
        target: 0x00FAC34,
        role: "send control byte",
    },
    Iq7000CommandTarget {
        index_or_command: 0x32,
        target: 0x00FA8E1,
        role: "TX state machine",
    },
    Iq7000CommandTarget {
        index_or_command: 0x34,
        target: 0x00FAD7C,
        role: "set EOH bit 0x20",
    },
    Iq7000CommandTarget {
        index_or_command: 0x35,
        target: 0x00FAD81,
        role: "clear EOH bit 0x20",
    },
    Iq7000CommandTarget {
        index_or_command: 0x36,
        target: 0x00FAD86,
        role: "set KOH bit 0x40",
    },
    Iq7000CommandTarget {
        index_or_command: 0x37,
        target: 0x00FAD8B,
        role: "clear KOH bit 0x40",
    },
    Iq7000CommandTarget {
        index_or_command: 0x38,
        target: 0x00FAD90,
        role: "set KOH bit 0x80",
    },
    Iq7000CommandTarget {
        index_or_command: 0x39,
        target: 0x00FAD95,
        role: "clear KOH bit 0x80",
    },
    Iq7000CommandTarget {
        index_or_command: 0x3A,
        target: 0x00FAD9A,
        role: "test EIH bit 0x10",
    },
    Iq7000CommandTarget {
        index_or_command: 0x3B,
        target: 0x00FADA4,
        role: "test EIH bit 0x04",
    },
    Iq7000CommandTarget {
        index_or_command: 0x3C,
        target: 0x00FADC3,
        role: "write UCR and OR line flags",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4F,
        target: 0x00FADDD,
        role: "clear COM ring head/tail",
    },
    Iq7000CommandTarget {
        index_or_command: 0x51,
        target: 0x00FADED,
        role: "read line status",
    },
    Iq7000CommandTarget {
        index_or_command: 0x54,
        target: 0x00FAE0A,
        role: "set lines by mask then read status",
    },
    Iq7000CommandTarget {
        index_or_command: 0x55,
        target: 0x00FAE2A,
        role: "clear lines by mask then read status",
    },
    Iq7000CommandTarget {
        index_or_command: 0x59,
        target: 0x00FAB9B,
        role: "handshake UCR loop",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5F,
        target: 0x00FAE76,
        role: "flow-control gate",
    },
];

pub const IQ7000_PRN_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x08,
        target: 0x00FB5DC,
        role: "clear printer buffer state",
    },
    Iq7000CommandTarget {
        index_or_command: 0x09,
        target: 0x00FB600,
        role: "status check then clear printer buffer",
    },
    Iq7000CommandTarget {
        index_or_command: 0x0A,
        target: 0x00FB4E6,
        role: "fixed error A=1",
    },
    Iq7000CommandTarget {
        index_or_command: 0x0B,
        target: 0x00FB4FC,
        role: "put character with CR/LF/TAB handling",
    },
    Iq7000CommandTarget {
        index_or_command: 0x0C,
        target: 0x00FB4E6,
        role: "fixed error A=1",
    },
    Iq7000CommandTarget {
        index_or_command: 0x0D,
        target: 0x00FB605,
        role: "write caller block through put-character loop",
    },
    Iq7000CommandTarget {
        index_or_command: 0x0E,
        target: 0x00FB4E6,
        role: "fixed error A=1",
    },
    Iq7000CommandTarget {
        index_or_command: 0x3F,
        target: 0x00FB9ED,
        role: "seed default printer/PANET config bytes",
    },
    Iq7000CommandTarget {
        index_or_command: 0x40,
        target: 0x00FB4EA,
        role: "config dispatch for mode A=0/1/2",
    },
    Iq7000CommandTarget {
        index_or_command: 0x41,
        target: 0x00FB6FD,
        role: "printing UI and bit-banged send workflow",
    },
    Iq7000CommandTarget {
        index_or_command: 0x42,
        target: 0x00FB95D,
        role: "packet transfer variant with shared prologue",
    },
    Iq7000CommandTarget {
        index_or_command: 0x43,
        target: 0x00FB953,
        role: "packet transfer variant with leading mode flag",
    },
    Iq7000CommandTarget {
        index_or_command: 0x44,
        target: 0x00FB5DC,
        role: "clear printer buffer state alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x45,
        target: 0x00FB600,
        role: "status check then clear alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x46,
        target: 0x00FB605,
        role: "write block alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x47,
        target: 0x00FB4FC,
        role: "put character alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x48,
        target: 0x00FBBC3,
        role: "mask EOH link bits to 0x3F",
    },
    Iq7000CommandTarget {
        index_or_command: 0x49,
        target: 0x00FBBAF,
        role: "set EOH bit 7",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4A,
        target: 0x00FBBB4,
        role: "clear EOH bit 7",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4B,
        target: 0x00FBBB9,
        role: "poll EIH bit 6",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4C,
        target: 0x00FB828,
        role: "send C0 handshake and expect link ACK",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4D,
        target: 0x00FBA15,
        role: "send AA handshake and read two response bytes",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4E,
        target: 0x00FBA38,
        role: "send A5 handshake and read two response bytes",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4F,
        target: 0x00FBA63,
        role: "wait for EIH.6 high with timeout",
    },
    Iq7000CommandTarget {
        index_or_command: 0x50,
        target: 0x00FBA81,
        role: "alternate wait for EIH.6 high with timeout",
    },
    Iq7000CommandTarget {
        index_or_command: 0x51,
        target: 0x00FB99A,
        role: "expect FA acknowledgement",
    },
    Iq7000CommandTarget {
        index_or_command: 0x52,
        target: 0x00FBB7C,
        role: "send byte with extra delay",
    },
    Iq7000CommandTarget {
        index_or_command: 0x53,
        target: 0x00FBB2F,
        role: "send byte bit-by-bit",
    },
    Iq7000CommandTarget {
        index_or_command: 0x54,
        target: 0x00FBB86,
        role: "send byte and add checksum",
    },
    Iq7000CommandTarget {
        index_or_command: 0x55,
        target: 0x00FBB91,
        role: "sync-send byte and add checksum",
    },
    Iq7000CommandTarget {
        index_or_command: 0x56,
        target: 0x00FBB9C,
        role: "sync-send byte",
    },
    Iq7000CommandTarget {
        index_or_command: 0x57,
        target: 0x00FBAA2,
        role: "receive byte bit-by-bit",
    },
    Iq7000CommandTarget {
        index_or_command: 0x58,
        target: 0x00FBAE9,
        role: "receive byte and add checksum",
    },
    Iq7000CommandTarget {
        index_or_command: 0x59,
        target: 0x00FBAF7,
        role: "sync-receive byte and add checksum",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5A,
        target: 0x00FBB05,
        role: "link RX wait loop",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5B,
        target: 0x00FBB20,
        role: "sync-receive byte",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5C,
        target: 0x00FBBCD,
        role: "shared ready/defer gate",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5D,
        target: 0x00FBBEB,
        role: "shared bit-bang prologue",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5E,
        target: 0x00FBBE0,
        role: "shared bit-bang epilogue",
    },
];

pub const IQ7000_PACOM_PANET_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x00,
        target: 0x00FBC3B,
        role: "no-op success",
    },
    Iq7000CommandTarget {
        index_or_command: 0x01,
        target: 0x00FBC3B,
        role: "no-op success",
    },
    Iq7000CommandTarget {
        index_or_command: 0x02,
        target: 0x00FBCC0,
        role: "prepare PACOM record/buffer backend",
    },
    Iq7000CommandTarget {
        index_or_command: 0x03,
        target: 0x00FB600,
        role: "printer-style status check then clear",
    },
    Iq7000CommandTarget {
        index_or_command: 0x04,
        target: 0x00FBD54,
        role: "receive frame, validate checksum, send ACK/NAK",
    },
    Iq7000CommandTarget {
        index_or_command: 0x05,
        target: 0x00FBCF9,
        role: "send frame with checksum and wait for ACK",
    },
    Iq7000CommandTarget {
        index_or_command: 0x06,
        target: 0x00FBC3F,
        role: "receive A5 handshake and reply 80 00",
    },
    Iq7000CommandTarget {
        index_or_command: 0x07,
        target: 0x00FBC7A,
        role: "send A5 handshake and parse response",
    },
    Iq7000CommandTarget {
        index_or_command: 0x08,
        target: 0x00FBD4B,
        role: "receive frame with first-byte guard",
    },
    Iq7000CommandTarget {
        index_or_command: 0x09,
        target: 0x00F069F,
        role: "stub return",
    },
];

pub const IQ7000_RAM_DISK_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x3F,
        target: 0x00F7D2D,
        role: "init/reset",
    },
    Iq7000CommandTarget {
        index_or_command: 0x40,
        target: 0x00F7D88,
        role: "reset/media select",
    },
    Iq7000CommandTarget {
        index_or_command: 0x41,
        target: 0x00F80AE,
        role: "workspace select/preflight",
    },
    Iq7000CommandTarget {
        index_or_command: 0x42,
        target: 0x00F80E3,
        role: "IMR-masked workspace resize/copy",
    },
    Iq7000CommandTarget {
        index_or_command: 0x43,
        target: 0x00F818E,
        role: "retry wrapper around resize/copy",
    },
    Iq7000CommandTarget {
        index_or_command: 0x44,
        target: 0x00F81AA,
        role: "length/free delta query",
    },
    Iq7000CommandTarget {
        index_or_command: 0x45,
        target: 0x00F81BC,
        role: "overlap-safe memmove",
    },
    Iq7000CommandTarget {
        index_or_command: 0x46,
        target: 0x00F823F,
        role: "record helper, pending full semantics",
    },
    Iq7000CommandTarget {
        index_or_command: 0x47,
        target: 0x00F8269,
        role: "record helper, pending full semantics",
    },
    Iq7000CommandTarget {
        index_or_command: 0x48,
        target: 0x00F8308,
        role: "record helper, pending full semantics",
    },
    Iq7000CommandTarget {
        index_or_command: 0x49,
        target: 0x00F8330,
        role: "IMR-masked record iterator",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4A,
        target: 0x00F7CD3,
        role: "media base select",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4B,
        target: 0x00F8360,
        role: "bounds/room check",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4C,
        target: 0x00F8390,
        role: "current record finder",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4D,
        target: 0x00F83C8,
        role: "record checksum/validator",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4E,
        target: 0x00F89BF,
        role: "pending classification",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4F,
        target: 0x00F8A65,
        role: "pending classification",
    },
    Iq7000CommandTarget {
        index_or_command: 0x50,
        target: 0x00F8AA2,
        role: "pending classification",
    },
    Iq7000CommandTarget {
        index_or_command: 0x51,
        target: 0x00F8469,
        role: "workspace select wrapper",
    },
    Iq7000CommandTarget {
        index_or_command: 0x52,
        target: 0x00F84BA,
        role: "workspace select wrapper",
    },
    Iq7000CommandTarget {
        index_or_command: 0x53,
        target: 0x00F8412,
        role: "pending classification",
    },
];

pub const IQ7000_CAS_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x20,
        target: 0x00FAF35,
        role: "record-builder frontend, backend mode 0",
    },
    Iq7000CommandTarget {
        index_or_command: 0x21,
        target: 0x00FAF35,
        role: "record-builder frontend, backend mode 0",
    },
    Iq7000CommandTarget {
        index_or_command: 0x22,
        target: 0x00FB24B,
        role: "record-builder clear/finish",
    },
    Iq7000CommandTarget {
        index_or_command: 0x23,
        target: 0x00FB041,
        role: "record read/load",
    },
    Iq7000CommandTarget {
        index_or_command: 0x24,
        target: 0x00FAF8D,
        role: "chunked record write",
    },
    Iq7000CommandTarget {
        index_or_command: 0x27,
        target: 0x00FB03D,
        role: "record read verify/alternate mode",
    },
    Iq7000CommandTarget {
        index_or_command: 0x2F,
        target: 0x00FAF25,
        role: "unconditional builder offset seed",
    },
    Iq7000CommandTarget {
        index_or_command: 0x30,
        target: 0x00FAF21,
        role: "conditional builder offset seed when A==0",
    },
    Iq7000CommandTarget {
        index_or_command: 0x31,
        target: 0x00FAF35,
        role: "record-builder frontend alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x32,
        target: 0x00FB24B,
        role: "record-builder clear/finish alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x33,
        target: 0x00FB041,
        role: "record read/load alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x34,
        target: 0x00FAF8D,
        role: "chunked record write alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x35,
        target: 0x00FB03D,
        role: "record verify alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x3F,
        target: 0x00FAF25,
        role: "unconditional builder offset seed alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x40,
        target: 0x00FAF21,
        role: "conditional builder offset seed alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x41,
        target: 0x00FAF35,
        role: "record-builder frontend alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x42,
        target: 0x00FB24B,
        role: "record-builder clear/finish alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x43,
        target: 0x00FB041,
        role: "record read/load alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x44,
        target: 0x00FAF8D,
        role: "chunked record write alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x45,
        target: 0x00FB03D,
        role: "record verify alias",
    },
    Iq7000CommandTarget {
        index_or_command: 0x46,
        target: 0x00F0108,
        role: "suspicious unused/filler target",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000MediaFormatField {
    pub offset: u8,
    pub width: u8,
    pub name: &'static str,
    pub role: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_CAS_RECORD_FIELDS: &[Iq7000MediaFormatField] = &[
    Iq7000MediaFormatField {
        offset: 0x00,
        width: 1,
        name: "record_type",
        role: "0xB2 marker required by read/write validators",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x01,
        width: 11,
        name: "name_8_3",
        role: "space-padded 8.3-style name/pattern field",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x12,
        width: 2,
        name: "chunk_payload_len",
        role: "payload length for this 0x1C00-byte-bounded chunk",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x18,
        width: 2,
        name: "sequence",
        role: "record sequence compared against builder offset +0x11",
        status: Iq7000ContractStatus::Confirmed,
    },
];

pub const IQ7000_RAM_DISK_METADATA_FIELDS: &[Iq7000MediaFormatField] = &[
    Iq7000MediaFormatField {
        offset: 0x00,
        width: 3,
        name: "workspace_base_ptr",
        role: "drive workspace pointer root at 1FD00/1FD09/1FD0F",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x03,
        width: 3,
        name: "media_base_ptr",
        role: "media/data base pointer, initialized from 1FDEB or selected card RAM",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000MediaFormatField {
        offset: 0x0C,
        width: 1,
        name: "busy_or_locked_flags",
        role: "bit0 blocks some public operations",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x11,
        width: 1,
        name: "record_flags",
        role: "bit0 marks not-ready/locked in readiness checks",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x12,
        width: 2,
        name: "capacity_or_end_guard",
        role: "used by init/repair path to validate available record space",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000MediaFormatField {
        offset: 0x16,
        width: 3,
        name: "record_region_start",
        role: "start offset for record/list data",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x19,
        width: 3,
        name: "record_region_end",
        role: "end/current offset for record/list data",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x26,
        width: 3,
        name: "record_data_base_offset",
        role: "base offset used by list/status/template commands",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000RtcCommandSpec {
    pub command: u8,
    pub payload_len: u8,
    pub response_len: u8,
    pub role: &'static str,
}

pub const IQ7000_RTC_COMMAND_SPECS: &[Iq7000RtcCommandSpec] = &[
    Iq7000RtcCommandSpec {
        command: 0xF0,
        payload_len: 6,
        response_len: 1,
        role: "write packed BCD/status",
    },
    Iq7000RtcCommandSpec {
        command: 0xF1,
        payload_len: 6,
        response_len: 1,
        role: "write packed BCD/status",
    },
    Iq7000RtcCommandSpec {
        command: 0xF2,
        payload_len: 2,
        response_len: 1,
        role: "short write/status",
    },
    Iq7000RtcCommandSpec {
        command: 0xF4,
        payload_len: 0,
        response_len: 6,
        role: "read current datetime BCD",
    },
    Iq7000RtcCommandSpec {
        command: 0xF5,
        payload_len: 0,
        response_len: 6,
        role: "read datetime-like BCD",
    },
    Iq7000RtcCommandSpec {
        command: 0xF6,
        payload_len: 0,
        response_len: 2,
        role: "read short status/value",
    },
    Iq7000RtcCommandSpec {
        command: 0xF7,
        payload_len: 0,
        response_len: 4,
        role: "read 4-byte status/value",
    },
    Iq7000RtcCommandSpec {
        command: 0xF8,
        payload_len: 0,
        response_len: 1,
        role: "read status",
    },
    Iq7000RtcCommandSpec {
        command: 0xFD,
        payload_len: 0,
        response_len: 1,
        role: "read status",
    },
];

pub const IQ7000_RTC_PAYLOAD_FIELDS: &[Iq7000MediaFormatField] = &[
    Iq7000MediaFormatField {
        offset: 0x00,
        width: 1,
        name: "century_status_bcd",
        role: "first packed byte used by F0/F1 writes and F4/F5/F7 ASCII formatter; low nibble selects 19xx/20xx prefix",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x01,
        width: 5,
        name: "packed_datetime_pairs",
        role: "remaining YY/MM/DD/HH/MM-style BCD pairs consumed by the common parser/formatter",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x00,
        width: 2,
        name: "short_alarm_value",
        role: "F2 write payload and F6 read response parsed/formatted through the short BCD path",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000MediaFormatField {
        offset: 0x00,
        width: 1,
        name: "one_byte_status",
        role: "F8 and FD status helpers read one byte through cas_rtc_preamble_f333d",
        status: Iq7000ContractStatus::Confirmed,
    },
];

pub const IQ7000_DEV0D_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x3F,
        target: 0x00F67D7,
        role: "noop success",
    },
    Iq7000CommandTarget {
        index_or_command: 0x40,
        target: 0x00F67D7,
        role: "noop success",
    },
    Iq7000CommandTarget {
        index_or_command: 0x41,
        target: 0x00F67D9,
        role: "record lookup and validate",
    },
    Iq7000CommandTarget {
        index_or_command: 0x42,
        target: 0x00F6805,
        role: "seek last matching record",
    },
    Iq7000CommandTarget {
        index_or_command: 0x43,
        target: 0x00F681B,
        role: "next record skip invalid",
    },
    Iq7000CommandTarget {
        index_or_command: 0x44,
        target: 0x00F683A,
        role: "previous record skip invalid",
    },
    Iq7000CommandTarget {
        index_or_command: 0x45,
        target: 0x00F6BC7,
        role: "field match mode 0",
    },
    Iq7000CommandTarget {
        index_or_command: 0x46,
        target: 0x00F6C0F,
        role: "field match mode 1",
    },
    Iq7000CommandTarget {
        index_or_command: 0x47,
        target: 0x00F6BBF,
        role: "next then mode 0",
    },
    Iq7000CommandTarget {
        index_or_command: 0x48,
        target: 0x00F6C07,
        role: "previous then mode 1",
    },
    Iq7000CommandTarget {
        index_or_command: 0x49,
        target: 0x00F6C44,
        role: "field match mode 2",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4A,
        target: 0x00F6C75,
        role: "field match mode 3",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4B,
        target: 0x00F6C3C,
        role: "next then mode 2",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4C,
        target: 0x00F6C6D,
        role: "previous then mode 3",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4D,
        target: 0x00F6853,
        role: "copy blocks",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4E,
        target: 0x00F68A9,
        role: "insert/copy worker",
    },
    Iq7000CommandTarget {
        index_or_command: 0x4F,
        target: 0x00F6A6C,
        role: "edit worker",
    },
    Iq7000CommandTarget {
        index_or_command: 0x50,
        target: 0x00F6974,
        role: "edit preflight",
    },
    Iq7000CommandTarget {
        index_or_command: 0x51,
        target: 0x00F6D30,
        role: "advance and apply",
    },
    Iq7000CommandTarget {
        index_or_command: 0x52,
        target: 0x00F6C9E,
        role: "count matching records",
    },
    Iq7000CommandTarget {
        index_or_command: 0x53,
        target: 0x00F69D3,
        role: "allocate/copy/insert",
    },
    Iq7000CommandTarget {
        index_or_command: 0x54,
        target: 0x00F6CC0,
        role: "apply changes",
    },
    Iq7000CommandTarget {
        index_or_command: 0x55,
        target: 0x00F6D54,
        role: "index to checked 7-byte offset",
    },
    Iq7000CommandTarget {
        index_or_command: 0x56,
        target: 0x00F70E3,
        role: "compare dispatch",
    },
    Iq7000CommandTarget {
        index_or_command: 0x57,
        target: 0x00F71A6,
        role: "daily alarm service",
    },
    Iq7000CommandTarget {
        index_or_command: 0x58,
        target: 0x00F7286,
        role: "alarm service",
    },
    Iq7000CommandTarget {
        index_or_command: 0x59,
        target: 0x00F8733,
        role: "user dictionary transfer/search",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5A,
        target: 0x00F8914,
        role: "user dictionary fixed compare search",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5B,
        target: 0x00F6A34,
        role: "cursor adjust",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5C,
        target: 0x00F70FD,
        role: "RTC alarm worker",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5D,
        target: 0x00F6FEC,
        role: "RTC alarm phase 2",
    },
    Iq7000CommandTarget {
        index_or_command: 0x5E,
        target: 0x00F6FB9,
        role: "RTC alarm phase 1",
    },
];

pub const IQ7000_DEV0E_COMMAND_TABLE: &[Iq7000CommandTarget] = &[
    Iq7000CommandTarget {
        index_or_command: 0x40,
        target: 0x00F5CA8,
        role: "store X to global 0x0CAE5C",
    },
    Iq7000CommandTarget {
        index_or_command: 0x41,
        target: 0x00F5CAE,
        role: "timed SCR/SSR wait variant",
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000SemanticContract {
    pub area: &'static str,
    pub address: u32,
    pub name: &'static str,
    pub inputs: &'static str,
    pub outputs: &'static str,
    pub side_effects: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_UI_CALLER_CONTRACTS: &[Iq7000SemanticContract] = &[
    Iq7000SemanticContract {
        area: "system setup menu",
        address: 0x00C2880,
        name: "system_setup_menu_labels",
        inputs: "ROM descriptor text",
        outputs: "SCHEDULE ALARM / DAILY ALARM / USER'S DIC / TEL FILE NAME / SET UP menu labels",
        side_effects: "feeds the setup UI that dispatches through dev0D services",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "user dictionary UI",
        address: 0x00F6740,
        name: "user_dict_menu_descriptor",
        inputs: "encoded descriptor with action ids 1/2/3",
        outputs: "USER'S DIC > ADD / DELETE / MODIFY / SELECT WORD",
        side_effects: "selects user-dictionary edit flows backed by dev0D record/list helpers",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "schedule alarm storage",
        address: 0x00E6F46,
        name: "schedule_alarm_file_name",
        inputs: "workspace selector in U+0x0B",
        outputs: "opens RAM-disk record namespace S_ALARM 1",
        side_effects: "calls ram_disk_workspace_preflight_f80ae and seeds DI with the record base",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "daily alarm storage",
        address: 0x00F719A,
        name: "daily_alarm_file_name",
        inputs: "workspace selector in U+0x0B",
        outputs: "opens RAM-disk record namespace D_ALARM 1",
        side_effects: "calls ram_disk_workspace_preflight_f80ae and seeds DI with the record base",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "alarm list UI",
        address: 0x00E6B82,
        name: "set_alarm_record_list_dispatch",
        inputs: "shared record/list workspace and alarm record pointers",
        outputs: "visible list of schedule/daily alarm records whose flag byte has bit 0x40 set",
        side_effects: "filters alarm records before rendering/selecting set-alarm entries",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "daily alarm UI",
        address: 0x00E1512,
        name: "daily_alarm_editor_entry",
        inputs: "C00A service request and D_ALARM 1 record namespace",
        outputs: "DAILY ALARM editor with up to eight rendered alarm records",
        side_effects: "opens D_ALARM 1, draws the title, and enters the edit/toggle loop at E16CB",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "schedule alarm UI",
        address: 0x00E70CA,
        name: "schedule_alarm_list_manager",
        inputs: "SCHEDULE1/ANN/S_ALARM workspace descriptors and current C004 clock data",
        outputs: "schedule-alarm list/editor state",
        side_effects:
            "uses record flag bit 0x40 for enabled alarms and routes set/unset through elapsed-time validation",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "user dictionary transfer UI",
        address: 0x00DBEE0,
        name: "transport_target_menu",
        inputs: "ROM descriptor text",
        outputs: "<CASSETTE TAPE> / <SIO> / <PRINTER> / <PC LINK> / <DIAG PROGRAM>",
        side_effects:
            "routes setup/user-dictionary transfer flows toward CAS/COM/PRN/PACOM-style transports",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
];

pub const IQ7000_RTC_SEMANTIC_CONTRACTS: &[Iq7000SemanticContract] = &[
    Iq7000SemanticContract {
        area: "RTC current time",
        address: 0x00F3454,
        name: "rtc_read_current_datetime_ascii",
        inputs: "low-level RTC command F4",
        outputs: "YYYYMMDDHHMM-style buffer via BCD nibble formatter",
        side_effects:
            "used by World/Home live-clock renderer and backed by the Rust E-port RTC peripheral",
        status: Iq7000ContractStatus::RuntimeCovered,
    },
    Iq7000SemanticContract {
        area: "RTC write",
        address: 0x00F33A2,
        name: "rtc_pack_bcd_then_write_f0",
        inputs: "12 ASCII date/time nibbles at X, D4=0x0C",
        outputs: "packed BCD payload sent with RTC opcode F0",
        side_effects: "waits for one-byte status response from the RTC device",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC write",
        address: 0x00F3410,
        name: "rtc_write_f1_from_workspace",
        inputs: "workspace buffer parsed through sub_f33d4",
        outputs: "six-byte payload sent with RTC opcode F1",
        side_effects: "waits for one-byte status response from the RTC device",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC short write",
        address: 0x00F3431,
        name: "rtc_write_f2_short_value",
        inputs: "two-byte/nibble value parsed through sub_f33ca",
        outputs: "short payload sent with RTC opcode F2",
        side_effects: "waits for one-byte status response from the RTC device",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm IRQ",
        address: 0x00F70FD,
        name: "rtc_alarm_worker",
        inputs: "BA flags from rtc_alarm_state_query_eca65 and state byte 1FE75",
        outputs:
            "queues the 0x0C-byte alarm payload in 1FE76 and dispatches phase1/phase2 handlers depending on BA&1 / BA&2",
        side_effects:
            "loads alarm payload via FFFDC BA=0x33, sets 1FE75 bits 0x01/0x02, then phase2 clears ISR bit 0x08",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm phase",
        address: 0x00F6FB9,
        name: "rtc_alarm_phase1",
        inputs: "scratch frame size 0x18, X=U, key C004, IL=0x41",
        outputs: "refreshes the C004 alarm/list record via CALLF FFFDC",
        side_effects: "masks IMR bit 0x08, preserves D6/D8, called by dev0D IL=0x5E and rtc_alarm_worker_f70fd",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm phase",
        address: 0x00F6FEC,
        name: "rtc_alarm_phase2",
        inputs: "A mode flag, 0x18-byte scratch frame, alarm payload state at 1FE76/1FE7E",
        outputs:
            "rebuilds the next daily or schedule 0x0C-byte alarm payload, calls C004 IL=0x32/0x33 and ECA65 IL=0x42/0x43",
        side_effects:
            "copies the success payload to 1FE76, sets 1FE75 bit 0x02, and clears ISR bit 0x08",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm refresh",
        address: 0x00E1211,
        name: "rtc_alarm_refresh_wrapper",
        inputs: "X points at 1FE76 alarm payload buffer",
        outputs: "refreshes alarm payload via CA69 IL=0x44 and F6F79 IL=0x0C",
        side_effects: "allocates 0x0D scratch bytes and stores failure byte at U on carry",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm payload",
        address: 0x00E02E1,
        name: "daily_alarm_payload_build",
        inputs: "source/destination scratch buffers and D_ALARM 1 namespace",
        outputs: "0x0C-byte daily-alarm payload with ':' field separators",
        side_effects: "walks D_ALARM 1 records via dev0D IL=0x45/0x47 and tests record flag bit 0x40",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm payload",
        address: 0x00E0239,
        name: "schedule_alarm_payload_process",
        inputs: "mode A, source/destination scratch buffers and S_ALARM 1 namespace",
        outputs: "0x0C-byte schedule-alarm payload with ':' field separators",
        side_effects: "walks S_ALARM 1 records via dev0D IL=0x45/0x47 and clears 1FE75 bit 0x10 on exit",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm event consumer",
        address: 0x00E000F,
        name: "alarm_service_c00a",
        inputs: "queued alarm state in 1FE75 plus payload buffers 1FE76/1FE7E",
        outputs: "schedule alarm callback dispatch or daily alarm alert screen",
        side_effects:
            "sets 1FE75 bit 0x10 while servicing, searches S_ALARM 1 or D_ALARM 1, and clears active service bits after daily alarm display",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm setting",
        address: 0x00E1A0F,
        name: "daily_alarm_commit_phase2",
        inputs: "D_ALARM 1 record changes and enabled flag bit 0x40",
        outputs: "dev0D IL=0x5D phase2 request with CL=0x000D and A=0x80",
        side_effects: "causes rtc_alarm_phase2_f6fec to rebuild/program the next alarm payload",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RTC alarm setting",
        address: 0x00E1278,
        name: "alarm_time_compare_current_clock",
        inputs: "candidate 0x0C-byte alarm timestamp and current C004 clock/alarm record from IL=0x41",
        outputs: "carry set when the candidate is elapsed/invalid",
        side_effects:
            "used by alarm_time_elapsed_guard_e6b52 before enabling schedule alarms; failure displays ALARM TIME ELAPSED",
        status: Iq7000ContractStatus::Confirmed,
    },
];

pub const IQ7000_STORAGE_SIDE_EFFECT_CONTRACTS: &[Iq7000SemanticContract] = &[
    Iq7000SemanticContract {
        area: "RAM disk init",
        address: 0x00F7D2D,
        name: "ram_disk_init_reset",
        inputs: "drive selector in 0x1000D7",
        outputs: "1FD00/1FD03/1FD09/1FD0C workspace/media pointers",
        side_effects: "probes 0x18000/0x20000-class RAM bases and clears stale pointers on failure",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RAM disk status",
        address: 0x00F8269,
        name: "ram_disk_status_template",
        inputs: "selected workspace and caller buffer",
        outputs: "0x2E-byte template, formatted 8.3 name, 0xFF terminator",
        side_effects: "returns A=0x00/0x09/0x0C/0xFF according to readiness and room checks",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RAM disk listing",
        address: 0x00F89BF,
        name: "ram_disk_insert_space_and_shift_records",
        inputs: "workspace offsets at Y+0x03/0x09 and IOCS buffer state",
        outputs: "shifted record region and adjusted Y+0x16/Y+0x26 offsets",
        side_effects: "memmoves existing records through F81BC; writes U+0x0E=A=0x0C/0xFF on bounds/preflight errors",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RAM disk copy",
        address: 0x00F8A65,
        name: "ram_disk_copy_record_body_from_offset",
        inputs: "caller length in DL and source pointer from U+0x00",
        outputs: "copies bytes from record body after 0x29-byte header using F6F88",
        side_effects: "writes U+0x08=A=0x0C on bounds failure",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RAM disk copy",
        address: 0x00F8AA2,
        name: "ram_disk_copy_record_body_to_offset",
        inputs: "caller length in DL and destination pointer from U+0x00",
        outputs: "copies bytes to record body after 0x29-byte header using F6F88",
        side_effects: "zero length returns current offset in U+0x00; bounds failure writes U+0x08=A=0x0C",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "RAM disk status",
        address: 0x00F8416,
        name: "ram_disk_workspace_status_check",
        inputs: "drive selector in CH and current workspace header",
        outputs: "carry clear when workspace is ready and caller pointer is in range",
        side_effects: "returns A=0x05 when Y+0x0C bit0 blocks public operations",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "CAS low-level",
        address: 0x00FB253,
        name: "cas_receive_record_primitive",
        inputs: "record buffer U and builder state at (*E6)+0x43",
        outputs: "received record header/payload or carry-set failure",
        side_effects: "feeds CAS load/verify paths through 0xB2 record validation",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000SemanticContract {
        area: "CAS low-level",
        address: 0x00FB32B,
        name: "cas_transmit_record_primitive",
        inputs: "record header/payload at U and builder counters",
        outputs: "transmitted chunk or carry-set failure",
        side_effects: "used by chunked writer 0xFAF8D with 0x1C00-byte chunks",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000SemanticContract {
        area: "CAS retry/status",
        address: 0x00F35DA,
        name: "cas_rtc_ready_ack_helper",
        inputs: "flag byte 1FDC6 bit 0x10",
        outputs: "clears pending flag or returns immediately",
        side_effects: "busy-waits, masks FD with 0x8F, ORs FD with 0x40 when acking",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "CAS/link retry",
        address: 0x00FBBCD,
        name: "shared_ready_defer_gate",
        inputs: "1FDC5 bit 0x40 and keyboard/link readiness helper F54D9",
        outputs: "A=0xFE carry-set when deferred, A=0xFF carry-clear on readiness helper failure",
        side_effects: "used inside send/receive timeout loops before retrying bit-banged link operations",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "CAS/link retry",
        address: 0x00FBB2F,
        name: "shared_bitbang_send_byte_retry",
        inputs: "A byte payload, EIH.6 handshake, EOH.7 data output",
        outputs: "carry clear after eight acknowledged bits",
        side_effects: "uses 0x00EB and 0x049E timeout loops; clears EOH bits and returns A=0 on timeout",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "CAS/link retry",
        address: 0x00FBAA2,
        name: "shared_bitbang_receive_byte_retry",
        inputs: "EIH.6 handshake and EOH.7 acknowledge output",
        outputs: "received byte after eight sampled bits",
        side_effects: "uses 0x00EB/0x049E timeout loops and shared ready/defer gate on stalls",
        status: Iq7000ContractStatus::Confirmed,
    },
];

pub const IQ7000_LINK_ABI_CONTRACTS: &[Iq7000SemanticContract] = &[
    Iq7000SemanticContract {
        area: "PACOM/PANET dispatch",
        address: 0x00FBC29,
        name: "pacom_panet_word_table",
        inputs: "IL index or IL-0x3F alias",
        outputs: "16-bit in-bank target pushed to S then RET",
        side_effects: "selects handshake/frame/record helper entries",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PACOM/PANET handshake",
        address: 0x00FBC3F,
        name: "receive_a5_reply_8000",
        inputs: "incoming 0xA5 byte over EIH/EOH bit-banged link",
        outputs: "reply bytes 0x80, 0x00; carry clear on success",
        side_effects: "uses IMR=0xC0 critical section and EOH.7/EIH.6 handshake",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PACOM/PANET handshake",
        address: 0x00FBC7A,
        name: "send_a5_parse_response",
        inputs: "link peer ready on EIH.6",
        outputs: "success on 0xF0/0x01 or 0x80 response pattern",
        side_effects: "common failures go through FBC72 cleanup and return carry set",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PACOM/PANET frame send",
        address: 0x00FBCF9,
        name: "send_frame_checksum_wait_fa",
        inputs: "payload at X and length in Y",
        outputs: "payload plus 16-bit additive checksum, expects ACK 0xFA",
        side_effects: "returns carry set on timeout/check failure",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PACOM/PANET frame receive",
        address: 0x00FBD54,
        name: "receive_frame_checksum_ack",
        inputs: "destination buffer X and length Y",
        outputs: "payload bytes and ACK 0xFA on checksum match, 0xF0 on mismatch",
        side_effects: "returns carry clear only when checksum matches",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PRN dispatch",
        address: 0x00FB477,
        name: "printer_word_table_dispatch",
        inputs: "IL<0x3F maps via IL-8, else IL-0x3F",
        outputs: "16-bit in-bank target pushed to S then RET",
        side_effects: "shares prologue/epilogue and bit-banged helpers with PACOM/PANET",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000SemanticContract {
        area: "PRN output",
        address: 0x00FB4FC,
        name: "printer_putchar_control_handler",
        inputs: "A character byte and IOCS workspace offsets +3E/+3F/+40/+41",
        outputs: "buffered character, padded line flush, or CR/LF/TAB expansion",
        side_effects:
            "uses 1FE52 bits to suppress NUL/TAB handling and calls B666/B62E/B624 buffer helpers",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PRN output",
        address: 0x00FB6FD,
        name: "printer_printing_ui_send_workflow",
        inputs: "printer/PANET config and current output buffer",
        outputs: "PRINTING status UI and bit-banged payload transfer",
        side_effects: "uses IOCS IL=0x5E/0x42/0x5F around the inline 'PRINTING' label",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PANET packet",
        address: 0x00FB8A7,
        name: "panet_packet_xfer_mode1",
        inputs: "packet buffer X, 1FE50 mode flags, 1FE51 zero-pad count",
        outputs: "packet bytes sent with 0x59/0x70 framing and FA ACK",
        side_effects:
            "may invert bytes when 1FE50 bit 0x40 is set; loops over 0x10 or 0x0C byte records",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "PANET packet",
        address: 0x00FB8AF,
        name: "panet_packet_xfer_mode0",
        inputs: "packet buffer X, 1FE50 mode flags, 1FE51 zero-pad count",
        outputs: "packet bytes sent with 0x59/0x70 framing and FA ACK",
        side_effects: "same transport as mode1 but without the leading BP+00 mode bit",
        status: Iq7000ContractStatus::Confirmed,
    },
];

pub const IQ7000_KEY_TRANSLATION_SEMANTICS: &[Iq7000SemanticContract] = &[
    Iq7000SemanticContract {
        area: "keyboard scanner",
        address: 0x00F55AE,
        name: "matrix_to_translation_code",
        inputs: "row, column from sweep loop",
        outputs: "idx=col*8+row; primary byte from 1FDC5+2+idx; optional secondary byte from 1FDCD",
        side_effects: "returns keycode/dispatch index to higher-level STDI paths",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "keyboard dequeue",
        address: 0x00F3652,
        name: "key_fetch_state_gate",
        inputs: "caller A flags, ring indices at (*E6)+4/+5, scan flags 1FDC5/1FDBE",
        outputs: "stable key byte or A=0xFE/carry-set when gated",
        side_effects: "may synthesize wake/poll event 0x8F0F when A&0x80 and ring is empty",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "keyboard state",
        address: 0x00F383C,
        name: "keycode_state_dispatch",
        inputs: "keycode from dequeue/scan path",
        outputs: "updates current code 1FD4A and returns repeat/state carry",
        side_effects: "handles special A==1/A==9 toggles through IOCS IL=0x46",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "keyboard queue",
        address: 0x00F55CF,
        name: "translated_key_queue_push",
        inputs: "translated keycode A and modifier/repeat byte B",
        outputs: "writes keycode into IOCS workspace ring at base [(*E6)+2] + head",
        side_effects: "advances head modulo 0x10, sets ring-full bit 0x80, and invokes optional callback 1FDD0",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "keyboard repeat",
        address: 0x00F5394,
        name: "keyboard_scan_repeat_manager",
        inputs: "previous scan state 1FDBC/1FDBE, repeat delay 1FDC1, repeat rate 1FDC2",
        outputs: "new key, release, or repeat events queued through F55CF",
        side_effects: "sets 1FDC5 bit 0x80 for ON/EA activity and updates scan state words",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000SemanticContract {
        area: "keyboard shell",
        address: 0x00E2974,
        name: "shell_read_input_event",
        inputs: "IOCS STDI device 1, IL=0x49",
        outputs: "foreground UI input event consumed by app selector loops",
        side_effects: "special translated app codes are routed by shell/app dispatchers",
        status: Iq7000ContractStatus::Confirmed,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000StatusContract {
    pub area: &'static str,
    pub code: u8,
    pub meaning: &'static str,
    pub producers: &'static str,
    pub status: Iq7000ContractStatus,
}

pub const IQ7000_STATUS_CONTRACTS: &[Iq7000StatusContract] = &[
    Iq7000StatusContract {
        area: "dev0D/list editor",
        code: 0x06,
        meaning: "not found / invalid record / end of linked record list",
        producers: "F681B, F683A, F6853 and related user-dictionary walkers",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "dev0D/list editor",
        code: 0x0C,
        meaning: "bounds or size failure during copy/insert/edit",
        producers: "F6853, F68A9, F6A6C and storage/list copy helpers",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "IOCS dispatch",
        code: 0xFF,
        meaning: "generic out-of-range or failed operation",
        producers: "dev0D out-of-range, RAM-disk dispatcher failures, several storage helpers",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "COM/link",
        code: 0xFE,
        meaning: "deferred/busy/timeout-style failure",
        producers: "COM gates and PACOM/PANET BBCD when 1FDC5&0x40 is set",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "PACOM/PANET",
        code: 0xFA,
        meaning: "frame checksum accepted ACK",
        producers: "receive frame FBD54/FBD4B",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "PACOM/PANET",
        code: 0xF0,
        meaning: "frame checksum mismatch or negative handshake response",
        producers: "receive frame FBD54/FBD4B and A5 response parser",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "RAM disk",
        code: 0x09,
        meaning: "workspace/template/status helper specific failure",
        producers: "F8269 readiness/template helper",
        status: Iq7000ContractStatus::StructurallyMapped,
    },
    Iq7000StatusContract {
        area: "RAM disk",
        code: 0x05,
        meaning: "public operation blocked by workspace header lock/busy bit",
        producers: "F8416 workspace status check when Y+0x0C bit0 is set",
        status: Iq7000ContractStatus::Confirmed,
    },
    Iq7000StatusContract {
        area: "PACOM/PANET",
        code: 0x0B,
        meaning: "guarded receive-frame first byte did not match caller expectation",
        producers: "FBD4B/FBD54 guarded receive path",
        status: Iq7000ContractStatus::Confirmed,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iq7000DeviceGap {
    pub area: &'static str,
    pub rust_contract: &'static str,
    pub remaining: &'static str,
}

pub const IQ7000_REMAINING_GAPS: &[Iq7000DeviceGap] = &[
    Iq7000DeviceGap {
        area: "STDO:SCRN",
        rust_contract: "rendering, exact STDO header fields, and decoded walker steps",
        remaining: "only mirrored-header/hardware-parity caller tracing remains",
    },
    Iq7000DeviceGap {
        area: "STDI:KYBD",
        rust_contract: "selector map, scan chain, F0/F1 contracts, and translation table anchors",
        remaining: "hardware keycap-to-row/column parity labels",
    },
    Iq7000DeviceGap {
        area: "COM/PANET/PACOM/PRN",
        rust_contract: "IOCS records, COM/PRN/PACOM/PANET IL tables, line/status/ring/flow/link helpers",
        remaining: "cycle-accurate external UART/printer/PANET/PACOM peripherals and user-facing call-site traces",
    },
    Iq7000DeviceGap {
        area: "CAS",
        rust_contract: "CAS IL table, record fields, chunked load/write/verify paths",
        remaining: "analog transport timing and hardware retry parity",
    },
    Iq7000DeviceGap {
        area: "S1:S2:S3:S4 / E:F:G",
        rust_contract: "command table, helper roles, and workspace metadata fields",
        remaining: "byte-for-byte external media image parity fixtures",
    },
    Iq7000DeviceGap {
        area: "RTC",
        rust_contract: "current-time peripheral, command lengths, and alarm worker entries",
        remaining: "stateful alarm write/status and ON-key wake scheduling",
    },
    Iq7000DeviceGap {
        area: "SYSTM/dev0D/dev0E",
        rust_contract: "dev0D/dev0E command tables and named setup/user-dictionary/list/alarm/SCR caller roles",
        remaining: "stateful list/alarm side effects and finer per-screen call-site labels",
    },
];

fn mask_for_bits(bits: u8) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else if bits == 0 {
        0
    } else {
        (1u32 << bits) - 1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iq7000ClockSeed {
    bytes: [u8; CLOCK_WORKSPACE_LEN],
}

impl Iq7000ClockSeed {
    pub fn from_yyyymmddhhmm(raw: &str) -> std::result::Result<Self, String> {
        if raw.len() != 12 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(format!(
                "IQ-7000 RTC seed must be YYYYMMDDHHMM, got '{raw}'"
            ));
        }
        let month: u32 = raw[4..6]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC month in '{raw}'"))?;
        let day: u32 = raw[6..8]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC day in '{raw}'"))?;
        let hour: u32 = raw[8..10]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC hour in '{raw}'"))?;
        let minute: u32 = raw[10..12]
            .parse()
            .map_err(|_| format!("invalid IQ-7000 RTC minute in '{raw}'"))?;
        if !(1..=12).contains(&month) {
            return Err(format!("invalid IQ-7000 RTC month in '{raw}'"));
        }
        if !(1..=31).contains(&day) {
            return Err(format!("invalid IQ-7000 RTC day in '{raw}'"));
        }
        if hour > 23 {
            return Err(format!("invalid IQ-7000 RTC hour in '{raw}'"));
        }
        if minute > 59 {
            return Err(format!("invalid IQ-7000 RTC minute in '{raw}'"));
        }

        let mut bytes = [0u8; CLOCK_WORKSPACE_LEN];
        bytes[..12].copy_from_slice(raw.as_bytes());
        Ok(Self { bytes })
    }

    pub fn read(&self, addr: u32, bits: u8) -> Option<u32> {
        if bits == 0 {
            return Some(0);
        }
        let width = usize::from(bits.div_ceil(8).clamp(1, 4));
        let start = addr.checked_sub(CLOCK_WORKSPACE_START)? as usize;
        if start >= self.bytes.len() || start + width > self.bytes.len() {
            return None;
        }
        let mut value = 0u32;
        for idx in 0..width {
            value |= (self.bytes[start + idx] as u32) << (idx * 8);
        }
        Some(value & mask_for_bits(bits))
    }

    pub fn apply_to_memory(&self, memory: &mut MemoryImage) {
        for (idx, byte) in self.bytes.iter().copied().enumerate() {
            memory.write_external_byte(CLOCK_WORKSPACE_START + idx as u32, byte);
        }
        memory.write_external_byte(CLOCK_INITIALIZED_FLAG, 1);
    }

    pub fn as_ascii(&self) -> &str {
        std::str::from_utf8(&self.bytes[..12]).unwrap_or("")
    }

    pub fn rtc_datetime_bcd(&self) -> [u8; 6] {
        let digits = &self.bytes[..12];
        [
            packed_bcd(digits[0], digits[1]),
            packed_bcd(digits[2], digits[3]),
            packed_bcd(digits[4], digits[5]),
            packed_bcd(digits[6], digits[7]),
            packed_bcd(digits[8], digits[9]),
            packed_bcd(digits[10], digits[11]),
        ]
    }
}

fn packed_bcd(tens: u8, ones: u8) -> u8 {
    ((tens - b'0') << 4) | (ones - b'0')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RtcWritePhase {
    Idle,
    ReadyHigh,
    AwaitData,
    ReadyLow,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RtcReadPhase {
    Idle,
    ReadyHigh,
    ReadyLow,
    Sample,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iq7000RtcPeripheral {
    seed: Iq7000ClockSeed,
    eol: u8,
    write_phase: RtcWritePhase,
    write_acc: u8,
    write_bits: u8,
    read_phase: RtcReadPhase,
    response_bits: Vec<bool>,
    response_index: usize,
    payload_remaining: u8,
    response_after_payload: Vec<u8>,
    current_read_data: u8,
    last_command: Option<u8>,
}

impl Iq7000RtcPeripheral {
    pub fn new(seed: Iq7000ClockSeed) -> Self {
        Self {
            seed,
            eol: 0,
            write_phase: RtcWritePhase::Idle,
            write_acc: 0,
            write_bits: 0,
            read_phase: RtcReadPhase::Idle,
            response_bits: Vec::new(),
            response_index: 0,
            payload_remaining: 0,
            response_after_payload: Vec::new(),
            current_read_data: 0,
            last_command: None,
        }
    }

    pub fn set_seed(&mut self, seed: Iq7000ClockSeed) {
        self.seed = seed;
        self.reset_protocol();
    }

    pub fn seed(&self) -> &Iq7000ClockSeed {
        &self.seed
    }

    pub fn handle_eol_write(&mut self, value: u8) {
        let strobe_was_low = (self.eol & EOL_STROBE) == 0;
        self.eol = value;

        if (value & EOL_STROBE) == 0 {
            self.write_phase = RtcWritePhase::Idle;
            self.read_phase = RtcReadPhase::Idle;
            return;
        }

        if self.has_pending_response() {
            if strobe_was_low || self.read_phase == RtcReadPhase::Idle {
                self.read_phase = RtcReadPhase::ReadyHigh;
            }
            return;
        }

        if strobe_was_low
            || matches!(
                self.write_phase,
                RtcWritePhase::Idle | RtcWritePhase::Complete
            )
        {
            self.write_phase = RtcWritePhase::ReadyHigh;
        } else if self.write_phase == RtcWritePhase::AwaitData {
            self.latch_host_bit((value & EOL_OUT_DATA) != 0);
            self.write_phase = RtcWritePhase::ReadyLow;
        }
    }

    pub fn handle_eil_read(&mut self) -> u8 {
        if self.write_phase == RtcWritePhase::ReadyLow {
            self.write_phase = RtcWritePhase::Complete;
            return 0;
        }

        if self.has_pending_response() && (self.eol & EOL_STROBE) != 0 {
            return self.next_response_eil();
        }

        match self.write_phase {
            RtcWritePhase::ReadyHigh => {
                self.write_phase = RtcWritePhase::AwaitData;
                EIL_READY
            }
            _ => 0,
        }
    }

    fn has_pending_response(&self) -> bool {
        self.response_index < self.response_bits.len()
    }

    fn reset_protocol(&mut self) {
        self.write_phase = RtcWritePhase::Idle;
        self.write_acc = 0;
        self.write_bits = 0;
        self.read_phase = RtcReadPhase::Idle;
        self.response_bits.clear();
        self.response_index = 0;
        self.payload_remaining = 0;
        self.response_after_payload.clear();
        self.current_read_data = 0;
        self.last_command = None;
    }

    fn latch_host_bit(&mut self, bit: bool) {
        if bit {
            self.write_acc |= 1 << self.write_bits;
        }
        self.write_bits += 1;
        if self.write_bits == 8 {
            let byte = self.write_acc;
            self.write_acc = 0;
            self.write_bits = 0;
            if self.payload_remaining > 0 {
                self.payload_remaining -= 1;
                if self.payload_remaining == 0 && !self.response_after_payload.is_empty() {
                    let response = std::mem::take(&mut self.response_after_payload);
                    self.queue_response_bytes(&response);
                }
            } else {
                self.accept_command(byte);
            }
        }
    }

    fn accept_command(&mut self, command: u8) {
        self.last_command = Some(command);
        self.response_bits.clear();
        self.response_index = 0;
        self.payload_remaining = 0;
        self.response_after_payload.clear();
        match command {
            0xF0 | 0xF1 => {
                self.payload_remaining = 6;
                self.response_after_payload.push(0);
            }
            0xF2 => {
                self.payload_remaining = 2;
                self.response_after_payload.push(0);
            }
            RTC_COMMAND_CURRENT_DATETIME | 0xF5 => {
                self.queue_response_bytes(&self.seed.rtc_datetime_bcd());
            }
            0xF6 => self.queue_response_bytes(&[0, 0]),
            0xF7 => self.queue_response_bytes(&[0, 0, 0, 0]),
            0xF8 | 0xFD => self.queue_response_bytes(&[0]),
            _ => {}
        }
    }

    fn queue_response_bytes(&mut self, bytes: &[u8]) {
        self.response_bits.clear();
        self.response_index = 0;
        for byte in bytes {
            // The ROM read helper XORs the assembled byte with 0xFF before returning it.
            let wire_byte = !byte;
            for bit in 0..8 {
                self.response_bits.push(((wire_byte >> bit) & 1) != 0);
            }
        }
    }

    fn next_response_eil(&mut self) -> u8 {
        match self.read_phase {
            RtcReadPhase::Idle | RtcReadPhase::ReadyHigh => {
                self.read_phase = RtcReadPhase::ReadyLow;
                EIL_READY
            }
            RtcReadPhase::ReadyLow => {
                self.current_read_data = if self.response_bits[self.response_index] {
                    EIL_IN_DATA
                } else {
                    0
                };
                self.read_phase = RtcReadPhase::Sample;
                self.current_read_data
            }
            RtcReadPhase::Sample => {
                let value = self.current_read_data;
                self.response_index += 1;
                self.read_phase = if self.has_pending_response() {
                    RtcReadPhase::ReadyHigh
                } else {
                    RtcReadPhase::Idle
                };
                value
            }
        }
    }
}

pub fn load_iq7000_rom_image(rt: &mut CoreRuntime, rom: &[u8]) -> Result<()> {
    load_iq7000_rom_image_into_memory(&mut rt.memory, rom);
    rt.memory
        .set_readonly_ranges(vec![(ROM_READONLY_START, ROM_READONLY_END)]);
    rt.memory.set_keyboard_bridge(false);
    Ok(())
}

pub fn load_iq7000_rom_image_into_memory(memory: &mut MemoryImage, rom: &[u8]) {
    let src_start = rom.len().saturating_sub(ROM_WINDOW_LEN);
    let slice = &rom[src_start..];
    let copy_len = slice.len().min(ROM_WINDOW_LEN);
    let start_in_slice = slice.len().saturating_sub(copy_len);
    memory.write_external_slice(
        ROM_WINDOW_START,
        &slice[start_in_slice..start_in_slice + copy_len],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_write_byte(peripheral: &mut Iq7000RtcPeripheral, byte: u8) {
        for bit in 0..8 {
            peripheral.handle_eol_write(EOL_STROBE);
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, EIL_READY);
            let data = if ((byte >> bit) & 1) != 0 {
                EOL_STROBE | EOL_OUT_DATA
            } else {
                EOL_STROBE
            };
            peripheral.handle_eol_write(data);
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, 0);
        }
        peripheral.handle_eol_write(0);
    }

    fn host_read_byte_like_rom(peripheral: &mut Iq7000RtcPeripheral) -> u8 {
        peripheral.handle_eol_write(EOL_STROBE);
        let mut assembled = 0u8;
        for _ in 0..8 {
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, EIL_READY);
            assert_eq!(peripheral.handle_eil_read() & EIL_READY, 0);
            let sample = peripheral.handle_eil_read();
            let carry = (sample & EIL_IN_DATA) != 0;
            assembled >>= 1;
            if carry {
                assembled |= 0x80;
            }
        }
        peripheral.handle_eol_write(0);
        assembled ^ 0xFF
    }

    #[test]
    fn iocs_device_records_lock_known_handlers() {
        assert_eq!(IQ7000_IOCS_DEVICE_RECORDS.len(), 12);

        let stdo = IQ7000_IOCS_DEVICE_RECORDS
            .iter()
            .find(|record| record.name == "STDO:SCRN:")
            .expect("STDO record");
        assert_eq!(stdo.device, 0x00);
        assert_eq!(stdo.attr, 0xA2);
        assert_eq!(stdo.handler, 0x00F04A5);
        assert_eq!(stdo.status, Iq7000ContractStatus::RuntimeCovered);

        let rtc = IQ7000_IOCS_DEVICE_RECORDS
            .iter()
            .find(|record| record.name == "RTC:")
            .expect("RTC record");
        assert_eq!(rtc.device, 0x0A);
        assert_eq!(rtc.handler, 0x00F31EF);

        let dev0d = IQ7000_IOCS_DEVICE_RECORDS
            .iter()
            .find(|record| record.device == 0x0D)
            .expect("dev0D record");
        assert_eq!(dev0d.handler, 0x00F6777);
        assert_eq!(dev0d.name, "dev0D:System/UserDic/Alarm");
    }

    #[test]
    fn header_records_lock_stdo_walker_and_trailers() {
        let stdo = IQ7000_HEADER_RECORDS
            .iter()
            .find(|record| record.address == 0x051720)
            .expect("STDO header record");
        assert_eq!(stdo.name, "STDO:SCRN:");
        assert_eq!(stdo.attr, Some(0xA2));
        assert_eq!(stdo.entry_or_walker, Some(IQ7000_STDO_WALKER_ENTRY));
        assert_eq!(stdo.trailer_ptr, Some(0x05174F));
        assert_eq!(stdo.trailer_index, Some(0x01));

        assert_eq!(IQ7000_STDO_WALKER_BLOB_START, 0x060200);
        assert_eq!(IQ7000_STDO_WALKER_ENTRY, 0x0602EC);
        assert_eq!(IQ7000_STDO_WALKER_DEVICE_STRING_ADDR, 0x060980);

        let system = IQ7000_HEADER_RECORDS
            .iter()
            .find(|record| record.name == "SYSTEM block")
            .expect("SYSTEM header block");
        assert_eq!(system.entry_or_walker, Some(0x048C31));
        assert_eq!(system.trailer_index, None);

        assert!(IQ7000_STDO_HEADER_FIELDS.iter().any(|field| {
            field.address == 0x051726
                && field.value == u64::from(IQ7000_STDO_WALKER_ENTRY)
                && field.name == "stdo_header_walker_entry"
        }));
        assert!(IQ7000_STDO_WALKER_STEPS.iter().any(|step| {
            step.address == 0x060892 && step.role.contains("initialize walker header tables")
        }));
    }

    #[test]
    fn keyboard_selector_contract_covers_builtin_apps() {
        let actual: Vec<(&str, u8, u8)> = IQ7000_APP_SELECTOR_MAP
            .iter()
            .map(|selector| (selector.label, selector.raw_code, selector.translated_code))
            .collect();
        assert_eq!(
            actual,
            vec![
                ("CALENDAR", 0x18, 0x51),
                ("SCHEDULE", 0x19, 0x50),
                ("TEL", 0x10, 0x52),
                ("MEMO", 0x08, 0x53),
                ("CALC", 0x00, 0x54),
                ("CARD/SAMPLES", 0x1A, 0x55),
                ("WORLD", 0x11, 0x56),
                ("HOME", 0x09, 0x57),
            ]
        );

        assert_eq!(IQ7000_KEYBOARD_SCAN_CHAIN[0], 0x00F5394);
        assert_eq!(IQ7000_KEYBOARD_SCAN_CHAIN[5], 0x00F55AE);
        assert!(IQ7000_KEYBOARD_ACTION_TARGETS
            .iter()
            .any(|target| target.address == 0x00F563C
                && target.role == "serial RX interrupt handler"));
        assert!(IQ7000_KEYBOARD_MATRIX_CONTRACTS.iter().any(|entry| {
            entry.address == 0x00F5548
                && entry.f0_effect == "F0 receives the active output-line mask"
        }));
        assert!(IQ7000_KEYBOARD_TRANSLATION_TABLES
            .iter()
            .any(|entry| entry.address == 0x00F4019));
    }

    #[test]
    fn ramdisk_and_rtc_command_contracts_cover_known_tables() {
        assert_eq!(IQ7000_RAM_DISK_COMMAND_TABLE.len(), 0x15);
        assert_eq!(
            IQ7000_RAM_DISK_COMMAND_TABLE[0],
            Iq7000CommandTarget {
                index_or_command: 0x3F,
                target: 0x00F7D2D,
                role: "init/reset",
            }
        );
        assert_eq!(
            IQ7000_RAM_DISK_COMMAND_TABLE[0x14],
            Iq7000CommandTarget {
                index_or_command: 0x53,
                target: 0x00F8412,
                role: "pending classification",
            }
        );

        let current_time = IQ7000_RTC_COMMAND_SPECS
            .iter()
            .find(|spec| spec.command == RTC_COMMAND_CURRENT_DATETIME)
            .expect("F4 RTC command");
        assert_eq!(current_time.payload_len, 0);
        assert_eq!(current_time.response_len, 6);

        let write_bcd = IQ7000_RTC_COMMAND_SPECS
            .iter()
            .find(|spec| spec.command == 0xF1)
            .expect("F1 RTC command");
        assert_eq!(write_bcd.payload_len, 6);
        assert_eq!(write_bcd.response_len, 1);
    }

    #[test]
    fn device_command_contracts_cover_serial_cas_storage_and_system_tables() {
        assert!(IQ7000_COM_COMMAND_TABLE.iter().any(|entry| {
            entry.index_or_command == 0x5F
                && entry.target == 0x00FAE76
                && entry.role == "flow-control gate"
        }));
        assert!(IQ7000_CAS_COMMAND_TABLE.iter().any(|entry| {
            entry.index_or_command == 0x24
                && entry.target == 0x00FAF8D
                && entry.role == "chunked record write"
        }));
        assert!(IQ7000_CAS_RECORD_FIELDS
            .iter()
            .any(|field| { field.offset == 0x18 && field.name == "sequence" }));
        assert!(IQ7000_RAM_DISK_METADATA_FIELDS
            .iter()
            .any(|field| { field.offset == 0x16 && field.name == "record_region_start" }));
        assert_eq!(IQ7000_DEV0D_COMMAND_TABLE.len(), 0x20);
        assert!(IQ7000_DEV0D_COMMAND_TABLE.iter().any(|entry| {
            entry.index_or_command == 0x5C
                && entry.target == 0x00F70FD
                && entry.role == "RTC alarm worker"
        }));
        assert_eq!(
            IQ7000_DEV0E_COMMAND_TABLE,
            &[
                Iq7000CommandTarget {
                    index_or_command: 0x40,
                    target: 0x00F5CA8,
                    role: "store X to global 0x0CAE5C",
                },
                Iq7000CommandTarget {
                    index_or_command: 0x41,
                    target: 0x00F5CAE,
                    role: "timed SCR/SSR wait variant",
                },
            ]
        );
    }

    #[test]
    fn semantic_contracts_cover_remaining_rom_solvable_iq7000_gaps() {
        assert!(IQ7000_UI_CALLER_CONTRACTS.iter().any(|entry| {
            entry.address == 0x00F6740 && entry.name == "user_dict_menu_descriptor"
        }));
        assert!(IQ7000_RTC_SEMANTIC_CONTRACTS
            .iter()
            .any(|entry| { entry.address == 0x00F70FD && entry.name == "rtc_alarm_worker" }));
        assert!(IQ7000_STORAGE_SIDE_EFFECT_CONTRACTS.iter().any(|entry| {
            entry.address == 0x00FB32B && entry.name == "cas_transmit_record_primitive"
        }));
        assert!(IQ7000_LINK_ABI_CONTRACTS.iter().any(|entry| {
            entry.address == 0x00FBD54 && entry.name == "receive_frame_checksum_ack"
        }));
        assert!(IQ7000_KEY_TRANSLATION_SEMANTICS.iter().any(|entry| {
            entry.address == 0x00F55AE && entry.outputs.contains("idx=col*8+row")
        }));
        assert!(IQ7000_STATUS_CONTRACTS
            .iter()
            .any(|entry| { entry.area == "PACOM/PANET" && entry.code == 0xFA }));
    }

    #[test]
    fn focused_disassembly_contracts_close_priority_iq7000_gaps() {
        assert!(IQ7000_RTC_PAYLOAD_FIELDS
            .iter()
            .any(|field| field.name == "century_status_bcd"));
        assert!(IQ7000_RTC_SEMANTIC_CONTRACTS.iter().any(|entry| {
            entry.address == 0x00E02E1 && entry.name == "daily_alarm_payload_build"
        }));
        assert!(IQ7000_STORAGE_SIDE_EFFECT_CONTRACTS.iter().any(|entry| {
            entry.address == 0x00F8AA2 && entry.name == "ram_disk_copy_record_body_to_offset"
        }));
        assert!(IQ7000_PRN_COMMAND_TABLE
            .iter()
            .any(|entry| { entry.index_or_command == 0x41 && entry.target == 0x00FB6FD }));
        assert!(IQ7000_PACOM_PANET_COMMAND_TABLE
            .iter()
            .any(|entry| { entry.index_or_command == 0x04 && entry.target == 0x00FBD54 }));
        assert!(IQ7000_KEY_TRANSLATION_SEMANTICS.iter().any(|entry| {
            entry.address == 0x00F55CF && entry.name == "translated_key_queue_push"
        }));
        assert!(IQ7000_STATUS_CONTRACTS
            .iter()
            .any(|entry| entry.area == "RAM disk" && entry.code == 0x05));
    }

    #[test]
    fn remaining_gap_contract_names_all_open_device_families() {
        for area in [
            "STDO:SCRN",
            "STDI:KYBD",
            "COM/PANET/PACOM/PRN",
            "CAS",
            "S1:S2:S3:S4 / E:F:G",
            "RTC",
            "SYSTM/dev0D/dev0E",
        ] {
            assert!(
                IQ7000_REMAINING_GAPS.iter().any(|gap| gap.area == area),
                "missing gap entry for {area}"
            );
        }
    }

    #[test]
    fn clock_seed_converts_to_rtc_bcd() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        assert_eq!(
            seed.rtc_datetime_bcd(),
            [0x20, 0x26, 0x04, 0x25, 0x21, 0x19]
        );
    }

    #[test]
    fn rtc_peripheral_accepts_lsb_first_command_bits() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut peripheral = Iq7000RtcPeripheral::new(seed);

        host_write_byte(&mut peripheral, RTC_COMMAND_CURRENT_DATETIME);

        assert_eq!(peripheral.last_command, Some(RTC_COMMAND_CURRENT_DATETIME));
        assert!(peripheral.has_pending_response());
    }

    #[test]
    fn rtc_peripheral_streams_current_datetime_like_rom_read_helper() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut peripheral = Iq7000RtcPeripheral::new(seed);

        host_write_byte(&mut peripheral, RTC_COMMAND_CURRENT_DATETIME);
        let actual = [
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
            host_read_byte_like_rom(&mut peripheral),
        ];

        assert_eq!(actual, [0x20, 0x26, 0x04, 0x25, 0x21, 0x19]);
        assert!(!peripheral.has_pending_response());
    }

    #[test]
    fn rtc_peripheral_waits_for_write_payload_before_status_response() {
        let seed = Iq7000ClockSeed::from_yyyymmddhhmm("202604252119").expect("seed parses");
        let mut peripheral = Iq7000RtcPeripheral::new(seed);

        host_write_byte(&mut peripheral, 0xF1);
        assert_eq!(peripheral.last_command, Some(0xF1));
        assert!(!peripheral.has_pending_response());

        for byte in [0x20, 0x26, 0x04, 0x25, 0x21] {
            host_write_byte(&mut peripheral, byte);
            assert!(!peripheral.has_pending_response());
        }
        host_write_byte(&mut peripheral, 0x19);

        assert!(peripheral.has_pending_response());
        assert_eq!(host_read_byte_like_rom(&mut peripheral), 0x00);
    }
}
