use crate::llama::dispatch;
use crate::llama::opcodes::InstrKind;
use crate::memory::ADDRESS_MASK;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

const RETI_OPCODE: u8 = 0x01;
const DEFAULT_MAX_LOOP_LEN: usize = 4096;
const DEFAULT_RECENT_POSITIONS: usize = 8;
const DEFAULT_MAIN_HISTORY_SLACK: usize = 64;
const DEFAULT_FULL_HISTORY_MULTIPLIER: usize = 4;
const DEFAULT_DETECT_STRIDE: u64 = 0;

#[derive(Clone, Copy, Debug)]
pub struct LoopDetectorConfig {
    pub max_loop_len: usize,
    pub main_history_len: usize,
    pub full_history_len: usize,
    pub recent_positions_len: usize,
    pub detect_stride: u64,
}

impl Default for LoopDetectorConfig {
    fn default() -> Self {
        let main_history_len = DEFAULT_MAX_LOOP_LEN
            .saturating_mul(3)
            .saturating_add(DEFAULT_MAIN_HISTORY_SLACK);
        let full_history_len = main_history_len.saturating_mul(DEFAULT_FULL_HISTORY_MULTIPLIER);
        Self {
            max_loop_len: DEFAULT_MAX_LOOP_LEN,
            main_history_len,
            full_history_len,
            recent_positions_len: DEFAULT_RECENT_POSITIONS,
            detect_stride: DEFAULT_DETECT_STRIDE,
        }
    }
}

impl LoopDetectorConfig {
    fn normalized(self) -> Self {
        let min_main = self
            .max_loop_len
            .saturating_mul(3)
            .saturating_add(DEFAULT_MAIN_HISTORY_SLACK);
        let main_history_len = if self.main_history_len == 0 {
            min_main
        } else {
            self.main_history_len.max(min_main)
        };
        let full_history_len = if self.full_history_len == 0 {
            main_history_len.saturating_mul(DEFAULT_FULL_HISTORY_MULTIPLIER)
        } else {
            self.full_history_len.max(main_history_len)
        };
        let recent_positions_len = self.recent_positions_len.max(2);
        Self {
            max_loop_len: self.max_loop_len.max(1),
            main_history_len,
            full_history_len,
            recent_positions_len,
            detect_stride: self.detect_stride,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoopStep {
    pub pc_before: u32,
    pub pc_after: u32,
    pub opcode: u8,
    pub instr_len: u8,
    pub in_interrupt: bool,
    pub irq_source: Option<LoopIrqSource>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopIrqSource {
    Key,
    Onk,
    Mti,
    Sti,
    Ir,
    Irq,
    Other,
}

impl LoopIrqSource {
    pub fn from_name(name: &str) -> Self {
        match name {
            "KEY" => Self::Key,
            "ONK" => Self::Onk,
            "MTI" => Self::Mti,
            "STI" => Self::Sti,
            "IR" => Self::Ir,
            "IRQ" => Self::Irq,
            _ => Self::Other,
        }
    }

    fn is_ir(self) -> bool {
        matches!(self, Self::Ir)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopSummary {
    pub start_pc: u32,
    pub len: usize,
    pub repeats: u32,
    pub start_index: u64,
    pub end_index: u64,
    pub detected_at: u64,
    pub candidate_lengths: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopCandidate {
    pub len: usize,
    pub repeats: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopBranchKind {
    Taken,
    NotTaken,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopBranchInfo {
    pub kind: LoopBranchKind,
    pub fallthrough: u32,
    pub target: u32,
    pub observed_targets: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopTraceEntry {
    pub index: u64,
    pub mainline_index: Option<u64>,
    pub pc_before: u32,
    pub pc_after: u32,
    pub opcode: u8,
    pub instr_len: u8,
    pub in_interrupt: bool,
    pub irq_source: Option<LoopIrqSource>,
    pub is_hardware_interrupt: bool,
    pub branch: Option<LoopBranchInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopReport {
    pub summary: LoopSummary,
    pub candidates: Vec<LoopCandidate>,
    pub mainline_start_index: u64,
    pub mainline_end_index: u64,
    pub full_start_index: Option<u64>,
    pub full_end_index: Option<u64>,
    pub full_trace_truncated: bool,
    pub trace: Vec<LoopTraceEntry>,
}

pub struct LoopDetector {
    config: LoopDetectorConfig,
    full_trace: VecDeque<FullEntry>,
    main_trace: VecDeque<MainEntry>,
    recent_positions: HashMap<u32, VecDeque<u64>>,
    edges: HashMap<u32, HashSet<u32>>,
    full_index: u64,
    main_index: u64,
    current_summary: Option<LoopSummary>,
    last_report: Option<LoopReport>,
}

impl LoopDetector {
    pub fn new(config: LoopDetectorConfig) -> Self {
        let config = config.normalized();
        Self {
            config,
            full_trace: VecDeque::with_capacity(config.full_history_len),
            main_trace: VecDeque::with_capacity(config.main_history_len),
            recent_positions: HashMap::new(),
            edges: HashMap::new(),
            full_index: 0,
            main_index: 0,
            current_summary: None,
            last_report: None,
        }
    }

    pub fn current_summary(&self) -> Option<&LoopSummary> {
        self.current_summary.as_ref()
    }

    pub fn last_report(&self) -> Option<&LoopReport> {
        self.last_report.as_ref()
    }

    pub fn set_detect_stride(&mut self, stride: u64) {
        self.config.detect_stride = stride;
    }

    pub fn record_step(&mut self, step: LoopStep) {
        let full_idx = self.full_index;
        self.full_index = self.full_index.saturating_add(1);
        let is_hardware_interrupt =
            step.in_interrupt && step.irq_source.is_none_or(|src| !src.is_ir());
        if self.full_trace.len() == self.config.full_history_len {
            self.full_trace.pop_front();
        }
        self.full_trace.push_back(FullEntry {
            idx: full_idx,
            pc_before: step.pc_before,
            pc_after: step.pc_after,
            opcode: step.opcode,
            instr_len: step.instr_len,
            in_interrupt: step.in_interrupt,
            irq_source: step.irq_source,
            is_hardware_interrupt,
            mainline_index: None,
        });

        let include_mainline = !is_hardware_interrupt && step.opcode != RETI_OPCODE;
        if !include_mainline {
            return;
        }

        let main_idx = self.main_index;
        self.main_index = self.main_index.saturating_add(1);
        if self.main_trace.len() == self.config.main_history_len {
            self.main_trace.pop_front();
        }
        self.main_trace.push_back(MainEntry {
            idx: main_idx,
            pc_before: step.pc_before,
            pc_after: step.pc_after,
            opcode: step.opcode,
            instr_len: step.instr_len,
            full_index: full_idx,
        });
        if let Some(last_full) = self.full_trace.back_mut() {
            last_full.mainline_index = Some(main_idx);
        }
        self.edges.entry(step.pc_before).or_default().insert(step.pc_after);
        self.update_recent_positions(step.pc_before, main_idx);
        self.maybe_detect(main_idx);
    }

    fn update_recent_positions(&mut self, pc: u32, idx: u64) {
        let front_idx = self.main_front_idx();
        let entries = self
            .recent_positions
            .entry(pc)
            .or_default();
        entries.push_back(idx);
        while entries.len() > self.config.recent_positions_len {
            entries.pop_front();
        }
        while entries.front().is_some_and(|v| *v < front_idx) {
            entries.pop_front();
        }
    }

    fn maybe_detect(&mut self, current_idx: u64) {
        let stride = self.config.detect_stride;
        if stride > 0 && !(current_idx + 1).is_multiple_of(stride) {
            return;
        }
        self.detect(current_idx);
    }

    fn detect(&mut self, current_idx: u64) {
        let current_pc = match self.main_trace.back() {
            Some(entry) => entry.pc_before,
            None => {
                self.current_summary = None;
                return;
            }
        };
        let candidates = self.collect_candidates(current_pc, current_idx);
        if candidates.is_empty() {
            self.current_summary = None;
            return;
        }

        let mut candidate_lengths: Vec<usize> = candidates.iter().map(|c| c.len).collect();
        candidate_lengths.sort_unstable();
        let primary = candidates
            .iter()
            .max_by_key(|candidate| candidate.len)
            .unwrap();
        let loop_len = primary.len;
        let loop_start_idx = current_idx + 1 - loop_len as u64;
        let loop_end_idx = current_idx;
        let loop_start_pc = self
            .main_at(loop_start_idx)
            .map(|entry| entry.pc_before)
            .unwrap_or(0);
        let summary = LoopSummary {
            start_pc: loop_start_pc,
            len: loop_len,
            repeats: primary.repeats,
            start_index: loop_start_idx,
            end_index: loop_end_idx,
            detected_at: current_idx,
            candidate_lengths,
        };
        let changed = self.current_summary.as_ref() != Some(&summary);
        self.current_summary = Some(summary.clone());
        if changed {
            self.last_report = Some(self.build_report(summary, candidates));
        }
    }

    fn collect_candidates(&self, current_pc: u32, current_idx: u64) -> Vec<LoopCandidate> {
        let mut candidates = Vec::new();
        let Some(positions) = self.recent_positions.get(&current_pc) else {
            return candidates;
        };
        let mut seen = BTreeSet::new();
        for &prev_idx in positions.iter().rev().skip(1) {
            let len = current_idx.saturating_sub(prev_idx) as usize;
            if len == 0 || len > self.config.max_loop_len {
                continue;
            }
            if !seen.insert(len) {
                continue;
            }
            if !self.check_repeat(current_idx, len) {
                continue;
            }
            let repeats = self.count_repeats(current_idx, len);
            if repeats >= 3 {
                candidates.push(LoopCandidate { len, repeats });
            }
        }
        candidates.sort_by_key(|candidate| candidate.len);
        candidates
    }

    fn check_repeat(&self, current_idx: u64, len: usize) -> bool {
        let len_u64 = len as u64;
        if current_idx + 1 < 3 * len_u64 {
            return false;
        }
        let start_a = current_idx + 1 - 3 * len_u64;
        if start_a < self.main_front_idx() {
            return false;
        }
        for offset in 0..len {
            let idx_a = start_a + offset as u64;
            let idx_b = idx_a + len_u64;
            let idx_c = idx_b + len_u64;
            let Some(a) = self.main_at(idx_a) else {
                return false;
            };
            let Some(b) = self.main_at(idx_b) else {
                return false;
            };
            let Some(c) = self.main_at(idx_c) else {
                return false;
            };
            if a.pc_before != b.pc_before || b.pc_before != c.pc_before {
                return false;
            }
        }
        true
    }

    fn count_repeats(&self, current_idx: u64, len: usize) -> u32 {
        let len_u64 = len as u64;
        let front_idx = self.main_front_idx();
        let mut repeats = 1u32;
        let mut start = current_idx + 1 - len_u64;
        while start >= len_u64 {
            let prev = start - len_u64;
            if prev < front_idx {
                break;
            }
            if !self.blocks_equal(prev, start, len) {
                break;
            }
            repeats = repeats.saturating_add(1);
            start = prev;
        }
        repeats
    }

    fn blocks_equal(&self, start_a: u64, start_b: u64, len: usize) -> bool {
        for offset in 0..len {
            let idx_a = start_a + offset as u64;
            let idx_b = start_b + offset as u64;
            let Some(a) = self.main_at(idx_a) else {
                return false;
            };
            let Some(b) = self.main_at(idx_b) else {
                return false;
            };
            if a.pc_before != b.pc_before {
                return false;
            }
        }
        true
    }

    fn main_front_idx(&self) -> u64 {
        self.main_trace.front().map(|entry| entry.idx).unwrap_or(0)
    }

    fn main_at(&self, idx: u64) -> Option<&MainEntry> {
        let front_idx = self.main_trace.front()?.idx;
        let offset = idx.checked_sub(front_idx)? as usize;
        self.main_trace.get(offset)
    }

    fn build_report(&self, summary: LoopSummary, candidates: Vec<LoopCandidate>) -> LoopReport {
        let mainline_start_index = summary.start_index;
        let mainline_end_index = summary.end_index;
        let full_start_index = self
            .main_at(mainline_start_index)
            .map(|entry| entry.full_index);
        let full_end_index = self
            .main_at(mainline_end_index)
            .map(|entry| entry.full_index);
        let mut trace = Vec::new();
        let mut full_trace_truncated = true;

        if let (Some(full_start), Some(full_end)) = (full_start_index, full_end_index) {
            if let (Some(front), Some(back)) = (self.full_trace.front(), self.full_trace.back()) {
                if full_start >= front.idx && full_end <= back.idx {
                    full_trace_truncated = false;
                    for entry in self.full_trace.iter() {
                        if entry.idx < full_start {
                            continue;
                        }
                        if entry.idx > full_end {
                            break;
                        }
                        trace.push(self.trace_entry_from_full(entry));
                    }
                }
            }
            if full_trace_truncated {
                for idx in mainline_start_index..=mainline_end_index {
                    if let Some(entry) = self.main_at(idx) {
                        trace.push(self.trace_entry_from_main(entry));
                    }
                }
            }
        }

        LoopReport {
            summary,
            candidates,
            mainline_start_index,
            mainline_end_index,
            full_start_index,
            full_end_index,
            full_trace_truncated,
            trace,
        }
    }

    fn trace_entry_from_full(&self, entry: &FullEntry) -> LoopTraceEntry {
        LoopTraceEntry {
            index: entry.idx,
            mainline_index: entry.mainline_index,
            pc_before: entry.pc_before,
            pc_after: entry.pc_after,
            opcode: entry.opcode,
            instr_len: entry.instr_len,
            in_interrupt: entry.in_interrupt,
            irq_source: entry.irq_source,
            is_hardware_interrupt: entry.is_hardware_interrupt,
            branch: self.branch_info(entry.opcode, entry.pc_before, entry.pc_after, entry.instr_len),
        }
    }

    fn trace_entry_from_main(&self, entry: &MainEntry) -> LoopTraceEntry {
        LoopTraceEntry {
            index: entry.full_index,
            mainline_index: Some(entry.idx),
            pc_before: entry.pc_before,
            pc_after: entry.pc_after,
            opcode: entry.opcode,
            instr_len: entry.instr_len,
            in_interrupt: false,
            irq_source: None,
            is_hardware_interrupt: false,
            branch: self.branch_info(entry.opcode, entry.pc_before, entry.pc_after, entry.instr_len),
        }
    }

    fn branch_info(
        &self,
        opcode: u8,
        pc_before: u32,
        pc_after: u32,
        instr_len: u8,
    ) -> Option<LoopBranchInfo> {
        if instr_len == 0 {
            return None;
        }
        let fallthrough = pc_before.wrapping_add(instr_len as u32) & ADDRESS_MASK;
        let taken = pc_after != fallthrough;
        let is_conditional = dispatch::lookup(opcode).is_some_and(|entry| {
            matches!(entry.kind, InstrKind::JpAbs | InstrKind::JpRel) && entry.cond.is_some()
        });
        let mut observed_targets = self
            .edges
            .get(&pc_before)
            .map(|targets| targets.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        observed_targets.sort_unstable();
        if observed_targets.is_empty() {
            observed_targets.push(fallthrough);
        }
        if taken {
            return Some(LoopBranchInfo {
                kind: LoopBranchKind::Taken,
                fallthrough,
                target: pc_after,
                observed_targets,
            });
        }
        if observed_targets.len() > 1 {
            return Some(LoopBranchInfo {
                kind: LoopBranchKind::NotTaken,
                fallthrough,
                target: fallthrough,
                observed_targets,
            });
        }
        if is_conditional {
            return Some(LoopBranchInfo {
                kind: LoopBranchKind::NotTaken,
                fallthrough,
                target: fallthrough,
                observed_targets,
            });
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
struct MainEntry {
    idx: u64,
    pc_before: u32,
    pc_after: u32,
    opcode: u8,
    instr_len: u8,
    full_index: u64,
}

#[derive(Clone, Copy, Debug)]
struct FullEntry {
    idx: u64,
    pc_before: u32,
    pc_after: u32,
    opcode: u8,
    instr_len: u8,
    in_interrupt: bool,
    irq_source: Option<LoopIrqSource>,
    is_hardware_interrupt: bool,
    mainline_index: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_step(
        detector: &mut LoopDetector,
        pc_before: u32,
        pc_after: u32,
        opcode: u8,
        instr_len: u8,
        in_interrupt: bool,
        irq_source: Option<LoopIrqSource>,
    ) {
        detector.record_step(LoopStep {
            pc_before,
            pc_after,
            opcode,
            instr_len,
            in_interrupt,
            irq_source,
        });
    }

    fn record_main(detector: &mut LoopDetector, pc: u32) {
        let pc_after = pc.wrapping_add(1) & ADDRESS_MASK;
        record_step(detector, pc, pc_after, 0x00, 1, false, None);
    }

    fn record_irq(detector: &mut LoopDetector, pc: u32, src: LoopIrqSource) {
        let pc_after = pc.wrapping_add(1) & ADDRESS_MASK;
        record_step(detector, pc, pc_after, 0x00, 1, true, Some(src));
    }

    #[test]
    fn detects_repeated_loop() {
        let config = LoopDetectorConfig {
            max_loop_len: 8,
            main_history_len: 32,
            full_history_len: 64,
            detect_stride: 0,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        let pcs = [0x10, 0x20, 0x30];
        for idx in 0..9 {
            let pc = pcs[idx % pcs.len()];
            detector.record_step(LoopStep {
                pc_before: pc,
                pc_after: pc.wrapping_add(1) & ADDRESS_MASK,
                opcode: 0x00,
                instr_len: 1,
                in_interrupt: false,
                irq_source: None,
            });
        }
        let summary = detector.current_summary().expect("loop summary");
        assert_eq!(summary.len, 3);
        assert!(summary.repeats >= 3);
        assert_eq!(summary.start_pc, 0x10);
    }

    #[test]
    fn detects_multiple_candidates_picks_longest() {
        let config = LoopDetectorConfig {
            max_loop_len: 4,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        for _ in 0..6 {
            record_main(&mut detector, 0x10);
        }
        let summary = detector.current_summary().expect("loop summary");
        assert_eq!(summary.len, 2);
        assert_eq!(summary.candidate_lengths, vec![1, 2]);
    }

    #[test]
    fn skips_hardware_interrupts_and_reti() {
        let config = LoopDetectorConfig {
            max_loop_len: 4,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        record_irq(&mut detector, 0x9000, LoopIrqSource::Key);
        record_main(&mut detector, 0x10);
        record_step(
            &mut detector,
            0x20,
            0x21,
            RETI_OPCODE,
            1,
            false,
            None,
        );
        record_main(&mut detector, 0x30);
        assert_eq!(detector.main_index, 2);
        assert_eq!(detector.main_trace.len(), 2);
        assert_eq!(detector.full_trace.len(), 4);
    }

    #[test]
    fn includes_ir_source_even_when_in_interrupt() {
        let config = LoopDetectorConfig {
            max_loop_len: 4,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        record_irq(&mut detector, 0x7000, LoopIrqSource::Ir);
        assert_eq!(detector.main_index, 1);
        assert_eq!(detector.main_trace.len(), 1);
        let last_full = detector.full_trace.back().expect("full entry");
        assert!(!last_full.is_hardware_interrupt);
    }

    #[test]
    fn detect_stride_gates_detection() {
        let config = LoopDetectorConfig {
            max_loop_len: 2,
            detect_stride: 2,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        for _ in 0..3 {
            record_main(&mut detector, 0x10);
        }
        assert!(detector.current_summary().is_none());
        record_main(&mut detector, 0x10);
        assert!(detector.current_summary().is_some());
    }

    #[test]
    fn ignores_short_repeats() {
        let config = LoopDetectorConfig {
            max_loop_len: 2,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        record_main(&mut detector, 0x10);
        record_main(&mut detector, 0x20);
        record_main(&mut detector, 0x10);
        record_main(&mut detector, 0x20);
        assert!(detector.current_summary().is_none());
    }

    #[test]
    fn respects_max_loop_len() {
        let config = LoopDetectorConfig {
            max_loop_len: 3,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        let pcs = [0x10, 0x20, 0x30, 0x40];
        for _ in 0..3 {
            for pc in pcs {
                record_main(&mut detector, pc);
            }
        }
        assert!(detector.current_summary().is_none());
    }

    #[test]
    fn branch_not_taken_when_multiple_edges() {
        let config = LoopDetectorConfig {
            max_loop_len: 2,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        record_step(
            &mut detector,
            0x40,
            0x60,
            0x00,
            1,
            false,
            None,
        );
        for _ in 0..3 {
            record_step(
                &mut detector,
                0x40,
                0x41,
                0x00,
                1,
                false,
                None,
            );
        }
        let report = detector.last_report().expect("loop report");
        let entry = report
            .trace
            .iter()
            .find(|item| item.pc_before == 0x40)
            .expect("loop entry");
        let branch = entry.branch.as_ref().expect("branch info");
        assert_eq!(branch.kind, LoopBranchKind::NotTaken);
        assert!(branch.observed_targets.contains(&0x41));
        assert!(branch.observed_targets.contains(&0x60));
    }

    #[test]
    fn branch_not_taken_when_conditional_only_fallthrough() {
        let config = LoopDetectorConfig {
            max_loop_len: 1,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        for _ in 0..3 {
            record_step(
                &mut detector,
                0x80,
                0x83,
                0x14,
                3,
                false,
                None,
            );
        }
        let report = detector.last_report().expect("loop report");
        let entry = report
            .trace
            .iter()
            .find(|item| item.pc_before == 0x80)
            .expect("loop entry");
        let branch = entry.branch.as_ref().expect("branch info");
        assert_eq!(branch.kind, LoopBranchKind::NotTaken);
    }

    #[test]
    fn report_truncates_when_full_trace_overflows() {
        let main_history_len = 5usize
            .saturating_mul(3)
            .saturating_add(DEFAULT_MAIN_HISTORY_SLACK);
        let config = LoopDetectorConfig {
            max_loop_len: 5,
            main_history_len,
            full_history_len: main_history_len,
            ..Default::default()
        };
        let mut detector = LoopDetector::new(config);
        let pattern = [0x10, 0x20, 0x30, 0x40, 0x50];
        let irq_fill = 20;
        for rep in 0..3 {
            for (idx, pc) in pattern.iter().enumerate() {
                record_main(&mut detector, *pc);
                let is_last = rep == 2 && idx == pattern.len() - 1;
                if !is_last {
                    for _ in 0..irq_fill {
                        record_irq(&mut detector, 0x9000, LoopIrqSource::Key);
                    }
                }
            }
        }
        let report = detector.last_report().expect("loop report");
        assert!(report.full_trace_truncated);
        assert_eq!(report.trace.len(), 5);
        assert!(report.trace.iter().all(|entry| !entry.in_interrupt));
    }
}
