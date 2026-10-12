// PY_SOURCE: pce500/pvf1/auxiliary.py
//! PV-F1 FF406/FF40E control transport. Absent mode exposes the acquired idle
//! status and lets ROM time out. The opt-in control candidate reuses the shared
//! handshake engine, gated to PV-F1-confirmed request/response contracts only.
use crate::handwriting::{Model, Strokes};
use crate::pi3000::auxiliary::Auxiliary as ControlEngine;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Provisional sample cadence, independent of the acquired register/IRQ ABI.
pub const CAPTURE_SAMPLE_PERIOD_CANDIDATE: u64 = 4096;

#[derive(Default)]
pub struct CaptureClock {
    period: Option<u64>,
    next: Option<u64>,
    ready: bool,
    events: u64,
}
impl CaptureClock {
    fn advance(&mut self, active: bool, cycle: u64) -> bool {
        if !active || self.period.is_none() {
            self.next = None;
            self.ready = false;
            return false;
        }
        if self.ready {
            return false;
        }
        let deadline = *self
            .next
            .get_or_insert_with(|| cycle.saturating_add(self.period.unwrap()));
        if cycle < deadline {
            return false;
        }
        self.next = None;
        self.ready = true;
        self.events += 1;
        true
    }
}

/// Explicit controller-response fixture. It does not recognize pen strokes.
#[derive(Default)]
pub struct CandidateResults {
    ready: bool,
    irq_sent: bool,
    pages: BTreeMap<u8, Vec<[u8; 3]>>,
    response: Option<Vec<u8>>,
}

/// Optional portable inference backend; protocol state remains PV-owned.
#[derive(Default)]
pub struct Recognition {
    model: Option<Arc<Model>>,
    width: u16,
    height: u16,
    active: Option<u8>,
    strokes: BTreeMap<u8, Strokes>,
    in_stroke: bool,
    classifications: u64,
    last_labels: Vec<String>,
}
type RecognitionResult = Option<(u8, Vec<[u8; 3]>)>;
impl Recognition {
    fn upload(&mut self, samples: &[[u16; 2]]) -> Result<(), String> {
        if self.model.is_none() {
            return Ok(());
        }
        for &[x, y] in samples {
            if x == 0x8000 && y < 4 {
                self.active = Some(y as u8);
                self.in_stroke = false;
            } else if x == 0xffff && y == 0xffff {
                self.in_stroke = false;
            } else if x < 0x8000 && y < 0x8000 {
                let index = self
                    .active
                    .ok_or("recognizer sample without start marker")?;
                let strokes = self.strokes.entry(index).or_default();
                if !self.in_stroke {
                    strokes.push(vec![]);
                    self.in_stroke = true;
                }
                if strokes.len() > 64 {
                    return Err("too many recognition strokes".into());
                }
                let stroke = strokes.last_mut().unwrap();
                if stroke.len() >= 16384 {
                    return Err("too many recognition points".into());
                }
                if stroke.last() != Some(&[x as f32, y as f32]) {
                    stroke.push([x as f32, y as f32]);
                }
            } else {
                return Err(format!("unrecovered recognition marker{x:04X}/{y:04X}"));
            }
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<RecognitionResult, String> {
        let Some(model) = &self.model else {
            return Ok(None);
        };
        let index = self
            .active
            .take()
            .ok_or("recognizer finish without character")?;
        self.in_stroke = false;
        let strokes = self
            .strokes
            .remove(&index)
            .ok_or("recognizer finish without ink")?;
        let labels = model.classify(
            &strokes,
            self.width as f32,
            self.height as f32,
            model.class_count(),
        )?;
        let mut records = Vec::new();
        self.last_labels.clear();
        for (label, _) in labels {
            let Some(mut c) = label.chars().next().filter(|_| label.chars().count() == 1) else {
                continue;
            };
            if c.is_ascii_graphic() {
                c = char::from_u32(c as u32 + 0xfee0).unwrap();
            }
            let text = c.to_string();
            let (bytes, _, error) = encoding_rs::SHIFT_JIS.encode(&text);
            if !error && bytes.len() == 2 {
                records.push([bytes[0], bytes[1], 0]);
                self.last_labels.push(text);
            }
            if records.len() == 10 {
                break;
            }
        }
        self.classifications += 1;
        Ok(Some((index, records)))
    }
}

pub enum Auxiliary {
    Absent {
        request: [u8; 32],
        controls: [u8; 3],
        enabled: bool,
        starts: u64,
    },
    Control {
        request: [u8; 32],
        controls: [u8; 3],
        enabled: bool,
        engine: ControlEngine,
        clock: CaptureClock,
        sample_uploads: u64,
        completed_strokes: u64,
        last_samples: Vec<[u16; 2]>,
        results: Box<CandidateResults>,
        recognition: Box<Recognition>,
    },
}
impl Auxiliary {
    pub fn absent() -> Self {
        Self::Absent {
            request: [0; 32],
            controls: [0; 3],
            enabled: true,
            starts: 0,
        }
    }
    pub fn control() -> Self {
        Self::Control {
            request: [0; 32],
            controls: [0; 3],
            enabled: true,
            engine: ControlEngine::default(),
            clock: CaptureClock::default(),
            sample_uploads: 0,
            completed_strokes: 0,
            last_samples: vec![],
            results: Box::default(),
            recognition: Box::default(),
        }
    }
    pub fn configure_recognizer(&mut self, model: Option<Arc<Model>>) -> Result<(), String> {
        let Self::Control {
            recognition,
            engine,
            ..
        } = self
        else {
            return Err("recognizer requires auxiliary candidate".into());
        };
        if engine.capture_active {
            return Err("Leave the handwriting editor before changing the model".into());
        }
        **recognition = Recognition {
            model,
            width: recognition.width,
            height: recognition.height,
            ..Recognition::default()
        };
        Ok(())
    }
    /// Supply an opaque type-1 result from a controller fixture or replay.
    /// Native FF463 and DD4DF assign four query indices; DD511 accepts 0..10
    /// three-byte records. No text, PC or guest workspace is modified here.
    pub fn supply_candidate_result_fixture(
        &mut self,
        index: u8,
        candidates: Vec<[u8; 3]>,
    ) -> Result<(), String> {
        if index >= 4 || candidates.len() > 10 {
            return Err("PV-F1 result requires index 0..3 and at most 10 records".into());
        }
        let Self::Control { results, .. } = self else {
            return Err("controller result fixture requires the auxiliary candidate".into());
        };
        if results.ready || results.pages.contains_key(&index) {
            return Err("PV-F1 result fixture would replace an unconsumed result".into());
        }
        results.pages.insert(index, candidates);
        results.ready = true;
        results.irq_sent = false;
        Ok(())
    }
    /// Separate GA source6 edge, enabled by native08A06.bit6. Acknowledgment
    /// at08A08=BF clears this notification but preserves indexed query data.
    pub fn take_result_irq_edge(&mut self) -> bool {
        let Self::Control {
            results,
            controls,
            enabled,
            ..
        } = self
        else {
            return false;
        };
        if *enabled && controls[0] & 0x40 != 0 && results.ready && !results.irq_sent {
            results.irq_sent = true;
            return true;
        }
        false
    }
    pub fn configure_sample_period(&mut self, period: Option<u64>) -> Result<(), String> {
        if period == Some(0) {
            return Err("PV-F1 capture sample period must be positive".into());
        }
        let Self::Control { clock, .. } = self else {
            return Err("capture clock requires the auxiliary control candidate".into());
        };
        clock.period = period;
        clock.next = None;
        clock.ready = false;
        Ok(())
    }
    pub fn snapshot(&self) -> Value {
        match self {
            Self::Absent {
                controls,
                enabled,
                starts,
                ..
            } => {
                json!({"control_candidate":false,"controls":controls.to_vec(),"enabled":enabled,"starts":starts})
            }
            Self::Control {
                controls,
                enabled,
                engine,
                clock,
                sample_uploads,
                completed_strokes,
                last_samples,
                results,
                recognition,
                ..
            } => {
                json!({"control_candidate":true,"controls":controls.to_vec(),"enabled":enabled,"capture_active":engine.capture_active,"accepted_controls":engine.accepted,"accepted_count":engine.accepted_count,"sample_period_candidate":clock.period,"sample_ready":clock.ready,"sample_events":clock.events,"sample_uploads":sample_uploads,"completed_strokes":completed_strokes,"last_uploaded_samples":last_samples,"result_fixture_ready":results.ready,"result_fixture_indices":results.pages.keys().collect::<Vec<_>>(),"recognizer_loaded":recognition.model.is_some(),"recognition_canvas":[recognition.width,recognition.height],"classifications":recognition.classifications,"recognized_labels":recognition.last_labels })
            }
        }
    }
    pub fn peek(&self, address: u32) -> Option<u8> {
        match (address, self) {
            (0x8a00, Self::Absent { enabled, .. }) => Some(if *enabled { 2 } else { 0 }),
            (
                0x8a00,
                Self::Control {
                    engine, enabled, ..
                },
            ) => Some((engine.status() & !2) | if *enabled { 2 } else { 0 }),
            // FE DE39B/DE3AB explicitly compare writes to06 with reads at04.
            (0x8a04, Self::Absent { controls, .. } | Self::Control { controls, .. }) => {
                Some(controls[0])
            }
            (0x8a05, Self::Control { clock, results, .. }) => {
                Some(if clock.ready { 0x80 } else { 0 } | if results.ready { 0x40 } else { 0 })
            }
            (0x8a09, Self::Control { results, .. }) => Some(u8::from(results.ready)),
            (
                0x8a41..=0x8a5f,
                Self::Control {
                    engine, results, ..
                },
            ) if engine.status() & 0x40 != 0 => results
                .response
                .as_ref()
                .and_then(|r| r.get((address - 0x8a41) as usize))
                .copied(),
            (0x8a05 | 0x8a09, _) => Some(0),
            (0x8a40, Self::Control { engine, .. }) => engine.response(0),
            _ => None,
        }
    }
    /// One fresh sample-ready edge. ROM acknowledges the80 status through07;
    /// peeks and the GA pending latch do not consume or repeatedly assert it.
    pub fn advance(&mut self, cycle: u64) -> bool {
        if let Self::Control {
            engine,
            clock,
            controls,
            enabled,
            ..
        } = self
        {
            engine.advance(cycle);
            return clock.advance(
                *enabled && engine.capture_active && controls[0] & 0x80 != 0,
                cycle,
            );
        }
        false
    }
    pub fn write(&mut self, address: u32, value: u8, cycle: u64) -> Result<(), String> {
        match address {
            // FE DD3AE/DD3BD drives FD, delay, 02, delay. Control bit1 feedback
            // and reset cancellation remain explicit controller hypotheses.
            0x8a01 if matches!(value, 2 | 0xfd) => {
                let model = match self {
                    Self::Control { recognition, .. } => recognition.model.clone(),
                    _ => None,
                };
                let period = match self {
                    Self::Control { clock, .. } => clock.period,
                    Self::Absent { .. } => None,
                };
                if matches!(self, Self::Absent { .. }) {
                    *self = Self::absent();
                } else {
                    *self = Self::control();
                    self.configure_sample_period(period)?;
                    self.configure_recognizer(model)?;
                }
                let enabled = match self {
                    Self::Absent { enabled, .. } | Self::Control { enabled, .. } => enabled,
                };
                *enabled = value == 2;
                Ok(())
            }
            0x8a20..=0x8a3f => {
                let offset = (address - 0x8a20) as usize;
                match self {
                    Self::Absent { request, .. } => request[offset] = value,
                    Self::Control {
                        request, engine, ..
                    } => {
                        request[offset] = value;
                        engine.mark_written(offset);
                    }
                }
                Ok(())
            }
            // Bank FE DB947/DB94D/DB953 initializes these latches.
            // Interrupt production from them is not yet recovered.
            0x8a06..=0x8a08 => {
                let controls = match self {
                    Self::Absent { controls, .. } | Self::Control { controls, .. } => controls,
                };
                controls[(address - 0x8a06) as usize] = value;
                if let Self::Control { clock, results, .. } = self {
                    if address == 0x8a07 && value & 0x80 == 0 {
                        clock.ready = false;
                        clock.next = None;
                    }
                    if address == 0x8a08 && value & 0x40 == 0 {
                        results.ready = false;
                        results.irq_sent = false;
                    }
                }
                Ok(())
            }
            0x8a02 if value == 0x80 => match self {
                Self::Absent { starts, .. } => {
                    *starts += 1;
                    Ok(())
                }
                Self::Control {
                    request,
                    engine,
                    sample_uploads,
                    completed_strokes,
                    last_samples,
                    results,
                    recognition,
                    ..
                } => {
                    let command = u16::from_le_bytes([request[0], request[1]]);
                    if command == 0x02fc {
                        let index = request[2];
                        let page = results.pages.get(&index).ok_or_else(|| {
                            format!("PV-F1 query index {index} has no controller result fixture")
                        })?;
                        engine.start_packet(request, cycle, 3, None, true)?;
                        let mut response = vec![page.len() as u8];
                        response.extend(page.iter().flatten());
                        results.response = Some(response);
                        results.pages.remove(&index);
                        return Ok(());
                    }
                    if command == 0x00fc {
                        // DBF57/DBF60 ends the upload with the same busy-only
                        // start. Candidate-result IRQ/data remain unresolved.
                        engine.start_packet(request, cycle, 2, None, false)?;
                        *completed_strokes += 1;
                        if let Some((index, records)) = recognition.finish()? {
                            if results.ready || results.pages.contains_key(&index) {
                                return Err("recognizer result would replace pending page".into());
                            }
                            results.pages.insert(index, records);
                            results.ready = true;
                            results.irq_sent = false;
                        }
                        return Ok(());
                    }
                    // FE:DBEBD..DBF20 uploads one to seven converted coordinate
                    // pairs at22 onward, then FF/count at20. No response payload
                    // is consumed here. This does not synthesize recognition.
                    let count = request[1] as usize;
                    if request[0] == 0xff && (1..=7).contains(&count) {
                        // DBEBD tests only busy80 before another upload. Unlike
                        // status commands, sample batches have no host BF ack.
                        engine.start_packet(request, cycle, 2 + 4 * count, None, false)?;
                        *sample_uploads += 1;
                        *last_samples = request[2..2 + 4 * count]
                            .chunks_exact(4)
                            .map(|p| {
                                [
                                    u16::from_le_bytes([p[0], p[1]]),
                                    u16::from_le_bytes([p[2], p[3]]),
                                ]
                            })
                            .collect();
                        recognition.upload(last_samples)?;
                        return Ok(());
                    }
                    // FE DD77B..DD7B3 sends two-byte controls; DD55E sends
                    // 04FE plus one parameter (3 bytes); DD589 sends 03FE plus geometry
                    // (8 bytes). These callers consume only status 01 at 08A40
                    // and acknowledge BF; recognition/query bytes remain absent.
                    // Their successful status is an explicit candidate.
                    if !matches!(
                        command,
                        0x03fc
                            | 0x05fe
                            | 0x00fe
                            | 0x01fe
                            | 0x02fe
                            | 0x30fb
                            | 0x00f9
                            | 0x01f9
                            | 0x02f9
                            | 0x04fe
                            | 0x03fe
                    ) {
                        return Err(format!("unrecovered PV-F1 auxiliary command {command:04X}"));
                    }
                    engine.start(request, cycle)?;
                    if command == 0x03fe {
                        recognition.width = u16::from_le_bytes([request[2], request[3]]);
                        recognition.height = u16::from_le_bytes([request[4], request[5]]);
                        recognition.active = None;
                        recognition.strokes.clear();
                        recognition.in_stroke = false;
                    }
                    Ok(())
                }
            },
            0x8a03 if value == 0xbf => match self {
                Self::Absent { .. } => Ok(()),
                Self::Control {
                    engine, results, ..
                } => {
                    engine.acknowledge(value)?;
                    results.response = None;
                    Ok(())
                }
            },
            _ => Err(format!(
                "unrecovered PV-F1 auxiliary write {address:05X}={value:02X}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start(a: &mut Auxiliary, header: u16, cycle: u64) {
        for (i, b) in header.to_le_bytes().into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, cycle).unwrap();
        }
        a.write(0x8a02, 0x80, cycle).unwrap();
    }

    #[test]
    fn changing_host_model_preserves_native_geometry_but_reset_clears_it() {
        let mut a = Auxiliary::control();
        if let Auxiliary::Control { recognition, .. } = &mut a {
            recognition.width = 138;
            recognition.height = 222;
            recognition.active = Some(1);
        }
        a.configure_recognizer(None).unwrap();
        assert_eq!(a.snapshot()["recognition_canvas"], json!([138, 222]));
        if let Auxiliary::Control { recognition, .. } = &a {
            assert_eq!(recognition.active, None);
        }
        a.write(0x8a01, 2, 0).unwrap();
        assert_eq!(a.snapshot()["recognition_canvas"], json!([0, 0]));
        if let Auxiliary::Control { engine, .. } = &mut a {
            engine.capture_active = true;
        }
        let before = a.snapshot();
        assert!(a.configure_recognizer(None).is_err());
        assert_eq!(a.snapshot(), before);
    }

    #[test]
    fn result_fixture_irq_ack_and_query_ack_are_independent() {
        let mut a = Auxiliary::control();
        a.supply_candidate_result_fixture(2, vec![[0x82, 0x60, 7]])
            .unwrap();
        assert!(!a.take_result_irq_edge());
        assert_eq!(a.peek(0x8a05), Some(0x40));
        assert_eq!(a.peek(0x8a09), Some(1));
        assert!(a.supply_candidate_result_fixture(2, vec![]).is_err());
        a.write(0x8a06, 0x40, 0).unwrap();
        assert!(a.take_result_irq_edge());
        assert!(!a.take_result_irq_edge());
        a.write(0x8a07, 0x7f, 1).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0x40));
        a.write(0x8a08, 0xbf, 2).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0));
        assert!(!a.take_result_irq_edge());
        for (i, b) in [0xfc, 2, 2].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 3).unwrap();
        }
        a.write(0x8a02, 0x80, 3).unwrap();
        assert_eq!(a.peek(0x8a41), None);
        a.advance(35);
        assert_eq!(a.peek(0x8a41), Some(1));
        assert_eq!(a.peek(0x8a42), Some(0x82));
        assert_eq!(a.peek(0x8a44), Some(7));
        a.write(0x8a03, 0xbf, 35).unwrap();
        assert_eq!(a.peek(0x8a41), None);
        assert!(a.supply_candidate_result_fixture(4, vec![]).is_err());
        assert!(a
            .supply_candidate_result_fixture(0, vec![[0; 3]; 11])
            .is_err());
        a.supply_candidate_result_fixture(2, vec![]).unwrap();
        a.write(0x8a01, 2, 40).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0));
        assert_eq!(a.snapshot()["result_fixture_indices"], json!([]));
    }

    #[test]
    fn native_sample_uploads_are_busy_only_and_require_exact_model_lengths() {
        let mut a = Auxiliary::control();
        for (cycle, count) in [(0, 1), (100, 7)] {
            let mut request = vec![0xff, count];
            for i in 0..count {
                request.extend_from_slice(&(i as u16).to_le_bytes());
                request.extend_from_slice(&(0x8000 + i as u16).to_le_bytes());
            }
            for (i, b) in request.iter().enumerate() {
                a.write(0x8a20 + i as u32, *b, cycle).unwrap();
            }
            a.write(0x8a02, 0x80, cycle).unwrap();
            assert_eq!(a.peek(0x8a00), Some(0x82));
            a.advance(cycle + 32);
            assert_eq!(a.peek(0x8a00), Some(2));
            assert_eq!(a.peek(0x8a40), None);
            assert_eq!(a.peek(0x8a41), None);
            assert_eq!(a.snapshot()["last_uploaded_samples"][0], json!([0, 0x8000]));
        }
        start(&mut a, 0x00fc, 200);
        a.advance(232);
        assert_eq!(a.snapshot()["completed_strokes"], 1);
        assert_eq!(a.peek(0x8a00), Some(2));
        // Capture controls remain acknowledged and never acquire query bytes.
        start(&mut a, 0x03fc, 300);
        a.advance(332);
        assert_eq!(a.peek(0x8a00), Some(0x42));
        a.write(0x8a03, 0xbf, 332).unwrap();
        for (i, b) in [0xff, 7, 0, 0, 0, 0].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 400).unwrap();
        }
        assert!(a.write(0x8a02, 0x80, 400).is_err());
        assert_eq!(a.snapshot()["sample_uploads"], 2);
        assert_eq!(a.peek(0x8a41), None);
        let mut pi = ControlEngine::default();
        let request = [0xff; 32];
        assert!(
            pi.start(&request, 0).is_err(),
            "PV uploads must not change PI command admission"
        );
    }

    #[test]
    fn capture_requires_an_explicit_cadence_and_preserves_ready_until_ack() {
        let mut a = Auxiliary::control();
        start(&mut a, 0x03fc, 0);
        assert!(!a.advance(32));
        a.write(0x8a03, 0xbf, 32).unwrap();
        a.write(0x8a06, 0x80, 32).unwrap();
        assert!(!a.advance(10000));
        assert_eq!(a.peek(0x8a05), Some(0));
        a.configure_sample_period(Some(4096)).unwrap();
        assert!(!a.advance(100));
        assert!(!a.advance(4195));
        assert!(a.advance(4196));
        for _ in 0..3 {
            assert_eq!(a.peek(0x8a04), Some(0x80));
            assert_eq!(a.peek(0x8a05), Some(0x80));
            assert_eq!(a.peek(0x8a41), None);
        }
        assert!(!a.advance(100000));
        a.write(0x8a07, 0xff, 100000).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0x80));
        a.write(0x8a07, 0x7f, 100000).unwrap();
        assert_eq!(a.peek(0x8a05), Some(0));
        assert!(!a.advance(100000));
        assert!(a.advance(104096));
        assert_eq!(a.snapshot()["sample_events"], 2);
        start(&mut a, 0x05fe, 104097);
        assert!(!a.advance(104129));
        assert_eq!(a.peek(0x8a05), Some(0));
        assert_eq!(a.peek(0x8a40), Some(1));
        assert_eq!(a.peek(0x8a41), None);
    }

    #[test]
    fn native_control_readback_and_reset_keep_configuration_but_cancel_capture() {
        let mut a = Auxiliary::control();
        a.configure_sample_period(Some(4096)).unwrap();
        for value in [0xaa, 0x55] {
            a.write(0x8a06, value, 0).unwrap();
            assert_eq!(a.peek(0x8a04), Some(value));
        }
        assert!(a.configure_sample_period(Some(0)).is_err());
        assert_eq!(a.snapshot()["sample_period_candidate"], 4096);
        start(&mut a, 0x03fc, 0);
        a.advance(32);
        a.write(0x8a01, 0xfd, 32).unwrap();
        a.write(0x8a01, 2, 33).unwrap();
        assert_eq!(a.snapshot()["sample_period_candidate"], 4096);
        assert_eq!(a.snapshot()["capture_active"], false);
        assert!(!a.advance(500000));
        a.configure_sample_period(None).unwrap();
        assert_eq!(a.snapshot()["sample_period_candidate"], Value::Null);
    }
}
