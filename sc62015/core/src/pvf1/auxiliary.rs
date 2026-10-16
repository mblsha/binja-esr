// PY_SOURCE: pce500/pvf1/auxiliary.py
//! PV-F1 FF406/FF40E control transport. Absent mode exposes the acquired idle
//! status and lets ROM time out. The opt-in control candidate reuses the shared
//! handshake engine, gated to PV-F1-confirmed request/response contracts only.
use crate::handwriting::{template_distance, Model, Strokes};
use crate::pi3000::auxiliary::Auxiliary as ControlEngine;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Provisional sample cadence, independent of the acquired register/IRQ ABI.
pub const CAPTURE_SAMPLE_PERIOD_CANDIDATE: u64 = 4096;

#[derive(Default, serde::Serialize, serde::Deserialize)]
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
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct CandidateResults {
    ready: bool,
    notification_type: u8,
    irq_sent: bool,
    pages: BTreeMap<u8, Vec<[u8; 3]>>,
    response: Option<Vec<u8>>,
}

/// Optional portable inference backend; protocol state remains PV-owned.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct Recognition {
    #[serde(skip)]
    model: Option<Arc<Model>>,
    width: u16,
    height: u16,
    configuration: u16,
    input_flags: u8,
    registration_mode: bool,
    active: Option<u8>,
    strokes: BTreeMap<u8, Strokes>,
    in_stroke: bool,
    classifications: u64,
    last_labels: Vec<String>,
}
type RecognitionResult = Option<(u8, Vec<[u8; 3]>)>;
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct RegisteredGlyph {
    code: u16,
    width: u16,
    height: u16,
    strokes: Strokes,
}
impl Recognition {
    fn upload(&mut self, samples: &[[u16; 2]]) -> Result<(), String> {
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
    fn finish(
        &mut self,
        bound_samples: &BTreeMap<u8, RegisteredGlyph>,
    ) -> Result<RecognitionResult, String> {
        if self.model.is_none() && bound_samples.is_empty() {
            return Ok(None);
        }
        let index = self
            .active
            .take()
            .ok_or("recognizer finish without character")?;
        self.in_stroke = false;
        let strokes = self
            .strokes
            .remove(&index)
            .ok_or("recognizer finish without ink")?;
        let mut matched = Vec::new();
        for glyph in bound_samples.values() {
            if let Some(distance) = template_distance(
                &strokes,
                [self.width as f32, self.height as f32],
                &glyph.strokes,
                [glyph.width as f32, glyph.height as f32],
            )? {
                // Host template policy, not recovered controller silicon.
                // Only near repeats precede the ordinary inference model.
                if distance <= 0.01 {
                    let bytes = glyph.code.to_le_bytes();
                    let (text, error) = encoding_rs::SHIFT_JIS.decode_without_bom_handling(&bytes);
                    if !error {
                        matched.push((text.into_owned(), -(distance as f32)));
                    }
                }
            }
        }
        matched.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut labels = matched;
        if let Some(model) = &self.model {
            labels.extend(model.classify(
                &strokes,
                self.width as f32,
                self.height as f32,
                model.class_count(),
            )?);
        }
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
                if self.last_labels.contains(&text) {
                    continue;
                }
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

#[derive(serde::Serialize, serde::Deserialize)]
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
        engine: Box<ControlEngine>,
        clock: Box<CaptureClock>,
        sample_uploads: u64,
        completed_strokes: u64,
        last_samples: Vec<[u16; 2]>,
        results: Box<CandidateResults>,
        recognition: Box<Recognition>,
        // FE DCAE1..DCAFA queries ten user-registration slots. DD5D7 binds
        // the entered two-byte character to captured ink through10FB.
        registrations: BTreeMap<u8, u16>,
        registration_response: Option<Vec<u8>>,
        registration_exchange: bool,
        registration_channel: Option<u8>,
        registered_glyphs: BTreeMap<u8, RegisteredGlyph>,
        raw_confirmation: bool,
        stroke_read: Option<(u8, Vec<u8>, usize)>,
    },
}
impl Auxiliary {
    pub(super) fn validate_live_state(&self) -> Result<(), String> {
        if let Self::Control {
            clock,
            recognition,
            registered_glyphs,
            stroke_read,
            ..
        } = self
        {
            if clock.period == Some(0)
                || recognition.active.is_some_and(|index| index >= 4)
                || recognition.strokes.len() > 4
                || registered_glyphs.len() > 10
                || stroke_read
                    .as_ref()
                    .is_some_and(|(_, bytes, offset)| *offset > bytes.len())
            {
                return Err("invalid live auxiliary clock/channel/cursor".into());
            }
            let valid_strokes = |strokes: &Strokes| {
                strokes.len() <= 64
                    && strokes.iter().all(|s| s.len() <= 16384)
                    && strokes.iter().flatten().flatten().all(|v| v.is_finite())
            };
            if recognition.strokes.values().any(|s| !valid_strokes(s))
                || registered_glyphs
                    .values()
                    .any(|glyph| !valid_strokes(&glyph.strokes))
            {
                return Err("invalid live auxiliary stroke geometry".into());
            }
        }
        Ok(())
    }
    pub(super) fn live_model_identity(&self) -> Option<String> {
        match self {
            Self::Control { recognition, .. } => {
                recognition.model.as_ref().map(|m| m.state_fingerprint())
            }
            _ => None,
        }
    }
    pub(super) fn restore_live_model(&mut self, source: &Self) {
        if let (
            Self::Control {
                recognition: dst, ..
            },
            Self::Control {
                recognition: src, ..
            },
        ) = (self, source)
        {
            dst.model = src.model.clone();
        }
    }
    pub(super) fn registered_state(&self) -> Option<Vec<u8>> {
        match self {
            Self::Control {
                registered_glyphs, ..
            } if !registered_glyphs.is_empty() => Some(
                serde_json::to_vec(registered_glyphs).expect("finite uploaded stroke geometry"),
            ),
            _ => None,
        }
    }
    pub(super) fn decode_registered_state(
        bytes: &[u8],
    ) -> Result<BTreeMap<u8, RegisteredGlyph>, String> {
        let glyphs: BTreeMap<u8, RegisteredGlyph> =
            serde_json::from_slice(bytes).map_err(|e| format!("PV-F1 controller backing: {e}"))?;
        for (channel, glyph) in &glyphs {
            if *channel >= 10 {
                return Err("PV-F1 controller backing channel outside directory".into());
            }
            crate::handwriting::features(&glyph.strokes, glyph.width as f32, glyph.height as f32)?;
        }
        Ok(glyphs)
    }
    pub(super) fn restore_registered_state(&mut self, glyphs: BTreeMap<u8, RegisteredGlyph>) {
        if let Self::Control {
            registrations,
            registered_glyphs,
            ..
        } = self
        {
            *registrations = glyphs
                .iter()
                .map(|(channel, glyph)| (*channel, glyph.code))
                .collect();
            *registered_glyphs = glyphs;
        }
    }
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
            engine: Box::default(),
            clock: Box::default(),
            sample_uploads: 0,
            completed_strokes: 0,
            last_samples: vec![],
            results: Box::default(),
            recognition: Box::default(),
            registrations: BTreeMap::new(),
            registration_response: None,
            registration_exchange: false,
            registration_channel: None,
            registered_glyphs: BTreeMap::new(),
            raw_confirmation: false,
            stroke_read: None,
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
            configuration: recognition.configuration,
            input_flags: recognition.input_flags,
            registration_mode: recognition.registration_mode,
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
        results.notification_type = 1;
        results.irq_sent = false;
        Ok(())
    }
    /// Replay the acquired type-2 notification. FF476 sets BFF20.bit0 and
    /// FE DB5FC/DBFBC then suppress uploads. Its physical trigger is unknown;
    /// no character page or inferred completion is supplied by this fixture.
    pub fn supply_type2_notification_fixture(&mut self) -> Result<(), String> {
        let Self::Control { results, .. } = self else {
            return Err("controller notification requires the auxiliary candidate".into());
        };
        if results.ready {
            return Err("PV-F1 notification would replace an unacknowledged notification".into());
        }
        results.ready = true;
        results.notification_type = 2;
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
                registered_glyphs,
                ..
            } => {
                let captured = registered_glyphs
                    .iter()
                    .map(|(channel, glyph)| {
                        json!({"channel":channel,"code_bytes":glyph.code.to_le_bytes(),
                        "canvas":[glyph.width,glyph.height],"strokes":glyph.strokes})
                    })
                    .collect::<Vec<_>>();
                json!({"control_candidate":true,"controls":controls.to_vec(),"enabled":enabled,"capture_active":engine.capture_active,"accepted_controls":engine.accepted,"accepted_count":engine.accepted_count,"sample_period_candidate":clock.period,"sample_ready":clock.ready,"sample_events":clock.events,"sample_uploads":sample_uploads,"completed_strokes":completed_strokes,"last_uploaded_samples":last_samples,"result_fixture_ready":results.ready,"result_fixture_indices":results.pages.keys().collect::<Vec<_>>(),"recognizer_loaded":recognition.model.is_some(),"recognition_canvas":[recognition.width,recognition.height],"recognition_configuration":recognition.configuration,"input_flags":recognition.input_flags,"classifications":recognition.classifications,"recognized_labels":recognition.last_labels,"bound_raw_samples":captured })
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
            (0x8a09, Self::Control { results, .. }) => Some(results.notification_type),
            (
                0x8a41..=0x8a5f,
                Self::Control {
                    engine,
                    results,
                    registration_response,
                    ..
                },
            ) if engine.status() & 0x40 != 0 => {
                let offset = (address - 0x8a41) as usize;
                registration_response
                    .as_ref()
                    .and_then(|r| r.get(offset))
                    .copied()
                    .or_else(|| {
                        results
                            .response
                            .as_ref()
                            .and_then(|r| r.get(offset))
                            .copied()
                    })
            }
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
                let glyphs = match self {
                    Self::Control {
                        registered_glyphs, ..
                    } => std::mem::take(registered_glyphs),
                    _ => BTreeMap::new(),
                };
                if matches!(self, Self::Absent { .. }) {
                    *self = Self::absent();
                } else {
                    *self = Self::control();
                    self.configure_sample_period(period)?;
                    self.configure_recognizer(model)?;
                    self.restore_registered_state(glyphs);
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
                        results.notification_type = 0;
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
                    registrations,
                    registration_response,
                    registration_exchange,
                    registration_channel,
                    registered_glyphs,
                    raw_confirmation,
                    stroke_read,
                    ..
                } => {
                    let command = u16::from_le_bytes([request[0], request[1]]);
                    if command == 0x22fb {
                        if !*registration_exchange
                            || stroke_read
                                .as_ref()
                                .is_none_or(|(_, data, cursor)| *cursor != data.len())
                        {
                            return Err("PV-F1 stored stroke finish before complete read".into());
                        }
                        engine.start_packet(request, cycle, 2, Some(1), true)?;
                        *stroke_read = None;
                        return Ok(());
                    }
                    if command == 0x20fb {
                        let channel = request[2];
                        if !*registration_exchange {
                            return Err("PV-F1 stored stroke read outside exchange".into());
                        }
                        let glyph = registered_glyphs
                            .get(&channel)
                            .ok_or("PV-F1 stored stroke channel is empty")?;
                        let mut packed = Vec::new();
                        for stroke in &glyph.strokes {
                            for [x, y] in stroke {
                                // DCA59..DCA79 multiplies each byte by four
                                // before adding the drawing origin.
                                packed.extend([(*x / 4.).round() as u8, (*y / 4.).round() as u8]);
                            }
                            packed.extend([0xff, 0xff]);
                        }
                        engine.start_packet(request, cycle, 3, Some(1), true)?;
                        *stroke_read = Some((channel, packed, 0));
                        return Ok(());
                    }
                    if command == 0x21fb {
                        let (channel, data, cursor) = stroke_read
                            .as_mut()
                            .ok_or("PV-F1 stored stroke data without selection")?;
                        if !*registration_exchange || request[2] != *channel {
                            return Err("PV-F1 stored stroke data outside selected exchange".into());
                        }
                        let end = (*cursor + 30).min(data.len());
                        let status = u8::from(end == data.len());
                        engine.start_packet(request, cycle, 3, Some(status), true)?;
                        let mut reply = vec![(end - *cursor) as u8];
                        reply.extend_from_slice(&data[*cursor..end]);
                        *cursor = end;
                        *registration_response = Some(reply);
                        return Ok(());
                    }
                    if command == 0x10fb {
                        // FE DC85C/DD5D7 supplies channel plus the entered
                        // two-byte character. The learned ink comes from the
                        // preceding real sample uploads, not a result fixture.
                        let channel = request[2];
                        if channel >= 10 || !recognition.registration_mode {
                            return Err(
                                "PV-F1 glyph binding outside controller44/channel range".into()
                            );
                        }
                        let index = recognition
                            .active
                            .ok_or("PV-F1 glyph binding without captured channel")?;
                        let strokes = recognition
                            .strokes
                            .get(&index)
                            .filter(|s| !s.is_empty())
                            .ok_or("PV-F1 glyph binding without captured ink")?
                            .clone();
                        let code = u16::from_le_bytes([request[3], request[4]]);
                        engine.start_packet(request, cycle, 5, Some(1), true)?;
                        registered_glyphs.insert(
                            channel,
                            RegisteredGlyph {
                                code,
                                width: recognition.width,
                                height: recognition.height,
                                strokes,
                            },
                        );
                        registrations.insert(channel, code);
                        *raw_confirmation = true;
                        recognition.active = None;
                        recognition.strokes.remove(&index);
                        recognition.in_stroke = false;
                        return Ok(());
                    }
                    if matches!(command, 0x01fb | 0x02fb) {
                        // FE DCA26/DCB35 bracket registration-data operations
                        // with D76C/D771. D79C requires status01 and BF ack.
                        // This exchange does not start normal pen capture.
                        engine.start_packet(request, cycle, 2, Some(1), true)?;
                        *registration_exchange = command == 0x01fb;
                        return Ok(());
                    }
                    if command == 0x31fb {
                        // FE DCB16 -> DD780 clears all handwritten registrations.
                        // It sends only31FB outside the per-slot01FB/02FB exchange,
                        // then expects timed status01 and a BF acknowledgement.
                        // Clearing model-owned learned ink is an explicit storage
                        // candidate; the supplied recognition model is retained.
                        engine.start_packet(request, cycle, 2, Some(1), true)?;
                        registrations.clear();
                        registered_glyphs.clear();
                        *registration_channel = None;
                        *registration_response = None;
                        *stroke_read = None;
                        *raw_confirmation = false;
                        return Ok(());
                    }
                    if command == 0x12fb {
                        // Native registration Cancel runs DCB35/DD607 with
                        // its channel within the exchange bracket. Remove that
                        // binding; other directory records remain intact.
                        if !*registration_exchange || request[2] >= 10 {
                            return Err(
                                "PV-F1 controller channel selection outside active exchange/range"
                                    .into(),
                            );
                        }
                        engine.start_packet(request, cycle, 3, Some(1), true)?;
                        *registration_channel = Some(request[2]);
                        registrations.remove(&request[2]);
                        registered_glyphs.remove(&request[2]);
                        *raw_confirmation = false;
                        return Ok(());
                    }
                    if command == 0x13fb {
                        let index = request[2];
                        if index >= 10 {
                            return Err(format!(
                                "PV-F1 registration slot {index} outside recovered directory"
                            ));
                        }
                        let entry = registrations.get(&index).copied();
                        engine.start_packet(
                            request,
                            cycle,
                            3,
                            Some(u8::from(entry.is_some())),
                            true,
                        )?;
                        *registration_response = Some(entry.unwrap_or(0).to_le_bytes().to_vec());
                        return Ok(());
                    }
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
                        // DC836 waits for capture-state4 before learning.
                        // A normal type1 result would advance it to5 and trap
                        // that wait.02F9 selects raw learning;00F9/01F9 select
                        // ordinary recognition, despite their sharedC2 bits.
                        if recognition.registration_mode && !*raw_confirmation {
                            recognition.in_stroke = false;
                            return Ok(());
                        }
                        if let Some((index, records)) = recognition.finish(registered_glyphs)? {
                            *raw_confirmation = false;
                            if results.ready || results.pages.contains_key(&index) {
                                return Err("recognizer result would replace pending page".into());
                            }
                            results.pages.insert(index, records);
                            results.ready = true;
                            results.notification_type = 1;
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
                        if recognition.model.is_some()
                            || !registered_glyphs.is_empty()
                            || recognition.registration_mode
                        {
                            recognition.upload(last_samples)?;
                        }
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
                    // DC6EF selects controller44 with02F9. Controller43 uses
                    // 00F9/01F9 atDCC2D/DCC32. Both then driveC2/42, so the
                    // shared control bits cannot distinguish raw learning.
                    if command == 0x02f9 {
                        recognition.registration_mode = true;
                    }
                    if matches!(command, 0x00f9 | 0x01f9) {
                        recognition.registration_mode = false;
                    }
                    if command == 0x04fe {
                        recognition.input_flags = request[2];
                    }
                    if command == 0x03fe {
                        recognition.width = u16::from_le_bytes([request[2], request[3]]);
                        recognition.height = u16::from_le_bytes([request[4], request[5]]);
                        recognition.configuration = u16::from_le_bytes([request[6], request[7]]);
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
                    engine,
                    results,
                    registration_response,
                    ..
                } => {
                    engine.acknowledge(value)?;
                    results.response = None;
                    *registration_response = None;
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

    #[test]
    fn controller44_collects_raw_ink_without_a_recognizer_or_result_notification() {
        let mut a = Auxiliary::control();
        a.write(0x8a06, 0x42, 0).unwrap();
        start(&mut a, 0x02f9, 0);
        a.advance(32);
        a.write(0x8a03, 0xbf, 32).unwrap();
        let strokes = vec![vec![[12., 18.], [90., 18.]], vec![[48., 18.], [48., 96.]]];
        if let Auxiliary::Control { recognition, .. } = &mut a {
            recognition.width = 138;
            recognition.height = 222;
            recognition
                .upload(&[
                    [0x8000, 0],
                    [12, 18],
                    [90, 18],
                    [0xffff, 0xffff],
                    [48, 18],
                    [48, 96],
                ])
                .unwrap();
        }
        start(&mut a, 0x00fc, 100);
        a.advance(132);
        assert!(!a.take_result_irq_edge());
        assert_eq!(a.snapshot()["classifications"], 0);
        for (i, b) in [0xfb, 0x10, 0, 0x82, 0x73].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 200).unwrap();
        }
        a.write(0x8a02, 0x80, 200).unwrap();
        assert_eq!(a.peek(0x8a40), None);
        a.advance(232);
        assert_eq!(a.peek(0x8a40), Some(1));
        if let Auxiliary::Control {
            registered_glyphs,
            recognition,
            ..
        } = &a
        {
            let sample = &registered_glyphs[&0];
            assert_eq!(sample.code.to_le_bytes(), [0x82, 0x73]);
            assert_eq!(sample.strokes, strokes);
            assert_eq!((sample.width, sample.height), (138, 222));
            assert!(recognition.active.is_none());
            assert!(recognition.strokes.is_empty());
        }
        a.write(0x8a03, 0xbf, 232).unwrap();
        a.write(0x8a22, 0, 300).unwrap();
        start(&mut a, 0x13fb, 300);
        a.advance(332);
        assert_eq!(a.peek(0x8a40), Some(1));
        assert_eq!(a.peek(0x8a41), Some(0x82));
        assert_eq!(a.peek(0x8a42), Some(0x73));
    }

    #[test]
    fn stored_stroke_read_chunks_have_native_lengths_and_final_status() {
        let mut a = Auxiliary::control();
        if let Auxiliary::Control {
            registered_glyphs, ..
        } = &mut a
        {
            registered_glyphs.insert(
                0,
                RegisteredGlyph {
                    code: 0x7382,
                    width: 138,
                    height: 222,
                    strokes: vec![(0..16).map(|i| [i as f32 * 4., 40.]).collect()],
                },
            );
        }
        start(&mut a, 0x01fb, 0);
        a.advance(32);
        a.write(0x8a03, 0xbf, 32).unwrap();
        for (i, b) in [0xfb, 0x20, 0].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 100).unwrap();
        }
        a.write(0x8a02, 0x80, 100).unwrap();
        a.advance(132);
        a.write(0x8a03, 0xbf, 132).unwrap();
        for (cycle, status, payload) in [
            (200, 0, (0..15).flat_map(|x| [x, 10]).collect::<Vec<u8>>()),
            (300, 1, vec![15, 10, 0xff, 0xff]),
        ] {
            for (i, b) in [0xfb, 0x21, 0].into_iter().enumerate() {
                a.write(0x8a20 + i as u32, b, cycle).unwrap();
            }
            a.write(0x8a02, 0x80, cycle).unwrap();
            assert_eq!(a.peek(0x8a41), None);
            a.advance(cycle + 32);
            assert_eq!(a.peek(0x8a40), Some(status));
            assert_eq!(a.peek(0x8a41), Some(payload.len() as u8));
            for (i, b) in payload.iter().enumerate() {
                assert_eq!(a.peek(0x8a42 + i as u32), Some(*b));
            }
            a.write(0x8a03, 0xbf, cycle + 32).unwrap();
            assert_eq!(a.peek(0x8a41), None);
        }
        start(&mut a, 0x22fb, 400);
        a.advance(432);
        assert_eq!(a.peek(0x8a40), Some(1));
    }

    #[test]
    fn retained_template_survives_transport_reset_and_recognizes_repeated_ink() {
        let strokes = vec![vec![[39., 40.], [97., 40.]], vec![[67., 40.], [67., 159.]]];
        let mut source = Auxiliary::control();
        if let Auxiliary::Control {
            registered_glyphs, ..
        } = &mut source
        {
            registered_glyphs.insert(
                0,
                RegisteredGlyph {
                    code: 0x7382,
                    width: 138,
                    height: 222,
                    strokes: strokes.clone(),
                },
            );
        }
        let state = source.registered_state().unwrap();
        let mut fresh = Auxiliary::control();
        fresh.restore_registered_state(Auxiliary::decode_registered_state(&state).unwrap());
        for (value, cycle) in [(0xfd, 0), (2, 100)] {
            fresh.write(0x8a01, value, cycle).unwrap();
        }
        assert_eq!(fresh.registered_state(), Some(state));
        fresh.write(0x8a06, 0x80, 200).unwrap();
        if let Auxiliary::Control { recognition, .. } = &mut fresh {
            recognition.width = 138;
            recognition.height = 222;
            recognition
                .upload(&[
                    [0x8000, 0],
                    [39, 40],
                    [97, 40],
                    [0xffff, 0xffff],
                    [67, 40],
                    [67, 159],
                ])
                .unwrap();
        }
        start(&mut fresh, 0x00fc, 300);
        fresh.advance(332);
        assert_eq!(fresh.snapshot()["recognized_labels"], json!(["Ｔ"]));
        fresh.write(0x8a06, 0xc0, 332).unwrap();
        assert!(fresh.take_result_irq_edge());
        let invalid = br#"{"10":{"code":29570,"width":138,"height":222,"strokes":[[[1.0,2.0]]]}}"#;
        assert!(Auxiliary::decode_registered_state(invalid).is_err());
    }

    #[test]
    fn registration_query_has_timed_status_word_and_ack_scoped_reply() {
        let mut a = Auxiliary::control();
        for index in [0, 9] {
            for (offset, byte) in [0xfb, 0x13, index].into_iter().enumerate() {
                a.write(0x8a20 + offset as u32, byte, 100).unwrap();
            }
            a.write(0x8a02, 0x80, 100).unwrap();
            assert_eq!(a.peek(0x8a41), None);
            a.advance(132);
            assert_eq!(a.peek(0x8a40), Some(0));
            assert_eq!(a.peek(0x8a41), Some(0));
            assert_eq!(a.peek(0x8a42), Some(0));
            assert_eq!(a.peek(0x8a43), None);
            a.write(0x8a03, 0xbf, 132).unwrap();
            assert_eq!(a.peek(0x8a41), None);
        }
        // Synthetic component entry tests wire ordering only; no training
        // or physical word meaning is claimed by this fixture.
        if let Auxiliary::Control { registrations, .. } = &mut a {
            registrations.insert(0, 0x8273);
        }
        for (offset, byte) in [0xfb, 0x13, 0].into_iter().enumerate() {
            a.write(0x8a20 + offset as u32, byte, 200).unwrap();
        }
        a.write(0x8a02, 0x80, 200).unwrap();
        a.advance(232);
        assert_eq!(a.peek(0x8a40), Some(1));
        assert_eq!(a.peek(0x8a41), Some(0x73));
        assert_eq!(a.peek(0x8a42), Some(0x82));
    }

    #[test]
    fn registration_exchange_controls_are_timed_and_do_not_enable_pen_capture() {
        let mut a = Auxiliary::control();
        for (header, open) in [(0x01fbu16, true), (0x02fb, false)] {
            for (i, b) in header.to_le_bytes().into_iter().enumerate() {
                a.write(0x8a20 + i as u32, b, 100).unwrap();
            }
            a.write(0x8a02, 0x80, 100).unwrap();
            assert_eq!(a.peek(0x8a40), None);
            a.advance(132);
            assert_eq!(a.peek(0x8a40), Some(1));
            assert_eq!(a.peek(0x8a41), None);
            if let Auxiliary::Control {
                engine,
                registration_exchange,
                ..
            } = &a
            {
                assert_eq!(*registration_exchange, open);
                assert!(!engine.capture_active);
            }
            a.write(0x8a03, 0xbf, 132).unwrap();
        }
    }

    #[test]
    fn controller_channel_removal_requires_exchange_and_preserves_other_records() {
        let mut a = Auxiliary::control();
        if let Auxiliary::Control { registrations, .. } = &mut a {
            registrations.insert(9, 0x1234);
            registrations.insert(0, 0x5678);
        }
        for (i, b) in [0xfb, 0x12, 9].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 0).unwrap();
        }
        assert!(a.write(0x8a02, 0x80, 0).is_err());
        // Begin has its own two-byte packet; failed selection did not consume
        // or clear a directory entry. Use a fresh transport for the valid path.
        a = Auxiliary::control();
        start(&mut a, 0x01fb, 100);
        a.advance(132);
        a.write(0x8a03, 0xbf, 132).unwrap();
        if let Auxiliary::Control { registrations, .. } = &mut a {
            registrations.insert(9, 0x1234);
            registrations.insert(0, 0x5678);
        }
        for (i, b) in [0xfb, 0x12, 9].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 200).unwrap();
        }
        a.write(0x8a02, 0x80, 200).unwrap();
        a.advance(232);
        assert_eq!(a.peek(0x8a40), Some(1));
        if let Auxiliary::Control {
            registrations,
            registration_channel,
            ..
        } = &a
        {
            assert_eq!(*registration_channel, Some(9));
            assert!(!registrations.contains_key(&9));
            assert_eq!(registrations.get(&0), Some(&0x5678));
        }
    }

    #[test]
    fn clear_all_registration_removes_labels_and_ink_with_timed_status() {
        let mut a = Auxiliary::control();
        if let Auxiliary::Control {
            registrations,
            registered_glyphs,
            registration_channel,
            registration_response,
            raw_confirmation,
            stroke_read,
            recognition,
            ..
        } = &mut a
        {
            for (channel, code) in [(0, 0x7382), (9, 0x6082)] {
                registrations.insert(channel, code);
                registered_glyphs.insert(
                    channel,
                    RegisteredGlyph {
                        code,
                        width: 138,
                        height: 222,
                        strokes: vec![vec![[1., 2.], [3., 4.]]],
                    },
                );
            }
            *registration_channel = Some(9);
            *registration_response = Some(vec![0x82, 0x60]);
            *stroke_read = Some((9, vec![1, 2], 0));
            *raw_confirmation = true;
            recognition.input_flags = 7;
        }
        // The native global erase is not a per-slot exchange operation.
        start(&mut a, 0x31fb, 100);
        assert_eq!(a.peek(0x8a40), None);
        a.advance(132);
        assert_eq!(a.peek(0x8a40), Some(1));
        assert_eq!(a.peek(0x8a41), None);
        if let Auxiliary::Control {
            registrations,
            registered_glyphs,
            registration_channel,
            registration_response,
            raw_confirmation,
            stroke_read,
            recognition,
            registration_exchange,
            engine,
            ..
        } = &a
        {
            assert!(registrations.is_empty() && registered_glyphs.is_empty());
            assert!(registration_channel.is_none() && registration_response.is_none());
            assert!(!raw_confirmation && stroke_read.is_none());
            assert_eq!(recognition.input_flags, 7);
            assert!(!registration_exchange && !engine.capture_active);
        }
        a.write(0x8a03, 0xbf, 132).unwrap();
        a.write(0x8a01, 0xfd, 200).unwrap();
        a.write(0x8a01, 2, 300).unwrap();
        assert_eq!(a.snapshot()["bound_raw_samples"], json!([]));
        for (i, b) in [0xfb, 0x13, 9].into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, 400).unwrap();
        }
        a.write(0x8a02, 0x80, 400).unwrap();
        a.advance(432);
        assert_eq!(a.peek(0x8a40), Some(0));
        assert_eq!(a.peek(0x8a41), Some(0));
        assert_eq!(a.peek(0x8a42), Some(0));
    }

    fn start(a: &mut Auxiliary, header: u16, cycle: u64) {
        for (i, b) in header.to_le_bytes().into_iter().enumerate() {
            a.write(0x8a20 + i as u32, b, cycle).unwrap();
        }
        a.write(0x8a02, 0x80, cycle).unwrap();
    }

    #[test]
    fn type2_notification_preserves_pages_and_uses_the_result_irq_ack() {
        let mut absent = Auxiliary::absent();
        assert!(absent.supply_type2_notification_fixture().is_err());
        let mut a = Auxiliary::control();
        a.supply_candidate_result_fixture(2, vec![[0x82, 0x60, 0]])
            .unwrap();
        let before = a.snapshot();
        assert!(a.supply_type2_notification_fixture().is_err());
        assert_eq!(a.snapshot(), before);
        a.write(0x8a08, 0xbf, 0).unwrap();
        a.supply_type2_notification_fixture().unwrap();
        assert_eq!(a.peek(0x8a09), Some(2));
        assert_eq!(a.peek(0x8a05), Some(0x40));
        assert!(!a.take_result_irq_edge());
        a.write(0x8a06, 0x40, 0).unwrap();
        assert!(a.take_result_irq_edge());
        assert!(!a.take_result_irq_edge());
        assert!(a.supply_candidate_result_fixture(1, vec![]).is_err());
        a.write(0x8a07, 0x7f, 1).unwrap();
        assert_eq!(a.peek(0x8a09), Some(2));
        a.write(0x8a08, 0xbf, 2).unwrap();
        assert_eq!(a.peek(0x8a09), Some(0));
        assert_eq!(a.peek(0x8a05), Some(0));
        assert!(!a.take_result_irq_edge());
        assert_eq!(a.snapshot()["result_fixture_indices"], json!([2]));
        a.supply_type2_notification_fixture().unwrap();
        a.write(0x8a01, 2, 3).unwrap();
        assert_eq!(a.peek(0x8a09), Some(0));
        assert_eq!(a.snapshot()["result_fixture_indices"], json!([]));
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
