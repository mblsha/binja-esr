use anyhow::Result;
use protobuf::{Message, MessageField};
use smol_str::SmolStr;
use std::{
    collections::HashMap,
    io::{self, Write},
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::{SystemTime, UNIX_EPOCH},
};

use perfetto_protos::{
    counter_descriptor::{CounterDescriptor, counter_descriptor::Unit},
    debug_annotation::{DebugAnnotation, DebugAnnotationName},
    interned_data::InternedData,
    profile_common::InternedString,
    source_location::SourceLocation,
    trace::Trace,
    trace_packet::{TracePacket, trace_packet::SequenceFlags},
    track_descriptor::TrackDescriptor,
    track_event::{EventCategory, EventName, TrackEvent, track_event::Type},
};

// Re-export Unit enum for counter tracks
pub use perfetto_protos::counter_descriptor::counter_descriptor::Unit as CounterUnit;

#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum InternID {
    New(u64),
    Existing(u64),
}

impl Into<u64> for InternID {
    fn into(self) -> u64 {
        match self {
            InternID::New(i) => i,
            InternID::Existing(i) => i,
        }
    }
}

impl Into<MessageField<EventName>> for InternID {
    fn into(self) -> MessageField<EventName> {
        MessageField::some(EventName {
            iid: Some(self.into()),
            ..Default::default()
        })
    }
}

impl InternID {
    fn new(self) -> bool {
        match self {
            InternID::New(_) => true,
            InternID::Existing(_) => false,
        }
    }

    fn as_u64(self) -> u64 {
        self.into()
    }
}

#[derive(Default, Debug)]
pub(crate) struct Intern<T> {
    next_id: u64,
    items: HashMap<T, u64>,
}

impl<T: Clone + std::hash::Hash + Eq> Intern<T> {
    pub(crate) fn intern(&mut self, value: T) -> InternID {
        if let Some(entry) = self.items.get(&value) {
            return InternID::Existing(*entry);
        }

        self.next_id += 1;
        let id = self.next_id;
        self.items.insert(value, id);
        InternID::New(id)
    }
}

pub struct Context<W: io::Write> {
    event_names: Intern<SmolStr>,
    debug_annotation_names: Intern<SmolStr>,
    debug_annotation_str_values: Intern<SmolStr>,
    categories: Intern<SmolStr>,
    source_locations: Intern<(SmolStr, u32)>,
    trace: Trace,
    seq: u32,
    writer: io::BufWriter<W>,
    next_id: AtomicU64,
    thread_tracks: HashMap<i32, u64>,
}
impl<W> Context<W>
where
    W: io::Write + std::fmt::Debug,
{
    pub fn into_inner(mut self) -> W {
        self.flush().unwrap();
        self.writer.into_inner().unwrap()
    }
}

impl<W> Context<W>
where
    W: io::Write,
{
    pub fn new(writer: W) -> Self {
        let mut s = Self {
            writer: io::BufWriter::new(writer),
            event_names: Default::default(),
            debug_annotation_names: Default::default(),
            debug_annotation_str_values: Default::default(),
            categories: Default::default(),
            thread_tracks: Default::default(),
            source_locations: Default::default(),
            seq: rand::random(),
            trace: Default::default(),
            next_id: 0.into(),
        };
        let init = s.init_packet();
        s.trace.packet.push(init);
        s
    }

    #[cfg(test)]
    pub(crate) fn new_with_seq(writer: W, seq: u32) -> Self {
        let mut s = Self {
            writer: io::BufWriter::new(writer),
            event_names: Default::default(),
            debug_annotation_names: Default::default(),
            debug_annotation_str_values: Default::default(),
            categories: Default::default(),
            thread_tracks: Default::default(),
            source_locations: Default::default(),
            seq,
            trace: Default::default(),
            next_id: 0.into(),
        };
        let init = s.init_packet();
        s.trace.packet.push(init);
        s
    }

    pub fn current_thread_track(&mut self) -> u64 {
        let current = current_thread();
        if let Some(track) = self.thread_tracks.get(&current) {
            return *track;
        }
        let track = self.track().current_process().current_thread().build();
        self.thread_tracks.insert(current, track);
        track
    }

    fn init_packet(&self) -> TracePacket {
        let mut tp = TracePacket::new();
        tp.set_sequence_flags(SequenceFlags::SEQ_INCREMENTAL_STATE_CLEARED as u32);
        tp.set_trusted_packet_sequence_id(self.seq);
        tp
    }

    pub fn flush(&mut self) -> Result<()> {
        let trace = std::mem::take(&mut self.trace);
        trace.write_to_writer(&mut self.writer)?;
        self.writer.flush()?;
        Ok(())
    }
    pub fn event<'a>(&'a mut self) -> EventBuilder<'a, W> {
        EventBuilder::new(self)
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Relaxed)
    }

    pub fn track<'a>(&'a mut self) -> TrackBuilder<'a, W> {
        let id = self.next_id();
        TrackBuilder::new(self).uuid(id)
    }

    fn source_location<'a>(&'a mut self, file: impl Into<SmolStr>, line: u32) -> u64 {
        let file = file.into();
        let id = self.source_locations.intern((file.clone(), line));
        match id {
            InternID::New(id) => {
                let mut tp = TracePacket::new();
                tp.interned_data
                    .mut_or_insert_default()
                    .source_locations
                    .push(SourceLocation {
                        iid: Some(id),
                        file_name: Some(file.to_string()),
                        line_number: Some(line),
                        ..Default::default()
                    });
                self.push_packet(tp);
                id
            }
            InternID::Existing(id) => id,
        }
    }

    fn intern_event_name(&mut self, name: impl Into<SmolStr>) -> InternID {
        let name = name.into();
        let id = self.event_names.intern(name.clone());
        if id.new() {
            let mut tp = TracePacket::new();
            let mut itd = InternedData::new();
            itd.event_names.push(EventName {
                iid: Some(id.as_u64()),
                name: Some(name.to_string()),
                ..Default::default()
            });
            tp.interned_data = MessageField::some(itd);
            self.push_packet(tp);
        }
        id
    }

    fn intern_debug_annotation_name(&mut self, name: impl Into<SmolStr>) -> InternID {
        let name = name.into();
        let id = self.debug_annotation_names.intern(name.clone());
        if id.new() {
            let mut tp = TracePacket::new();
            let mut itd = InternedData::new();
            itd.debug_annotation_names.push(DebugAnnotationName {
                iid: Some(id.into()),
                name: Some(name.to_string()),
                ..Default::default()
            });
            tp.interned_data = MessageField::some(itd);
            self.push_packet(tp);
        }
        id
    }

    fn intern_debug_annotation_str_value(&mut self, value: impl Into<SmolStr>) -> InternID {
        let value = value.into();
        let id = self.debug_annotation_str_values.intern(value.clone());
        if id.new() {
            let mut tp = TracePacket::new();
            let mut itd = InternedData::new();
            itd.debug_annotation_string_values.push(InternedString {
                iid: Some(id.into()),
                str: Some(value.as_bytes().to_vec()),
                ..Default::default()
            });
            tp.interned_data = MessageField::some(itd);
            self.push_packet(tp);
        }
        id
    }

    fn intern_category(&mut self, category: impl Into<SmolStr>) -> InternID {
        let category = category.into();
        let id = self.categories.intern(category.clone());
        if id.new() {
            let mut tp = TracePacket::new();
            let mut itd = InternedData::new();
            itd.event_categories.push(EventCategory {
                iid: Some(id.as_u64()),
                name: Some(category.to_string()),
                ..Default::default()
            });
            tp.interned_data = MessageField::some(itd);
            self.push_packet(tp);
        }
        id
    }
}

impl<'a, W: Write> Context<W> {
    fn push_packet(&'a mut self, mut packet: TracePacket) {
        if !packet.has_trusted_packet_sequence_id() {
            packet.set_trusted_packet_sequence_id(self.seq);
        }
        self.trace.packet.push(packet);
    }
}
pub fn current_thread() -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        // wasm32-unknown-unknown has no native threads; a stable synthetic tid is fine.
        0
    }

    #[cfg(target_os = "linux")]
    {
        nix::unistd::gettid().as_raw()
    }

    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "linux")))]
    {
        (nix::sys::pthread::pthread_self() as i32).abs()
    }
}

pub struct TrackBuilder<'a, W: io::Write> {
    track: TrackDescriptor,
    ctx: &'a mut Context<W>,
}

impl<'a, W: io::Write> TrackBuilder<'a, W> {
    fn new(ctx: &'a mut Context<W>) -> Self {
        Self {
            track: TrackDescriptor::new(),
            ctx,
        }
    }
    pub fn uuid(mut self, id: u64) -> Self {
        self.track.set_uuid(id);
        self
    }
    pub fn parent_uuid(mut self, id: u64) -> Self {
        self.track.set_parent_uuid(id);
        self
    }

    pub fn pid(mut self, pid: i32) -> Self {
        self.track.thread.mut_or_insert_default().set_pid(pid);
        self
    }
    pub fn tid(mut self, tid: i32) -> Self {
        self.track.thread.mut_or_insert_default().set_tid(tid);
        self
    }

    pub fn name<T: Into<String>>(mut self, name: T) -> Self {
        self.track.set_name(name.into());
        self
    }

    pub fn current_process(self) -> Self {
        let pid = std::process::id();
        self.pid(pid as i32)
    }

    pub fn current_thread(self) -> Self {
        self.tid(current_thread())
    }

    pub fn counter(mut self) -> Self {
        self.track.counter = protobuf::MessageField::some(CounterDescriptor::new());
        self
    }

    pub fn unit(mut self, unit: Unit) -> Self {
        self.track.counter.mut_or_insert_default().set_unit(unit);
        self
    }

    pub fn unit_name<T: Into<String>>(mut self, name: T) -> Self {
        self.track
            .counter
            .mut_or_insert_default()
            .set_unit_name(name.into());
        self
    }

    pub fn unit_multiplier(mut self, multiplier: i64) -> Self {
        self.track
            .counter
            .mut_or_insert_default()
            .set_unit_multiplier(multiplier);
        self
    }

    pub fn is_incremental(mut self, incremental: bool) -> Self {
        self.track
            .counter
            .mut_or_insert_default()
            .set_is_incremental(incremental);
        self
    }

    pub fn build(self) -> u64 {
        let mut tp = TracePacket::new();
        let id = self.track.uuid();
        assert!(
            self.track.has_uuid(),
            "track_uuid is required for a track event"
        );
        tp.set_track_descriptor(self.track);
        self.ctx.push_packet(tp);
        id
    }
}

pub struct EventBuilder<'a, W: io::Write> {
    event: TrackEvent,
    ctx: &'a mut Context<W>,
}

impl<'a, W: io::Write> EventBuilder<'a, W> {
    fn new(ctx: &'a mut Context<W>) -> Self {
        Self {
            event: TrackEvent::new(),
            ctx,
        }
    }

    pub fn timestamp_us(&mut self, us: i64) {
        self.event.set_timestamp_absolute_us(us);
    }

    pub fn now(&mut self) {
        let us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        self.timestamp_us(us);
    }

    pub fn begin(&mut self) {
        self.event.set_type(Type::TYPE_SLICE_BEGIN);
    }

    pub fn end(&mut self) {
        self.event.set_type(Type::TYPE_SLICE_END);
    }

    pub fn instant(&mut self) {
        self.event.set_type(Type::TYPE_INSTANT);
    }

    pub fn counter(&mut self) {
        self.event.set_type(Type::TYPE_COUNTER);
    }

    pub fn category(&mut self, category: impl Into<SmolStr>) {
        let id = self.ctx.intern_category(category);
        self.event.category_iids.push(id.into());
    }

    pub fn source_location(&mut self, file: impl Into<SmolStr>, line: u32) {
        let loc = self.ctx.source_location(file, line);
        self.event.set_source_location_iid(loc);
    }

    pub fn name(&mut self, name: impl Into<SmolStr>) {
        let id = self.ctx.intern_event_name(name);
        self.event.set_name_iid(id.into());
    }

    pub fn debug_str(&mut self, name: impl Into<SmolStr>, value: impl Into<SmolStr>) {
        let id = self.ctx.intern_debug_annotation_name(name);
        let vid = self.ctx.intern_debug_annotation_str_value(value);
        let mut da = DebugAnnotation::new();
        da.set_name_iid(id.into());
        da.set_string_value_iid(vid.into());
        self.event.debug_annotations.push(da);
    }

    pub fn debug_bool(&mut self, name: impl Into<SmolStr>, value: bool) {
        let id = self.ctx.intern_debug_annotation_name(name);
        let mut da = DebugAnnotation::new();
        da.set_name_iid(id.into());
        da.set_bool_value(value);
        self.event.debug_annotations.push(da);
    }

    pub fn debug_int(&mut self, name: impl Into<SmolStr>, value: i64) {
        let id = self.ctx.intern_debug_annotation_name(name);
        let mut da = DebugAnnotation::new();
        da.set_name_iid(id.into());
        da.set_int_value(value);
        self.event.debug_annotations.push(da);
    }

    pub fn debug_uint(&mut self, name: impl Into<SmolStr>, value: u64) {
        let id = self.ctx.intern_debug_annotation_name(name);
        let mut da = DebugAnnotation::new();
        da.set_name_iid(id.into());
        da.set_uint_value(value);
        self.event.debug_annotations.push(da);
    }

    pub fn debug_double(&mut self, name: impl Into<SmolStr>, value: f64) {
        let id = self.ctx.intern_debug_annotation_name(name);
        let mut da = DebugAnnotation::new();
        da.set_name_iid(id.into());
        da.set_double_value(value);
        self.event.debug_annotations.push(da);
    }

    pub fn debug_pointer(&mut self, name: impl Into<SmolStr>, value: u64) {
        let id = self.ctx.intern_debug_annotation_name(name);
        let mut da = DebugAnnotation::new();
        da.set_name_iid(id.into());
        da.set_pointer_value(value);
        self.event.debug_annotations.push(da);
    }

    pub fn track_uuid(&mut self, id: u64) {
        self.event.set_track_uuid(id);
    }

    pub fn counter_value(&mut self, value: i64) {
        self.event.set_counter_value(value);
    }

    pub fn double_counter_value(&mut self, value: f64) {
        self.event.set_double_counter_value(value);
    }

    pub fn flow_id(&mut self, id: u64) {
        self.event.flow_ids.push(id);
    }

    pub fn terminating_flow_id(&mut self, id: u64) {
        self.event.terminating_flow_ids.push(id);
    }

    pub fn extra_counter(&mut self, track_uuid: u64, value: i64) {
        self.event.extra_counter_track_uuids.push(track_uuid);
        self.event.extra_counter_values.push(value);
    }

    pub fn extra_double_counter(&mut self, track_uuid: u64, value: f64) {
        self.event.extra_double_counter_track_uuids.push(track_uuid);
        self.event.extra_double_counter_values.push(value);
    }

    pub fn with_timestamp_us(mut self, us: i64) -> Self {
        self.timestamp_us(us);
        self
    }

    pub fn with_now(mut self) -> Self {
        self.now();
        self
    }

    pub fn with_begin(mut self) -> Self {
        self.begin();
        self
    }

    pub fn with_end(mut self) -> Self {
        self.end();
        self
    }

    pub fn with_instant(mut self) -> Self {
        self.instant();
        self
    }

    pub fn with_counter(mut self) -> Self {
        self.counter();
        self
    }

    pub fn with_category(mut self, category: impl Into<SmolStr>) -> Self {
        self.category(category);
        self
    }

    pub fn with_source_location(mut self, file: impl Into<SmolStr>, line: u32) -> Self {
        self.source_location(file, line);
        self
    }

    pub fn with_name(mut self, name: impl Into<SmolStr>) -> Self {
        self.name(name);
        self
    }

    pub fn with_track_uuid(mut self, id: u64) -> Self {
        self.track_uuid(id);
        self
    }

    pub fn with_debug_str(mut self, name: impl Into<SmolStr>, value: impl Into<SmolStr>) -> Self {
        self.debug_str(name, value);
        self
    }

    pub fn with_debug_bool(mut self, name: impl Into<SmolStr>, value: bool) -> Self {
        self.debug_bool(name, value);
        self
    }

    pub fn with_debug_int(mut self, name: impl Into<SmolStr>, value: i64) -> Self {
        self.debug_int(name, value);
        self
    }

    pub fn with_debug_uint(mut self, name: impl Into<SmolStr>, value: u64) -> Self {
        self.debug_uint(name, value);
        self
    }

    pub fn with_debug_double(mut self, name: impl Into<SmolStr>, value: f64) -> Self {
        self.debug_double(name, value);
        self
    }

    pub fn with_debug_pointer(mut self, name: impl Into<SmolStr>, value: u64) -> Self {
        self.debug_pointer(name, value);
        self
    }

    pub fn with_counter_value(mut self, value: i64) -> Self {
        self.counter_value(value);
        self
    }

    pub fn with_double_counter_value(mut self, value: f64) -> Self {
        self.double_counter_value(value);
        self
    }

    pub fn with_flow_id(mut self, id: u64) -> Self {
        self.flow_id(id);
        self
    }

    pub fn with_terminating_flow_id(mut self, id: u64) -> Self {
        self.terminating_flow_id(id);
        self
    }

    pub fn with_extra_counter(mut self, track_uuid: u64, value: i64) -> Self {
        self.extra_counter(track_uuid, value);
        self
    }

    pub fn with_extra_double_counter(mut self, track_uuid: u64, value: f64) -> Self {
        self.extra_double_counter(track_uuid, value);
        self
    }

    pub fn build(self) {
        let mut tp = TracePacket::new();
        assert!(
            self.event.has_track_uuid(),
            "track_uuid is required for a track event"
        );
        tp.set_track_event(self.event);
        self.ctx.push_packet(tp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use assert_matches::assert_matches;
    use bytes::{Buf, BufMut, BytesMut};
    use perfetto_protos::trace::Trace;
    use protobuf::Message;
    use std::fs;
    use std::path::Path;

    /// Assert that a Trace matches the text representation stored in a golden file.
    ///
    /// This function converts the given Trace to protobuf text format and compares it
    /// with the contents of the file at the specified path. If the contents don't match,
    /// the test will fail with a detailed diff.
    ///
    /// Set the `UPDATE_GOLDEN` environment variable to update golden files:
    /// ```bash
    /// UPDATE_GOLDEN=1 cargo test
    /// ```
    ///
    /// # Arguments
    ///
    /// * `trace` - The Trace to compare
    /// * `golden_path` - Path to the golden file (relative to workspace root)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let buf = BytesMut::new();
    /// let mut ctx = Context::new(buf.writer());
    /// ctx.event().with_begin().with_name("test").with_track_uuid(1).build();
    /// let buf: BytesMut = ctx.into_inner().into_inner();
    /// let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;
    /// assert_golden_text(&trace, "tests/golden/simple_event.txtpb")?;
    /// ```
    fn assert_golden_text(trace: &Trace, golden_path: impl AsRef<Path>) -> Result<()> {
        use protobuf::text_format;

        let golden_path = golden_path.as_ref();
        let actual_text = text_format::print_to_string_pretty(trace);

        // Check if we should update golden files
        if std::env::var("UPDATE_GOLDEN").is_ok() {
            // Ensure parent directory exists
            if let Some(parent) = golden_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(golden_path, &actual_text)?;
            eprintln!("Updated golden file: {}", golden_path.display());
            return Ok(());
        }

        // Read the expected golden file
        let expected_text = fs::read_to_string(golden_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to read golden file at '{}': {}.\n\
                 Hint: Run with UPDATE_GOLDEN=1 to create this file.",
                golden_path.display(),
                e
            )
        })?;

        // Compare the texts
        if actual_text != expected_text {
            // Create a helpful error message
            let diff_msg = format!(
                "\nGolden file mismatch for: {}\n\
                 \n\
                 Expected:\n\
                 {}\n\
                 \n\
                 Actual:\n\
                 {}\n\
                 \n\
                 Hint: Run with UPDATE_GOLDEN=1 to update the golden file.",
                golden_path.display(),
                expected_text,
                actual_text
            );
            anyhow::bail!(diff_msg);
        }

        Ok(())
    }

    #[test]
    fn same_id() -> Result<()> {
        let mut i = Intern::default();
        let id = i.intern("a");
        assert_eq!(id.as_u64(), i.intern("a").as_u64());
        Ok(())
    }

    #[test]
    fn owned_strings_work() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());
        let track = ctx.track().current_process().current_thread().build();

        // Test with owned strings (e.g., from Python or user input)
        let event_name = String::from("dynamic_event");
        let category = String::from("custom_category");
        let annotation_name = String::from("user_data");
        let annotation_value = String::from("hello from owned string!");

        ctx.event()
            .with_begin()
            .with_name(event_name)
            .with_category(category)
            .with_track_uuid(track)
            .with_debug_str(annotation_name, annotation_value)
            .build();

        // Test with static strings (still works!)
        ctx.event()
            .with_end()
            .with_name("static_event")
            .with_category("static_category")
            .with_track_uuid(track)
            .with_debug_str("key", "value")
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // Verify we got the events
        assert!(trace.packet.len() > 4);
        Ok(())
    }

    #[test]
    fn event_round_trip() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());
        ctx.event()
            .with_begin()
            .with_name("test")
            .with_track_uuid(1)
            .build();
        ctx.event()
            .with_end()
            .with_name("test")
            .with_track_uuid(1)
            .build();
        ctx.event()
            .with_begin()
            .with_name("best")
            .with_track_uuid(1)
            .build();
        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;
        // packet[0] is the init packet
        assert_matches!(trace.packet[1].interned_data.as_ref(), Some(InternedData{ event_names, .. }) if event_names[0].name() == "test");
        assert_eq!(trace.packet[2].track_event().name_iid(), 1);
        assert_eq!(trace.packet[3].track_event().name_iid(), 1);
        assert_matches!(trace.packet[4].interned_data.as_ref(), Some(InternedData{ event_names, .. }) if event_names[0].name() == "best");
        assert_eq!(trace.packet[5].track_event().name_iid(), 2);

        Ok(())
    }

    #[test]
    fn track_round_trip() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());
        ctx.track()
            .uuid(100)
            .name("main_thread")
            .pid(100)
            .tid(101)
            .build();
        ctx.track()
            .uuid(101)
            .name("worker_thread")
            .pid(100)
            .tid(102)
            .build();
        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // First track descriptor
        let track1 = trace.packet[1].track_descriptor();
        assert_eq!(track1.uuid(), 100);
        assert_eq!(track1.name(), "main_thread");
        assert_eq!(track1.thread.pid(), 100);
        assert_eq!(track1.thread.tid(), 101);

        // Second track descriptor
        let track2 = trace.packet[2].track_descriptor();
        assert_eq!(track2.uuid(), 101);
        assert_eq!(track2.name(), "worker_thread");
        assert_eq!(track2.thread.pid(), 100);
        assert_eq!(track2.thread.tid(), 102);

        Ok(())
    }

    #[test]
    fn track_current_process() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());
        let expected_pid = std::process::id();

        ctx.track()
            .uuid(200)
            .name("process_track")
            .current_process()
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 200);
        assert_eq!(track.name(), "process_track");
        assert_eq!(track.thread.pid(), expected_pid as i32);

        Ok(())
    }

    #[test]
    fn track_current_thread() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());
        let expected_tid = current_thread();

        ctx.track()
            .uuid(201)
            .name("thread_track")
            .current_thread()
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 201);
        assert_eq!(track.name(), "thread_track");
        assert_eq!(track.thread.tid(), expected_tid as i32);

        Ok(())
    }

    #[test]
    fn track_current_process_and_thread() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());
        let expected_pid = std::process::id();
        let expected_tid = current_thread();

        ctx.track()
            .uuid(202)
            .name("full_track")
            .current_process()
            .current_thread()
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 202);
        assert_eq!(track.name(), "full_track");
        assert_eq!(track.thread.pid(), expected_pid as i32);
        assert_eq!(track.thread.tid(), expected_tid as i32);

        Ok(())
    }

    #[test]
    fn category_interning() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        ctx.event()
            .with_begin()
            .with_name("event1")
            .with_category("rendering")
            .with_track_uuid(1)
            .build();

        ctx.event()
            .with_end()
            .with_name("event2")
            .with_category("rendering")
            .with_track_uuid(1)
            .build();

        ctx.event()
            .with_instant()
            .with_name("event3")
            .with_category("networking")
            .with_track_uuid(1)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // Find events and verify categories are interned correctly
        let mut event_count = 0;
        let mut category_rendering_found = false;
        let mut category_networking_found = false;

        for packet in &trace.packet {
            if let Some(interned) = packet.interned_data.as_ref() {
                if !interned.event_categories.is_empty() {
                    if interned.event_categories[0].name() == "rendering" {
                        category_rendering_found = true;
                        assert_eq!(interned.event_categories[0].iid(), 1);
                    } else if interned.event_categories[0].name() == "networking" {
                        category_networking_found = true;
                        assert_eq!(interned.event_categories[0].iid(), 2);
                    }
                }
            }
            if packet.has_track_event() {
                event_count += 1;
                let event = packet.track_event();
                if event_count <= 2 {
                    // First two events use category_iid 1 (rendering)
                    assert_eq!(event.category_iids[0], 1);
                } else {
                    // Third event uses category_iid 2 (networking)
                    assert_eq!(event.category_iids[0], 2);
                }
            }
        }

        assert!(category_rendering_found);
        assert!(category_networking_found);
        assert_eq!(event_count, 3);

        Ok(())
    }

    #[test]
    fn counter_values() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        ctx.event()
            .with_counter()
            .with_name("int_counter")
            .with_counter_value(42)
            .with_track_uuid(1)
            .build();

        ctx.event()
            .with_counter()
            .with_name("double_counter")
            .with_double_counter_value(3.14)
            .with_track_uuid(1)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the interned name "int_counter"
        // packet[2] is the counter event with int value
        assert_eq!(trace.packet[2].track_event().counter_value(), 42);
        // packet[3] is the interned name "double_counter"
        // packet[4] is the counter event with double value
        assert_eq!(trace.packet[4].track_event().double_counter_value(), 3.14);

        Ok(())
    }

    #[test]
    fn flow_ids() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        ctx.event()
            .with_begin()
            .with_name("flow_start")
            .with_flow_id(100)
            .with_track_uuid(1)
            .build();

        ctx.event()
            .with_end()
            .with_name("flow_end")
            .with_terminating_flow_id(100)
            .with_track_uuid(1)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the interned name "flow_start"
        // packet[2] is the event with flow_id
        assert_eq!(trace.packet[2].track_event().flow_ids[0], 100);
        // packet[3] is the interned name "flow_end"
        // packet[4] is the event with terminating_flow_id
        assert_eq!(trace.packet[4].track_event().terminating_flow_ids[0], 100);

        Ok(())
    }

    #[test]
    fn extra_counters() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        ctx.event()
            .with_instant()
            .with_name("event_with_extras")
            .with_extra_counter(200, 123)
            .with_extra_double_counter(201, 45.67)
            .with_track_uuid(1)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the interned name
        // packet[2] is the event with extra counters
        let event = trace.packet[2].track_event();
        assert_eq!(event.extra_counter_track_uuids[0], 200);
        assert_eq!(event.extra_counter_values[0], 123);
        assert_eq!(event.extra_double_counter_track_uuids[0], 201);
        assert_eq!(event.extra_double_counter_values[0], 45.67);

        Ok(())
    }

    #[test]
    fn debug_annotations_types() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        ctx.event()
            .with_instant()
            .with_name("annotated_event")
            .with_debug_bool("is_enabled", true)
            .with_debug_int("count", -42)
            .with_debug_uint("size", 1024)
            .with_debug_double("ratio", 0.75)
            .with_debug_pointer("ptr", 0xdeadbeef)
            .with_track_uuid(1)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packets 1-5 are interned debug annotation names
        // packet[6] is the interned event name
        // packet[7] is the event with debug annotations
        let event = trace.packet[7].track_event();
        assert_eq!(event.debug_annotations.len(), 5);

        // Check bool annotation
        assert_eq!(event.debug_annotations[0].bool_value(), true);
        // Check int annotation
        assert_eq!(event.debug_annotations[1].int_value(), -42);
        // Check uint annotation
        assert_eq!(event.debug_annotations[2].uint_value(), 1024);
        // Check double annotation
        assert_eq!(event.debug_annotations[3].double_value(), 0.75);
        // Check pointer annotation
        assert_eq!(event.debug_annotations[4].pointer_value(), 0xdeadbeef);

        Ok(())
    }

    #[test]
    fn counter_track_basic() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        // Create a basic counter track
        ctx.track().uuid(300).name("memory_usage").counter().build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the track descriptor
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 300);
        assert_eq!(track.name(), "memory_usage");
        assert!(track.counter.is_some());

        Ok(())
    }

    #[test]
    fn counter_track_with_unit() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        // Create a counter track with unit
        ctx.track()
            .uuid(301)
            .name("bytes_allocated")
            .counter()
            .unit(Unit::UNIT_SIZE_BYTES)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the track descriptor
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 301);
        assert_eq!(track.name(), "bytes_allocated");
        assert!(track.counter.is_some());
        assert_eq!(track.counter.unit(), Unit::UNIT_SIZE_BYTES);

        Ok(())
    }

    #[test]
    fn counter_track_with_custom_unit_name() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        // Create a counter track with custom unit name
        ctx.track()
            .uuid(302)
            .name("cpu_temperature")
            .counter()
            .unit_name("celsius")
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the track descriptor
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 302);
        assert_eq!(track.name(), "cpu_temperature");
        assert!(track.counter.is_some());
        assert_eq!(track.counter.unit_name(), "celsius");

        Ok(())
    }

    #[test]
    fn counter_track_with_multiplier() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        // Create a counter track with unit multiplier (e.g., kilobytes instead of bytes)
        ctx.track()
            .uuid(303)
            .name("memory_kb")
            .counter()
            .unit(Unit::UNIT_SIZE_BYTES)
            .unit_multiplier(1024)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the track descriptor
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 303);
        assert_eq!(track.name(), "memory_kb");
        assert!(track.counter.is_some());
        assert_eq!(track.counter.unit(), Unit::UNIT_SIZE_BYTES);
        assert_eq!(track.counter.unit_multiplier(), 1024);

        Ok(())
    }

    #[test]
    fn counter_track_incremental() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        // Create an incremental counter track (for delta values)
        ctx.track()
            .uuid(304)
            .name("packet_count_delta")
            .counter()
            .unit(Unit::UNIT_COUNT)
            .is_incremental(true)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the track descriptor
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 304);
        assert_eq!(track.name(), "packet_count_delta");
        assert!(track.counter.is_some());
        assert_eq!(track.counter.unit(), Unit::UNIT_COUNT);
        assert_eq!(track.counter.is_incremental(), true);

        Ok(())
    }

    #[test]
    fn counter_track_full_configuration() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new(buf.writer());

        // Create a fully configured counter track
        ctx.track()
            .uuid(305)
            .name("network_throughput")
            .counter()
            .unit(Unit::UNIT_SIZE_BYTES)
            .unit_name("bytes_per_second")
            .unit_multiplier(1)
            .is_incremental(false)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        // packet[0] is the init packet
        // packet[1] is the track descriptor
        let track = trace.packet[1].track_descriptor();
        assert_eq!(track.uuid(), 305);
        assert_eq!(track.name(), "network_throughput");
        assert!(track.counter.is_some());
        assert_eq!(track.counter.unit(), Unit::UNIT_SIZE_BYTES);
        assert_eq!(track.counter.unit_name(), "bytes_per_second");
        assert_eq!(track.counter.unit_multiplier(), 1);
        assert_eq!(track.counter.is_incremental(), false);

        Ok(())
    }

    #[test]
    fn golden_simple_event() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new_with_seq(buf.writer(), 12345);

        // Use a fixed track UUID for deterministic output
        let track = ctx.track().uuid(100).name("test_thread").build();

        ctx.event()
            .with_begin()
            .with_name("test_event")
            .with_category("test_category")
            .with_track_uuid(track)
            .build();

        ctx.event()
            .with_end()
            .with_name("test_event")
            .with_category("test_category")
            .with_track_uuid(track)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        assert_golden_text(&trace, "tests/golden/simple_event.txtpb")?;

        Ok(())
    }

    #[test]
    fn golden_counter_track() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new_with_seq(buf.writer(), 12345);

        // Create a counter track with full configuration
        ctx.track()
            .uuid(100)
            .name("cpu_usage")
            .counter()
            .unit(Unit::UNIT_COUNT)
            .unit_name("percent")
            .unit_multiplier(1)
            .is_incremental(false)
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        assert_golden_text(&trace, "tests/golden/counter_track.txtpb")?;

        Ok(())
    }

    #[test]
    fn golden_debug_annotations() -> Result<()> {
        let buf = BytesMut::new();
        let mut ctx = Context::new_with_seq(buf.writer(), 12345);

        // Use a fixed track UUID for deterministic output
        let track = ctx.track().uuid(200).name("annotation_thread").build();

        ctx.event()
            .with_begin()
            .with_name("annotated_event")
            .with_category("annotations")
            .with_track_uuid(track)
            .with_debug_bool("is_enabled", true)
            .with_debug_int("count", 42)
            .with_debug_uint("id", 123)
            .with_debug_double("ratio", 3.14)
            .with_debug_str("message", "Hello, Perfetto!")
            .build();

        let buf: BytesMut = ctx.into_inner().into_inner();
        let trace: Trace = Trace::parse_from_reader(&mut buf.reader())?;

        assert_golden_text(&trace, "tests/golden/debug_annotations.txtpb")?;

        Ok(())
    }
}
