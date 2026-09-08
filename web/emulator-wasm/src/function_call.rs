// Resumable debugger calls; host yields never execute guest instructions.
// PY_SOURCE: pce500/emulator.py:PCE500Emulator
use super::*;
use sc62015_core::llama::state::CallMetricsSnapshot;

#[cfg(test)]
#[path = "function_call_tests.rs"]
mod tests;

pub(super) struct FunctionCallSession {
    id: u32,
    addr: u32,
    max_instructions: u32,
    opts: CallOptions,
    stub_map: HashMap<u32, u32>,
    before_pc: u32,
    before_sp: u32,
    before_call_metrics: CallMetricsSnapshot,
    before_regs: HashMap<String, u32>,
    sentinel_pc: u32,
    saved_stack: Vec<(u32, u8)>,
    previous_tracer: Option<sc62015_core::PerfettoTracer>,
    steps: u32,
    scheduler_boundaries: u32,
    reason: String,
    fault: Option<CallFault>,
    probe_samples: Vec<ProbeSample>,
    probe_hits: u32,
    stub_hits: HashMap<(u32, u32), u32>,
    forced_writes: HashMap<u32, u8>,
    pending_stub: Option<CallStubRequest>,
}

#[derive(Clone, Serialize)]
struct CallStubRequest {
    id: u32,
    sequence: u32,
    pc: u32,
    regs: Vec<StubRegEntry>,
    flags: Vec<StubRegEntry>,
}

#[derive(Serialize)]
struct CallSliceStatus<'a> {
    state: &'static str,
    // Compatibility call budget: successful scheduler boundaries OR explicit
    // debugger stub actions. It is not the CPU's retired-instruction counter.
    steps: u32,
    scheduler_boundaries: u32,
    stub_request: Option<&'a CallStubRequest>,
}

impl FunctionCallSession {
    fn fail(&mut self, kind: &str, message: String) {
        self.reason = "fault".into();
        self.fault = Some(CallFault {
            kind: kind.into(),
            message,
        });
        self.pending_stub = None;
    }

    fn complete_boundary(&mut self, pc: u32) {
        self.steps += 1;
        if pc == self.sentinel_pc {
            self.reason = "returned".into();
        } else if self.steps == self.max_instructions {
            self.reason = "timeout".into();
        }
    }
}

#[wasm_bindgen]
impl Sc62015Emulator {
    /// Synchronous compatibility helper for offline callers. Interactive hosts
    /// must use begin/slice/finish and yield their event loop between slices.
    pub fn call_function(&mut self, addr: u32, max_instructions: u32) -> Result<JsValue, JsValue> {
        self.call_function_ex(addr, max_instructions, JsValue::UNDEFINED)
    }

    pub fn call_function_ex(
        &mut self,
        addr: u32,
        max_instructions: u32,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        let id = self.call_function_begin(addr, max_instructions, options)?;
        let mut call = self.call_session.take().expect("just began call");
        while call.reason == "running" {
            self.advance_function_call(&mut call, u32::MAX, || Ok(false))?;
            if let Some(request) = call.pending_stub.clone() {
                let patch = (|| {
                    let regs = serde_wasm_bindgen::to_value(&request.regs)?;
                    let flags = serde_wasm_bindgen::to_value(&request.flags)?;
                    js_stub_dispatch(request.id, regs, flags)
                        .map_err(|error| serde_wasm_bindgen::Error::new(js_error_to_string(error)))
                })();
                match patch {
                    Ok(patch) => self.apply_call_stub(&mut call, patch),
                    Err(error) => call.fail("StubError", format!("stub dispatch failed: {error}")),
                }
            }
        }
        debug_assert_eq!(call.id, id);
        self.finish_function_call(call)
            .map(|json| JsValue::from_str(&json))
    }

    /// Begin a debugger invocation. Only PC, S, the three sentinel stack bytes
    /// and call bookkeeping are restored on finish/cancel. Other guest effects
    /// are intentionally retained, as with the synchronous diagnostic helper.
    pub fn call_function_begin(
        &mut self,
        addr: u32,
        max_instructions: u32,
        options: JsValue,
    ) -> Result<u32, JsValue> {
        self.require_no_active_call()?;
        let id = self
            .next_call_id
            .checked_add(1)
            .ok_or_else(|| JsValue::from_str("function call ID space exhausted"))?;
        let addr = addr & 0x000f_ffff;
        let max_instructions = max_instructions.max(1);
        if options.is_instance_of::<js_sys::Map>() || js_sys::Array::is_array(&options) {
            return Err(JsValue::from_str(
                "call options must be an object, not a Map or array",
            ));
        }
        let mut opts: CallOptions = if options.is_null() || options.is_undefined() {
            CallOptions::default()
        } else {
            serde_wasm_bindgen::from_value(options)
                .map_err(|error| JsValue::from_str(&format!("invalid call options: {error}")))?
        };
        if opts.probe_max_samples == 0 {
            opts.probe_max_samples = 256;
        }
        let mut stub_map = HashMap::with_capacity(opts.stubs.len());
        for stub in &opts.stubs {
            let pc = stub.pc & 0x000f_ffff;
            if stub_map.insert(pc, stub.id).is_some() {
                return Err(JsValue::from_str(&format!(
                    "duplicate stub address after 20-bit masking: 0x{pc:05X}"
                )));
            }
        }

        if opts.trace {
            let mut guard = sc62015_core::PERFETTO_TRACER.enter();
            if let Some(existing) = guard.take() {
                guard.replace(Some(existing));
                return Err(JsValue::from_str(
                    "Perfetto trace already recording (nested tracing is unsupported)",
                ));
            }
        }

        let before_pc = self.runtime.state.pc();
        let before_sp = self.runtime.state.get_reg(RegName::S);
        let before_call_metrics = self.runtime.state.snapshot_call_metrics();
        let before_regs = sc62015_core::collect_registers(&self.runtime.state);

        let sentinel_low16: u32 = 0xD00D;
        let sentinel_pc = ((addr & 0x0f_0000) | sentinel_low16) & 0x000f_ffff;

        // Push a 20-bit sentinel return address (little-endian) onto the S stack.
        // Preserve the overwritten bytes so this debugger helper does not
        // perturb later machine execution.
        let stack_mask = mask_for(RegName::S);
        let new_sp = before_sp.wrapping_sub(3) & stack_mask;
        let mut saved_stack = Vec::with_capacity(3);
        for i in 0..3u32 {
            let stack_addr = new_sp.wrapping_add(i) & stack_mask;
            if self.runtime.memory.is_read_only_range(stack_addr, 1)
                || !self.runtime.memory.instruction_byte_is_stable(stack_addr)
            {
                return Err(JsValue::from_str(&format!(
                    "call sentinel stack byte 0x{stack_addr:05X} is not writable static RAM"
                )));
            }
            let previous = self
                .runtime
                .memory
                .read_byte_for_preflight(stack_addr, None)
                .ok_or_else(|| {
                    JsValue::from_str(&format!(
                        "call sentinel stack byte 0x{stack_addr:05X} is unavailable"
                    ))
                })?;
            saved_stack.push((stack_addr, previous));
        }
        for (i, (stack_addr, _)) in saved_stack.iter().enumerate() {
            let byte = (sentinel_pc >> (8 * i)) & 0xff;
            let _ = self.runtime.memory.store(*stack_addr, 8, byte);
        }
        self.runtime.state.set_reg(RegName::S, new_sp);
        // Bookkeeping for call-stack tracing; RET/RETF will unwind this.
        self.runtime.state.push_call_stack(addr);
        self.runtime.state.call_depth_inc();

        // Enable last-value write capture.
        self.runtime.memory.begin_write_capture();
        if let Some(lcd) = self.runtime.lcd.as_mut() {
            lcd.begin_display_write_capture();
        }

        // Optional perfetto capture for this call (serialized on wasm32).
        let mut previous_tracer: Option<sc62015_core::PerfettoTracer> = None;
        if opts.trace {
            let mut guard = sc62015_core::PERFETTO_TRACER.enter();
            previous_tracer = guard.replace(Some(sc62015_core::PerfettoTracer::new(
                std::path::PathBuf::from("call.perfetto-trace"),
            )));
        }

        // Enter the function.
        self.runtime.state.set_pc(addr);

        self.next_call_id = id;
        self.call_session = Some(FunctionCallSession {
            id,
            addr,
            max_instructions,
            opts,
            stub_map,
            before_pc,
            before_sp,
            before_call_metrics,
            before_regs,
            sentinel_pc,
            saved_stack,
            previous_tracer,
            steps: 0,
            scheduler_boundaries: 0,
            reason: "running".into(),
            fault: None,
            probe_samples: Vec::new(),
            probe_hits: 0,
            stub_hits: HashMap::new(),
            forced_writes: HashMap::new(),
            pending_stub: None,
        });
        Ok(id)
    }

    /// Advance only at complete scheduler boundaries; no user JavaScript runs
    /// inside this mutable WASM borrow. A stub is returned as a host request.
    pub fn call_function_slice(
        &mut self,
        id: u32,
        boundaries: u32,
        max_host_ms: f64,
    ) -> Result<JsValue, JsValue> {
        self.require_call_id(id)?;
        if !max_host_ms.is_finite() || !(0.0..=16.0).contains(&max_host_ms) {
            return Err(JsValue::from_str(
                "host slice target must be finite and in 0..=16 ms",
            ));
        }
        let started = host_monotonic_now()?;
        let mut call = self.call_session.take().expect("checked call");
        let advanced = self.advance_function_call(&mut call, boundaries, || {
            host_monotonic_now().map(|now| now - started >= max_host_ms)
        });
        self.call_session = Some(call); // Also preserve ownership on clock failure.
        advanced?;
        let call = self.call_session.as_ref().expect("restored call");
        serde_wasm_bindgen::to_value(&CallSliceStatus {
            state: if call.reason != "running" {
                "complete"
            } else if call.pending_stub.is_some() {
                "stub"
            } else {
                "running"
            },
            steps: call.steps,
            scheduler_boundaries: call.scheduler_boundaries,
            stub_request: call.pending_stub.as_ref(),
        })
        .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Apply the response to the current stub request. The caller must not run
    /// guest instructions while the response is pending.
    pub fn call_function_apply_stub(
        &mut self,
        id: u32,
        sequence: u32,
        patch: JsValue,
    ) -> Result<(), JsValue> {
        self.require_call_id(id)?;
        let call = self.call_session.as_ref().expect("checked call");
        if call.pending_stub.as_ref().map(|stub| stub.sequence) != Some(sequence) {
            return Err(JsValue::from_str("no matching pending function stub"));
        }
        let mut call = self.call_session.take().expect("checked call");
        self.apply_call_stub(&mut call, patch);
        self.call_session = Some(call);
        Ok(())
    }

    /// Convert a host callback failure into a normal call fault with cleanup.
    pub fn call_function_stub_failed(
        &mut self,
        id: u32,
        sequence: u32,
        message: &str,
    ) -> Result<(), JsValue> {
        self.require_call_id(id)?;
        let call = self.call_session.as_mut().expect("checked call");
        if call.pending_stub.as_ref().map(|stub| stub.sequence) != Some(sequence) {
            return Err(JsValue::from_str("no pending function stub"));
        }
        call.fail("StubError", message.into());
        Ok(())
    }

    pub fn call_function_finish(&mut self, id: u32) -> Result<String, JsValue> {
        let artifacts = self.take_finished_function_call(id)?;
        serde_json::to_string(&artifacts).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Structured result endpoint. Build ordinary JS objects, not generic JS
    /// Maps; no callbacks run while the owned report is converted.
    pub fn call_function_finish_object(&mut self, id: u32) -> Result<JsValue, JsValue> {
        let artifacts = self.take_finished_function_call(id)?;
        artifacts
            .serialize(
                &serde_wasm_bindgen::Serializer::new()
                    .serialize_maps_as_objects(true)
                    .serialize_missing_as_null(true),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Structured cancellation result, with the same restoration guarantees.
    pub fn call_function_cancel_object(&mut self, id: u32) -> Result<JsValue, JsValue> {
        let artifacts = self.take_cancelled_function_call(id)?;
        artifacts
            .serialize(
                &serde_wasm_bindgen::Serializer::new()
                    .serialize_maps_as_objects(true)
                    .serialize_missing_as_null(true),
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Cancel at a host boundary; retain the JSON endpoint for old callers.
    pub fn call_function_cancel(&mut self, id: u32) -> Result<String, JsValue> {
        let artifacts = self.take_cancelled_function_call(id)?;
        serde_json::to_string(&artifacts).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Sc62015Emulator {
    fn take_finished_function_call(&mut self, id: u32) -> Result<CallArtifacts, JsValue> {
        self.require_call_id(id)?;
        if self.call_session.as_ref().expect("checked call").reason == "running" {
            return Err(JsValue::from_str(
                "function call is still running; cancel it explicitly",
            ));
        }
        let call = self.call_session.take().expect("checked call");
        self.finish_function_call_artifacts(call)
    }

    /// Cancel at a host boundary. This restores debugger scaffolding, not a
    /// whole-machine snapshot; it cannot undo bus/peripheral effects.
    fn take_cancelled_function_call(&mut self, id: u32) -> Result<CallArtifacts, JsValue> {
        self.require_call_id(id)?;
        let mut call = self.call_session.take().expect("checked call");
        if call.reason == "running" {
            call.reason = "cancelled".into();
        }
        call.pending_stub = None;
        self.finish_function_call_artifacts(call)
    }
}

impl Sc62015Emulator {
    pub(super) fn require_no_active_call(&self) -> Result<(), JsValue> {
        if self.call_session.is_some() {
            Err(JsValue::from_str(
                "function call owns the machine; finish or cancel it first",
            ))
        } else {
            Ok(())
        }
    }

    fn require_call_id(&self, id: u32) -> Result<(), JsValue> {
        if self.call_session.as_ref().map(|call| call.id) == Some(id) {
            Ok(())
        } else {
            Err(JsValue::from_str("stale or missing function call ID"))
        }
    }

    fn advance_function_call(
        &mut self,
        call: &mut FunctionCallSession,
        boundaries: u32,
        mut should_yield: impl FnMut() -> Result<bool, JsValue>,
    ) -> Result<(), JsValue> {
        if call.reason != "running" || call.pending_stub.is_some() {
            return Ok(());
        }
        for used in 0..boundaries {
            if used % sc62015_core::run_control::RUN_CONTROL_POLL_BOUNDARIES as u32 == 0
                && should_yield()?
            {
                break;
            }
            let current_pc = self.runtime.state.pc() & 0x000f_ffff;
            if self.runtime.state.is_halted() && !self.runtime.timer.irq_pending {
                call.reason = "halted".into();
                break;
            }
            if let Some(probe_pc) = call.opts.probe_pc {
                if current_pc == (probe_pc & 0x000f_ffff)
                    && (call.probe_samples.len() as u32) < call.opts.probe_max_samples
                {
                    call.probe_hits = call.probe_hits.saturating_add(1);
                    call.probe_samples.push(ProbeSample {
                        pc: current_pc,
                        count: call.probe_hits,
                        regs: sc62015_core::collect_registers(&self.runtime.state),
                    });
                }
            }
            if let Some(stub_id) = call.stub_map.get(&current_pc).copied() {
                let hits = call.stub_hits.entry((stub_id, current_pc)).or_default();
                *hits = hits.saturating_add(1);
                call.pending_stub = Some(CallStubRequest {
                    id: stub_id,
                    sequence: call.steps + 1,
                    pc: current_pc,
                    regs: sc62015_core::collect_registers(&self.runtime.state)
                        .into_iter()
                        .map(|(name, value)| StubRegEntry { name, value })
                        .collect(),
                    flags: vec![
                        StubRegEntry {
                            name: "C".into(),
                            value: self.runtime.state.get_reg(RegName::FC) & 1,
                        },
                        StubRegEntry {
                            name: "Z".into(),
                            value: self.runtime.state.get_reg(RegName::FZ) & 1,
                        },
                    ],
                });
                break;
            }
            if let Err(error) = self.runtime.step(1) {
                call.fail("CoreError", error.to_string());
                break;
            }
            call.complete_boundary(self.runtime.state.pc());
            call.scheduler_boundaries += 1;
            if call.reason != "running" {
                break;
            }
        }
        Ok(())
    }

    fn apply_call_stub(&mut self, call: &mut FunctionCallSession, patch: JsValue) {
        if patch.is_instance_of::<js_sys::Map>() || js_sys::Array::is_array(&patch) {
            call.fail(
                "StubError",
                "stub patch must be an object, not a Map or array".into(),
            );
            return;
        }
        let patch: Result<StubPatch, _> = if patch.is_null() || patch.is_undefined() {
            Ok(StubPatch::default())
        } else {
            serde_wasm_bindgen::from_value(patch)
        };
        let patch = match patch {
            Ok(patch) => patch,
            Err(error) => {
                call.fail("StubError", format!("stub patch decode failed: {error}"));
                return;
            }
        };
        let current_pc = call.pending_stub.take().expect("pending stub").pc;
        for write in patch.mem_writes {
            let size = match write.size {
                2 | 3 => write.size,
                _ => 1,
            };
            let bits = size * 8;
            let addr = write.addr & 0x000f_ffff;
            let _ = self
                .runtime
                .memory
                .store_with_pc(addr, bits, write.value, Some(current_pc));
            for offset in 0..size {
                let byte = ((write.value >> (8 * offset)) & 0xFF) as u8;
                call.forced_writes
                    .insert(addr.wrapping_add(offset as u32), byte);
            }
        }
        for entry in patch.regs {
            if let Some(reg) = reg_from_name(&entry.name) {
                self.runtime.state.set_reg(reg, entry.value);
            }
        }
        for entry in patch.flags {
            match entry.name.to_ascii_uppercase().as_str() {
                "C" | "FC" => self.runtime.state.set_reg(RegName::FC, entry.value & 1),
                "Z" | "FZ" => self.runtime.state.set_reg(RegName::FZ, entry.value & 1),
                _ => {}
            }
        }
        let ret = patch.ret.unwrap_or(StubReturn::Ret { pc: None });
        match ret {
            StubReturn::Ret { pc } => {
                let ret_addr = pop_stack(&mut self.runtime.state, &mut self.runtime.memory, 16);
                let _ = self.runtime.state.pop_call_page();
                let page = current_pc & 0xFF0000;
                let mut dest = (page | (ret_addr & 0xFFFF)) & 0xFFFFF;
                if let Some(override_pc) = pc {
                    dest = override_pc & 0x000f_ffff;
                }
                self.runtime.state.set_pc(dest);
                self.runtime.state.call_depth_dec();
                let _ = self.runtime.state.pop_call_stack();
            }
            StubReturn::Retf { pc } => {
                let mut dest =
                    pop_stack(&mut self.runtime.state, &mut self.runtime.memory, 24) & 0xFFFFF;
                if let Some(override_pc) = pc {
                    dest = override_pc & 0x000f_ffff;
                }
                self.runtime.state.set_pc(dest);
                self.runtime.state.call_depth_dec();
                let _ = self.runtime.state.pop_call_stack();
            }
            StubReturn::Jump { pc } => {
                self.runtime.state.set_pc(pc & 0x000f_ffff);
            }
            StubReturn::Stay => {}
        }

        call.complete_boundary(self.runtime.state.pc());
    }

    fn finish_function_call(&mut self, call: FunctionCallSession) -> Result<String, JsValue> {
        let artifacts = self.finish_function_call_artifacts(call)?;
        serde_json::to_string(&artifacts).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    fn finish_function_call_artifacts(
        &mut self,
        call: FunctionCallSession,
    ) -> Result<CallArtifacts, JsValue> {
        let FunctionCallSession {
            addr,
            opts,
            before_pc,
            before_sp,
            before_call_metrics,
            before_regs,
            saved_stack,
            mut previous_tracer,
            steps,
            scheduler_boundaries,
            reason,
            fault,
            probe_samples,
            stub_hits,
            forced_writes,
            ..
        } = call;
        let after_pc = self.runtime.state.pc();
        let after_sp = self.runtime.state.get_reg(RegName::S);
        let mut memory_writes_map: HashMap<u32, u8> = self
            .runtime
            .memory
            .take_write_capture()
            .into_iter()
            .collect();
        if !forced_writes.is_empty() {
            memory_writes_map.extend(forced_writes);
        }
        let mut memory_writes: Vec<MemoryWriteByte> = memory_writes_map
            .into_iter()
            .map(|(addr, value)| MemoryWriteByte { addr, value })
            .collect();
        memory_writes.sort_by_key(|entry| entry.addr);
        let lcd_writes = self
            .runtime
            .lcd
            .as_mut()
            .map(|lcd| lcd.take_display_write_capture())
            .unwrap_or_default();

        for (stack_addr, previous) in saved_stack {
            let _ = self
                .runtime
                .memory
                .store(stack_addr, 8, u32::from(previous));
        }

        // Restore debugger-owned state before any fallible trace/artifact encoding.
        // Guest writes, non-PC/S registers, timers and peripheral effects are retained.
        self.runtime.state.set_pc(before_pc);
        self.runtime.state.set_reg(RegName::S, before_sp);
        self.runtime.state.restore_call_metrics(before_call_metrics);

        let perfetto_trace_b64 = if opts.trace {
            let mut guard = sc62015_core::PERFETTO_TRACER.enter();
            let tracer = guard.take();
            guard.replace(previous_tracer.take());
            let trace_bytes = tracer
                .map(|tracer| tracer.serialize())
                .transpose()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            trace_bytes.map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes))
        } else {
            None
        };

        let after_regs = sc62015_core::collect_registers(&self.runtime.state);
        let mut stubs_used: Vec<StubUse> = stub_hits
            .into_iter()
            .map(|((id, pc), hits)| StubUse { id, pc, hits })
            .collect();
        stubs_used.sort_by_key(|stub| (stub.pc, stub.id));

        let report = CallReport {
            reason,
            steps,
            scheduler_boundaries,
            pc: after_pc,
            sp: after_sp,
            halted: self.runtime.state.is_halted(),
            fault,
        };
        let artifacts = CallArtifacts {
            address: addr,
            before_pc,
            after_pc,
            before_sp,
            after_sp,
            before_regs,
            after_regs,
            memory_writes,
            lcd_writes,
            probe_samples,
            stubs_used,
            perfetto_trace_b64,
            report,
        };
        Ok(artifacts)
    }
}
