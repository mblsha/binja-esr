"""Device-boundary reference; the shared scheduler owns CPU/IRQ execution."""

from dataclasses import dataclass


@dataclass
class ExecutionState:
    mapped_selector: int | None = None
    diagnostic_on_irq_edge_only: bool = False
    diagnostic_irq_imr_only: bool = False
    diagnostic_lcc7_halt_main_timer: bool = False
    halt_main_timer_units: int = 0
    diagnostic_rtc_a2_period: int | None = None
    rtc_a2_phase_units: int = 0
    rtc_a2_ticks: int = 0
    experimental_rtc_second_period: int | None = None
    rtc_second_phase_units: int = 0
    rtc_elapsed_seconds: int = 0
    rtc_held_units: int = 0
    rtc_timing_units: int = 0
    rtc_off_elapsed_units: int = 0
    elapsed_before: int = 0
    audio_scr_before: int = 0
    audio_off_before: bool = False

    halt_main_deadline: int | None = None

    def rtc_progression_report(self):
        if self.experimental_rtc_second_period is None:
            return None
        return {
            "second_period_units": self.experimental_rtc_second_period,
            "second_phase_units": self.rtc_second_phase_units,
            "elapsed_seconds": self.rtc_elapsed_seconds,
            "held_units": self.rtc_held_units,
            "elapsed_timing_units": self.rtc_timing_units,
            "off_elapsed_units": self.rtc_off_elapsed_units,
            "half_second_period_units": self.diagnostic_rtc_a2_period,
            "half_second_phase_units": self.rtc_a2_phase_units,
            "half_second_ticks": self.rtc_a2_ticks,
            "qualification": "Experimental nominal scheduler rate, freeze-on-HOLD and inferred cyclic/deadline/seconds-3F policy; not physical calibration or complete wildcard support",
        }

    def before_boundary(
        self,
        hw,
        cycles,
        *,
        pc=None,
        halted=False,
        off=False,
        lcc=0,
        next_mti=0,
        memory=None,
        elapsed_timing_units=None,
    ):
        if pc == 0:
            raise ValueError(
                "OZ-9600 reached an unqualified zero code target; empty-RAM reset/ESR-P policies remain unresolved"
            )
        self.halt_main_deadline = (
            next_mti
            if self.diagnostic_lcc7_halt_main_timer and halted and not off and lcc & 128
            else None
        )
        self.elapsed_before = (
            cycles if elapsed_timing_units is None else elapsed_timing_units
        )
        if hw.audio.enabled:
            if memory is None:
                raise ValueError(
                    "audio capture requires the owning internal-memory view"
                )
            self.audio_scr_before = memory.read_internal_byte_silent(0xFD)
            self.audio_off_before = off
        # Caller publishes a static ROM mapping before executable preflight;
        # missing selectors never provide executable bytes.
        changed = self.mapped_selector != hw.selector
        self.mapped_selector = hw.selector
        if hw.card:
            if memory is None:
                raise ValueError(
                    "mounted card requires the owning internal-memory view"
                )
            ssr = memory.read_internal_byte_silent(0xFF)
            memory.write_internal_byte(0xFF, hw.card.presence_ssr(ssr))
        hw.cycle = cycles
        return changed, hw.refresh_irq_inputs()

    def after_boundary(
        self,
        hw,
        cycles,
        elapsed,
        *,
        timer=None,
        memory=None,
        pc=None,
        elapsed_timing_units=None,
    ):
        rtc_elapsed = (
            (elapsed_timing_units - self.elapsed_before) & (2**64 - 1)
            if self.experimental_rtc_second_period is not None
            and elapsed_timing_units is not None
            else elapsed
        )
        if self.experimental_rtc_second_period is not None:
            self.rtc_timing_units = (self.rtc_timing_units + rtc_elapsed) & (2**64 - 1)
            self.rtc_off_elapsed_units = (
                self.rtc_off_elapsed_units + rtc_elapsed - elapsed
            ) & (2**64 - 1)
        if self.halt_main_deadline is not None:
            # The owner supplies its existing timer allocation, with the same
            # restricted selected-tick API as the shared Rust driver contract.
            if timer is None:
                raise ValueError("HALT timer experiment requires the owning timer")
            timer.next_mti = self.halt_main_deadline
            self.halt_main_deadline = None
            timer.tick_timers_selected(memory, cycles, pc, True, False)
            self.halt_main_timer_units = (self.halt_main_timer_units + elapsed) & (
                2**64 - 1
            )
        if self.diagnostic_rtc_a2_period is not None:
            ticks, self.rtc_a2_phase_units = divmod(
                self.rtc_a2_phase_units + rtc_elapsed, self.diagnostic_rtc_a2_period
            )
            if ticks:
                self.rtc_a2_ticks = (self.rtc_a2_ticks + ticks) & (2**64 - 1)
                hw.rtc.raise_causes(4, 0)
                hw.cycle = cycles
                hw.log("diagnostic_rtc_cause", 0x800A, 4, None)
        if self.experimental_rtc_second_period is not None:
            if hw.rtc.read(12) & 128:
                self.rtc_held_units = (self.rtc_held_units + rtc_elapsed) & (2**64 - 1)
            else:
                seconds, phase = divmod(
                    self.rtc_second_phase_units + rtc_elapsed,
                    self.experimental_rtc_second_period,
                )
                if seconds:
                    hw.rtc.advance_experimental_seconds(seconds)
                    self.rtc_elapsed_seconds = (self.rtc_elapsed_seconds + seconds) & (
                        2**64 - 1
                    )
                self.rtc_second_phase_units = phase
        if hw.fault:
            raise ValueError(hw.fault)
        hw.audio.advance(
            elapsed
            if elapsed_timing_units is None
            else (elapsed_timing_units - self.elapsed_before) & (2**64 - 1),
            self.audio_scr_before,
            self.audio_off_before,
            hw.audio_ci_input,
        )
