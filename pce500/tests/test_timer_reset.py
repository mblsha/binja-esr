"""Regression tests for timer state handling in the PC-E500 emulator."""

from pce500 import PCE500Emulator


def test_reset_reinitializes_timer_schedule() -> None:
    """Reset should restore timer targets and clear stale IRQ state."""

    emu = PCE500Emulator(perfetto_trace=False, save_lcd_on_exit=False)

    # Advance cycle count beyond both timer thresholds and force a tick.
    threshold = max(emu._timer_next_mti, emu._timer_next_sti)  # type: ignore[attr-defined]
    emu.cycle_count = threshold + 123  # type: ignore[attr-defined]
    emu._tick_timers()  # type: ignore[attr-defined]

    assert emu._timer_next_mti > emu._timer_mti_period  # type: ignore[attr-defined]
    assert emu._timer_next_sti > emu._timer_sti_period  # type: ignore[attr-defined]

    emu.reset()

    assert emu._timer_next_mti == emu._timer_mti_period  # type: ignore[attr-defined]
    assert emu._timer_next_sti == emu._timer_sti_period  # type: ignore[attr-defined]
    assert not emu._irq_pending  # type: ignore[attr-defined]
    assert emu._irq_source is None  # type: ignore[attr-defined]
