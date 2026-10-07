"""Fresh-runtime CPU/timing profile reference; Rust remains the executing CPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionProfile:
    block_transfer: str
    byte_arithmetic: str
    isr_software_write: str
    timers_enabled: bool
    halt_main_timer: bool
    rtc_a2_period: int | None
    rtc_second_period: int | None
    initial_bp: int
    on_irq_edge_only: bool
    irq_imr_only: bool

    @classmethod
    def configure(
        cls, name, *, retained_loaded=False, empty_sram=None, instructions=0, cycles=0
    ):
        if instructions or cycles:
            raise ValueError("OZ profile must be set before the first CPU boundary")
        if name not in (
            "strict",
            "experimental",
            "experimental-isr-clear-only",
            "experimental-isr-mti-writable",
            "experimental-on-edge",
            "experimental-irq-imr",
            "experimental-rtc",
            "experimental-rtc-irq-imr",
            "provisional-v1",
        ):
            raise ValueError("Unknown execution profile")
        experiment = name != "strict"
        clear_only = name in {
            "experimental-isr-clear-only",
            "experimental-isr-mti-writable",
            "experimental-on-edge",
            "experimental-irq-imr",
            "experimental-rtc",
            "experimental-rtc-irq-imr",
            "provisional-v1",
        }
        return cls(
            "coupled_predecrement" if experiment else "independent",
            "low_byte" if experiment else "strict",
            "clear_only_except_mti"
            if name
            in {
                "experimental-isr-mti-writable",
                "experimental-on-edge",
                "experimental-irq-imr",
                "experimental-rtc",
                "experimental-rtc-irq-imr",
                "provisional-v1",
            }
            else "clear_only"
            if clear_only
            else "replace",
            experiment,
            experiment,
            512_000
            if name
            in {"experimental-rtc", "experimental-rtc-irq-imr", "provisional-v1"}
            else 1_000_000
            if experiment
            else None,
            1_024_000
            if name
            in {"experimental-rtc", "experimental-rtc-irq-imr", "provisional-v1"}
            else None,
            0xD0
            if (
                name == "provisional-v1"
                and (not retained_loaded if empty_sram is None else empty_sram)
            )
            or (name == "experimental" and not retained_loaded)
            else 0,
            name
            in {
                "experimental-on-edge",
                "experimental-irq-imr",
                "experimental-rtc-irq-imr",
                "provisional-v1",
            },
            name
            in {"experimental-irq-imr", "experimental-rtc-irq-imr", "provisional-v1"},
        )


def on_irq_should_reassert(held: bool, *, edge_only: bool) -> bool:
    """Profile reference for the Rust scheduler's ONKI assertion decision.

    Edge-only leaves physical SSR polling intact. Press latches ONKI once;
    release cannot acknowledge it, and another rising contact can latch again.
    This opt-in hypothesis does not qualify the physical RTC/CPU ON waveform.
    """
    return held and not edge_only


def handler_blocks_irq(in_handler: bool, *, imr_only: bool) -> bool:
    """Optional metadata guard; architectural IMR/source/ISR checks still apply."""
    return in_handler and not imr_only
