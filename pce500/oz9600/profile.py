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
    external_off_wake: bool
    irq_recognition_delay: bool

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
            "experimental-external-wake",
            "experimental-irq-recognition",
        ):
            raise ValueError("Unknown execution profile")
        experiment = name != "strict"
        return cls(
            # Device instruction measurements, independent of clock/reset profile.
            "coupled_predecrement",
            "low_byte",
            "clear_only_except_mti"
            if name == "experimental-isr-mti-writable"
            else "clear_only"
            if name == "experimental-isr-clear-only"
            else "replace",
            experiment,
            experiment,
            512_000
            if name
            in {
                "experimental-rtc",
                "experimental-rtc-irq-imr",
                "provisional-v1",
                "experimental-external-wake",
                "experimental-irq-recognition",
            }
            else 1_000_000
            if experiment
            else None,
            1_024_000
            if name
            in {
                "experimental-rtc",
                "experimental-rtc-irq-imr",
                "provisional-v1",
                "experimental-external-wake",
                "experimental-irq-recognition",
            }
            else None,
            0xD0
            if (
                name
                in {
                    "provisional-v1",
                    "experimental-external-wake",
                    "experimental-irq-recognition",
                }
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
                "experimental-external-wake",
                "experimental-irq-recognition",
            },
            name
            in {
                "experimental-irq-imr",
                "experimental-rtc-irq-imr",
                "provisional-v1",
                "experimental-external-wake",
                "experimental-irq-recognition",
            },
            name == "experimental-external-wake",
            name == "experimental-irq-recognition",
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


def off_wake_mask(*, external_enabled: bool, external_experiment: bool) -> int:
    """OFF wake-source reference. EXI extension is an explicit ESR-P hypothesis."""
    return 0x08 | (0x40 if external_enabled and external_experiment else 0)


@dataclass
class IrqRecognition:
    """OZ boundary reference; Rust executes instructions and constructs frames.

    A transfer consumes an earlier recognition; otherwise recognize immediately
    before retiring the following instruction. Its later IMR/ISR changes do not
    alter this mask. Software MTI/STI is measured; electrical and low-power
    boundary composition remains an explicit experiment.
    """

    enabled: bool = False
    recognized_mask: int = 0

    def recognize_before_instruction(self, imr: int, isr: int) -> None:
        if self.enabled and not self.recognized_mask and imr & 0x80:
            pending = imr & isr & 0x7F
            if pending:
                self.recognized_mask = next(
                    mask for mask in (0x20, 0x40, 0x10, 8, 4, 2, 1) if pending & mask
                )

    def transfer_selected(self, *, running: bool = True) -> bool:
        return self.enabled and running and bool(self.recognized_mask)

    def entered_or_reset(self) -> None:
        self.recognized_mask = 0
