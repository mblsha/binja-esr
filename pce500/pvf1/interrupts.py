"""PV-F1 interrupt admission candidate, distinct from physical IRQ timing."""


def handler_blocks_irq(in_interrupt, imr_only_candidate=False):
    """Architectural masks remain checked by the CPU after this monitor policy."""
    return bool(in_interrupt and not imr_only_candidate)


def sample_on_input(backing, held, live_candidate=False):
    """PV-F1 SSR.ONK is an external input, separate from latched ISR.ONKI."""
    value = backing & ~0x08 if live_candidate else backing
    return value | (0x08 if held else 0)


def on_irq_should_reassert(held, *, contact_edge_candidate=False, alarm_wake=False):
    """A held SSR input is separate from a guest-acknowledged ON edge latch."""
    return bool((held or alarm_wake) and not (held and contact_edge_candidate))
