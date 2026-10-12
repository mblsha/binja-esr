"""PV-F1 interrupt admission candidate, distinct from physical IRQ timing."""


def handler_blocks_irq(in_interrupt, imr_only_candidate=False):
    """Architectural masks remain checked by the CPU after this monitor policy."""
    return bool(in_interrupt and not imr_only_candidate)
