from pce500.scheduler import TimerScheduler, TimerSource


def test_scheduler_fires_mti_and_sti() -> None:
    sched = TimerScheduler(mti_period=5, sti_period=8)

    assert list(sched.advance(0)) == []
    assert list(sched.advance(5)) == [TimerSource.MTI]

    fired = list(sched.advance(8))
    assert TimerSource.STI in fired


def test_scheduler_respects_enable_flag() -> None:
    sched = TimerScheduler(mti_period=3, sti_period=7)
    sched.enabled = False
    assert list(sched.advance(10)) == []


def test_scheduler_reset_uses_cycle_base() -> None:
    sched = TimerScheduler(mti_period=4, sti_period=6)
    sched.next_mti = 100
    sched.reset(cycle_base=10)
    assert sched.next_mti == 14
