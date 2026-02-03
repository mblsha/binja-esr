use std::cell::Cell;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

thread_local! {
    static CURRENT_CYCLE: Cell<u64> = const { Cell::new(0) };
    static NEXT_WAKE_CYCLE: Cell<Option<u64>> = const { Cell::new(None) };
    static PENDING_EVENT: Cell<Option<DriverEvent>> = const { Cell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverEvent {
    MaxCycles,
    User(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverRunResult {
    pub event: DriverEvent,
    pub cycles_executed: u64,
}

pub struct CycleSleep {
    cycles: u64,
    initialized: bool,
}

impl Future for CycleSleep {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let current_cycle = current_cycle();
        if !this.initialized {
            NEXT_WAKE_CYCLE.with(|cell| {
                cell.set(Some(current_cycle.saturating_add(this.cycles)));
            });
            this.initialized = true;
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

pub fn sleep_cycles(cycles: u64) -> CycleSleep {
    CycleSleep {
        cycles,
        initialized: false,
    }
}

pub fn block_on<F: Future>(mut future: F) -> F::Output {
    let mut clock = current_cycle();
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut future = unsafe { Pin::new_unchecked(&mut future) };

    loop {
        CURRENT_CYCLE.with(|cell| cell.set(clock));
        NEXT_WAKE_CYCLE.with(|cell| cell.set(None));
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        let next = NEXT_WAKE_CYCLE
            .with(|cell| cell.get())
            .unwrap_or(clock.saturating_add(1));
        clock = if next <= clock {
            clock.saturating_add(1)
        } else {
            next
        };
        let _ = take_pending_event();
    }
}

pub fn current_cycle() -> u64 {
    CURRENT_CYCLE.with(|cell| cell.get())
}

pub fn emit_event(event: DriverEvent) {
    PENDING_EVENT.with(|cell| {
        if cell.get().is_none() {
            cell.set(Some(event));
        }
    });
}

type FutureQueue = BTreeMap<u64, Vec<Pin<Box<dyn Future<Output = ()>>>>>;

pub struct AsyncDriver {
    clock: u64,
    futures_queue: FutureQueue,
    events_queue: VecDeque<DriverEvent>,
}

impl AsyncDriver {
    pub fn new() -> Self {
        Self {
            clock: 0,
            futures_queue: BTreeMap::new(),
            events_queue: VecDeque::new(),
        }
    }

    pub fn with_clock(clock: u64) -> Self {
        let mut driver = Self::new();
        driver.clock = clock;
        driver
    }

    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn spawn<F>(&mut self, future: F)
    where
        F: Future<Output = ()> + 'static,
    {
        self.futures_queue
            .entry(self.clock)
            .or_default()
            .push(Box::pin(future));
    }

    pub fn run_for(&mut self, max_cycles: u64) -> DriverRunResult {
        if let Some(event) = self.events_queue.pop_front() {
            return DriverRunResult {
                event,
                cycles_executed: 0,
            };
        }

        let start_cycle = self.clock;
        let target_cycle = start_cycle.saturating_add(max_cycles);

        while !self.futures_queue.is_empty() && self.clock < target_cycle {
            let next_cycle = *self.futures_queue.keys().next().unwrap();
            if next_cycle >= target_cycle {
                break;
            }

            self.clock = next_cycle;
            CURRENT_CYCLE.with(|cell| cell.set(self.clock));

            let futures = self.futures_queue.remove(&next_cycle).unwrap();
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);

            for mut future in futures {
                NEXT_WAKE_CYCLE.with(|cell| cell.set(None));
                if future.as_mut().poll(&mut cx).is_pending() {
                    let wake_cycle = NEXT_WAKE_CYCLE
                        .with(|cell| cell.get())
                        .unwrap_or(self.clock.saturating_add(1));
                    self.futures_queue
                        .entry(wake_cycle)
                        .or_default()
                        .push(future);
                }

                if let Some(event) = take_pending_event() {
                    self.events_queue.push_back(event);
                }
            }

            if let Some(event) = self.events_queue.pop_front() {
                return DriverRunResult {
                    event,
                    cycles_executed: self.clock - start_cycle,
                };
            }
        }

        DriverRunResult {
            event: DriverEvent::MaxCycles,
            cycles_executed: self.clock - start_cycle,
        }
    }
}

fn take_pending_event() -> Option<DriverEvent> {
    PENDING_EVENT.with(|cell| {
        let event = cell.get();
        if event.is_some() {
            cell.set(None);
        }
        event
    })
}

fn noop_waker() -> Waker {
    unsafe { Waker::from_raw(noop_raw_waker()) }
}

fn noop_raw_waker() -> RawWaker {
    RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE)
}

fn noop_clone(_: *const ()) -> RawWaker {
    noop_raw_waker()
}

fn noop(_: *const ()) {}

static NOOP_WAKER_VTABLE: RawWakerVTable =
    RawWakerVTable::new(noop_clone, noop, noop, noop);

impl Default for AsyncDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn sleep_resumes_on_target_cycles() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let log_task = log.clone();
        let waits = vec![3_u64, 2_u64];
        let mut driver = AsyncDriver::new();

        driver.spawn(async move {
            for wait in waits {
                sleep_cycles(wait).await;
                log_task.borrow_mut().push(current_cycle());
            }
        });

        let result = driver.run_for(10);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(*log.borrow(), vec![3, 5]);
    }

    #[test]
    fn emit_event_interrupts_run() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let log_task = log.clone();
        let mut driver = AsyncDriver::new();

        driver.spawn(async move {
            sleep_cycles(4).await;
            emit_event(DriverEvent::User(7));
            log_task.borrow_mut().push("event");
            sleep_cycles(2).await;
            log_task.borrow_mut().push("after");
        });

        let result = driver.run_for(10);
        assert_eq!(result.event, DriverEvent::User(7));
        assert_eq!(result.cycles_executed, 4);
        assert_eq!(*log.borrow(), vec!["event"]);

        let result = driver.run_for(10);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(*log.borrow(), vec!["event", "after"]);
    }

    #[test]
    fn pending_without_sleep_advances_by_one() {
        struct PendingOnce {
            log: Rc<RefCell<Vec<u64>>>,
            polled: bool,
        }

        impl Future for PendingOnce {
            type Output = ();

            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                let this = self.get_mut();
                this.log.borrow_mut().push(current_cycle());
                if this.polled {
                    Poll::Ready(())
                } else {
                    this.polled = true;
                    Poll::Pending
                }
            }
        }

        let log = Rc::new(RefCell::new(Vec::new()));
        let mut driver = AsyncDriver::new();
        driver.spawn(PendingOnce {
            log: log.clone(),
            polled: false,
        });

        let result = driver.run_for(5);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(*log.borrow(), vec![0, 1]);
    }

    #[test]
    fn driver_starts_at_configured_clock() {
        let mut driver = AsyncDriver::with_clock(7);
        driver.spawn(async move {
            sleep_cycles(0).await;
        });
        let result = driver.run_for(1);
        assert_eq!(result.event, DriverEvent::MaxCycles);
        assert_eq!(driver.clock(), 7);
    }

    #[test]
    fn block_on_advances_cycles() {
        CURRENT_CYCLE.with(|cell| cell.set(0));
        let result = block_on(async {
            sleep_cycles(2).await;
            current_cycle()
        });
        assert_eq!(result, 2);
    }
}
