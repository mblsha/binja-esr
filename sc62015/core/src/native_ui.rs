// PY_SOURCE: pce500/run_pce500.py
//! Host-only native UI ownership. No architectural state or bus access.
//! Locks protect short queue operations only; no user callback or I/O runs
//! under a lock. A blocked writer retains one frame plus one newest pending
//! frame, not a growing queue, and cannot hold up priority control requests.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const INPUT_CAPACITY: usize = 128;
pub const INPUT_BATCH: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlState {
    pub revision: u64,
    pub paused: bool,
    pub release_epoch: u64,
    pub quit: bool,
}

pub struct ControlInbox<E> {
    // Low bit is desired pause state; upper bits identify the request.
    pause: AtomicU64,
    release_epoch: AtomicU64,
    quit: AtomicBool,
    events: Mutex<VecDeque<E>>,
    error: Mutex<Option<String>>,
    dropped: AtomicU64,
}

impl<E> Default for ControlInbox<E> {
    fn default() -> Self {
        Self {
            pause: AtomicU64::new(0),
            release_epoch: AtomicU64::new(0),
            quit: AtomicBool::new(false),
            events: Mutex::new(VecDeque::new()),
            error: Mutex::new(None),
            dropped: AtomicU64::new(0),
        }
    }
}

impl<E> ControlInbox<E> {
    pub fn state(&self) -> ControlState {
        let pause = self.pause.load(Ordering::Acquire);
        ControlState {
            revision: pause >> 1,
            paused: pause & 1 != 0,
            release_epoch: self.release_epoch.load(Ordering::Acquire),
            quit: self.quit.load(Ordering::Acquire),
        }
    }
    pub fn changed(&self, observed: ControlState) -> bool {
        self.state() != observed
    }
    pub fn request_quit(&self) {
        self.quit.store(true, Ordering::Release);
    }
    pub fn toggle_pause(&self) {
        self.update_pause(|old| ((old & !1).wrapping_add(2)) | ((old & 1) ^ 1));
    }
    pub fn request_pause(&self) {
        self.update_pause(|old| (old & !1).wrapping_add(2) | 1);
    }
    /// Atomically replace the pause word with `next(current)` (the
    /// `fetch_update(AcqRel, Acquire, ..)` loop, spelled out so it builds
    /// without deprecation warnings on every toolchain).
    fn update_pause(&self, next: impl Fn(u64) -> u64) {
        let mut current = self.pause.load(Ordering::Acquire);
        while let Err(observed) = self.pause.compare_exchange_weak(
            current,
            next(current),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            current = observed;
        }
    }
    pub fn release_all(&self) {
        let mut events = self.events.lock().unwrap();
        events.clear();
        self.release_epoch.fetch_add(1, Ordering::Release);
    }
    pub fn push(&self, event: E) -> bool {
        let mut events = self.events.lock().unwrap();
        if events.len() >= INPUT_CAPACITY {
            self.dropped
                .fetch_add(events.len() as u64 + 1, Ordering::Relaxed);
            events.clear();
            self.release_epoch.fetch_add(1, Ordering::Release);
            self.request_pause();
            return false;
        }
        events.push_back(event);
        true
    }
    /// Capture the release epoch under the same lock as the events. A caller
    /// must release old contacts before applying a newer epoch, and discard an
    /// already-drained batch if focus loss/overflow advances that epoch again.
    /// Pause requests alone do not invalidate key releases in a drained batch.
    pub fn take_batch(&self) -> (u64, Vec<E>) {
        self.take_batch_limit(INPUT_BATCH)
    }
    /// A tap-based terminal consumes at most one event until its previous
    /// contact has released. Keep the backlog here so overflow stays bounded.
    pub fn take_batch_limit(&self, limit: usize) -> (u64, Vec<E>) {
        let mut events = self.events.lock().unwrap();
        let count = events.len().min(INPUT_BATCH).min(limit);
        let epoch = self.release_epoch.load(Ordering::Acquire);
        (epoch, events.drain(..count).collect())
    }
    pub fn dropped_inputs(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
    pub fn fail(&self, error: impl AsRef<str>) {
        *self.error.lock().unwrap() = Some(error.as_ref().chars().take(512).collect());
        self.release_all();
        self.request_quit();
    }
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }
}

struct OutputState<T> {
    pending: Option<T>,
    closed: bool,
    done: bool,
    error: Option<String>,
    published: u64,
    presented: u64,
    coalesced: u64,
}
struct OutputShared<T> {
    state: Mutex<OutputState<T>>,
    ready: Condvar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputStats {
    pub published: u64,
    pub presented: u64,
    pub coalesced: u64,
    pub pending: bool,
    pub done: bool,
    pub error: Option<String>,
}

pub struct LatestOutput<T> {
    shared: Arc<OutputShared<T>>,
    thread: Option<JoinHandle<()>>,
}

impl<T: Send + 'static> LatestOutput<T> {
    pub fn spawn(
        mut write: impl FnMut(Option<T>) -> Result<(), String> + Send + 'static,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(OutputShared {
            state: Mutex::new(OutputState {
                pending: None,
                closed: false,
                done: false,
                error: None,
                published: 0,
                presented: 0,
                coalesced: 0,
            }),
            ready: Condvar::new(),
        });
        let worker = shared.clone();
        let thread = thread::Builder::new()
            .name("lcd-output".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let result = (|| {
                        loop {
                            let next = {
                                let mut state = worker.state.lock().unwrap();
                                while state.pending.is_none() && !state.closed {
                                    state = worker.ready.wait(state).unwrap();
                                }
                                state.pending.take()
                            };
                            let Some(next) = next else {
                                break;
                            };
                            write(Some(next))?;
                            worker.state.lock().unwrap().presented += 1;
                        }
                        Ok(())
                    })();
                    // Try terminal restoration even after an ordinary write error.
                    let restored = write(None);
                    result.and(restored)
                }));
                let mut state = worker.state.lock().unwrap();
                state.error = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error.chars().take(512).collect()),
                    Err(_) => Some("native output worker panicked".into()),
                };
                state.pending = None;
                state.closed = true;
                state.done = true;
                worker.ready.notify_all();
            })?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    pub fn publish(&self, frame: T) -> bool {
        let mut state = self.shared.state.lock().unwrap();
        if state.closed {
            return false;
        }
        if state.pending.replace(frame).is_some() {
            state.coalesced += 1;
        }
        state.published += 1;
        self.shared.ready.notify_one();
        true
    }

    pub fn stats(&self) -> OutputStats {
        let state = self.shared.state.lock().unwrap();
        OutputStats {
            published: state.published,
            presented: state.presented,
            coalesced: state.coalesced,
            pending: state.pending.is_some(),
            done: state.done,
            error: state.error.clone(),
        }
    }

    /// Stop after presenting the latest pending frame. Return false rather
    /// than joining a thread stuck in user/terminal I/O. Its output handles must
    /// NOT own the process-global stdout lock, including at process shutdown.
    pub fn finish(&mut self, timeout: Duration) -> bool {
        let started = Instant::now();
        let mut state = self.shared.state.lock().unwrap();
        state.closed = true;
        self.shared.ready.notify_all();
        while !state.done {
            let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
                return false;
            };
            let (next, _) = self.shared.ready.wait_timeout(state, remaining).unwrap();
            state = next;
        }
        drop(state);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        true
    }
}

impl<T> Drop for LatestOutput<T> {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap();
        state.closed = true;
        self.shared.ready.notify_all();
        // JoinHandle drop detaches; never implicitly wait for blocked output.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn input_overflow_cannot_drop_a_release_and_leave_a_key_stuck() {
        let inbox = ControlInbox::default();
        for event in 0..INPUT_CAPACITY {
            assert!(inbox.push(event));
        }
        assert!(!inbox.push(99));
        assert_eq!(inbox.dropped_inputs(), INPUT_CAPACITY as u64 + 1);
        assert!(inbox.take_batch().1.is_empty());
        let observed = inbox.state();
        assert!(observed.paused);
        assert_eq!(observed.release_epoch, 1);
        inbox.toggle_pause();
        assert!(inbox.changed(observed));
        assert!(!inbox.state().paused);
        inbox.request_quit();
        assert!(inbox.state().quit);
    }

    #[test]
    fn input_batches_are_bounded_and_focus_loss_discards_stale_downs() {
        let inbox = ControlInbox::default();
        for event in 0..100 {
            inbox.push(event);
        }
        assert_eq!(
            inbox.take_batch(),
            (0, (0..INPUT_BATCH).collect::<Vec<_>>())
        );
        inbox.release_all();
        assert_eq!(inbox.take_batch(), (1, Vec::new()));
        inbox.fail("reader disconnected");
        assert!(inbox.state().quit);
        assert_eq!(inbox.error().as_deref(), Some("reader disconnected"));
    }

    #[test]
    fn drained_input_is_invalidated_by_focus_loss_but_not_pause() {
        let inbox = ControlInbox::default();
        inbox.push("down");
        inbox.push("up");
        let (epoch, batch) = inbox.take_batch();
        inbox.toggle_pause();
        assert_eq!(inbox.state().release_epoch, epoch);
        assert_eq!(batch, ["down", "up"]);
        inbox.release_all();
        assert_ne!(inbox.state().release_epoch, epoch);
        inbox.push("new down");
        assert_eq!(inbox.take_batch(), (epoch + 1, vec!["new down"]));
    }

    #[test]
    fn serialized_taps_leave_the_remainder_in_the_bounded_queue() {
        let inbox = ControlInbox::default();
        for key in ['A', 'B', 'C'] {
            inbox.push(key);
        }
        assert_eq!(inbox.take_batch_limit(0), (0, vec![]));
        assert_eq!(inbox.take_batch_limit(1), (0, vec!['A']));
        assert_eq!(inbox.take_batch_limit(1), (0, vec!['B']));
        inbox.release_all();
        assert_eq!(inbox.take_batch_limit(1), (1, vec![]));
    }

    #[test]
    fn blocked_output_is_latest_only_and_never_blocks_priority_controls_or_shutdown() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (gate_tx, gate_rx) = mpsc::channel();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let written = seen.clone();
        let mut output = LatestOutput::spawn(move |frame| {
            let Some(frame) = frame else {
                return Ok(());
            };
            if frame == 0 {
                entered_tx.send(()).unwrap();
                gate_rx.recv().unwrap();
            }
            written.lock().unwrap().push(frame);
            Ok(())
        })
        .unwrap();
        output.publish(0);
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let inbox = ControlInbox::<u8>::default();
        for frame in 1..=1000 {
            assert!(output.publish(frame));
        }
        inbox.toggle_pause();
        assert!(inbox.state().paused);
        inbox.request_quit();
        assert!(inbox.state().quit);
        let stats = output.stats();
        assert_eq!(stats.coalesced, 999);
        assert!(stats.pending);
        assert!(!output.finish(Duration::from_millis(5)));
        assert!(!output.publish(1001));
        gate_tx.send(()).unwrap();
        assert!(output.finish(Duration::from_secs(1)));
        assert_eq!(*seen.lock().unwrap(), [0, 1000]);
    }

    #[test]
    fn output_errors_are_reported_and_close_the_queue() {
        let mut output = LatestOutput::spawn(|frame: Option<u8>| {
            if frame.is_some() {
                Err("broken pipe".into())
            } else {
                Ok(())
            }
        })
        .unwrap();
        output.publish(1);
        assert!(output.finish(Duration::from_secs(1)));
        assert_eq!(output.stats().error.as_deref(), Some("broken pipe"));
        assert!(!output.publish(2));
    }
}
