// PY_SOURCE: pce500/oz9600/native_audio.py
//! Bounded host playback of the existing nominal-clock PCM. No guest mutation.
use sc62015_core::oz9600::audio::AudioChunk;
#[cfg(any(feature = "audio", test))]
use sc62015_core::oz9600::audio::SAMPLE_RATE;
#[cfg(any(feature = "audio", test))]
use std::collections::VecDeque;

#[cfg(any(feature = "audio", test))]
const CAPACITY: usize = SAMPLE_RATE as usize / 10;

#[cfg(any(feature = "audio", test))]
pub struct Queue {
    samples: VecDeque<i16>,
    expected: Option<u64>,
    phase: u64,
    current: Option<f32>,
    previous_input: f32,
    previous_output: f32,
    output_rate: u32,
    alpha: f32,
    pub dropped: u64,
    pub rendered: u64,
    pub nonzero: u64,
}
#[cfg(any(feature = "audio", test))]
impl Queue {
    pub fn new(output_rate: u32) -> Self {
        assert!(output_rate > 0);
        Self {
            samples: VecDeque::with_capacity(CAPACITY),
            expected: None,
            phase: 0,
            current: None,
            previous_input: 0.0,
            previous_output: 0.0,
            output_rate,
            alpha: (-std::f32::consts::TAU * 40.0 / output_rate as f32).exp(),
            dropped: 0,
            rendered: 0,
            nonzero: 0,
        }
    }
    pub fn clear(&mut self) {
        self.samples.clear();
        self.expected = None;
        self.phase = 0;
        self.current = None;
        self.previous_input = 0.0;
        self.previous_output = 0.0;
    }
    pub fn push(&mut self, chunk: &AudioChunk) {
        if chunk.sample_rate != SAMPLE_RATE {
            self.clear();
            return;
        }
        if self.expected.is_some_and(|next| next != chunk.first_sample) {
            self.clear();
        }
        self.expected = Some(chunk.first_sample.wrapping_add(chunk.samples.len() as u64));
        let drop = self
            .samples
            .len()
            .saturating_add(chunk.samples.len())
            .saturating_sub(CAPACITY);
        if drop > 0 {
            let old = drop.min(self.samples.len());
            self.samples.drain(..old);
            self.dropped = self.dropped.saturating_add(drop as u64);
            self.phase = 0;
            self.current = None;
            self.previous_input = 0.0;
            self.previous_output = 0.0;
        }
        let skip = chunk.samples.len().saturating_sub(CAPACITY);
        self.samples.extend(chunk.samples[skip..].iter().copied());
    }
    /// Zero-order rational rate conversion, then a 40 Hz host DC blocker.
    /// Empty queues emit immediate zero, rather than repeating the last tone.
    pub fn next(&mut self) -> f32 {
        if self.current.is_none() {
            self.current = self.samples.pop_front().map(|x| f32::from(x) / 32768.0);
        }
        let Some(input) = self.current else {
            self.phase = 0;
            self.previous_input = 0.0;
            self.previous_output = 0.0;
            self.rendered = self.rendered.saturating_add(1);
            return 0.0;
        };
        let output = self.alpha * (self.previous_output + input - self.previous_input);
        self.previous_input = input;
        self.previous_output = output;
        self.phase += u64::from(SAMPLE_RATE);
        while self.phase >= u64::from(self.output_rate) {
            self.phase -= u64::from(self.output_rate);
            self.current = self.samples.pop_front().map(|x| f32::from(x) / 32768.0);
            if self.current.is_none() {
                self.phase = 0;
                break;
            }
        }
        self.rendered = self.rendered.saturating_add(1);
        if output.abs() > 0.0001 {
            self.nonzero = self.nonzero.saturating_add(1);
        }
        output.clamp(-1.0, 1.0)
    }
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({"queued_samples":self.samples.len(),"dropped_samples":self.dropped,
            "rendered_frames":self.rendered,"nonzero_frames":self.nonzero,"output_rate":self.output_rate})
    }
}

#[cfg(feature = "audio")]
mod backend {
    use super::*;
    use cpal::{
        traits::{DeviceTrait, HostTrait, StreamTrait},
        FromSample, SizedSample,
    };
    use std::sync::{Arc, Mutex};
    pub struct Output {
        queue: Arc<Mutex<Queue>>,
        error: Arc<Mutex<Option<String>>>,
        _stream: cpal::Stream,
    }
    impl Output {
        pub fn open() -> Result<Self, String> {
            let device = cpal::default_host()
                .default_output_device()
                .ok_or("No default audio output device")?;
            let supported = device.default_output_config().map_err(|e| e.to_string())?;
            if supported.sample_rate() == 0 {
                return Err("Audio device has zero sample rate".into());
            }
            let queue = Arc::new(Mutex::new(Queue::new(supported.sample_rate())));
            let error = Arc::new(Mutex::new(None));
            let format = supported.sample_format();
            let config = supported.config();
            if config.channels == 0 {
                return Err("Audio device has zero channels".into());
            }
            let stream = match format {
                cpal::SampleFormat::F32 => {
                    build::<f32>(&device, &config, queue.clone(), error.clone())
                }
                cpal::SampleFormat::I16 => {
                    build::<i16>(&device, &config, queue.clone(), error.clone())
                }
                cpal::SampleFormat::U16 => {
                    build::<u16>(&device, &config, queue.clone(), error.clone())
                }
                _ => return Err(format!("Unsupported audio sample format: {format}")),
            }
            .map_err(|e| e.to_string())?;
            stream.play().map_err(|e| e.to_string())?;
            Ok(Self {
                queue,
                error,
                _stream: stream,
            })
        }
        pub fn push(&self, chunk: &AudioChunk) {
            if let Ok(mut queue) = self.queue.lock() {
                queue.push(chunk);
            }
        }
        pub fn clear(&self) {
            if let Ok(mut queue) = self.queue.lock() {
                queue.clear();
            }
        }
        pub fn error(&self) -> Option<String> {
            self.error.lock().ok().and_then(|mut value| value.take())
        }
        pub fn status(&self) -> serde_json::Value {
            self.queue
                .lock()
                .map(|queue| queue.status())
                .unwrap_or_else(|_| serde_json::json!({"error":"Audio queue unavailable"}))
        }
    }
    fn build<T: SizedSample + FromSample<f32>>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        queue: Arc<Mutex<Queue>>,
        error: Arc<Mutex<Option<String>>>,
    ) -> Result<cpal::Stream, cpal::Error> {
        let channels = usize::from(config.channels);
        device.build_output_stream(
            *config,
            move |data: &mut [T], _| {
                // Never wait for the producer on the device callback thread.
                if let Ok(mut queue) = queue.try_lock() {
                    for frame in data.chunks_mut(channels) {
                        let sample = T::from_sample(queue.next());
                        frame.fill(sample);
                    }
                } else {
                    data.fill(T::EQUILIBRIUM);
                }
            },
            move |failure| {
                if let Ok(mut value) = error.lock() {
                    *value = Some(failure.to_string());
                }
            },
            None,
        )
    }
}
#[cfg(feature = "audio")]
pub use backend::Output;
#[cfg(not(feature = "audio"))]
pub struct Output;
#[cfg(not(feature = "audio"))]
impl Output {
    pub fn open() -> Result<Self, String> {
        Err("This build has no audio backend; build with --features audio".into())
    }
    pub fn push(&self, _: &AudioChunk) {}
    pub fn clear(&self) {}
    pub fn error(&self) -> Option<String> {
        None
    }
    pub fn status(&self) -> serde_json::Value {
        serde_json::json!({"enabled":false})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chunk(first_sample: u64, samples: Vec<i16>) -> AudioChunk {
        AudioChunk {
            sample_rate: SAMPLE_RATE,
            first_sample,
            total_samples: first_sample + samples.len() as u64,
            dropped_samples: 0,
            samples,
        }
    }
    #[test]
    fn python_reference_matches_streaming_rate_conversion_and_filter() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
        let code = r#"import json,runpy,sys
Q=runpy.run_path(sys.argv[1])["Queue"]
output=[]
for rate in (24000,44100,48000,96000):
 q=Q(rate)
 samples=[8192 if n%37<19 else 0 for n in range(500)]
 q.push(0,samples[:250])
 values=[q.next() for _ in range(100)]
 q.push(250,samples[250:])
 values += [q.next() for _ in range(1000)]
 q.push(10000,[0]*10)
 values += [q.next() for _ in range(20)]
 q.clear()
 values += [q.next()]
 output.append(values)
print(json.dumps(output))"#;
        let output = std::process::Command::new(python)
            .arg("-c")
            .arg(code)
            .arg(root.join("pce500/oz9600/native_audio.py"))
            .output()
            .expect("Python reference is required");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected: Vec<Vec<f32>> = serde_json::from_slice(&output.stdout).unwrap();
        for (rate, expected) in [24_000, 44_100, 48_000, 96_000].into_iter().zip(expected) {
            let mut queue = Queue::new(rate);
            let samples = (0..500)
                .map(|n| if n % 37 < 19 { 8192 } else { 0 })
                .collect::<Vec<_>>();
            queue.push(&chunk(0, samples[..250].to_vec()));
            let mut actual = (0..100).map(|_| queue.next()).collect::<Vec<_>>();
            queue.push(&chunk(250, samples[250..].to_vec()));
            actual.extend((0..1000).map(|_| queue.next()));
            queue.push(&chunk(10000, vec![0; 10]));
            actual.extend((0..20).map(|_| queue.next()));
            queue.clear();
            actual.push(queue.next());
            assert_eq!(actual.len(), expected.len());
            for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - b).abs() < 0.000001,
                    "rate {rate}, sample {index}: {a} != {b}"
                );
            }
        }
    }
    #[test]
    fn backlog_and_discontinuity_keep_latest_and_clear_old_tone() {
        let mut queue = Queue::new(SAMPLE_RATE);
        queue.push(&chunk(0, vec![8192; CAPACITY + 17]));
        assert_eq!(queue.samples.len(), CAPACITY);
        assert_eq!(queue.dropped, 17);
        assert!(queue.next() > 0.2);
        queue.push(&chunk(100_000, vec![0; 16]));
        assert_eq!(queue.next(), 0.0);
        assert_eq!(queue.samples.len(), 14);
        queue.clear();
        assert_eq!(queue.next(), 0.0);
        assert!(queue.samples.is_empty());
        assert_eq!(queue.status()["queued_samples"], 0);
        assert_eq!(queue.status()["dropped_samples"], 17);
    }
    #[test]
    fn mismatched_source_rate_clears_previous_tone() {
        let mut queue = Queue::new(SAMPLE_RATE);
        queue.push(&chunk(0, vec![8192; 16]));
        assert!(queue.next() > 0.2);
        let mut wrong_rate = chunk(16, vec![8192; 16]);
        wrong_rate.sample_rate = 44_100;
        queue.push(&wrong_rate);
        assert_eq!(queue.next(), 0.0);
        assert_eq!(queue.status()["queued_samples"], 0);
    }
    #[test]
    fn drain_silence_dc_filter_and_rate_conversion_are_bounded() {
        for rate in [24_000, 44_100, 48_000, 96_000] {
            let mut queue = Queue::new(rate);
            queue.push(&chunk(0, vec![8192; 4800]));
            let output = (0..rate / 10).map(|_| queue.next()).collect::<Vec<_>>();
            assert!(output[0] > 0.2);
            assert!(output.last().unwrap().abs() < 0.0001);
            assert_eq!(queue.next(), 0.0);
            assert!(queue.samples.is_empty());
            assert!(output
                .iter()
                .all(|sample| sample.is_finite() && sample.abs() <= 1.0));
        }
    }
}
