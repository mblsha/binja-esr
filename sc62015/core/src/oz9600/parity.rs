// PY_SOURCE: pce500/oz9600/parity.py
// PY_SOURCE: pce500/tests/test_oz9600_audio.py
//! Execute the independent Python controller corpus in the Rust bus.
use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[test]
fn audio_pcm_chunks_and_all_phase_counters_match_python_reference() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../data/oz9600_audio_reference.json")).unwrap();
    let mut audio = audio::AudioCapture::default();
    for (index, op) in fixture["operations"].as_array().unwrap().iter().enumerate() {
        let chunk = match op[0].as_str().unwrap() {
            "advance" => {
                audio.advance(
                    op[1].as_u64().unwrap(),
                    op[2].as_u64().unwrap() as u8,
                    op[3].as_bool().unwrap(),
                    op[4].as_bool().unwrap(),
                );
                Value::Null
            }
            "enable" => {
                audio.set_enabled(op[1].as_bool().unwrap());
                Value::Null
            }
            "reset" => {
                audio.reset();
                Value::Null
            }
            "take" => {
                let chunk = audio.take();
                let pcm = chunk
                    .samples
                    .iter()
                    .flat_map(|s| s.to_le_bytes())
                    .collect::<Vec<_>>();
                json!({"sample_rate":chunk.sample_rate,"first_sample":chunk.first_sample,
                    "total_samples":chunk.total_samples,"dropped_samples":chunk.dropped_samples,
                    "sample_count":chunk.samples.len(),"pcm_sha256":format!("{:x}",Sha256::digest(pcm))})
            }
            unknown => panic!("unknown audio operation {unknown}"),
        };
        assert_eq!(
            json!({"chunk":chunk,"status":audio.status()}),
            fixture["replies"][index],
            "audio operation {index}: {op}"
        );
    }
}

#[test]
fn controller_bus_matches_python_reference_through_all_observed_operations() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../data/oz9600_controller_reference.json")).unwrap();
    let mut hw = Hardware::default();
    hw.banks.insert(0xf0, vec![0x21; 0x20000]);
    hw.banks.insert(0xf1, vec![0x43; 0x20000]);
    for (index, op) in fixture["operations"].as_array().unwrap().iter().enumerate() {
        let n = |i: usize| op[i].as_u64().unwrap();
        let result = match op[0].as_str().unwrap() {
            "write" => {
                hw.write(n(1) as u32, n(2) as u8, None);
                Value::Null
            }
            "read" => json!(hw.architectural_read(n(1) as u32, None)),
            "peek" => json!(hw.read(n(1) as u32)),
            "cycle" => {
                hw.cycle = n(1);
                Value::Null
            }
            "clear_fault" => {
                hw.fault = None;
                Value::Null
            }
            "causes" => {
                hw.rtc.raise_causes(n(1) as u8, n(2) as u8);
                Value::Null
            }
            "irq" => json!(hw.refresh_irq_inputs()),
            "contact" => {
                match hw
                    .tablet
                    .set_contact(n(1) as u16, n(2) as u16, op[3].as_bool().unwrap())
                {
                    Ok(edge) => {
                        if edge {
                            hw.gate[18] |= 2;
                        }
                        Value::Null
                    }
                    Err(_) => json!({"error":true}),
                }
            }
            "seconds" => match hw.rtc.advance_seconds(n(1)) {
                Ok(()) => Value::Null,
                Err(_) => json!({"error":true}),
            },
            unknown => panic!("unknown corpus operation {unknown}"),
        };
        assert_eq!(result, fixture["replies"][index], "operation {index}: {op}");
        for checkpoint in fixture["checkpoints"].as_array().unwrap() {
            if checkpoint["index"] != index {
                continue;
            }
            let actual = json!({
                "selector":hw.selector,"gate":hw.gate.to_vec(),"gpio":hw.gpio.to_vec(),
                "rtc":hw.rtc.registers(),"lcd":hw.lcd.registers.to_vec(),
                "windows":(0..16).map(|n|hw.lcd.window_descriptor(n)).collect::<Vec<_>>(),
                "lcd_counts":[hw.lcd.data_writes,hw.lcd.data_reads,hw.lcd.block_operations],
                "tablet":[json!(hw.tablet.x),json!(hw.tablet.y),json!(hw.tablet.pressed),json!(hw.tablet.conversion_control),json!(hw.tablet.drive_control),json!(hw.tablet.data_reads)],
                "fault":hw.fault.is_some(),
                "pbm_sha256":format!("{:x}",Sha256::digest(hw.lcd.pbm())),
                "ram_sha256":format!("{:x}",Sha256::digest(&hw.ram)),
                "events_sha256":format!("{:x}",Sha256::digest(serde_json::to_vec(&hw.events).unwrap())),
            });
            assert_eq!(
                actual, checkpoint["state"],
                "checkpoint at operation {index}"
            );
        }
    }
}
