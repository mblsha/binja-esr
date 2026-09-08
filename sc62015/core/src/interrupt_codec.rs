// PY_SOURCE: pce500/scheduler.py:TimerScheduler
//! Legacy interrupt metadata adapter. Runtime snapshots contain no JSON values.
use crate::timer::{BitWatch, InterruptSnapshot};
use crate::InterruptInfo;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;

#[cfg(test)]
pub(crate) fn default_bit_watch_table() -> serde_json::Map<String, serde_json::Value> {
    let mut table = serde_json::Map::new();
    for reg in ["IMR", "ISR"] {
        let mut reg_map = serde_json::Map::new();
        for bit in 0..8u8 {
            reg_map.insert(
                bit.to_string(),
                json!({
                    "set": [],
                    "clear": [],
                }),
            );
        }
        table.insert(reg.to_string(), serde_json::Value::Object(reg_map));
    }
    table
}

impl BitWatch {
    #[cfg(any(test, feature = "json-compat"))]
    pub(crate) fn from_json(
        table: serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, String> {
        serde_json::from_value(serde_json::Value::Object(table)).map_err(|e| e.to_string())
    }

    #[cfg(any(test, feature = "json-compat"))]
    pub(crate) fn to_json(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::to_value(self)
            .expect("typed history")
            .as_object()
            .unwrap()
            .clone()
    }

    pub fn addresses_valid(&self) -> bool {
        self.histories
            .iter()
            .flatten()
            .flatten()
            .all(|h| h.pcs[..h.len].iter().all(|pc| *pc <= 0xfffff))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Actions {
    set: Vec<u32>,
    clear: Vec<u32>,
}

impl Serialize for BitWatch {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use std::collections::BTreeMap;
        let mut registers = BTreeMap::new();
        for (r, name) in ["IMR", "ISR"].into_iter().enumerate() {
            let mut bits = BTreeMap::new();
            for bit in 0..8 {
                let [set, clear] = &self.histories[r][bit];
                bits.insert(
                    bit.to_string(),
                    Actions {
                        set: set.pcs[..set.len].to_vec(),
                        clear: clear.pcs[..clear.len].to_vec(),
                    },
                );
            }
            registers.insert(name, bits);
        }
        registers.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BitWatch {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        use std::collections::BTreeMap;
        let registers = BTreeMap::<String, BTreeMap<String, Actions>>::deserialize(deserializer)?;
        if registers.len() != 2 {
            return Err(D::Error::custom("IRQ history requires IMR and ISR"));
        }
        let mut result = Self::default();
        for (r, name) in ["IMR", "ISR"].into_iter().enumerate() {
            let bits = registers
                .get(name)
                .ok_or_else(|| D::Error::custom("missing IRQ history register"))?;
            if bits.len() != 8 {
                return Err(D::Error::custom("IRQ history requires eight bits"));
            }
            for bit in 0..8 {
                let actions = bits
                    .get(&bit.to_string())
                    .ok_or_else(|| D::Error::custom("missing IRQ history bit"))?;
                for (a, pcs) in [&actions.set, &actions.clear].into_iter().enumerate() {
                    if pcs.len() > 10 {
                        return Err(D::Error::custom("IRQ history exceeds ten PCs"));
                    }
                    result.histories[r][bit][a].pcs[..pcs.len()].copy_from_slice(pcs);
                    result.histories[r][bit][a].len = pcs.len();
                }
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub total: u32,
    #[serde(rename = "KEY")]
    pub key: u32,
    #[serde(rename = "MTI")]
    pub mti: u32,
    #[serde(rename = "STI")]
    pub sti: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LastIrq {
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub src: Option<String>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub pc: Option<u32>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub vector: Option<u32>,
}

impl From<InterruptSnapshot> for InterruptInfo {
    fn from(s: InterruptSnapshot) -> Self {
        let [total, key, mti, sti] = s.counts;
        Self {
            pending: s.pending,
            in_interrupt: s.in_interrupt,
            key_irq_latched: s.key_irq_latched,
            source: s.source,
            last_fired: s.last_fired,
            stack: s.stack,
            next_id: s.next_id,
            imr: s.imr,
            isr: s.isr,
            irq_counts: Some(Counts {
                total,
                key,
                mti,
                sti,
            }),
            last_irq: Some(LastIrq {
                src: s.last_source,
                pc: s.last_pc,
                vector: s.last_vector,
            }),
            irq_bit_watch: Some(s.history.unwrap_or_default()),
            delivered_masks: s.delivered_masks,
        }
    }
}

impl TryFrom<&InterruptInfo> for InterruptSnapshot {
    type Error = String;

    fn try_from(s: &InterruptInfo) -> Result<Self, Self::Error> {
        let counts = s.irq_counts.clone().unwrap_or_default();
        let last = s.last_irq.clone().unwrap_or_default();
        if last.pc.is_some_and(|v| v > 0xfffff) || last.vector.is_some_and(|v| v > 0xfffff) {
            return Err("last IRQ address exceeds 20 bits".into());
        }
        let history = s.irq_bit_watch.clone();
        if history.as_ref().is_some_and(|h| !h.addresses_valid()) {
            return Err("IRQ history address exceeds 20 bits".into());
        }
        Ok(Self {
            pending: s.pending,
            in_interrupt: s.in_interrupt,
            key_irq_latched: s.key_irq_latched,
            source: s.source.clone(),
            last_fired: s.last_fired.clone(),
            stack: s.stack.clone(),
            next_id: s.next_id,
            imr: s.imr,
            isr: s.isr,
            counts: [counts.total, counts.key, counts.mti, counts.sti],
            last_source: last.src,
            last_pc: last.pc,
            last_vector: last.vector,
            history,
            delivered_masks: s.delivered_masks.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_diagnostics_keep_required_fields_and_numeric_bounds() {
        assert!(serde_json::from_str::<Counts>(r#"{"total":0,"KEY":0,"MTI":0}"#).is_err());
        assert!(
            serde_json::from_str::<Counts>(r#"{"total":4294967296,"KEY":0,"MTI":0,"STI":0}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<LastIrq>(r#"{"src":null,"pc":null}"#).is_err());
        assert!(serde_json::from_str::<LastIrq>(
            r#"{"src":null,"pc":null,"vector":null,"extra":0}"#
        )
        .is_err());
        let valid = r#"{"src":null,"pc":null,"vector":null}"#;
        assert_eq!(
            serde_json::from_str::<LastIrq>(valid).unwrap(),
            LastIrq::default()
        );
    }
}
