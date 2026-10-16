// PY_SOURCE: pce500/emulator.py
//! Owned-state codecs, separate from host pacing, callbacks and caches.
#[cfg(feature = "pa9500")]
pub(crate) mod array {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    pub fn serialize<T: Serialize, S: Serializer, const N: usize>(
        value: &[T; N],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        value.as_slice().serialize(s)
    }
    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>, const N: usize>(
        d: D,
    ) -> Result<[T; N], D::Error> {
        Vec::<T>::deserialize(d)?.try_into().map_err(|v: Vec<T>| {
            serde::de::Error::custom(format!("expected {N} array members, got {}", v.len()))
        })
    }
}
pub(crate) mod register_map {
    use crate::llama::opcodes::RegName;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;
    pub fn serialize<S: Serializer>(map: &HashMap<RegName, u32>, s: S) -> Result<S::Ok, S::Error> {
        let mut pairs: Vec<_> = map.iter().collect();
        pairs.sort_by_key(|(key, _)| format!("{key:?}"));
        pairs.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<HashMap<RegName, u32>, D::Error> {
        let pairs = Vec::<(RegName, u32)>::deserialize(d)?;
        let mut map = HashMap::new();
        for (key, value) in pairs {
            if map.insert(key, value).is_some() {
                return Err(serde::de::Error::custom("duplicate register"));
            }
        }
        Ok(map)
    }
}
