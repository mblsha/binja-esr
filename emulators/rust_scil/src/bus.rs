use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Space {
    Int,
    Ext,
    Code,
}

pub trait Bus {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32;
    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32);
}

#[derive(Default)]
pub struct MemoryBus {
    int_mem: HashMap<u32, u8>,
    ext_mem: HashMap<u32, u8>,
}

impl MemoryBus {
    pub fn preload_int(&mut self, pairs: impl IntoIterator<Item = (u32, u8)>) {
        for (addr, value) in pairs {
            self.int_mem.insert(addr & 0xFF, value);
        }
    }

    pub fn preload_ext(&mut self, pairs: impl IntoIterator<Item = (u32, u8)>) {
        for (addr, value) in pairs {
            self.ext_mem.insert(addr & 0xFF_FFFF, value);
        }
    }

    pub fn dump_int(&self) -> Vec<(u32, u8)> {
        self.int_mem.iter().map(|(k, v)| (*k, *v)).collect()
    }

    pub fn dump_ext(&self) -> Vec<(u32, u8)> {
        self.ext_mem.iter().map(|(k, v)| (*k, *v)).collect()
    }

    fn select(&mut self, space: Space) -> &mut HashMap<u32, u8> {
        match space {
            Space::Int => &mut self.int_mem,
            Space::Ext | Space::Code => &mut self.ext_mem,
        }
    }
}

impl Bus for MemoryBus {
    fn load(&mut self, space: Space, addr: u32, bits: u8) -> u32 {
        assert!(bits % 8 == 0, "load width must be multiple of 8");
        let bytes = (bits / 8).max(1);
        let base = match space {
            Space::Int => addr & 0xFF,
            _ => addr & 0xFF_FFFF,
        };
        let map = self.select(space);
        let mut value = 0u32;
        for i in 0..bytes {
            let byte = *map.get(&(base + i as u32)).unwrap_or(&0);
            value |= (byte as u32) << (i * 8);
        }
        value
    }

    fn store(&mut self, space: Space, addr: u32, bits: u8, value: u32) {
        assert!(bits % 8 == 0, "store width must be multiple of 8");
        let bytes = (bits / 8).max(1);
        let base = match space {
            Space::Int => addr & 0xFF,
            _ => addr & 0xFF_FFFF,
        };
        let map = self.select(space);
        for i in 0..bytes {
            let byte = ((value >> (i * 8)) & 0xFF) as u8;
            map.insert(base + i as u32, byte);
        }
    }
}
