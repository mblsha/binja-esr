// PY_SOURCE: pce500/ram_media.py
//! Passive RAM media backing. Device profiles own capacity and bus mapping.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RamMedia {
    bytes: Vec<u8>,
    writable: bool,
}
impl RamMedia {
    pub fn from_image(image: &[u8], capacity: usize, writable: bool) -> Result<Self, String> {
        if capacity == 0 || image.len() != capacity {
            return Err(format!(
                "RAM medium requires an exact {capacity}-byte image"
            ));
        }
        Ok(Self {
            bytes: image.to_vec(),
            writable,
        })
    }
    pub fn read(&self, offset: usize) -> Option<u8> {
        self.bytes.get(offset).copied()
    }
    pub fn write(&mut self, offset: usize, value: u8) -> bool {
        let Some(byte) = self.bytes.get_mut(offset) else {
            return false;
        };
        if self.writable {
            *byte = value;
        }
        true
    }
    pub fn image(&self) -> &[u8] {
        &self.bytes
    }
    pub fn writable(&self) -> bool {
        self.writable
    }
}
