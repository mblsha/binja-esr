// PY_SOURCE: pce500/tablet_geometry.py
//! Shared quarter-coordinate/ADC arithmetic. Layout and live coefficients
//! belong to the model; this does not manufacture guest pen events.
#[derive(Clone, Copy, Debug)]
pub struct TabletCalibration(pub [i16; 6]);
impl TabletCalibration {
    pub fn from_quarter(&self, point: [i16; 2]) -> Result<[u16; 2], String> {
        let mut result = [0; 2];
        for axis in 0..2 {
            let [denominator, numerator, offset] =
                [self.0[axis * 3], self.0[axis * 3 + 1], self.0[axis * 3 + 2]];
            if denominator <= 0 || numerator <= 0 {
                return Err("tablet calibration is unavailable or unsupported".into());
            }
            let value = (i32::from(point[axis]) - i32::from(offset)) * i32::from(numerator)
                / i32::from(denominator);
            if !(0..=1023).contains(&value) {
                return Err("calibrated contact outside ten-bit ADC range".into());
            }
            result[axis] = value as u16;
        }
        Ok(result)
    }
}
