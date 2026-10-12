// PY_SOURCE: pce500/handwriting.py
//! Portable Zinnia-compatible inference. Model data is supplied by the host.
//! Feature extraction adapted from Taku Kudo's BSD-licensed Zinnia (2008),
//! commit581faa8f6f15e4a7b21964be3a5ec36265c80e5b. See third_party/zinnia/COPYING.
#[cfg(test)]
use std::collections::BTreeMap;

pub type Point = [f32; 2];
pub type Strokes = Vec<Vec<Point>>;
#[derive(Clone)]
struct Class {
    label: String,
    bias: f32,
    weights: Vec<(u32, f32)>,
}
#[derive(Clone)]
pub struct Model {
    classes: Vec<Class>,
}
impl Model {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut at = 0;
        fn word(b: &[u8], at: &mut usize) -> Result<u32, String> {
            let p = b.get(*at..*at + 4).ok_or("truncated handwriting model")?;
            *at += 4;
            Ok(u32::from_le_bytes(p.try_into().unwrap()))
        }
        if word(bytes, &mut at)? ^ 0x0ef71821 != bytes.len() as u32 || word(bytes, &mut at)? != 1 {
            return Err("handwriting model identity/version mismatch".into());
        }
        let count = word(bytes, &mut at)? as usize;
        if count == 0 || count > 65536 {
            return Err("invalid handwriting class count".into());
        }
        let mut classes = Vec::with_capacity(count);
        for _ in 0..count {
            let label = bytes.get(at..at + 16).ok_or("truncated model label")?;
            at += 16;
            let end = label
                .iter()
                .position(|v| *v == 0)
                .ok_or("unterminated model label")?;
            let label = std::str::from_utf8(&label[..end])
                .map_err(|e| e.to_string())?
                .to_owned();
            if label.is_empty() {
                return Err("empty model label".into());
            }
            let bias = f32::from_bits(word(bytes, &mut at)?);
            if !bias.is_finite() {
                return Err("nonfinite model bias".into());
            }
            let mut weights = Vec::new();
            let mut previous = None;
            loop {
                let index = word(bytes, &mut at)?;
                let value = f32::from_bits(word(bytes, &mut at)?);
                if index == u32::MAX {
                    break;
                }
                if index > 2100000 || !value.is_finite() || previous.is_some_and(|p| index <= p) {
                    return Err("invalid model feature".into());
                }
                previous = Some(index);
                weights.push((index, value));
            }
            classes.push(Class {
                label,
                bias,
                weights,
            });
        }
        if at != bytes.len() {
            return Err("trailing handwriting model bytes".into());
        }
        Ok(Self { classes })
    }
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }
    pub fn classify(
        &self,
        strokes: &[Vec<Point>],
        width: f32,
        height: f32,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        let features = features(strokes, width, height)?;
        let mut scored = Vec::with_capacity(self.classes.len());
        for c in &self.classes {
            let mut a = 0;
            let mut b = 0;
            let mut score = 0f64;
            while a < c.weights.len() && b < features.len() {
                match c.weights[a].0.cmp(&features[b].0) {
                    std::cmp::Ordering::Equal => {
                        score += (c.weights[a].1 * features[b].1) as f64;
                        a += 1;
                        b += 1;
                    }
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                }
            }
            scored.push((c.label.clone(), (c.bias as f64 + score) as f32));
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        scored.truncate(limit);
        Ok(scored)
    }
}
fn basic(out: &mut Vec<(u32, f32)>, offset: u32, a: Point, b: Point) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    for (i, v) in [
        10. * (dx * dx + dy * dy).sqrt(),
        dy.atan2(dx),
        10. * (a[0] - 0.5),
        10. * (a[1] - 0.5),
        10. * (b[0] - 0.5),
        10. * (b[1] - 0.5),
        (a[1] - 0.5).atan2(a[0] - 0.5),
        (b[1] - 0.5).atan2(b[0] - 0.5),
        10. * ((a[0] - 0.5).powi(2) + (a[1] - 0.5).powi(2)).sqrt(),
        10. * ((b[0] - 0.5).powi(2) + (b[1] - 0.5).powi(2)).sqrt(),
        5. * dx,
        5. * dy,
    ]
    .into_iter()
    .enumerate()
    {
        out.push((offset + i as u32 + 1, v));
    }
}
fn vertex(
    out: &mut Vec<(u32, f32)>,
    points: &[Point],
    first: usize,
    last: usize,
    id: u32,
    sid: u32,
) {
    if id > 50 {
        return;
    }
    basic(out, sid * 1000 + id * 20, points[first], points[last]);
    if first == last {
        return;
    }
    let a = points[last][0] - points[first][0];
    let b = points[last][1] - points[first][1];
    let c = ((points[last][1] * points[first][0]) as f64
        - (points[last][0] * points[first][1]) as f64) as f32;
    let mut distance = -1f32;
    let mut best = first;
    for (i, p) in points.iter().enumerate().take(last).skip(first) {
        let d = (a * p[1] - b * p[0] + c).abs();
        if d > distance {
            distance = d;
            best = i;
        }
    }
    if distance * distance / (a * a + b * b) > 0.001 && best > first && best < last {
        vertex(out, points, first, best, 2 * id + 1, sid);
        vertex(out, points, best, last, 2 * id + 2, sid);
    }
}
pub fn features(
    strokes: &[Vec<Point>],
    width: f32,
    height: f32,
) -> Result<Vec<(u32, f32)>, String> {
    if !width.is_finite()
        || !height.is_finite()
        || width <= 0.
        || height <= 0.
        || strokes.is_empty()
        || strokes.len() > 64
        || strokes
            .iter()
            .any(|s| s.is_empty() || s.len() > 16384 || s.iter().flatten().any(|v| !v.is_finite()))
    {
        return Err("invalid handwriting geometry/strokes".into());
    }
    let mut out = vec![(0, 1.)];
    let mut previous = None;
    for (sid, stroke) in strokes.iter().enumerate() {
        let p: Vec<Point> = stroke
            .iter()
            .map(|v| [v[0] / width, v[1] / height])
            .collect();
        vertex(&mut out, &p, 0, p.len() - 1, 0, sid as u32);
        if let Some(last) = previous {
            basic(&mut out, 100000 + sid as u32 * 1000, last, p[0]);
        }
        previous = p.last().copied();
    }
    out.push((2000000, strokes.len() as f32));
    out.push((2000000 + strokes.len() as u32, 10.));
    if out.iter().any(|v| !v.1.is_finite()) {
        return Err("overflowed handwriting features".into());
    }
    out.sort_by_key(|v| v.0);
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_layout_and_sparse_scores_follow_independent_linear_expectations() {
        let mut b = vec![0; 12];
        for (label, bias, weights) in [("A", -1f32, vec![(3u32, 2f32)]), ("B", 0f32, vec![])] {
            let mut name = [0; 16];
            name[..label.len()].copy_from_slice(label.as_bytes());
            b.extend(name);
            b.extend(bias.to_le_bytes());
            for (i, v) in weights {
                b.extend(i.to_le_bytes());
                b.extend(v.to_le_bytes());
            }
            b.extend(u32::MAX.to_le_bytes());
            b.extend(0f32.to_le_bytes());
        }
        let size = b.len() as u32;
        b[..4].copy_from_slice(&(size ^ 0x0ef71821).to_le_bytes());
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        let m = Model::from_bytes(&b).unwrap();
        assert_eq!(m.class_count(), 2);
        let high = m
            .classify(&[vec![[90., 20.], [90., 80.]]], 100., 100., 2)
            .unwrap();
        assert_eq!(high[0].0, "A");
        assert!((high[0].1 - 7.).abs() < 1e-6);
        assert_eq!(
            m.classify(&[vec![[10., 20.], [10., 80.]]], 100., 100., 1)
                .unwrap()[0]
                .0,
            "B"
        );
        assert!(Model::from_bytes(&b[..b.len() - 1]).is_err());
        b.push(0);
        assert!(Model::from_bytes(&b).is_err());
    }
    #[test]
    fn straight_strokes_keep_position_and_count_features() {
        let f = features(
            &[vec![[10., 20.], [90., 20.]], vec![[50., 30.], [50., 90.]]],
            100.,
            100.,
        )
        .unwrap();
        let f: BTreeMap<_, _> = f.into_iter().collect();
        assert_eq!(f[&0], 1.);
        assert!((f[&1] - 8.).abs() < 1e-6);
        assert_eq!(f[&2000000], 2.);
        assert_eq!(f[&2000002], 10.);
        assert!(f.contains_key(&101001));
        assert!(features(&[], 100., 100.).is_err());
        assert!(features(&[vec![[f32::NAN, 0.]]], 100., 100.).is_err());
    }
}
