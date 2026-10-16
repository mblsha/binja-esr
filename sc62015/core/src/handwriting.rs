// PY_SOURCE: pce500/handwriting.py
//! Portable Zinnia-compatible inference. Model data is supplied by the host.
//! Feature extraction adapted from Taku Kudo's BSD-licensed Zinnia (2008),
//! commit581faa8f6f15e4a7b21964be3a5ec36265c80e5b. See third_party/zinnia/COPYING.
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;

const HYBRID_MAGIC: &[u8; 8] = b"PVHW02\0\0";
const SHAPE_POINTS: usize = 32;

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
    hybrid: Option<Hybrid>,
    #[cfg(feature = "pvf1")]
    fingerprint: std::sync::OnceLock<String>,
}
#[derive(Clone)]
struct Template {
    label: String,
    strokes: usize,
    points: Vec<Point>,
}
#[derive(Clone)]
struct Hybrid {
    templates: Vec<Template>,
    latin_ratio: f32,
    extra_classes: usize,
}
impl Model {
    /// Stable identity of parsed inference inputs; no prediction/test data.
    #[cfg(feature = "pvf1")]
    pub fn state_fingerprint(&self) -> String {
        self.fingerprint
            .get_or_init(|| self.compute_state_fingerprint())
            .clone()
    }
    #[cfg(feature = "pvf1")]
    fn compute_state_fingerprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for class in &self.classes {
            h.update((class.label.len() as u64).to_le_bytes());
            h.update(class.label.as_bytes());
            h.update(class.bias.to_bits().to_le_bytes());
            h.update((class.weights.len() as u64).to_le_bytes());
            for &(index, weight) in &class.weights {
                h.update(index.to_le_bytes());
                h.update(weight.to_bits().to_le_bytes());
            }
        }
        if let Some(hybrid) = &self.hybrid {
            h.update([1]);
            h.update(hybrid.latin_ratio.to_bits().to_le_bytes());
            h.update((hybrid.extra_classes as u64).to_le_bytes());
            for template in &hybrid.templates {
                h.update((template.label.len() as u64).to_le_bytes());
                h.update(template.label.as_bytes());
                h.update((template.strokes as u64).to_le_bytes());
                h.update((template.points.len() as u64).to_le_bytes());
                for point in &template.points {
                    for v in point {
                        h.update(v.to_bits().to_le_bytes());
                    }
                }
            }
        } else {
            h.update([0]);
        }
        format!("{:x}", h.finalize())
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.starts_with(HYBRID_MAGIC) {
            return Self::hybrid_from_bytes(bytes);
        }
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
        Ok(Self {
            classes,
            hybrid: None,
            #[cfg(feature = "pvf1")]
            fingerprint: std::sync::OnceLock::new(),
        })
    }
    fn hybrid_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        fn word(bytes: &[u8], at: &mut usize) -> Result<u32, String> {
            let value = bytes.get(*at..*at + 4).ok_or("truncated hybrid model")?;
            *at += 4;
            Ok(u32::from_le_bytes(value.try_into().unwrap()))
        }
        let mut at = HYBRID_MAGIC.len();
        let base_len = word(bytes, &mut at)? as usize;
        let count = word(bytes, &mut at)? as usize;
        let points = word(bytes, &mut at)? as usize;
        let latin_ratio = f32::from_bits(word(bytes, &mut at)?);
        if count == 0
            || count > 65536
            || points != SHAPE_POINTS
            || !latin_ratio.is_finite()
            || latin_ratio <= 0.
            || latin_ratio >= 1.
        {
            return Err("invalid hybrid model configuration".into());
        }
        let base = bytes
            .get(at..at + base_len)
            .ok_or("truncated hybrid base model")?;
        if base.starts_with(HYBRID_MAGIC) {
            return Err("nested hybrid model".into());
        }
        let mut model = Self::from_bytes(base)?;
        at += base_len;
        let mut templates = Vec::with_capacity(count);
        let mut labels: BTreeSet<_> = model.classes.iter().map(|c| c.label.clone()).collect();
        let base_labels = labels.len();
        for _ in 0..count {
            let raw = bytes.get(at..at + 16).ok_or("truncated hybrid label")?;
            at += 16;
            let end = raw
                .iter()
                .position(|b| *b == 0)
                .ok_or("unterminated hybrid label")?;
            let label = std::str::from_utf8(&raw[..end])
                .map_err(|e| e.to_string())?
                .to_owned();
            if label.chars().count() != 1 {
                return Err("invalid hybrid label".into());
            }
            let strokes = word(bytes, &mut at)? as usize;
            if !(1..=64).contains(&strokes) {
                return Err("invalid hybrid stroke count".into());
            }
            let mut points = Vec::with_capacity(strokes * SHAPE_POINTS);
            for _ in 0..strokes * SHAPE_POINTS {
                let p = [
                    f32::from_bits(word(bytes, &mut at)?),
                    f32::from_bits(word(bytes, &mut at)?),
                ];
                if p.iter().any(|v| !v.is_finite() || v.abs() > 1.) {
                    return Err("invalid hybrid shape point".into());
                }
                points.push(p);
            }
            // Non-Latin templates select the original engine; they do not
            // introduce additional output classes into that engine.
            if label.len() == 1 && label.as_bytes()[0].is_ascii_alphanumeric() {
                labels.insert(label.clone());
            }
            templates.push(Template {
                label,
                strokes,
                points,
            });
        }
        if at != bytes.len() {
            return Err("trailing hybrid model bytes".into());
        }
        model.hybrid = Some(Hybrid {
            templates,
            latin_ratio,
            extra_classes: labels.len() - base_labels,
        });
        Ok(model)
    }
    pub fn class_count(&self) -> usize {
        self.classes.len() + self.hybrid.as_ref().map_or(0, |h| h.extra_classes)
    }
    pub fn classify(
        &self,
        strokes: &[Vec<Point>],
        width: f32,
        height: f32,
        limit: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        let features = features(strokes, width, height)?;
        if let Some(hybrid) = &self.hybrid {
            if let Some(scored) = hybrid.classify(strokes, width, height, limit)? {
                return Ok(scored);
            }
        }
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
impl Hybrid {
    fn classify(
        &self,
        strokes: &[Vec<Point>],
        width: f32,
        height: f32,
        limit: usize,
    ) -> Result<Option<Vec<(String, f32)>>, String> {
        let query = shape(strokes, width, height)?;
        let mut latin = Vec::new();
        let mut japanese = f64::INFINITY;
        for template in self.templates.iter().filter(|t| t.strokes == strokes.len()) {
            let distance = query
                .iter()
                .zip(&template.points)
                .map(|(a, b)| {
                    (a[0] as f64 - b[0] as f64).powi(2) + (a[1] as f64 - b[1] as f64).powi(2)
                })
                .sum::<f64>()
                / (2 * query.len()) as f64;
            if template.label.len() == 1 && template.label.as_bytes()[0].is_ascii_alphanumeric() {
                latin.push((template.label.clone(), distance));
            } else {
                japanese = japanese.min(distance);
            }
        }
        latin.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        if !latin
            .first()
            .is_some_and(|(_, d)| *d < self.latin_ratio as f64 * japanese)
        {
            return Ok(None);
        }
        let mut seen = BTreeSet::new();
        Ok(Some(
            latin
                .into_iter()
                .filter(|(label, _)| seen.insert(label.clone()))
                .take(limit)
                .map(|(label, d)| (label, -(d as f32)))
                .collect(),
        ))
    }
}
/// Centered, aspect-preserving geometry, resampled by arc length within each
/// ordered stroke. Geometry validation is shared with the linear extractor.
pub(crate) fn template_distance(
    strokes: &[Vec<Point>],
    canvas: [f32; 2],
    template: &[Vec<Point>],
    template_canvas: [f32; 2],
) -> Result<Option<f64>, String> {
    features(strokes, canvas[0], canvas[1])?;
    features(template, template_canvas[0], template_canvas[1])?;
    if strokes.len() != template.len() {
        return Ok(None);
    }
    let a = shape(strokes, canvas[0], canvas[1])?;
    let b = shape(template, template_canvas[0], template_canvas[1])?;
    Ok(Some(
        a.iter()
            .zip(&b)
            .map(|(a, b)| (a[0] as f64 - b[0] as f64).powi(2) + (a[1] as f64 - b[1] as f64).powi(2))
            .sum::<f64>()
            / (2 * a.len()) as f64,
    ))
}
fn shape(strokes: &[Vec<Point>], width: f32, height: f32) -> Result<Vec<Point>, String> {
    let normalized: Vec<Vec<[f64; 2]>> = strokes
        .iter()
        .map(|stroke| {
            stroke
                .iter()
                .map(|p| [p[0] as f64 / width as f64, p[1] as f64 / height as f64])
                .collect()
        })
        .collect();
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for p in normalized.iter().flatten() {
        for j in 0..2 {
            low[j] = low[j].min(p[j]);
            high[j] = high[j].max(p[j]);
        }
    }
    let span = (high[0] - low[0]).max(high[1] - low[1]).max(1e-6);
    let center = [(low[0] + high[0]) / 2., (low[1] + high[1]) / 2.];
    let mut output = Vec::with_capacity(strokes.len() * SHAPE_POINTS);
    for stroke in normalized {
        let mut p = Vec::new();
        let mut distances = Vec::new();
        let mut length = 0.;
        for point in stroke {
            let point = [(point[0] - center[0]) / span, (point[1] - center[1]) / span];
            if let Some(last) = p.last() {
                let last: &[f64; 2] = last;
                let step = (point[0] - last[0]).hypot(point[1] - last[1]);
                if step == 0. {
                    continue;
                }
                length += step;
            }
            p.push(point);
            distances.push(length);
        }
        let mut segment = 0;
        for i in 0..SHAPE_POINTS {
            let target = length * i as f64 / (SHAPE_POINTS - 1) as f64;
            while segment + 1 < distances.len() && distances[segment + 1] < target {
                segment += 1;
            }
            let point = if segment + 1 == p.len() {
                p[segment]
            } else {
                let fraction =
                    (target - distances[segment]) / (distances[segment + 1] - distances[segment]);
                [
                    p[segment][0] + (p[segment + 1][0] - p[segment][0]) * fraction,
                    p[segment][1] + (p[segment + 1][1] - p[segment][1]) * fraction,
                ]
            };
            output.push([point[0] as f32, point[1] as f32]);
        }
    }
    if output.iter().flatten().any(|v| !v.is_finite()) {
        return Err("overflowed handwriting shape".into());
    }
    Ok(output)
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
    fn constant_model() -> Vec<u8> {
        let mut bytes = vec![0; 12];
        let mut label = [0; 16];
        label[.."い".len()].copy_from_slice("い".as_bytes());
        bytes.extend(label);
        bytes.extend(2.5f32.to_le_bytes());
        bytes.extend(u32::MAX.to_le_bytes());
        bytes.extend(0f32.to_le_bytes());
        let size = bytes.len() as u32;
        bytes[..4].copy_from_slice(&(size ^ 0x0ef71821).to_le_bytes());
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes
    }
    fn hybrid_fixture(templates: &[(&str, Vec<Point>)]) -> Vec<u8> {
        let base = constant_model();
        let mut bytes = HYBRID_MAGIC.to_vec();
        for word in [
            base.len() as u32,
            templates.len() as u32,
            32,
            0.75f32.to_bits(),
        ] {
            bytes.extend(word.to_le_bytes());
        }
        bytes.extend(base);
        for (name, points) in templates {
            let mut label = [0; 16];
            label[..name.len()].copy_from_slice(name.as_bytes());
            bytes.extend(label);
            bytes.extend(1u32.to_le_bytes());
            for p in points {
                for value in p {
                    bytes.extend(value.to_le_bytes());
                }
            }
        }
        bytes
    }
    fn line(horizontal: bool) -> Vec<Point> {
        (0..32)
            .map(|i| {
                let value = i as f32 / 31. - 0.5;
                if horizontal {
                    [value, 0.]
                } else {
                    [0., value]
                }
            })
            .collect()
    }
    #[test]
    #[cfg(feature = "pvf1")]
    fn live_identity_cache_preserves_exact_parsed_model_fingerprint() {
        let base = Model::from_bytes(&constant_model()).unwrap();
        let hybrid =
            Model::from_bytes(&hybrid_fixture(&[("A", line(true)), ("い", line(false))])).unwrap();
        for model in [&base, &hybrid] {
            let expected = model.compute_state_fingerprint();
            assert_eq!(model.state_fingerprint(), expected);
            assert_eq!(model.state_fingerprint(), expected);
            assert_eq!(model.clone().state_fingerprint(), expected);
        }
        assert_ne!(base.state_fingerprint(), hybrid.state_fingerprint());
    }
    #[test]
    fn hybrid_latin_clear_match_deduplicates_and_japanese_fallback_is_unchanged() {
        let model = Model::from_bytes(&hybrid_fixture(&[
            ("A", line(true)),
            ("A", line(true)),
            ("い", line(false)),
        ]))
        .unwrap();
        assert_eq!(model.class_count(), 2);
        let routing_only =
            Model::from_bytes(&hybrid_fixture(&[("A", line(true)), ("字", line(false))])).unwrap();
        assert_eq!(routing_only.class_count(), 2);
        let result = model
            .classify(&[vec![[10., 20.], [30., 20.]]], 100., 100., 10)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "A");
        assert!(result[0].1.abs() < 1e-12);
        let base = Model::from_bytes(&constant_model()).unwrap();
        for input in [
            vec![vec![[20., 10.], [20., 30.]]],
            vec![vec![[0., 0.]], vec![[1., 1.]]],
        ] {
            assert_eq!(
                model.classify(&input, 100., 100., 10),
                base.classify(&input, 100., 100., 10)
            );
        }
        let tie =
            Model::from_bytes(&hybrid_fixture(&[("A", line(true)), ("い", line(true))])).unwrap();
        assert_eq!(
            tie.classify(&[vec![[0., 0.], [100., 0.]]], 100., 100., 10)
                .unwrap(),
            vec![("い".into(), 2.5)]
        );
        assert!(model
            .classify(&[vec![[f32::NAN, 0.]]], 100., 100., 10)
            .is_err());
    }
    #[test]
    fn hybrid_shape_preserves_aspect_and_arc_length_with_duplicate_points() {
        let result = shape(&[vec![[10., 20.], [10., 20.], [30., 20.]]], 100., 100.).unwrap();
        for (a, b) in result.iter().zip(line(true)) {
            assert!((a[0] - b[0]).abs() < 1e-7);
            assert_eq!(a[1], b[1]);
        }
        assert_eq!(
            shape(&[vec![[10., 20.]]], 100., 100.).unwrap(),
            vec![[0., 0.]; 32]
        );
    }
    #[test]
    fn hybrid_rejects_bad_configuration_truncated_points_and_trailing_data() {
        let bytes = hybrid_fixture(&[("A", line(true)), ("い", line(false))]);
        for length in [8, 23, 24, 30, bytes.len() - 1] {
            assert!(Model::from_bytes(&bytes[..length]).is_err());
        }
        for (offset, word) in [
            (12, 0u32),
            (16, 31u32),
            (20, f32::NAN.to_bits()),
            (20, 1f32.to_bits()),
            (24 + constant_model().len() + 16, 65),
        ] {
            let mut bad = bytes.clone();
            bad[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
            assert!(Model::from_bytes(&bad).is_err());
        }
        let mut bad = bytes.clone();
        let offset = 24 + constant_model().len() + 20;
        bad[offset..offset + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(Model::from_bytes(&bad).is_err());
        let mut bad = bytes;
        bad.push(0);
        assert!(Model::from_bytes(&bad).is_err());
    }
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
