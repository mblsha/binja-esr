// PY_SOURCE: pce500/display/lcd_visualization.py
//! Presentation only: crisp ROM pixels plus scalable IQ-7000 glass segments.
//! Shapes are manually drawn from the supplied reference, not a device font.

use crate::iq7000_annunciators::Iq7000Annunciators;
use tiny_skia::{Color, FillRule, Paint, Path, PathBuilder, Pixmap, Rect, Stroke, Transform};

pub const LCD_BACKGROUND: u8 = 192;
pub const LCD_FOREGROUND: u8 = 0;
pub const IQ_PANEL_WIDTH: usize = 26; // 2-column gap + 24-column glass area.
pub const IQ_DISPLAY_SCALE: usize = 4;

/// Rasterize once at output resolution. Do not enlarge a 3x5 segment bitmap.
/// Output is grayscale, not the legacy logical 0/1 framebuffer.
pub fn render_lcd(
    matrix: &[Vec<u8>],
    annunciators: Option<&Iq7000Annunciators>,
    scale: usize,
) -> Result<Vec<Vec<u8>>, &'static str> {
    let width = matrix.first().map_or(0, Vec::len);
    if width == 0
        || matrix.is_empty()
        || !(1..=16).contains(&scale)
        || matrix.iter().any(|row| row.len() != width)
    {
        return Err("LCD render requires a rectangular matrix and scale 1..16");
    }
    let full_width = (width + annunciators.map_or(0, |_| IQ_PANEL_WIDTH)) * scale;
    let mut image = vec![vec![LCD_BACKGROUND; full_width]; matrix.len() * scale];
    for (y, row) in matrix.iter().enumerate() {
        for (x, pixel) in row.iter().enumerate() {
            let shade = if *pixel == 0 {
                LCD_BACKGROUND
            } else {
                LCD_FOREGROUND
            };
            for dy in 0..scale {
                image[y * scale + dy][x * scale..(x + 1) * scale].fill(shade);
            }
        }
    }
    if let Some(ann) = annunciators {
        let panel = render_panel(ann, scale);
        for (y, row) in image.iter_mut().enumerate().take(64 * scale) {
            for (x, dest) in row[width * scale..].iter_mut().enumerate() {
                *dest = panel.pixels()[y * IQ_PANEL_WIDTH * scale + x].red();
            }
        }
    }
    Ok(image)
}

struct Glass {
    pixmap: Pixmap,
    transform: Transform,
}

impl Glass {
    fn path(&mut self, path: Path, x: f32, y: f32, stroke: Option<f32>, inverse: bool) {
        let mut paint = Paint::default();
        let c = if inverse {
            LCD_BACKGROUND
        } else {
            LCD_FOREGROUND
        };
        paint.set_color_rgba8(c, c, c, 255);
        let transform = self.transform.pre_translate(x + 2.0, y);
        if let Some(width) = stroke {
            self.pixmap.stroke_path(
                &path,
                &paint,
                &Stroke {
                    width,
                    line_join: tiny_skia::LineJoin::Round,
                    ..Stroke::default()
                },
                transform,
                None,
            );
        } else {
            self.pixmap
                .fill_path(&path, &paint, FillRule::Winding, transform, None);
        }
    }

    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, stroke: Option<f32>) {
        self.path(
            PathBuilder::from_rect(Rect::from_xywh(0., 0., w, h).unwrap()),
            x,
            y,
            stroke,
            false,
        );
    }

    fn word(&mut self, text: &str, x: f32, y: f32, inverse: bool, stroke: f32) {
        let mut cursor = x;
        for ch in text.chars() {
            self.path(letter(ch), cursor, y, Some(stroke), inverse);
            cursor += if ch == 'I' { 1.8 } else { 4.0 };
        }
    }
}

// A small vector alphabet: curved C/S/B/P/R/D and continuous diagonals.
// These paths are artwork, not ROM glyphs, and require no host-installed font.
fn letter(ch: char) -> Path {
    let mut p = PathBuilder::new();
    match ch {
        'A' => {
            p.move_to(0., 5.);
            p.line_to(1.5, 0.);
            p.line_to(3., 5.);
            p.move_to(0.5, 3.3);
            p.line_to(2.5, 3.3);
        }
        'B' => {
            p.move_to(0., 5.);
            p.line_to(0., 0.);
            p.line_to(1.5, 0.);
            p.cubic_to(3.5, 0., 3.5, 2.4, 1.5, 2.4);
            p.line_to(0., 2.4);
            p.move_to(1.5, 2.4);
            p.cubic_to(3.8, 2.4, 3.8, 5., 1.5, 5.);
            p.line_to(0., 5.);
        }
        'C' => {
            p.move_to(3., 0.6);
            p.cubic_to(2.6, 0.05, 2.1, 0., 1.5, 0.);
            p.cubic_to(0.55, 0., 0., 0.95, 0., 2.5);
            p.cubic_to(0., 4.05, 0.55, 5., 1.5, 5.);
            p.cubic_to(2.1, 5., 2.6, 4.95, 3., 4.4);
        }
        'D' => {
            p.move_to(0., 5.);
            p.line_to(0., 0.);
            p.line_to(1., 0.);
            p.cubic_to(4., 0., 4., 5., 1., 5.);
            p.close();
        }
        'E' | 'F' => {
            p.move_to(3., 0.);
            p.line_to(0., 0.);
            p.line_to(0., 5.);
            if ch == 'E' {
                p.line_to(3., 5.);
            }
            p.move_to(0., 2.4);
            p.line_to(2.7, 2.4);
        }
        'H' => {
            p.move_to(0., 0.);
            p.line_to(0., 5.);
            p.move_to(3., 0.);
            p.line_to(3., 5.);
            p.move_to(0., 2.4);
            p.line_to(3., 2.4);
        }
        'I' => {
            p.move_to(0.5, 0.);
            p.line_to(0.5, 5.);
        }
        'P' | 'R' => {
            p.move_to(0., 5.);
            p.line_to(0., 0.);
            p.line_to(1.5, 0.);
            p.cubic_to(3.6, 0., 3.6, 2.5, 1.5, 2.5);
            p.line_to(0., 2.5);
            if ch == 'R' {
                p.move_to(1.4, 2.5);
                p.line_to(3.1, 5.);
            }
        }
        'S' => {
            p.move_to(3., 0.6);
            p.cubic_to(0., -1.5, -1.5, 2.1, 1.4, 2.5);
            p.cubic_to(4.8, 3., 3.2, 6.4, 0., 4.4);
        }
        'T' => {
            p.move_to(0., 0.);
            p.line_to(3.4, 0.);
            p.move_to(1.7, 0.);
            p.line_to(1.7, 5.);
        }
        _ => unreachable!("fixed glass alphabet"),
    }
    p.finish().unwrap()
}

fn polygon(points: &[(f32, f32)]) -> Path {
    let mut p = PathBuilder::new();
    p.move_to(points[0].0, points[0].1);
    for &(x, y) in &points[1..] {
        p.line_to(x, y);
    }
    p.close();
    p.finish().unwrap()
}

fn render_panel(ann: &Iq7000Annunciators, scale: usize) -> Pixmap {
    let mut g = Glass {
        pixmap: Pixmap::new((IQ_PANEL_WIDTH * scale) as u32, (64 * scale) as u32).unwrap(),
        transform: Transform::from_scale(scale as f32, scale as f32),
    };
    g.pixmap.fill(Color::from_rgba8(
        LCD_BACKGROUND,
        LCD_BACKGROUND,
        LCD_BACKGROUND,
        255,
    ));
    // Inactive segments are invisible: outlines are not permanently printed ink.
    if ann.batt {
        g.rect(1.4, 0.5, 17., 6.2, None);
        g.word("BATT", 2.2, 1.1, true, 0.36);
    }
    for (on, text, y) in [
        (ann.card, "CARD", 8.5),
        (ann.edit, "EDIT", 15.5),
        (ann.shift, "SHIFT", 22.5),
        (ann.caps, "CAPS", 29.5),
    ] {
        if on {
            g.word(text, 2., y, false, 0.42);
        }
    }
    if ann.secret_data {
        let mut p = PathBuilder::new();
        p.move_to(2.2, 0.);
        p.line_to(2.2, 5.);
        p.move_to(0., 1.2);
        p.line_to(4.4, 3.8);
        p.move_to(0., 3.8);
        p.line_to(4.4, 1.2);
        g.path(p.finish().unwrap(), 2., 37., Some(0.65), false);
    }
    if ann.secret_mode {
        g.rect(7.1, 36.5, 5.6, 6.6, Some(0.35));
        g.word("S", 8.4, 37.3, false, 0.60);
    }
    if ann.key_beep {
        let mut p = PathBuilder::new();
        p.move_to(1.8, 0.);
        p.line_to(1.8, 3.5);
        p.cubic_to(-0.4, 2.8, -0.7, 5.4, 1.1, 4.4);
        p.cubic_to(2., 4., 2.4, 3.7, 2.4, 3.);
        p.line_to(2.4, 1.2);
        p.cubic_to(3.5, 1.3, 3.5, 2.2, 3.2, 2.6);
        p.cubic_to(4.5, 1.4, 2.9, 0.9, 2.4, 0.);
        p.close();
        g.path(p.finish().unwrap(), 2.4, 45., None, false);
    }
    if ann.alarm {
        let mut p = PathBuilder::new();
        p.move_to(0., 4.);
        p.cubic_to(1.1, 3.3, 0.2, 0., 2.3, 0.);
        p.cubic_to(4.4, 0., 3.5, 3.3, 4.6, 4.);
        p.close();
        p.move_to(1.6, 4.4);
        p.quad_to(2.3, 5.4, 3., 4.4);
        let p = p
            .finish()
            .unwrap()
            .transform(Transform::from_row(0.96, 0.22, -0.22, 0.96, 0., 0.))
            .unwrap();
        g.path(p, 8.4, 44.6, Some(0.45), false);
    }
    let up = [
        (1., 5.),
        (1., 2.),
        (0., 2.),
        (2., 0.),
        (4., 2.),
        (3., 2.),
        (3., 5.),
    ];
    if ann.more_up {
        g.path(polygon(&up), 2., 51., None, false);
    }
    if ann.more_down {
        let points: Vec<_> = up.iter().map(|&(x, y)| (x, 5. - y)).collect();
        g.path(polygon(&points), 8.1, 51., None, false);
    }
    let left = [
        (0., 1.7),
        (2.2, 0.),
        (2.2, 0.7),
        (4.8, 0.7),
        (4.8, 2.7),
        (2.2, 2.7),
        (2.2, 3.4),
    ];
    if ann.more_left {
        g.path(polygon(&left), 1.5, 59., None, false);
    }
    if ann.more_right {
        let points: Vec<_> = left.iter().map(|&(x, y)| (4.8 - x, y)).collect();
        g.path(polygon(&points), 7.5, 59., None, false);
    }
    g.pixmap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_thirteen_vectors_are_independent_and_confined_to_their_rows() {
        let cases = [
            (0, 0x80, 0..7),
            (0, 0x40, 8..14),
            (0, 0x20, 15..21),
            (0, 0x10, 22..28),
            (0, 0x08, 29..35),
            (0, 0x04, 36..43),
            (1, 0x04, 36..44),
            (0, 0x02, 44..50),
            (1, 0x02, 44..51),
            (0, 0x01, 51..57),
            (1, 0x01, 51..57),
            (2, 0x80, 59..63),
            (3, 0x80, 59..63),
        ];
        for (byte, bit, rows) in cases {
            let mut shadows = [0; 4];
            shadows[byte] = bit;
            let ann = Iq7000Annunciators::from_sources([0; 4], shadows);
            let panel = render_panel(&ann, 4);
            let mut ink = 0;
            for (index, pixel) in panel.pixels().iter().enumerate() {
                if pixel.red() != LCD_BACKGROUND {
                    let y = index / (26 * 4);
                    assert!(
                        rows.contains(&(y / 4)),
                        "byte {byte} bit {bit:x}, row {}",
                        y as f32 / 4.
                    );
                    ink += 1;
                }
            }
            assert!(ink > 20, "byte {byte}, bit {bit:x} did not draw");
        }
    }

    #[test]
    fn curves_have_subpixel_edges_and_are_not_scaled_bitmap_blocks() {
        let ann = Iq7000Annunciators::from_sources([0; 4], [0x08, 0, 0, 0]);
        let panel = render_panel(&ann, 4);
        assert!(panel.pixels().iter().any(|p| p.red() > 0 && p.red() < 192));
        // At least one logical 4x4 cell has mixed shades: real output-resolution paths.
        assert!((0..64).any(|y| (0..26).any(|x| {
            let first = panel.pixels()[y * 4 * 104 + x * 4].red();
            (0..4).any(|dy| {
                (0..4).any(|dx| panel.pixels()[(y * 4 + dy) * 104 + x * 4 + dx].red() != first)
            })
        })));
    }

    #[test]
    fn inactive_unknown_and_workspace_bits_do_not_draw_ghosts() {
        let ann = Iq7000Annunciators::from_sources([0xff; 4], [0, 0x80, 1, 1]);
        assert!(render_panel(&ann, 4)
            .pixels()
            .iter()
            .all(|p| p.red() == 192));
    }

    #[test]
    fn zero_scale_and_ragged_frames_are_rejected() {
        assert!(render_lcd(&[vec![0]], None, 0).is_err());
        assert!(render_lcd(&[vec![0], vec![]], None, 4).is_err());
    }
}
