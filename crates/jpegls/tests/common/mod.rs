//! Helpers shared by the integration tests.

// Each test crate uses a different subset.
#![allow(dead_code)]

use std::path::Path;

/// A binary PGM (`P5`) or PPM (`P6`) image.
pub struct Pnm {
    pub width: u32,
    pub height: u32,
    /// 1 for PGM, 3 for PPM.
    pub components: usize,
    pub maxval: u32,
    /// Pixel-interleaved samples (`RGBRGB…` for PPM), row-major.
    pub samples: Vec<u16>,
}

/// Read a binary PGM/PPM, 8-bit or 16-bit big-endian (maxval > 255).
pub fn read_pnm(path: &Path) -> Pnm {
    let d = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let components = match &d[..2] {
        b"P5" => 1,
        b"P6" => 3,
        other => panic!("{}: not a binary PGM/PPM ({other:?})", path.display()),
    };
    let mut i = 2usize;
    let mut toks = Vec::new();
    while toks.len() < 3 {
        while d[i].is_ascii_whitespace() {
            i += 1;
        }
        if d[i] == b'#' {
            while d[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while !d[i].is_ascii_whitespace() {
            i += 1;
        }
        toks.push(
            std::str::from_utf8(&d[s..i])
                .unwrap()
                .parse::<u32>()
                .unwrap(),
        );
    }
    i += 1;
    let (width, height, maxval) = (toks[0], toks[1], toks[2]);
    let samples: Vec<u16> = if maxval > 255 {
        d[i..]
            .chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect()
    } else {
        d[i..].iter().map(|&b| u16::from(b)).collect()
    };
    assert_eq!(
        samples.len(),
        width as usize * height as usize * components,
        "{}: sample count",
        path.display()
    );
    Pnm {
        width,
        height,
        components,
        maxval,
        samples,
    }
}

/// Convert component-planar samples (as `jpegls::decode_image` returns them)
/// to pixel-interleaved order.
pub fn interleave(planar: &[u16], components: usize) -> Vec<u16> {
    let plane = planar.len() / components;
    (0..plane)
        .flat_map(|i| (0..components).map(move |c| planar[c * plane + i]))
        .collect()
}
