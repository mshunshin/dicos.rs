//! Helpers shared by the integration tests.

// Each test crate uses a different subset.
#![allow(dead_code)]

use std::path::Path;

/// A binary PGM (`P5`), PPM (`P6`) or PAM (`P7`) image.
pub struct Pnm {
    pub width: u32,
    pub height: u32,
    /// 1 for PGM, 3 for PPM, `DEPTH` for PAM.
    pub components: usize,
    pub maxval: u32,
    /// Pixel-interleaved samples (`RGBRGB…` for PPM), row-major.
    pub samples: Vec<u16>,
}

/// Read a binary PGM/PPM/PAM, 8-bit or 16-bit big-endian (maxval > 255).
pub fn read_pnm(path: &Path) -> Pnm {
    let d = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let components = match &d[..2] {
        b"P5" => 1,
        b"P6" => 3,
        b"P7" => return read_pam(path, &d),
        other => panic!("{}: not a binary PGM/PPM/PAM ({other:?})", path.display()),
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

/// Read the PAM (`P7`) header lines up to `ENDHDR`, then the samples.
fn read_pam(path: &Path, d: &[u8]) -> Pnm {
    let end = d
        .windows(7)
        .position(|w| w == b"ENDHDR\n")
        .unwrap_or_else(|| panic!("{}: PAM without ENDHDR", path.display()));
    let header = std::str::from_utf8(&d[..end]).unwrap();
    let field = |key: &str| -> u32 {
        header
            .lines()
            .find_map(|l| l.strip_prefix(key)?.trim().parse().ok())
            .unwrap_or_else(|| panic!("{}: PAM {key} missing", path.display()))
    };
    let (width, height, depth, maxval) = (
        field("WIDTH"),
        field("HEIGHT"),
        field("DEPTH"),
        field("MAXVAL"),
    );
    let body = &d[end + 7..];
    let samples: Vec<u16> = if maxval > 255 {
        body.chunks(2)
            .map(|c| u16::from_be_bytes([c[0], c[1]]))
            .collect()
    } else {
        body.iter().map(|&b| u16::from(b)).collect()
    };
    assert_eq!(
        samples.len(),
        (width * height * depth) as usize,
        "{}: sample count",
        path.display()
    );
    Pnm {
        width,
        height,
        components: depth as usize,
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
