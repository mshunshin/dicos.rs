//! Decoder throughput, for comparing changes to the scan decoder.
//!
//! ```text
//! cargo run --release -p pure_jpegls --example bench_decode [DIR...]
//! ```
//!
//! Times `decode_image` over every `.jls` file in each `DIR` (by default the
//! fixture sets under `tests/fixtures`) and over two synthesized images that
//! exercise regular mode heavily (16-bit noise, 8-bit RGB gradient+noise).
//! Each set is decoded repeatedly; the best of `ROUNDS` (env, default 15)
//! runs is reported as decoded megasamples per second.
//!
//! Timings on a laptop move by a few percent with unrelated code layout
//! changes; compare builds by alternating them within one process when the
//! difference is small.

use std::path::{Path, PathBuf};
use std::time::Instant;

use jpegls::EncodeOptions;

fn main() {
    let rounds: usize = std::env::var("ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(15);
    let mut dirs: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if dirs.is_empty() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        dirs = ["charls", "charls-gen", "dcmtk"]
            .iter()
            .map(|d| fixtures.join(d))
            .collect();
    }

    for dir in &dirs {
        let mut streams: Vec<Vec<u8>> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "jls"))
            .map(|p| std::fs::read(p).unwrap())
            .filter(|s| jpegls::decode_image(s).is_ok())
            .collect();
        streams.sort();
        report(&dir.display().to_string(), &streams, rounds);
    }

    report(
        "synthetic 16-bit noise 512x512",
        &[noise_16bit(512, 512)],
        rounds,
    );
    report("synthetic 8-bit RGB 512x512", &[rgb_8bit(512, 512)], rounds);
}

/// Decode `streams` `rounds` times and print the best throughput.
fn report(name: &str, streams: &[Vec<u8>], rounds: usize) {
    let samples: usize = streams
        .iter()
        .map(|s| jpegls::decode_image(s).unwrap().samples.len())
        .sum();
    let best = (0..rounds)
        .map(|_| {
            let start = Instant::now();
            for s in streams {
                std::hint::black_box(jpegls::decode_image(s).unwrap());
            }
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min);
    println!(
        "{name}: {} streams, {:.1} Msamples/s ({:.2} ms)",
        streams.len(),
        samples as f64 / best / 1e6,
        best * 1e3
    );
}

/// A cheap deterministic generator (xorshift), so runs are comparable.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn noise_16bit(w: u32, h: u32) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let pixels: Vec<u16> = (0..w * h)
        .map(|i| ((i % w) * 40 + (xorshift(&mut state) % 4096) as u32) as u16)
        .collect();
    let mut out = Vec::new();
    jpegls::encode(&pixels, w, h, &mut out).unwrap();
    out
}

fn rgb_8bit(w: u32, h: u32) -> Vec<u8> {
    let mut state = 0x2545_F491_4F6C_DD1D;
    let plane = (w * h) as usize;
    let planar: Vec<u16> = (0..3 * plane)
        .map(|i| {
            let (c, p) = (i / plane, (i % plane) as u32);
            (((p % w) + c as u32 * 50 + (xorshift(&mut state) % 24) as u32) % 256) as u16
        })
        .collect();
    let mut out = Vec::new();
    jpegls::encode_image(&planar, w, h, 3, &EncodeOptions::default(), &mut out).unwrap();
    out
}
