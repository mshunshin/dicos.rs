//! Interop tests against a third-party JPEG-LS codec (charls / pillow-jpls).
//!
//! There is no charls CLI in the maintainer's local environment, so this
//! harness is **fixture-driven and self-skipping**: it decodes every
//! `tests/fixtures/interop/*.jls` and compares against a same-stem `.pgm`
//! expectation. When the directory holds no `.jls` files (the checked-in
//! state), it skips with a note so a plain `cargo test` passes everywhere.
//!
//! CI, or a maintainer with charls/pillow-jpls, populates the directory using
//! the recipes in `fixtures/interop/README.md`; the test then really runs.
//!
//! ## Fixture pairing
//!
//! For a stream `foo.jls`, the expectation is `foo.pgm` (binary P5, 8- or
//! 16-bit big-endian). A stream may declare a near-lossless bound in its file
//! name as `...-nearN.jls`; the comparison then allows `|decoded - expected| ≤
//! N` per sample instead of exact equality.

mod common;

use std::path::PathBuf;

use common::read_pnm;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/interop")
}

/// Parse an optional `-nearN` suffix from a stream stem.
fn near_bound(stem: &str) -> i32 {
    stem.rsplit_once("-near")
        .and_then(|(_, n)| n.parse::<i32>().ok())
        .unwrap_or(0)
}

#[test]
fn decode_charls_fixtures() {
    let dir = fixture_dir();
    let mut streams: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().map(|x| x == "jls").unwrap_or(false))
                .collect()
        })
        .unwrap_or_default();
    streams.sort();

    if streams.is_empty() {
        eprintln!(
            "skipping decode_charls_fixtures: no *.jls in {} \
             (populate via fixtures/interop/README.md)",
            dir.display()
        );
        return;
    }

    for jls in streams {
        let stem = jls.file_stem().unwrap().to_str().unwrap().to_string();
        let pgm = dir.join(format!("{stem}.pgm"));
        let reference = read_pnm(&pgm);
        let (w, h, expected) = (reference.width, reference.height, reference.samples);
        let near = near_bound(&stem);
        let bytes = std::fs::read(&jls).unwrap();
        let (decoded, dw, dh) =
            jpegls::decode(&bytes, w, h).unwrap_or_else(|e| panic!("{stem}: decode failed: {e}"));
        assert_eq!((dw, dh), (w, h), "{stem}: dimensions");
        assert_eq!(decoded.len(), expected.len(), "{stem}: sample count");
        for (idx, (a, b)) in decoded.iter().zip(&expected).enumerate() {
            let diff = (*a as i32 - *b as i32).abs();
            assert!(
                diff <= near,
                "{stem}: sample {idx} diff {diff} exceeds near bound {near} \
                 (decoded {a}, expected {b})"
            );
        }
    }
}
