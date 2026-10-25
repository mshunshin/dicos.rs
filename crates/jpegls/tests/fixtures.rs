//! Decode third-party JPEG-LS streams and compare with their references.
//!
//! - `fixtures/dcmtk`: streams written by DCMTK's `dcmcjpls` (colour is always
//!   line- or sample-interleaved there), with the `dcmdjpls` output as the
//!   bit-exact expectation. Regenerate with `fixtures/dcmtk/generate.py`.
//! - `fixtures/charls`: the T.87 conformance images (HP encoder) and CharLS
//!   test streams, vendored from CharLS (see its `README.md` and `LICENSE`).
//!
//! Both directories are left out of the published package, so a test skips
//! (with a note) when its directory is absent, but fails when it is present
//! and empty.

mod common;

use std::path::{Path, PathBuf};

use common::{interleave, read_pnm, Pnm};
use jpegls::{CodecError, DecodedImage};

fn fixture_dir(name: &str) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    if dir.is_dir() {
        Some(dir)
    } else {
        eprintln!("skipping: {} not present (packaged crate?)", dir.display());
        None
    }
}

/// Assert `img` matches `reference` sample for sample, within `near`.
fn assert_matches(stem: &str, img: &DecodedImage, reference: &Pnm, near: u16) {
    assert_eq!(
        (img.width, img.height, usize::from(img.components)),
        (reference.width, reference.height, reference.components),
        "{stem}: geometry"
    );
    let decoded = interleave(&img.samples, usize::from(img.components));
    for (i, (&got, &want)) in decoded.iter().zip(&reference.samples).enumerate() {
        assert!(
            got.abs_diff(want) <= near,
            "{stem}: sample {i} decoded {got}, expected {want} (near {near})"
        );
    }
}

// ---------------------------------------------------------------------------
// DCMTK
// ---------------------------------------------------------------------------

/// What a DCMTK fixture stem (`{pattern}-{w}x{h}-{bits}-{rgb|mono}-ilv{N}
/// [-near{N}][-lse]`) says about the stream.
struct DcmtkStem {
    components: u8,
    ilv: u8,
    near: u8,
}

fn parse_dcmtk_stem(stem: &str) -> DcmtkStem {
    let parts: Vec<&str> = stem.split('-').collect();
    let components = match parts[3] {
        "rgb" => 3,
        "mono" => 1,
        other => panic!("{stem}: unknown kind {other}"),
    };
    let ilv = parts[4].strip_prefix("ilv").unwrap().parse().unwrap();
    let near = parts[5..]
        .iter()
        .find_map(|p| p.strip_prefix("near"))
        .map_or(0, |n| n.parse().unwrap());
    DcmtkStem {
        components,
        ilv,
        near,
    }
}

fn dcmtk_streams(dir: &Path) -> Vec<(String, Vec<u8>, Pnm)> {
    let mut streams: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jls"))
        .collect();
    streams.sort();
    assert!(
        streams.len() >= 20,
        "{}: expected the generated fixtures, found {}",
        dir.display(),
        streams.len()
    );
    streams
        .into_iter()
        .map(|jls| {
            let stem = jls.file_stem().unwrap().to_str().unwrap().to_string();
            let ext = if stem.contains("-mono-") {
                "pgm"
            } else {
                "ppm"
            };
            let reference = read_pnm(&dir.join(format!("{stem}.{ext}")));
            (stem, std::fs::read(&jls).unwrap(), reference)
        })
        .collect()
}

#[test]
fn dcmtk_streams_decode_exactly() {
    let Some(dir) = fixture_dir("dcmtk") else {
        return;
    };
    let mut interleaved = 0;
    for (stem, bytes, reference) in dcmtk_streams(&dir) {
        let want = parse_dcmtk_stem(&stem);
        let img = jpegls::decode_image(&bytes).unwrap_or_else(|e| panic!("{stem}: {e}"));
        // dcmdjpls is deterministic, so even near-lossless output must match
        // exactly.
        assert_matches(&stem, &img, &reference, 0);
        assert_eq!(img.components, want.components, "{stem}: components");
        assert_eq!(img.interleave_mode, want.ilv, "{stem}: interleave mode");
        assert_eq!(img.near, want.near, "{stem}: NEAR");
        assert_eq!(
            (1u32 << img.bits_per_sample) - 1,
            reference.maxval,
            "{stem}: precision"
        );

        // The grayscale entry agrees on mono streams and refuses colour.
        let decoded = jpegls::decode(&bytes, img.width, img.height);
        if img.components == 1 {
            assert_eq!(decoded.unwrap().0, img.samples, "{stem}: decode()");
        } else {
            assert!(
                matches!(decoded, Err(CodecError::Unsupported(_))),
                "{stem}: decode() should refuse colour"
            );
            interleaved += 1;
        }
    }
    assert!(interleaved >= 16, "only {interleaved} interleaved streams");
}

#[test]
fn dcmtk_interleaved_truncation_is_an_error() {
    let Some(dir) = fixture_dir("dcmtk") else {
        return;
    };
    for (stem, bytes, _) in dcmtk_streams(&dir) {
        if !stem.contains("-rgb-") {
            continue;
        }
        for cut in [bytes.len() / 4, bytes.len() / 2, bytes.len() - 3] {
            assert!(
                jpegls::decode_image(&bytes[..cut]).is_err(),
                "{stem}: truncated to {cut} bytes decoded"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// T.87 conformance + CharLS
// ---------------------------------------------------------------------------

enum Expect {
    /// Decodes to the reference within NEAR, with this interleave mode.
    Matches {
        reference: &'static str,
        near: u16,
        ilv: u8,
    },
    /// Decodes (no reference image).
    Decodes,
    /// Rejected as `Unsupported`.
    Unsupported,
    /// Rejected with any error.
    Fails,
}

const CHARLS_STREAMS: &[(&str, Expect)] = &[
    (
        "t8c0e0",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 0,
        },
    ),
    (
        "t8c1e0",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 1,
        },
    ),
    (
        "t8c2e0",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 2,
        },
    ),
    (
        "t8c0e3",
        Expect::Matches {
            reference: "test8.ppm",
            near: 3,
            ilv: 0,
        },
    ),
    (
        "t8c1e3",
        Expect::Matches {
            reference: "test8.ppm",
            near: 3,
            ilv: 1,
        },
    ),
    (
        "t8c2e3",
        Expect::Matches {
            reference: "test8.ppm",
            near: 3,
            ilv: 2,
        },
    ),
    (
        "t16e0",
        Expect::Matches {
            reference: "test16.pgm",
            near: 0,
            ilv: 0,
        },
    ),
    (
        "t16e3",
        Expect::Matches {
            reference: "test16.pgm",
            near: 3,
            ilv: 0,
        },
    ),
    (
        "t8nde0",
        Expect::Matches {
            reference: "test8bs2.pgm",
            near: 0,
            ilv: 0,
        },
    ),
    (
        "t8nde3",
        Expect::Matches {
            reference: "test8bs2.pgm",
            near: 3,
            ilv: 0,
        },
    ),
    (
        "banny",
        Expect::Matches {
            reference: "banny.ppm",
            near: 0,
            ilv: 1,
        },
    ),
    ("8bit-monochrome-2x2", Expect::Decodes),
    ("t8sse0", Expect::Unsupported),
    ("banny-hp1", Expect::Unsupported),
    ("banny-hp2", Expect::Unsupported),
    ("banny-hp3", Expect::Unsupported),
    ("fuzzy-input-bad-run-mode-golomb-code", Expect::Fails),
    ("fuzzy-input-no-valid-bits-at-the-end", Expect::Fails),
    ("fuzzy_input_golomb_16", Expect::Fails),
    ("no_start_byte_after_encoded_scan", Expect::Fails),
];

#[test]
fn charls_reference_streams() {
    let Some(dir) = fixture_dir("charls") else {
        return;
    };
    for (stem, expect) in CHARLS_STREAMS {
        let bytes = std::fs::read(dir.join(format!("{stem}.jls")))
            .unwrap_or_else(|e| panic!("{stem}.jls: {e}"));
        let result = jpegls::decode_image(&bytes);
        match expect {
            Expect::Matches {
                reference,
                near,
                ilv,
            } => {
                let img = result.unwrap_or_else(|e| panic!("{stem}: {e}"));
                assert_matches(stem, &img, &read_pnm(&dir.join(reference)), *near);
                assert_eq!(img.interleave_mode, *ilv, "{stem}: interleave mode");
                assert_eq!(u16::from(img.near), *near, "{stem}: NEAR");
            }
            Expect::Decodes => {
                result.unwrap_or_else(|e| panic!("{stem}: {e}"));
            }
            Expect::Unsupported => assert!(
                matches!(result, Err(CodecError::Unsupported(_))),
                "{stem}: {result:?}"
            ),
            Expect::Fails => assert!(result.is_err(), "{stem} decoded"),
        }
    }
}
