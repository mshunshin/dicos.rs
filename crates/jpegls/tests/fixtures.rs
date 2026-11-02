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

#[test]
fn dnl_spliced_into_dcmtk_streams() {
    // Each DCMTK stream with its SOF55 height moved into a DNL segment after
    // the scan (T.87 C.2.6) still decodes to `dcmdjpls`'s output.
    let Some(dir) = fixture_dir("dcmtk") else {
        return;
    };
    for (stem, bytes, reference) in dcmtk_streams(&dir) {
        let sof = bytes.windows(2).position(|p| p == [0xFF, 0xF7]).unwrap();
        let height = [bytes[sof + 5], bytes[sof + 6]];
        let mut stream = bytes.clone();
        stream[sof + 5..sof + 7].copy_from_slice(&[0, 0]);
        // DCMTK writes one scan, which ends at the EOI (after any fill).
        let mut end = stream.len() - 2;
        while stream[end - 1] == 0xFF {
            end -= 1;
        }
        stream.splice(end..end, [0xFF, 0xDC, 0x00, 0x04, height[0], height[1]]);
        let img = jpegls::decode_image(&stream).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_matches(&stem, &img, &reference, 0);
    }
}

/// The SOS segment and entropy data of a single-scan stream (everything from
/// SOS up to EOI, without any 0xFF fill before EOI).
fn scan_segment(stream: &[u8]) -> Vec<u8> {
    let sos = stream.windows(2).position(|p| p == [0xFF, 0xDA]).unwrap();
    let mut end = stream.len() - 2;
    while stream[end - 1] == 0xFF {
        end -= 1;
    }
    stream[sos..end].to_vec()
}

/// `scan` with its component selectors replaced by `selectors`.
fn with_selectors(scan: &[u8], selectors: &[u8]) -> Vec<u8> {
    let mut out = scan.to_vec();
    assert_eq!(usize::from(out[4]), selectors.len());
    for (k, &cs) in selectors.iter().enumerate() {
        out[5 + 2 * k] = cs;
    }
    out
}

#[test]
fn mixed_interleave_spliced_from_dcmtk_scans() {
    // Nf = 4 coded as DCMTK's 3-component line- or sample-interleaved scan
    // plus a 1-component scan from our encoder. A scan's coding depends only
    // on P, the presets and NEAR (both 8-bit defaults here), so the spliced
    // stream is valid T.87 with known samples.
    let Some(dir) = fixture_dir("dcmtk") else {
        return;
    };
    for ilv in [1u8, 2] {
        let stem = format!("mixed-37x24-8-rgb-ilv{ilv}");
        let rgb = std::fs::read(dir.join(format!("{stem}.jls"))).unwrap();
        let reference = read_pnm(&dir.join(format!("{stem}.ppm")));
        let (w, h) = (reference.width, reference.height);
        let mono: Vec<u16> = (0..w * h)
            .map(|i| ((i * 13 + i / w) % 256) as u16)
            .collect();
        let mut mono_stream = Vec::new();
        jpegls::encode_image(
            &mono,
            w,
            h,
            1,
            &jpegls::EncodeOptions::default(),
            &mut mono_stream,
        )
        .unwrap();
        let (rgb_scan, mono_scan) = (scan_segment(&rgb), scan_segment(&mono_stream));
        let rgb_planes: Vec<Vec<u16>> = (0..3)
            .map(|c| {
                reference
                    .samples
                    .iter()
                    .skip(c)
                    .step_by(3)
                    .copied()
                    .collect()
            })
            .collect();

        let mut frame = vec![0xFF, 0xD8, 0xFF, 0xF7, 0x00, 20, 8];
        frame.extend_from_slice(&(h as u16).to_be_bytes());
        frame.extend_from_slice(&(w as u16).to_be_bytes());
        frame.push(4);
        for id in 1..=4u8 {
            frame.extend_from_slice(&[id, 0x11, 0x00]);
        }
        let splice = |scans: [Vec<u8>; 2]| {
            let mut s = frame.clone();
            s.extend(scans.concat());
            s.extend_from_slice(&[0xFF, 0xD9]);
            jpegls::decode_image(&s).unwrap_or_else(|e| panic!("ILV {ilv}: {e}"))
        };
        let in_frame_order = [rgb_planes.concat(), mono.clone()].concat();

        // RGB scan first, selectors 1-3 then 4.
        let img = splice([
            with_selectors(&rgb_scan, &[1, 2, 3]),
            with_selectors(&mono_scan, &[4]),
        ]);
        assert_eq!((img.components, img.interleave_mode), (4, ilv));
        assert_eq!(img.samples, in_frame_order, "ILV {ilv}, frame order");

        // Component 4's scan first: the selectors still place it last.
        let img = splice([
            with_selectors(&mono_scan, &[4]),
            with_selectors(&rgb_scan, &[1, 2, 3]),
        ]);
        assert_eq!(img.interleave_mode, 0);
        assert_eq!(img.samples, in_frame_order, "ILV {ilv}, by selector");

        // Selectors naming no component: scan order decides.
        let img = splice([
            with_selectors(&mono_scan, &[9]),
            with_selectors(&rgb_scan, &[9, 9, 9]),
        ]);
        assert_eq!(
            img.samples,
            [mono.clone(), rgb_planes.concat()].concat(),
            "ILV {ilv}, fallback"
        );
    }
}

#[test]
fn near_per_scan_spliced_from_dcmtk_scans() {
    // Nf = 6 coded as two of DCMTK's 3-component scans of one image, one
    // lossless and one at NEAR 2 (T.87 keeps NEAR per scan). Neither stream
    // has an LSE, so each scan decodes as it did alone and `dcmdjpls`'s
    // output stays the bit-exact expectation.
    let Some(dir) = fixture_dir("dcmtk") else {
        return;
    };
    for ilv in [1u8, 2] {
        let lossless = format!("mixed-37x24-8-rgb-ilv{ilv}");
        let lossy = format!("{lossless}-near2");
        let read = |stem: &str| {
            let scan = scan_segment(&std::fs::read(dir.join(format!("{stem}.jls"))).unwrap());
            let reference = read_pnm(&dir.join(format!("{stem}.ppm")));
            let planes: Vec<u16> = (0..3)
                .flat_map(|c| reference.samples.iter().skip(c).step_by(3).copied())
                .collect();
            (scan, planes, reference.width, reference.height)
        };
        let (lossless_scan, lossless_planes, w, h) = read(&lossless);
        let (lossy_scan, lossy_planes, ..) = read(&lossy);

        let mut frame = vec![0xFF, 0xD8, 0xFF, 0xF7, 0x00, 26, 8];
        frame.extend_from_slice(&(h as u16).to_be_bytes());
        frame.extend_from_slice(&(w as u16).to_be_bytes());
        frame.push(6);
        for id in 1..=6u8 {
            frame.extend_from_slice(&[id, 0x11, 0x00]);
        }
        let splice = |scans: [Vec<u8>; 2]| {
            let mut s = frame.clone();
            s.extend(scans.concat());
            s.extend_from_slice(&[0xFF, 0xD9]);
            jpegls::decode_image(&s).unwrap_or_else(|e| panic!("ILV {ilv}: {e}"))
        };
        let in_frame_order = [lossless_planes, lossy_planes].concat();

        // Lossless scan first, then the NEAR 2 scan; and the other way round,
        // where the selectors still place the lossy scan's components last.
        for img in [
            splice([
                with_selectors(&lossless_scan, &[1, 2, 3]),
                with_selectors(&lossy_scan, &[4, 5, 6]),
            ]),
            splice([
                with_selectors(&lossy_scan, &[4, 5, 6]),
                with_selectors(&lossless_scan, &[1, 2, 3]),
            ]),
        ] {
            assert_eq!(img.samples, in_frame_order, "ILV {ilv}");
            assert_eq!(img.component_near, [0, 0, 0, 2, 2, 2], "ILV {ilv}");
            assert_eq!(img.near, 2, "ILV {ilv}");
        }
    }
}

#[test]
fn charls_generated_streams() {
    // Streams CharLS (via pillow-jpls) encoded, with CharLS's own decode as
    // the expectation: four components in each interleave mode, the SPIFF
    // header, custom presets. `-mx` streams carry an LSE MAXVAL below 2^P - 1,
    // which CharLS codes differently from T.87: they decode through the
    // CharLS retry.
    let Some(dir) = fixture_dir("charls-gen") else {
        return;
    };
    let mut streams: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jls"))
        .collect();
    streams.sort();
    assert!(streams.len() >= 15, "found {} streams", streams.len());
    for jls in streams {
        let stem = jls.file_stem().unwrap().to_str().unwrap().to_string();
        let parts: Vec<&str> = stem.split('-').collect();
        let (components, ext) = match parts[3] {
            "gray" => (1u8, "pgm"),
            "rgb" => (3, "ppm"),
            "rgba" => (4, "pam"),
            other => panic!("{stem}: unknown kind {other}"),
        };
        let ilv: u8 = parts[4].strip_prefix("ilv").unwrap().parse().unwrap();
        let near: u8 = parts[5..]
            .iter()
            .find_map(|p| p.strip_prefix("near"))
            .map_or(0, |n| n.parse().unwrap());
        let img = jpegls::decode_image(&std::fs::read(&jls).unwrap())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_matches(
            &stem,
            &img,
            &read_pnm(&dir.join(format!("{stem}.{ext}"))),
            0,
        );
        assert_eq!(
            (img.components, img.interleave_mode, img.near),
            (components, ilv, near),
            "{stem}: metadata"
        );
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
    (
        "banny-hp1",
        Expect::Matches {
            reference: "banny.ppm",
            near: 0,
            ilv: 1,
        },
    ),
    (
        "banny-hp2",
        Expect::Matches {
            reference: "banny.ppm",
            near: 0,
            ilv: 1,
        },
    ),
    (
        "banny-hp3",
        Expect::Matches {
            reference: "banny.ppm",
            near: 0,
            ilv: 1,
        },
    ),
    (
        "test8_ilv_none_rm_7",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 0,
        },
    ),
    (
        "test8_ilv_line_rm_7",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 1,
        },
    ),
    (
        "test8_ilv_sample_rm_7",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 2,
        },
    ),
    (
        "test8_ilv_sample_rm_300",
        Expect::Matches {
            reference: "test8.ppm",
            near: 0,
            ilv: 2,
        },
    ),
    (
        "test16_rm_5",
        Expect::Matches {
            reference: "test16.pgm",
            near: 0,
            ilv: 0,
        },
    ),
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

#[test]
fn feature_caps_lists_every_capability() {
    // Downstream tripwires match on these names; keep them exact.
    assert_eq!(
        jpegls::FEATURE_CAPS.split(';').collect::<Vec<_>>(),
        [
            "decode_image",
            "rgb-ilv0-planar",
            "ilv1-line",
            "ilv2-sample",
            "nf-any",
            "mixed-ilv",
            "restart",
            "hp-transform",
            "charls-retry",
            "near-per-scan",
            "dnl"
        ]
    );
}
