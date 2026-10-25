//! JPEG-LS lossless codec (ISO/IEC 14495-1 / ITU-T T.87).
//!
//! Implements the LOCO-I algorithm with context-based Golomb-Rice coding
//! and run-length mode for uniform regions.
//!
//! # Conformance scope
//!
//! Conformant T.87 for **single-component** images and **3-component** images
//! (e.g. RGB) through [`decode_image`], precision `2..=16`, with LSE ID=1
//! presets, run mode, and near-lossless. The decoder reads all three
//! interleave modes: one scan per component (`ILV = 0`), line-interleaved
//! (`ILV = 1`, DCMTK's default for colour) and sample-interleaved (`ILV = 2`).
//! [`encode_image`] writes `ILV = 0`. Restart markers inside a scan are
//! rejected as `Unsupported`; a `DRI` segment on its own is skipped. Verified
//! against the T.87 conformance streams, CharLS and DCMTK output.
//!
//! The decoder also accepts what CharLS accepts from real-world encoders:
//! `0x00` or `0xFF` padding before the next marker, LSE / COM / APPn segments
//! between scans, an LSE `RESET` of 0 (the default), and a non-zero (reserved)
//! SOF `Tq` or SOS `Ah`. An HP colour transform (APP8 `"mrfx"`) is rejected
//! rather than decoded to the wrong colours.
//!
//! # Profiles
//!
//! The default [`Profile::T87`] path is ITU-T T.87 conformant. The
//! [`Profile::LegacyGo`] path reproduces the frozen 1.0.0 / Go-compatible
//! bitstream (T.81-style `0xFF00` stuffing, no run mode, uncapped Golomb) for
//! reading and writing files produced by earlier releases.

pub mod error;

// Retains a few reader/writer helpers (write_byte/write_u16be/pending_marker/…)
// exercised only by unit tests or reserved for marker handling.
#[allow(dead_code)]
mod bitstream;
mod context;
mod decode;
mod encode;
// Frozen 1.0.0 coder. Retains verbatim helpers (write_byte/write_u16be/...)
// that the public path no longer calls; see legacy.rs. DO NOT MODIFY.
#[allow(dead_code)]
mod legacy;
mod predictor;
mod run_mode;

pub use decode::{decode, decode_image, decode_with_options, DecodedImage};
pub use encode::{encode, encode_image, encode_with_options};
pub use error::CodecError;

/// DICOM Transfer Syntax UID for JPEG-LS Lossless.
pub const TRANSFER_SYNTAX_UID: &str = "1.2.840.10008.1.2.4.80";

/// Capability marker for this feature set — present in forks/releases that
/// ship [`decode_image`], 3-component planar (ILV 0) decoding and line- /
/// sample-interleaved (ILV 1 / 2) decoding, and absent from codecs that
/// predate them.
///
/// Downstreams that pull this crate from a fork (pending an upstream release)
/// should use a git dependency pinned to an exact `rev` — a
/// `[patch.crates-io]` override applies only in the root manifest being built,
/// not when this crate is a dependency of a library — and can reference this
/// constant in a build-time assertion or tripwire test, so the dependency
/// cannot silently resolve to a version without these capabilities. The value
/// lists the capabilities, newest last.
pub const FEATURE_CAPS: &str = "decode_image;rgb-ilv0-planar;ilv1-line;ilv2-sample";

/// Bitstream profile selecting the entropy-coded scan format.
///
/// The two formats are not bridgeable by a single flag — single-bit stuffing
/// and the length-limited Golomb code differ from the legacy path — so the
/// choice is a whole-stream profile.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// ITU-T T.87 conformant (default).
    #[default]
    T87,
    /// Frozen 1.0.0 / Go-compatible bytes (lossless only).
    LegacyGo,
}

/// Options controlling [`encode_with_options`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EncodeOptions {
    /// Near-lossless error bound (`0` = lossless). Bounded by
    /// `min(255, MAXVAL/2)`. Must be `0` for [`Profile::LegacyGo`].
    pub near: u8,
    /// Bitstream profile.
    pub profile: Profile,
    /// Explicit sample precision in bits (`2..=16`). `None` derives it from the
    /// data (at least 8), preserving legacy behavior.
    pub precision: Option<u8>,
}

/// Options controlling [`decode_with_options`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeOptions {
    /// Bitstream profile to decode with.
    pub profile: Profile,
}
