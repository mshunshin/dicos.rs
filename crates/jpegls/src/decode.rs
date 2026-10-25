//! JPEG-LS decoder.
//!
//! Parses SOI, SOF55, LSE, SOS markers and then entropy-decodes the scan data
//! to reconstruct 16-bit samples. The default [`Profile::T87`] path is ITU-T
//! T.87 conformant; [`Profile::LegacyGo`] reads frozen 1.0.0 bytes.

use crate::bitstream::BitReader;
use crate::context::ContextModel;
use crate::error::CodecError;
use crate::predictor::{clamp, predict_med};
use crate::run_mode;
use crate::{DecodeOptions, Profile};

use crate::legacy;

// ---------------------------------------------------------------------------
// JPEG-LS markers
// ---------------------------------------------------------------------------

const MARKER_SOI: u16 = 0xFFD8;
const MARKER_EOI: u16 = 0xFFD9;
const MARKER_SOS: u16 = 0xFFDA;
const MARKER_SOF55: u16 = 0xFFF7;
const MARKER_LSE: u16 = 0xFFF8;
const MARKER_DRI: u16 = 0xFFDD;
const MARKER_APP8: u16 = 0xFFE8;

/// Default statistics reset threshold (T.87 A.2.1 RESET).
const DEFAULT_RESET: i32 = 64;

/// Hard cap on decoded image size (pixels), guarding the row-major
/// `vec![0u16; width*height]` against untrusted SOF55 dimensions. Matches the
/// `jpegli` / `jpegrle` decoders (`1 << 28` = 268 435 456 pixels ~= 512 MiB).
const MAX_PIXELS: usize = 1 << 28;

/// Minimum permitted LSE RESET value (T.87 C.2.4.1.1). A RESET below this
/// would suppress statistics halving and risk i32 overflow on long scans.
const MIN_RESET: i32 = 3;

/// Most components one scan may carry (T.87 C.2.3: `1 <= Ns <= 4`).
const MAX_SCAN_COMPONENTS: usize = 4;

// ---------------------------------------------------------------------------
// Frame / scan headers + LSE presets
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct FrameHeader {
    precision: u32,
    height: usize,
    width: usize,
    /// Number of components (Nf). 1 (grayscale) or 3 (e.g. RGB); other counts
    /// are rejected in [`Decoder::read_sof`].
    components: usize,
}

#[derive(Clone, Copy)]
struct ScanHeader {
    near: i32,
    /// Components coded in this scan (Ns, `1..=4`).
    components: usize,
    /// Interleave mode: 0 (one component), 1 (line) or 2 (sample).
    ilv: u8,
}

/// LSE ID=1 coding-parameter presets (T.87 C.2.4.1.1); `None` = spec default.
#[derive(Default)]
struct Presets {
    maxval: Option<i32>,
    t1: Option<i32>,
    t2: Option<i32>,
    t3: Option<i32>,
    reset: Option<i32>,
}

/// Everything parsed up to and including the first SOS.
struct Header {
    frame: FrameHeader,
    presets: Presets,
    scan: ScanHeader,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decode a JPEG-LS compressed bitstream into a 16-bit grayscale pixel buffer.
///
/// `width` and `height` are the *expected* image dimensions, cross-checked
/// against the SOF55 marker. Uses [`DecodeOptions::default`] ([`Profile::T87`]).
///
/// Returns `(pixels, width, height)` where pixels is a row-major `Vec<u16>`.
pub fn decode(data: &[u8], width: u32, height: u32) -> Result<(Vec<u16>, u32, u32), CodecError> {
    decode_with_options(data, width, height, &DecodeOptions::default())
}

/// Decode a JPEG-LS bitstream with explicit [`DecodeOptions`].
pub fn decode_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    opts: &DecodeOptions,
) -> Result<(Vec<u16>, u32, u32), CodecError> {
    let mut dec = Decoder::new(data);
    dec.decode(width, height, opts.profile)
}

/// A decoded JPEG-LS image with the frame metadata a DICOM caller needs.
///
/// Supports grayscale (`components == 1`) and 3-component frames, coded as
/// one scan per component (ILV 0) or as a single line- (ILV 1) or
/// sample-interleaved (ILV 2) scan. The samples are component-planar whatever
/// the interleave mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedImage {
    /// Samples in component-planar order — all of component 0, then component
    /// 1, … (`RRR…GGG…BBB…`). Length `width * height * components`.
    pub samples: Vec<u16>,
    /// Image width in pixels (from the codestream).
    pub width: u32,
    /// Image height in pixels (from the codestream).
    pub height: u32,
    /// Sample precision in bits (2..=16).
    pub bits_per_sample: u8,
    /// Component count (1 or 3).
    pub components: u8,
    /// Interleave mode of the (first) scan: `0` none, `1` line, `2` sample.
    /// Informational only: [`Self::samples`] is planar in every mode.
    pub interleave_mode: u8,
    /// NEAR error bound (0 = lossless).
    pub near: u8,
}

/// Decode a JPEG-LS bitstream, reading the geometry from the codestream.
///
/// Unlike [`decode`], this takes no expected dimensions and returns the full
/// frame metadata in [`DecodedImage`], including 3-component frames in any
/// interleave mode. T.87 conformant path only (no `LegacyGo`).
pub fn decode_image(data: &[u8]) -> Result<DecodedImage, CodecError> {
    let mut dec = Decoder::new(data);
    let header = dec.read_header()?;
    dec.decode_frame(header)
}

// ---------------------------------------------------------------------------
// Decoder state
// ---------------------------------------------------------------------------

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    // -- byte-level helpers ------------------------------------------------

    fn read_byte(&mut self) -> Result<u8, CodecError> {
        if self.pos >= self.data.len() {
            return Err(CodecError::InvalidData(
                "unexpected end of JPEG-LS data".into(),
            ));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u16be(&mut self) -> Result<u16, CodecError> {
        let hi = self.read_byte()?;
        let lo = self.read_byte()?;
        Ok(u16::from(hi) << 8 | u16::from(lo))
    }

    fn skip(&mut self, n: usize) -> Result<(), CodecError> {
        if self.pos + n > self.data.len() {
            return Err(CodecError::InvalidData(
                "unexpected end of data during skip".into(),
            ));
        }
        self.pos += n;
        Ok(())
    }

    /// Skip `0x00` bytes left between the end of a scan and the next marker.
    /// Not conformant, but some encoders pad this way (CharLS #365 cites
    /// official DICOM samples; the HP codec and CharLS accept it).
    fn skip_zero_padding(&mut self) {
        while self.data.get(self.pos) == Some(&0) {
            self.pos += 1;
        }
    }

    // -- marker parsing ----------------------------------------------------

    fn expect_marker(&mut self, expected: u16) -> Result<(), CodecError> {
        let b1 = self.read_byte()?;
        let b2 = self.read_byte()?;
        let marker = u16::from(b1) << 8 | u16::from(b2);
        if marker != expected {
            return Err(CodecError::InvalidData(format!(
                "expected marker 0x{expected:04X}, got 0x{marker:04X}"
            )));
        }
        Ok(())
    }

    fn read_marker(&mut self) -> Result<(u16, usize), CodecError> {
        let mut b = self.read_byte()?;
        if b != 0xFF {
            return Err(CodecError::InvalidData(format!(
                "expected 0xFF marker prefix, got 0x{b:02X}"
            )));
        }
        // A marker may be preceded by any number of 0xFF fill bytes
        // (T.81 B.1.1.2); skip them. The first non-0xFF byte is the code.
        while b == 0xFF {
            b = self.read_byte()?;
        }
        let marker = 0xFF00u16 | u16::from(b);

        let length = self.read_u16be()? as usize;
        // Length field includes its own 2 bytes.
        Ok((marker, length.saturating_sub(2)))
    }

    /// Read marker segments up to and including the next SOS, applying SOF55,
    /// LSE and APP8 as they come and skipping DRI, COM and the other APPn.
    ///
    /// Used before the first scan (`frame` still `None`) and between scans,
    /// where `done` components have already been decoded.
    fn read_segments(
        &mut self,
        frame: &mut Option<FrameHeader>,
        presets: &mut Presets,
        done: usize,
    ) -> Result<ScanHeader, CodecError> {
        loop {
            let (marker, length) = self.read_marker()?;
            match marker {
                MARKER_SOF55 => {
                    if frame.is_some() {
                        return Err(CodecError::InvalidData("duplicate SOF55 marker".into()));
                    }
                    *frame = Some(self.read_sof(length)?);
                }
                MARKER_LSE => self.read_lse(length, presets)?,
                MARKER_APP8 => self.read_app8(length)?,
                MARKER_DRI => {
                    // Tolerate and skip DRI, as CharLS does: a zero restart
                    // interval is a no-op. A non-zero interval that actually
                    // emits RSTn markers in the scan is rejected by the entropy
                    // decoder (scan_error), so restarts are still unsupported —
                    // but a benign DRI segment no longer fails an otherwise
                    // decodable stream.
                    self.skip(length)?;
                }
                MARKER_SOS => {
                    let remaining = frame.as_ref().map(|f| f.components - done);
                    return self.read_sos(length, remaining);
                }
                MARKER_EOI => {
                    return Err(CodecError::InvalidData(if done == 0 {
                        "unexpected EOI before SOS".into()
                    } else {
                        format!("unexpected EOI after {done} decoded component(s)")
                    }));
                }
                m if is_other_sof(m) => {
                    return Err(CodecError::Unsupported(format!(
                        "frame marker 0x{m:04X} is not JPEG-LS baseline (SOF55)"
                    )));
                }
                _ => self.skip(length)?,
            }
        }
    }

    fn read_sof(&mut self, payload_len: usize) -> Result<FrameHeader, CodecError> {
        let p = self.read_byte()?;
        if !(2..=16).contains(&p) {
            return Err(CodecError::InvalidParameter {
                name: "precision",
                value: i64::from(p),
                allowed: "2..=16",
            });
        }
        let height = self.read_u16be()? as usize;
        let width = self.read_u16be()? as usize;

        // Reject degenerate / oversized frames before any large allocation.
        // T.87 permits 1..=65535 per axis; also cap the pixel product so a
        // crafted 65535x65535 header cannot request an ~8 GiB buffer.
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidData(
                "SOF55 dimensions must be non-zero".into(),
            ));
        }
        if width
            .checked_mul(height)
            .filter(|&n| n <= MAX_PIXELS)
            .is_none()
        {
            return Err(CodecError::InvalidParameter {
                name: "width*height",
                value: (width as i64).saturating_mul(height as i64),
                allowed: "product <= 268435456 pixels (1<<28)",
            });
        }

        let nf = self.read_byte()?;
        // Grayscale (1) and 3-component (e.g. RGB) frames are supported. The
        // single-scan `decode()` entry remains grayscale-only (it re-checks
        // below); `decode_image()` handles 3-component frames.
        if nf != 1 && nf != 3 {
            return Err(CodecError::Unsupported(format!(
                "JPEG-LS component count {nf} not supported (only 1 or 3)"
            )));
        }

        // Component specs: [Ci, sampling (Hi<<4 | Vi), Tqi] per component.
        // Require 1:1 sampling (no subsampling); otherwise the scan would be
        // mis-decoded. Tqi is reserved in JPEG-LS and ignored, as CharLS does
        // (mapping tables are selected per scan, in SOS). The component id Ci
        // is not checked.
        for _ in 0..nf {
            let _ci = self.read_byte()?;
            let sampling = self.read_byte()?;
            let _tq = self.read_byte()?;
            if sampling != 0x11 {
                return Err(CodecError::Unsupported(format!(
                    "JPEG-LS component sampling {sampling:#04x} not supported (must be 1:1 = 0x11)"
                )));
            }
        }
        // Skip any trailing bytes in an over-long segment (payload is 6 + Nf*3).
        let consumed = 6 + nf as usize * 3;
        if payload_len > consumed {
            self.skip(payload_len - consumed)?;
        }

        Ok(FrameHeader {
            precision: u32::from(p),
            height,
            width,
            components: nf as usize,
        })
    }

    /// Parse an LSE segment. Only ID=1 (coding parameters) is supported.
    fn read_lse(&mut self, payload_len: usize, presets: &mut Presets) -> Result<(), CodecError> {
        if payload_len < 1 {
            return Err(CodecError::InvalidData("empty LSE segment".into()));
        }
        let id = self.read_byte()?;
        match id {
            1 => {
                if payload_len < 11 {
                    return Err(CodecError::InvalidData("LSE ID=1 segment too short".into()));
                }
                let maxval = self.read_u16be()? as i32;
                let t1 = self.read_u16be()? as i32;
                let t2 = self.read_u16be()? as i32;
                let t3 = self.read_u16be()? as i32;
                let reset = self.read_u16be()? as i32;
                // A field of 0 means "use the default" (T.87 C.2.4.1.1), for
                // RESET as for MAXVAL/T1/T2/T3; CharLS reads it the same way.
                presets.maxval = (maxval > 0).then_some(maxval);
                presets.t1 = (t1 > 0).then_some(t1);
                presets.t2 = (t2 > 0).then_some(t2);
                presets.t3 = (t3 > 0).then_some(t3);
                // An explicit RESET of 1 or 2 is below the T.87 minimum and
                // would suppress statistics halving, letting the A/B
                // accumulators overflow on a long scan. The upper bound needs
                // MAXVAL and is checked in `validate_params`.
                if reset != 0 && reset < MIN_RESET {
                    return Err(CodecError::InvalidData(format!(
                        "LSE RESET={reset} invalid (must be 0 or >= {MIN_RESET})"
                    )));
                }
                presets.reset = (reset > 0).then_some(reset);
                // Skip any trailing bytes in an over-long segment.
                self.skip(payload_len - 11)?;
                Ok(())
            }
            2..=4 => Err(CodecError::Unsupported(format!(
                "LSE ID={id} (mapping tables) not supported"
            ))),
            other => Err(CodecError::InvalidData(format!("unknown LSE ID={other}"))),
        }
    }

    /// Parse an APP8 segment. The 5-byte HP colour-transform segment
    /// (`"mrfx"` + transform id, written by the HP LOCO-I codec and CharLS)
    /// changes how the samples must be read back, so a transform other than
    /// "none" is rejected rather than decoded to the wrong colours. Any other
    /// APP8 (e.g. a SPIFF header) is skipped.
    fn read_app8(&mut self, payload_len: usize) -> Result<(), CodecError> {
        let end = self.pos + payload_len;
        if payload_len == 5 && self.data.get(self.pos..self.pos + 4) == Some(b"mrfx") {
            self.pos += 4;
            match self.read_byte()? {
                0 => {}
                id @ 1..=5 => {
                    return Err(CodecError::Unsupported(format!(
                        "HP colour transform {id} (APP8 \"mrfx\") not supported"
                    )))
                }
                other => {
                    return Err(CodecError::InvalidData(format!(
                        "unknown HP colour transform {other} (APP8 \"mrfx\")"
                    )))
                }
            }
        }
        self.skip(end.saturating_sub(self.pos))
    }

    /// Parse an SOS header. `remaining` is the number of frame components not
    /// yet decoded when SOF55 has been read (`None` if SOS precedes it; that
    /// stream fails later as missing SOF55).
    fn read_sos(
        &mut self,
        payload_len: usize,
        remaining: Option<usize>,
    ) -> Result<ScanHeader, CodecError> {
        let ns = usize::from(self.read_byte()?);
        // T.87 C.2.3: 1 <= Ns <= 4, and a scan cannot cover more components
        // than the frame still has. Such a header is malformed, not merely a
        // scan shape this decoder lacks, so it is InvalidData rather than
        // Unsupported.
        if ns == 0 || ns > MAX_SCAN_COMPONENTS || remaining.is_some_and(|r| ns > r) {
            let frame = remaining.map_or(String::new(), |r| {
                format!(" with {r} frame component(s) left to decode")
            });
            return Err(CodecError::InvalidData(format!(
                "SOS component count Ns = {ns} is invalid{frame}"
            )));
        }
        // Ns component specs, then NEAR, ILV and Ah/Al. CharLS checks this
        // size exactly too.
        if payload_len != 2 * ns + 4 {
            return Err(CodecError::InvalidData(format!(
                "SOS segment length {} does not match Ns = {ns} (expected {})",
                payload_len + 2,
                2 * ns + 6
            )));
        }
        // Component spec: [Csj, mapping-table selector]. The component selector
        // Csj is not checked (CharLS ignores it); a non-zero mapping-table
        // selector references an LSE mapping table this decoder does not apply,
        // so reject it rather than silently mis-decode.
        for _ in 0..ns {
            let _csj = self.read_byte()?;
            let mapping = self.read_byte()?;
            if mapping != 0 {
                return Err(CodecError::Unsupported(format!(
                    "SOS mapping-table selector {mapping} not supported (must be 0)"
                )));
            }
        }

        let near = self.read_byte()?;
        let ilv = self.read_byte()?;
        let al_ah = self.read_byte()?;

        // ILV 0 codes one component per scan; ILV 1 (line) and 2 (sample)
        // interleave several (T.87 C.2.3). Values above 2 are reserved.
        let valid_ilv = match ilv {
            0 => ns == 1,
            1 | 2 => ns > 1,
            _ => false,
        };
        if !valid_ilv {
            return Err(CodecError::InvalidData(format!(
                "SOS interleave mode ILV = {ilv} is invalid for Ns = {ns}"
            )));
        }
        // A multi-component scan that leaves frame components for a later scan
        // (mixed interleave) is legal but not supported yet.
        if ns > 1 && remaining.is_some_and(|r| ns < r) {
            return Err(CodecError::Unsupported(format!(
                "partial multi-component scan (Ns = {ns} of {} remaining) not supported",
                remaining.unwrap_or(ns)
            )));
        }
        // Point transform Al (the low nibble) must be 0: a non-zero value
        // shifts the samples, which this decoder does not apply. Ah has no
        // meaning in JPEG-LS and is ignored, as CharLS does.
        if al_ah & 0x0F != 0 {
            return Err(CodecError::InvalidData(format!(
                "SOS point transform Al = {} must be 0",
                al_ah & 0x0F
            )));
        }

        Ok(ScanHeader {
            near: i32::from(near),
            components: ns,
            ilv,
        })
    }

    fn expect_eoi(&mut self) -> Result<(), CodecError> {
        self.skip_zero_padding();
        let b1 = self.read_byte()?;
        let mut b2 = self.read_byte()?;
        // As in `read_marker`, 0xFF fill bytes may precede the EOI (T.81
        // B.1.1.2). DCMTK pads an odd-length stream to even this way by
        // default (`FF FF D9`); skip them. The first non-0xFF byte is the code.
        if b1 == 0xFF {
            while b2 == 0xFF {
                b2 = self.read_byte()?;
            }
        }
        let marker = u16::from(b1) << 8 | u16::from(b2);
        if marker != MARKER_EOI {
            return Err(CodecError::InvalidData(format!(
                "expected EOI (0xFFD9) after scan, got 0x{marker:04X}"
            )));
        }
        Ok(())
    }

    // -- main decode flow --------------------------------------------------

    /// Parse SOI and the marker segments up to the first SOS.
    fn read_header(&mut self) -> Result<Header, CodecError> {
        self.expect_marker(MARKER_SOI)?;
        let mut frame: Option<FrameHeader> = None;
        let mut presets = Presets::default();
        let scan = self.read_segments(&mut frame, &mut presets, 0)?;
        let frame = frame.ok_or_else(|| CodecError::InvalidData("missing SOF55 marker".into()))?;
        Ok(Header {
            frame,
            presets,
            scan,
        })
    }

    fn decode(
        &mut self,
        exp_width: u32,
        exp_height: u32,
        profile: Profile,
    ) -> Result<(Vec<u16>, u32, u32), CodecError> {
        let header = self.read_header()?;
        let frame = header.frame;

        if frame.width != exp_width as usize || frame.height != exp_height as usize {
            return Err(CodecError::DimensionMismatch {
                expected: (exp_width as usize) * (exp_height as usize),
                actual: frame.width * frame.height,
            });
        }

        // Grayscale-only contract for this entry; 3-component streams go
        // through `decode_image`.
        if frame.components != 1 {
            return Err(CodecError::Unsupported(format!(
                "decode() is grayscale-only; {}-component frames need decode_image()",
                frame.components
            )));
        }

        let pixels = match profile {
            Profile::LegacyGo => {
                let (max_val, _) = validate_params(&frame, &header.presets, header.scan.near)?;
                let mut pixels = vec![0u16; frame.width * frame.height];
                let scan_data = &self.data[self.pos..];
                legacy::decode_scan(
                    scan_data,
                    &mut pixels,
                    frame.width,
                    frame.height,
                    max_val,
                    header.scan.near,
                )?;
                pixels
            }
            Profile::T87 => self.decode_frame(header)?.samples,
        };

        Ok((pixels, exp_width, exp_height))
    }

    /// Decode every scan of the frame (the first SOS is already parsed) into
    /// component-planar samples, then check the trailing EOI. T.87 path only.
    ///
    /// LSE, DRI, COM and APPn segments may sit between scans (CharLS reads
    /// them too); an LSE there applies to the scans that follow.
    fn decode_frame(&mut self, header: Header) -> Result<DecodedImage, CodecError> {
        let Header {
            frame,
            mut presets,
            scan: first,
        } = header;
        let (w, h, nf) = (frame.width, frame.height, frame.components);
        // w*h*nf cannot overflow: read_sof caps w*h at MAX_PIXELS (1<<28) and
        // nf <= 3.
        let plane = w * h;
        let mut samples = vec![0u16; plane * nf];

        let mut scan = first;
        let mut done = 0usize;
        loop {
            if scan.near != first.near {
                // Legal (T.87 keeps NEAR per scan), but DecodedImage carries a
                // single bound.
                return Err(CodecError::Unsupported(format!(
                    "per-scan NEAR differs across components ({} then {})",
                    first.near, scan.near
                )));
            }
            let (max_val, reset) = validate_params(&frame, &presets, scan.near)?;
            let ctx = ContextModel::with_presets(
                max_val, scan.near, reset, presets.t1, presets.t2, presets.t3,
            );
            let ns = scan.components;
            let scan_start = self.pos;
            let consumed = decode_scan_t87(
                &self.data[scan_start..],
                &mut samples[done * plane..(done + ns) * plane],
                w,
                h,
                scan.ilv,
                ctx,
                scan_start,
            )?;
            self.pos = scan_start + consumed;
            done += ns;
            if done == nf {
                break;
            }
            self.skip_zero_padding();
            scan = self.read_segments(&mut Some(frame), &mut presets, done)?;
        }
        self.expect_eoi()?;

        Ok(DecodedImage {
            samples,
            width: w as u32,
            height: h as u32,
            bits_per_sample: frame.precision as u8,
            components: nf as u8,
            interleave_mode: first.ilv,
            near: first.near as u8,
        })
    }
}

/// SOFn markers of the other JPEG processes (T.81 B.1.1.3; DHT 0xFFC4, JPG
/// 0xFFC8 and DAC 0xFFCC are not frames) and JPEG-LS Part 2's SOF57
/// (T.870). Their scans cannot be decoded as T.87 baseline.
fn is_other_sof(marker: u16) -> bool {
    matches!(marker, 0xFFC0..=0xFFCF if !matches!(marker, 0xFFC4 | 0xFFC8 | 0xFFCC))
        || marker == 0xFFF9
}

/// Validate MAXVAL / NEAR / threshold ordering (T.87 C.2.4.1.1) for a scan and
/// return the effective `(max_val, reset)`. `near` is the scan's NEAR;
/// `presets` carries any LSE ID=1 overrides in force for that scan (a `None`
/// field means "use the spec default").
fn validate_params(
    frame: &FrameHeader,
    presets: &Presets,
    near: i32,
) -> Result<(i32, i32), CodecError> {
    let sof_max_val = (1i32 << frame.precision) - 1;
    let max_val = presets.maxval.unwrap_or(sof_max_val);

    // Effective MAXVAL must be positive and fit the SOF precision range; a
    // preset MAXVAL above (1<<precision)-1 would desync context arithmetic.
    if max_val <= 0 || max_val > sof_max_val {
        return Err(CodecError::InvalidData(format!(
            "LSE MAXVAL={max_val} outside 1..={sof_max_val} for precision {}",
            frame.precision
        )));
    }

    // NEAR validity depends on the effective MAXVAL.
    let near_max = (max_val / 2).min(255);
    if near < 0 || near > near_max {
        return Err(CodecError::InvalidParameter {
            name: "near",
            value: i64::from(near),
            allowed: "0..=min(255, MAXVAL/2)",
        });
    }

    // Thresholds must satisfy NEAR < T1 <= T2 <= T3 <= MAXVAL. Preset fields
    // left at their default fall back to the spec defaults (already valid).
    let (dt1, dt2, dt3) = ContextModel::default_thresholds(max_val, near);
    let t1 = presets.t1.unwrap_or(dt1);
    let t2 = presets.t2.unwrap_or(dt2);
    let t3 = presets.t3.unwrap_or(dt3);
    if !(near < t1 && t1 <= t2 && t2 <= t3 && t3 <= max_val) {
        return Err(CodecError::InvalidData(format!(
            "LSE thresholds violate NEAR<T1<=T2<=T3<=MAXVAL \
             (NEAR={near}, T1={t1}, T2={t2}, T3={t3}, MAXVAL={max_val})"
        )));
    }

    // T.87 C.2.4.1.1: RESET <= max(255, MAXVAL).
    let reset = presets.reset.unwrap_or(DEFAULT_RESET);
    if reset > max_val.max(255) {
        return Err(CodecError::InvalidData(format!(
            "LSE RESET={reset} exceeds max(255, MAXVAL={max_val})"
        )));
    }
    Ok((max_val, reset))
}

// ---------------------------------------------------------------------------
// T.87 conformant scan decode (regular + run mode)
// ---------------------------------------------------------------------------

/// Classify a bit-reader error hit mid-scan (before all samples are decoded).
fn scan_error(e: CodecError, offset: usize) -> CodecError {
    match e {
        CodecError::Marker(m) if (0xFFD0..=0xFFD7).contains(&m) => {
            CodecError::Unsupported("JPEG-LS restart markers (RSTn) in scan not supported".into())
        }
        CodecError::Marker(_) | CodecError::InvalidData(_) => CodecError::Truncated {
            offset,
            context: "entropy-coded scan",
        },
        other => other,
    }
}

/// Decode one entropy-coded scan (T.87 A.4-A.7) into `planes`, the scan's
/// `Ns` component planes back to back (each `width * height`); returns the
/// bytes consumed.
///
/// - ILV 0: one component, line after line.
/// - ILV 1 (line interleaved): each line of every component in turn. The
///   components share the context statistics but keep their own `RUNindex`.
/// - ILV 2 (sample interleaved): every component of a pixel before the next
///   pixel; see [`decode_line_sample`].
///
/// This is how CharLS (and the bundled CharLS 1.x in DCMTK) code the
/// interleaved modes, and it decodes the HP-encoded T.87 conformance streams.
pub(crate) fn decode_scan_t87(
    scan_data: &[u8],
    planes: &mut [u16],
    width: usize,
    height: usize,
    ilv: u8,
    mut ctx: ContextModel,
    base_offset: usize,
) -> Result<usize, CodecError> {
    let mut br = BitReader::new(scan_data);
    match ilv {
        0 => decode_planar(&mut br, &mut ctx, planes, width, base_offset)?,
        1 => decode_line_interleaved(&mut br, &mut ctx, planes, width, height, base_offset)?,
        _ => decode_sample_interleaved(&mut br, &mut ctx, planes, width, height, base_offset)?,
    }
    Ok(br.marker_pos())
}

/// ILV 0: one component, line after line. Kept out of line (as are the
/// interleaved loops) so each mode's hot loop is optimised on its own.
#[inline(never)]
fn decode_planar(
    br: &mut BitReader<'_>,
    ctx: &mut ContextModel,
    plane: &mut [u16],
    width: usize,
    base_offset: usize,
) -> Result<(), CodecError> {
    let mut prev = vec![0i32; width + 2];
    let mut cur = vec![0i32; width + 2];
    for row in plane.chunks_exact_mut(width) {
        decode_line(br, ctx, &mut prev, &mut cur, row, base_offset)?;
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(())
}

/// ILV 1: each line of every component in turn, sharing the context
/// statistics, each component with its own `RUNindex`.
#[inline(never)]
fn decode_line_interleaved(
    br: &mut BitReader<'_>,
    ctx: &mut ContextModel,
    planes: &mut [u16],
    width: usize,
    height: usize,
    base_offset: usize,
) -> Result<(), CodecError> {
    let plane = width * height;
    let ns = planes.len() / plane;
    let mut lines = vec![[vec![0i32; width + 2], vec![0i32; width + 2]]; ns];
    let mut run_index = [0usize; MAX_SCAN_COMPONENTS];
    for y in 0..height {
        for (c, [prev, cur]) in lines.iter_mut().enumerate() {
            ctx.run_index = run_index[c];
            let row = &mut planes[c * plane + y * width..][..width];
            decode_line(br, ctx, prev, cur, row, base_offset)?;
            run_index[c] = ctx.run_index;
            std::mem::swap(prev, cur);
        }
    }
    Ok(())
}

/// ILV 2: every component of a pixel before the next pixel; see
/// [`decode_line_sample`].
#[inline(never)]
fn decode_sample_interleaved(
    br: &mut BitReader<'_>,
    ctx: &mut ContextModel,
    planes: &mut [u16],
    width: usize,
    height: usize,
    base_offset: usize,
) -> Result<(), CodecError> {
    let plane = width * height;
    let ns = planes.len() / plane;
    let mut prev = vec![vec![0i32; width + 2]; ns];
    let mut cur = prev.clone();
    for y in 0..height {
        decode_line_sample(br, ctx, &mut prev, &mut cur, width, base_offset)?;
        for (c, line) in cur.iter().enumerate() {
            let row = &mut planes[c * plane + y * width..][..width];
            for (dst, &v) in row.iter_mut().zip(&line[1..=width]) {
                *dst = v as u16;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(())
}

/// Decode one line of one component into `cur` (offset-1 layout, see
/// [`run_mode`]) and `out_row`, whose length is the line width. Sets the
/// line's edge slots (T.87 A.2.1) first.
#[inline(always)]
fn decode_line(
    br: &mut BitReader<'_>,
    ctx: &mut ContextModel,
    prev: &mut [i32],
    cur: &mut [i32],
    out_row: &mut [u16],
    base_offset: usize,
) -> Result<(), CodecError> {
    let width = out_row.len();
    let near = ctx.near;
    cur[0] = prev[1];
    prev[width + 1] = prev[width];

    let mut x = 0usize;
    while x < width {
        let ra = cur[x];
        let rb = prev[x + 1];
        let rc = prev[x];
        let rd = prev[x + 2];

        if (rd - rb).abs() <= near && (rb - rc).abs() <= near && (rc - ra).abs() <= near {
            let before = x;
            x = match run_mode::decode_run(br, ctx, cur, prev, x, width) {
                Ok(v) => v,
                Err(e) => return Err(scan_error(e, base_offset + br.pos())),
            };
            for (dst, slot) in out_row[before..x].iter_mut().zip(&cur[before + 1..x + 1]) {
                *dst = *slot as u16;
            }
            continue;
        }

        let recon = decode_regular(br, ctx, [ra, rb, rc, rd], base_offset)?;
        cur[x + 1] = recon;
        out_row[x] = recon as u16;
        x += 1;
    }
    Ok(())
}

/// Decode one sample-interleaved (ILV 2) line of every component: `prev` and
/// `cur` hold one offset-1 line buffer per component.
///
/// Each pixel takes the gradients of every component. Run mode applies only
/// when all of them are within NEAR (the pixel then joins a shared run, see
/// [`run_mode::decode_run_sample`]); otherwise each component is decoded in
/// turn in regular mode with its own context, which may be context 0.
fn decode_line_sample(
    br: &mut BitReader<'_>,
    ctx: &mut ContextModel,
    prev: &mut [Vec<i32>],
    cur: &mut [Vec<i32>],
    width: usize,
    base_offset: usize,
) -> Result<(), CodecError> {
    let near = ctx.near;
    for (above, line) in prev.iter_mut().zip(cur.iter_mut()) {
        line[0] = above[1];
        above[width + 1] = above[width];
    }

    let mut x = 0usize;
    while x < width {
        let flat = prev.iter().zip(cur.iter()).all(|(above, line)| {
            let (ra, rb, rc, rd) = (line[x], above[x + 1], above[x], above[x + 2]);
            (rd - rb).abs() <= near && (rb - rc).abs() <= near && (rc - ra).abs() <= near
        });
        if flat {
            x = match run_mode::decode_run_sample(br, ctx, cur, prev, x, width) {
                Ok(v) => v,
                Err(e) => return Err(scan_error(e, base_offset + br.pos())),
            };
            continue;
        }

        for (above, line) in prev.iter().zip(cur.iter_mut()) {
            let neighbours = [line[x], above[x + 1], above[x], above[x + 2]];
            line[x + 1] = decode_regular(br, ctx, neighbours, base_offset)?;
        }
        x += 1;
    }
    Ok(())
}

/// Decode one regular-mode sample (T.87 A.4-A.6) from its causal neighbours
/// `[Ra, Rb, Rc, Rd]` and return the reconstructed value.
#[inline(always)]
fn decode_regular(
    br: &mut BitReader<'_>,
    ctx: &mut ContextModel,
    [ra, rb, rc, rd]: [i32; 4],
    base_offset: usize,
) -> Result<i32, CodecError> {
    let (near, max_val, range) = (ctx.near, ctx.max_val, ctx.range);
    let (q, sign) = ctx.get_context_index(rd - rb, rb - rc, rc - ra);
    let k = ctx.compute_k(q);
    let mut px = predict_med(ra, rb, rc);
    px += sign * ctx.c[q];
    px = clamp(px, 0, max_val);

    let correction = if k == 0 && near == 0 && 2 * ctx.b[q] <= -ctx.n[q] {
        -1
    } else {
        0
    };
    let mapped = match br.read_limited_golomb(k, ctx.limit, ctx.qbpp) {
        Ok(v) => v,
        Err(e) => return Err(scan_error(e, base_offset + br.pos())),
    };
    let err_q = run_mode::unmap_error(mapped, correction);
    run_mode::check_err_in_range(err_q, range)?;
    ctx.update_stats(q, err_q);

    Ok(run_mode::fix_reconstructed(
        px + sign * err_q * (2 * near + 1),
        near,
        range,
        max_val,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_image, encode_scan_t87, encode_with_options};
    use crate::{EncodeOptions, Profile};

    fn roundtrip(pixels: &[u16], w: u32, h: u32) -> Vec<u16> {
        let mut buf = Vec::new();
        encode_with_options(pixels, w, h, &EncodeOptions::default(), &mut buf).expect("encode");
        let (decoded, dw, dh) = decode(&buf, w, h).expect("decode");
        assert_eq!((dw, dh), (w, h));
        decoded
    }

    #[test]
    fn roundtrip_small_gradient() {
        let (w, h) = (7u32, 5u32);
        let pixels: Vec<u16> = (0..w * h).map(|i| (i * 3 % 256) as u16).collect();
        assert_eq!(roundtrip(&pixels, w, h), pixels);
    }

    #[test]
    fn roundtrip_constant_forces_run() {
        let pixels = vec![777u16; 24 * 8];
        assert_eq!(roundtrip(&pixels, 24, 8), pixels);
    }

    #[test]
    fn roundtrip_1x1() {
        assert_eq!(roundtrip(&[42u16], 1, 1), vec![42u16]);
    }

    /// Hand-build a stream carrying a non-default LSE ID=1 preset wrapping our
    /// own scan bytes, and verify it decodes exactly.
    #[test]
    fn lse_id1_presets_honored() {
        let (w, h) = (16usize, 12usize);
        let pixels: Vec<u16> = (0..(w * h) as u32)
            .map(|i| ((i * 5) % 200) as u16)
            .collect();
        let (maxval, t1, t2, t3, reset) = (255i32, 5i32, 12i32, 40i32, 32i32);

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&[0xFF, 0xD8]); // SOI
                                              // SOF55
        out.extend_from_slice(&[0xFF, 0xF7]);
        out.extend_from_slice(&11u16.to_be_bytes());
        out.push(8); // precision
        out.extend_from_slice(&(h as u16).to_be_bytes());
        out.extend_from_slice(&(w as u16).to_be_bytes());
        out.extend_from_slice(&[1, 1, 0x11, 0x00]); // Nf + component spec
                                                    // LSE ID=1
        out.extend_from_slice(&[0xFF, 0xF8]);
        out.extend_from_slice(&13u16.to_be_bytes());
        out.push(1);
        for v in [maxval, t1, t2, t3, reset] {
            out.extend_from_slice(&(v as u16).to_be_bytes());
        }
        // SOS
        out.extend_from_slice(&[0xFF, 0xDA]);
        out.extend_from_slice(&8u16.to_be_bytes());
        out.extend_from_slice(&[1, 1, 0x00, 0, 0, 0]);
        // scan encoded with the same presets
        let ctx = ContextModel::with_presets(maxval, 0, reset, Some(t1), Some(t2), Some(t3));
        encode_scan_t87(&mut out, &pixels, w, h, ctx).unwrap();
        out.extend_from_slice(&[0xFF, 0xD9]); // EOI

        let (decoded, _, _) = decode(&out, w as u32, h as u32).expect("LSE decode");
        assert_eq!(decoded, pixels);
    }

    /// Build SOI + SOF55 + LSE(ID=1) + SOS carrying the given preset fields.
    /// Validation fires before scan decode, so no scan/EOI bytes are needed.
    #[allow(clippy::too_many_arguments)]
    fn lse_preset_stream(
        precision: u8,
        w: u16,
        h: u16,
        maxval: u16,
        t1: u16,
        t2: u16,
        t3: u16,
        reset: u16,
        near: u8,
    ) -> Vec<u8> {
        let mut out: Vec<u8> = vec![0xFF, 0xD8]; // SOI
        out.extend_from_slice(&[0xFF, 0xF7]); // SOF55
        out.extend_from_slice(&11u16.to_be_bytes());
        out.push(precision);
        out.extend_from_slice(&h.to_be_bytes());
        out.extend_from_slice(&w.to_be_bytes());
        out.extend_from_slice(&[1, 1, 0x11, 0x00]);
        out.extend_from_slice(&[0xFF, 0xF8]); // LSE ID=1
        out.extend_from_slice(&13u16.to_be_bytes());
        out.push(1);
        for v in [maxval, t1, t2, t3, reset] {
            out.extend_from_slice(&v.to_be_bytes());
        }
        out.extend_from_slice(&[0xFF, 0xDA]); // SOS
        out.extend_from_slice(&8u16.to_be_bytes());
        out.extend_from_slice(&[1, 1, 0x00, near, 0, 0]);
        out
    }

    #[test]
    fn sof_zero_dimension_rejected() {
        // height = 0, width = 4 -> reject before allocation.
        let mut out: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xF7];
        out.extend_from_slice(&11u16.to_be_bytes());
        out.push(8);
        out.extend_from_slice(&0u16.to_be_bytes()); // height
        out.extend_from_slice(&4u16.to_be_bytes()); // width
        out.extend_from_slice(&[1, 1, 0x11, 0x00]);
        assert!(matches!(
            decode(&out, 4, 0),
            Err(CodecError::InvalidData(_))
        ));
    }

    #[test]
    fn sof_huge_dimensions_rejected_not_aborted() {
        // 65535 x 65535 would demand ~8 GiB; the pixel cap must reject it as a
        // clean Err rather than abort or OOM.
        let mut out: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xF7];
        out.extend_from_slice(&11u16.to_be_bytes());
        out.push(8);
        out.extend_from_slice(&65535u16.to_be_bytes()); // height
        out.extend_from_slice(&65535u16.to_be_bytes()); // width
        out.extend_from_slice(&[1, 1, 0x11, 0x00]);
        assert!(matches!(
            decode(&out, 65535, 65535),
            Err(CodecError::InvalidParameter {
                name: "width*height",
                ..
            })
        ));
    }

    #[test]
    fn lse_maxval_exceeds_precision_rejected() {
        // precision 8 -> MAXVAL must be <= 255; a preset MAXVAL of 65535 fails.
        let out = lse_preset_stream(8, 4, 4, 65535, 0, 0, 0, 64, 0);
        assert!(matches!(
            decode(&out, 4, 4),
            Err(CodecError::InvalidData(_))
        ));
    }

    #[test]
    fn lse_threshold_ordering_rejected() {
        // T1 > T2 violates NEAR < T1 <= T2 <= T3 <= MAXVAL.
        let out = lse_preset_stream(8, 4, 4, 0, 100, 50, 200, 64, 0);
        assert!(matches!(
            decode(&out, 4, 4),
            Err(CodecError::InvalidData(_))
        ));
    }

    #[test]
    fn lse_t3_exceeds_maxval_rejected() {
        // Explicit MAXVAL 200, but T3 = 250 > MAXVAL.
        let out = lse_preset_stream(8, 4, 4, 200, 5, 12, 250, 64, 0);
        assert!(matches!(
            decode(&out, 4, 4),
            Err(CodecError::InvalidData(_))
        ));
    }

    #[test]
    fn lse_reset_zero_means_default() {
        // T.87 C.2.4.1.1 (and CharLS): a RESET field of 0 selects the default
        // (64), as a zero MAXVAL/T1/T2/T3 does. The scan is coded with RESET
        // 64 and is large enough for contexts to reach it.
        let (w, h) = (64usize, 48usize);
        let pixels: Vec<u16> = (0..(w * h) as u32)
            .map(|i| ((i * 37 + (i / 64) * 11) % 251) as u16)
            .collect();
        let mut out = lse_preset_stream(8, w as u16, h as u16, 0, 0, 0, 0, 0, 0);
        let ctx = ContextModel::new(255, 0, DEFAULT_RESET);
        encode_scan_t87(&mut out, &pixels, w, h, ctx).unwrap();
        out.extend_from_slice(&[0xFF, 0xD9]);
        let (decoded, _, _) = decode(&out, w as u32, h as u32).expect("RESET 0 = default");
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn lse_reset_out_of_range_rejected() {
        // An explicit RESET of 1 or 2 is below the T.87 minimum; 256 exceeds
        // max(255, MAXVAL) for 8-bit samples.
        for reset in [1u16, 2, 256] {
            let out = lse_preset_stream(8, 4, 4, 0, 0, 0, 0, reset, 0);
            assert!(
                matches!(decode(&out, 4, 4), Err(CodecError::InvalidData(_))),
                "RESET {reset}"
            );
        }
    }

    #[test]
    fn lse_id2_unsupported() {
        // SOI, SOF55, LSE ID=2 -> Unsupported.
        let mut out: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xF7];
        out.extend_from_slice(&11u16.to_be_bytes());
        out.push(8);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&[1, 1, 0x11, 0x00]);
        out.extend_from_slice(&[0xFF, 0xF8]);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.push(2); // ID=2
        out.push(0);
        assert!(matches!(
            decode(&out, 4, 4),
            Err(CodecError::Unsupported(_))
        ));
    }

    #[test]
    fn dri_zero_interval_is_tolerated() {
        // A benign DRI segment (restart interval 0) before the scan must not
        // fail an otherwise decodable stream — CharLS accepts it. (A non-zero
        // interval that actually emits RSTn markers is still rejected by the
        // scan decoder.)
        let (w, h) = (4u32, 4u32);
        let pixels: Vec<u16> = (0..16).map(|i| i * 4).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, w, h, &EncodeOptions::default(), &mut buf).unwrap();
        let sos = buf
            .windows(2)
            .position(|p| p == [0xFF, 0xDA])
            .expect("SOS present");
        let mut with_dri = buf[..sos].to_vec();
        with_dri.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04, 0x00, 0x00]); // DRI, interval 0
        with_dri.extend_from_slice(&buf[sos..]);
        let (decoded, _, _) = decode(&with_dri, w, h).expect("benign DRI should decode");
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn fill_bytes_before_marker_tolerated() {
        // 0xFF fill bytes may precede any marker (T.81 B.1.1.2); they must be
        // skipped, not mistaken for a marker.
        let (w, h) = (4u32, 4u32);
        let pixels: Vec<u16> = (0..16).map(|i| i * 4).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, w, h, &EncodeOptions::default(), &mut buf).unwrap();
        // Two 0xFF fill bytes right after SOI, before SOF55.
        let mut filled = buf[..2].to_vec();
        filled.extend_from_slice(&[0xFF, 0xFF]);
        filled.extend_from_slice(&buf[2..]);
        let (decoded, _, _) = decode(&filled, w, h).expect("fill bytes should be skipped");
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn fill_bytes_before_eoi_tolerated() {
        // The EOI after the last scan may be preceded by 0xFF fill bytes too:
        // DCMTK's default padding of an odd-length stream is `FF FF D9`.
        let (w, h) = (4u32, 4u32);
        let pixels: Vec<u16> = (0..16).map(|i| i * 4).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, w, h, &EncodeOptions::default(), &mut buf).unwrap();
        let (scan, eoi) = buf.split_at(buf.len() - 2);
        assert_eq!(eoi, [0xFF, 0xD9]);
        for fill in 1..=3 {
            let mut filled = scan.to_vec();
            filled.extend_from_slice(&vec![0xFF; fill]);
            filled.extend_from_slice(eoi);
            let (decoded, _, _) = decode(&filled, w, h).expect("fill bytes should be skipped");
            assert_eq!(decoded, pixels, "{fill} fill byte(s), decode");
            let image = decode_image(&filled).expect("fill bytes should be skipped");
            assert_eq!(image.samples, pixels, "{fill} fill byte(s), decode_image");
        }
    }

    #[test]
    fn fill_bytes_without_eoi_rejected() {
        // A run of 0xFF fill bytes that ends the data, or that ends in a
        // marker other than EOI, is still not an EOI.
        let (w, h) = (4u32, 4u32);
        let pixels: Vec<u16> = (0..16).map(|i| i * 4).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, w, h, &EncodeOptions::default(), &mut buf).unwrap();
        let scan = &buf[..buf.len() - 2];
        for tail in [&[0xFF][..], &[0xFF, 0xFF], &[0xFF, 0xFF, 0xD8]] {
            let mut bad = scan.to_vec();
            bad.extend_from_slice(tail);
            assert!(decode(&bad, w, h).is_err(), "tail {tail:02X?}");
            assert!(decode_image(&bad).is_err(), "tail {tail:02X?}");
        }
    }

    fn encode_mono_4x4() -> (Vec<u8>, Vec<u16>) {
        let pixels: Vec<u16> = (0..16).map(|i| i * 4).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, 4, 4, &EncodeOptions::default(), &mut buf).unwrap();
        (buf, pixels)
    }

    #[test]
    fn sos_mapping_table_and_point_transform_rejected() {
        // SOS: FF DA | Ls(2) | Ns | Csj | mapping | NEAR | ILV | Ah/Al.
        let (buf, _) = encode_mono_4x4();
        let sos = buf.windows(2).position(|p| p == [0xFF, 0xDA]).unwrap();
        let mut mapping = buf.clone();
        mapping[sos + 6] = 1; // non-zero mapping-table selector
        assert!(matches!(
            decode(&mapping, 4, 4),
            Err(CodecError::Unsupported(_))
        ));
        let mut point = buf.clone();
        point[sos + 9] = 3; // non-zero point transform Al
        assert!(matches!(
            decode(&point, 4, 4),
            Err(CodecError::InvalidData(_))
        ));
        // Ah (the high nibble) has no meaning in JPEG-LS and is ignored.
        let mut ah = buf.clone();
        ah[sos + 9] = 0x30;
        assert!(decode(&ah, 4, 4).is_ok());
    }

    #[test]
    fn sos_component_count_is_validated_against_the_frame() {
        // An impossible Ns (0, > 4, or more than the frame's Nf) is malformed;
        // a legal multi-component scan (ILV 1/2) is merely unsupported.
        let (mono, _) = encode_mono_4x4();
        let sos = mono.windows(2).position(|p| p == [0xFF, 0xDA]).unwrap();
        for ns in [0u8, 2, 5, 7] {
            let mut bad = mono.clone();
            bad[sos + 4] = ns;
            match decode(&bad, 4, 4) {
                Err(CodecError::InvalidData(m)) => assert!(m.contains("Ns = "), "{m}"),
                other => panic!("mono Ns = {ns}: {other:?}"),
            }
        }

        let planes: Vec<Vec<u16>> = (0..3).map(|c| vec![c * 40; 16]).collect();
        let rgb = build_planar_rgb(&planes, 4, 4, 8);
        let sos = rgb.windows(2).position(|p| p == [0xFF, 0xDA]).unwrap();
        // Ns = 3 is legal for Nf = 3 (an interleaved scan), but this SOS
        // segment only has room for one component spec.
        let mut short = rgb.clone();
        short[sos + 4] = 3;
        match decode_image(&short) {
            Err(CodecError::InvalidData(m)) => assert!(m.contains("length"), "{m}"),
            other => panic!("short SOS: {other:?}"),
        }
        let mut too_many = rgb.clone();
        too_many[sos + 4] = 4;
        assert!(matches!(
            decode_image(&too_many),
            Err(CodecError::InvalidData(_))
        ));
    }

    #[test]
    fn sof_subsampling_rejected_and_tq_ignored() {
        // SOF55: FF F7 | Ls(2) | P | H(2) | W(2) | Nf | [Ci, sampling, Tq]...
        let (buf, pixels) = encode_mono_4x4();
        let sof = buf.windows(2).position(|p| p == [0xFF, 0xF7]).unwrap();
        let mut sub = buf.clone();
        sub[sof + 11] = 0x22; // 2:2 subsampling, not 1:1
        assert!(matches!(
            decode(&sub, 4, 4),
            Err(CodecError::Unsupported(_))
        ));
        // Tq is reserved in JPEG-LS (mapping tables are chosen per scan, in
        // SOS). CharLS ignores it; so do we.
        let mut tq = buf.clone();
        tq[sof + 12] = 2;
        assert_eq!(decode(&tq, 4, 4).expect("Tq ignored").0, pixels);
    }

    /// SOI + SOF55 (8-bit 4x4, `nf` components) + one SOS with `ns` component
    /// specs and the given ILV and Ah/Al byte. Header checks fire before any
    /// scan data is read, so none is appended.
    fn sos_header_stream(nf: u8, ns: u8, ilv: u8, al_ah: u8) -> Vec<u8> {
        let mut out: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xF7];
        out.extend_from_slice(&(8 + 3 * u16::from(nf)).to_be_bytes());
        out.push(8);
        out.extend_from_slice(&4u16.to_be_bytes());
        out.extend_from_slice(&4u16.to_be_bytes());
        out.push(nf);
        for c in 1..=nf {
            out.extend_from_slice(&[c, 0x11, 0x00]);
        }
        out.extend_from_slice(&[0xFF, 0xDA]);
        out.extend_from_slice(&(6 + 2 * u16::from(ns)).to_be_bytes());
        out.push(ns);
        for c in 1..=ns {
            out.extend_from_slice(&[c, 0x00]);
        }
        out.extend_from_slice(&[0, ilv, al_ah]);
        out
    }

    #[test]
    fn sos_interleave_mode_rules() {
        // One component per scan requires ILV 0, several require ILV 1 or 2,
        // and ILV values above 2 are reserved.
        for (nf, ns, ilv) in [(1, 1, 1), (1, 1, 2), (3, 1, 1), (3, 3, 0), (3, 3, 3)] {
            match decode_image(&sos_header_stream(nf, ns, ilv, 0)) {
                Err(CodecError::InvalidData(m)) => assert!(m.contains("ILV"), "{m}"),
                other => panic!("Nf {nf} Ns {ns} ILV {ilv}: {other:?}"),
            }
        }
        // A multi-component scan that leaves components for a later scan
        // (mixed interleave) is legal, but not supported.
        assert!(matches!(
            decode_image(&sos_header_stream(3, 2, 1, 0)),
            Err(CodecError::Unsupported(_))
        ));
    }

    #[test]
    fn other_sof_markers_unsupported_but_dht_skipped() {
        // SOF3 (lossless JPEG) and SOF57 (JPEG-LS Part 2) frames cannot be
        // decoded as T.87 baseline: say so, rather than failing later with a
        // missing SOF55.
        for code in [0xC3u8, 0xC0, 0xF9] {
            let s = [
                0xFF, 0xD8, 0xFF, code, 0x00, 0x0B, 8, 0, 4, 0, 4, 1, 1, 0x11, 0,
            ];
            match decode(&s, 4, 4) {
                Err(CodecError::Unsupported(m)) => assert!(m.contains("SOF55"), "{m}"),
                other => panic!("marker FF{code:02X}: {other:?}"),
            }
        }
        // DHT (0xFFC4) sits in the SOFn range but is not a frame; it is
        // skipped like any other unknown segment.
        let (buf, pixels) = encode_mono_4x4();
        let mut with_dht = buf[..2].to_vec();
        with_dht.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x04, 0x00, 0x00]);
        with_dht.extend_from_slice(&buf[2..]);
        assert_eq!(decode(&with_dht, 4, 4).expect("DHT skipped").0, pixels);
    }

    #[test]
    fn duplicate_sof55_rejected() {
        let (buf, _) = encode_mono_4x4();
        let sof = buf.windows(2).position(|p| p == [0xFF, 0xF7]).unwrap();
        let sof_len = 2 + usize::from(u16::from_be_bytes([buf[sof + 2], buf[sof + 3]]));
        let mut twice = buf[..sof + sof_len].to_vec();
        twice.extend_from_slice(&buf[sof..]);
        assert!(matches!(
            decode(&twice, 4, 4),
            Err(CodecError::InvalidData(_))
        ));
    }

    #[test]
    fn zero_padding_after_a_scan_tolerated() {
        // Some encoders leave 0x00 bytes between the scan data and the next
        // marker (CharLS #365); CharLS and the HP codec accept them.
        let (buf, pixels) = encode_mono_4x4();
        let (scan, eoi) = buf.split_at(buf.len() - 2);
        for pad in [&[0x00][..], &[0x00, 0x00], &[0x00, 0xFF]] {
            let mut padded = scan.to_vec();
            padded.extend_from_slice(pad);
            padded.extend_from_slice(eoi);
            assert_eq!(decode(&padded, 4, 4).expect("pad").0, pixels, "{pad:02X?}");
        }

        // ... and before the next scan's SOS.
        let planes: Vec<Vec<u16>> = (0..3u16)
            .map(|c| (0..16).map(|i| (i * 9 + c * 50) % 256).collect())
            .collect();
        let rgb = build_planar_rgb(&planes, 4, 4, 8);
        let second_sos = nth_marker(&rgb, 0xDA, 1);
        let mut padded = rgb[..second_sos].to_vec();
        padded.extend_from_slice(&[0x00, 0x00]);
        padded.extend_from_slice(&rgb[second_sos..]);
        assert_eq!(decode_image(&padded).expect("pad").samples, planes.concat());
    }

    /// Byte offset of the `n`th (0-based) `FF <code>` marker in `data`.
    fn nth_marker(data: &[u8], code: u8, n: usize) -> usize {
        data.windows(2)
            .enumerate()
            .filter(|(_, p)| *p == [0xFF, code])
            .nth(n)
            .expect("marker present")
            .0
    }

    #[test]
    fn segments_between_scans_are_read() {
        let planes: Vec<Vec<u16>> = (0..3u16)
            .map(|c| (0..16).map(|i| (i * 13 + c * 70) % 256).collect())
            .collect();
        let rgb = build_planar_rgb(&planes, 4, 4, 8);
        let second_sos = nth_marker(&rgb, 0xDA, 1);
        // COM, APP0 and a benign DRI between scan 1 and scan 2.
        let mut with_segments = rgb[..second_sos].to_vec();
        with_segments.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x04, b'h', b'i']);
        with_segments.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x02]);
        with_segments.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04, 0x00, 0x00]);
        with_segments.extend_from_slice(&rgb[second_sos..]);
        assert_eq!(
            decode_image(&with_segments).expect("segments").samples,
            planes.concat()
        );
    }

    #[test]
    fn lse_between_scans_applies_to_later_scans() {
        // Scan 1 uses the default presets; an LSE after it switches scans 2
        // and 3 to T1/T2/T3/RESET = 5/12/40/32 (CharLS reads presets per scan).
        let (w, h) = (16usize, 12usize);
        let planes: Vec<Vec<u16>> = (0..3u32)
            .map(|c| {
                (0..(w * h) as u32)
                    .map(|i| ((i * (5 + c) + c * 31) % 200) as u16)
                    .collect()
            })
            .collect();
        let mut out: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xF7, 0x00, 17, 8];
        out.extend_from_slice(&(h as u16).to_be_bytes());
        out.extend_from_slice(&(w as u16).to_be_bytes());
        out.push(3);
        for c in 1..=3u8 {
            out.extend_from_slice(&[c, 0x11, 0x00]);
        }
        for (c, plane) in planes.iter().enumerate() {
            let ctx = if c == 0 {
                ContextModel::new(255, 0, DEFAULT_RESET)
            } else {
                ContextModel::with_presets(255, 0, 32, Some(5), Some(12), Some(40))
            };
            if c == 1 {
                out.extend_from_slice(&[0xFF, 0xF8, 0x00, 13, 1]);
                for v in [0u16, 5, 12, 40, 32] {
                    out.extend_from_slice(&v.to_be_bytes());
                }
            }
            out.extend_from_slice(&[0xFF, 0xDA, 0x00, 8, 1, c as u8 + 1, 0, 0, 0, 0]);
            encode_scan_t87(&mut out, plane, w, h, ctx).unwrap();
        }
        out.extend_from_slice(&[0xFF, 0xD9]);
        assert_eq!(
            decode_image(&out).expect("per-scan presets").samples,
            planes.concat()
        );
    }

    #[test]
    fn near_differing_across_scans_unsupported() {
        // Legal in T.87, but DecodedImage carries a single NEAR.
        let planes: Vec<Vec<u16>> = (0..3).map(|c| vec![c * 40; 16]).collect();
        let rgb = build_planar_rgb(&planes, 4, 4, 8);
        let mut mixed = rgb.clone();
        mixed[nth_marker(&rgb, 0xDA, 1) + 7] = 1; // second scan's NEAR
        assert!(matches!(
            decode_image(&mixed),
            Err(CodecError::Unsupported(_))
        ));
    }

    #[test]
    fn hp_colour_transform_segment_detected() {
        // The HP/CharLS APP8 "mrfx" segment changes how samples read back; a
        // transform other than 0 must not be skipped silently.
        let (buf, pixels) = encode_mono_4x4();
        let with_app8 = |payload: &[u8]| {
            let mut s = buf[..2].to_vec();
            s.extend_from_slice(&[0xFF, 0xE8]);
            s.extend_from_slice(&(payload.len() as u16 + 2).to_be_bytes());
            s.extend_from_slice(payload);
            s.extend_from_slice(&buf[2..]);
            s
        };
        assert_eq!(decode(&with_app8(b"mrfx\x00"), 4, 4).unwrap().0, pixels);
        for id in [1u8, 3, 5] {
            assert!(matches!(
                decode(&with_app8(&[b'm', b'r', b'f', b'x', id]), 4, 4),
                Err(CodecError::Unsupported(_))
            ));
        }
        assert!(matches!(
            decode(&with_app8(b"mrfx\x09"), 4, 4),
            Err(CodecError::InvalidData(_))
        ));
        // Any other APP8 (here a SPIFF-like header) is skipped.
        let mut spiff = b"SPIFF\x00".to_vec();
        spiff.resize(30, 0);
        assert_eq!(decode(&with_app8(&spiff), 4, 4).unwrap().0, pixels);
    }

    /// Assemble a 3-component, planar (ILV 0) JPEG-LS stream: one SOS + scan
    /// per plane, sharing one SOF55 (Nf = 3). Lossless, default presets.
    fn build_planar_rgb(planes: &[Vec<u16>], w: usize, h: usize, precision: u8) -> Vec<u8> {
        let nf = planes.len();
        let max_val = (1i32 << precision) - 1;
        let mut out: Vec<u8> = vec![0xFF, 0xD8]; // SOI
                                                 // SOF55: Ls = 8 + 3*Nf.
        out.extend_from_slice(&[0xFF, 0xF7]);
        out.extend_from_slice(&((8 + 3 * nf) as u16).to_be_bytes());
        out.push(precision);
        out.extend_from_slice(&(h as u16).to_be_bytes());
        out.extend_from_slice(&(w as u16).to_be_bytes());
        out.push(nf as u8);
        for c in 0..nf {
            out.extend_from_slice(&[(c as u8) + 1, 0x11, 0x00]); // id, sampling 1:1, Tq 0
        }
        // One SOS + entropy scan per component (ILV 0).
        for (c, plane) in planes.iter().enumerate() {
            out.extend_from_slice(&[0xFF, 0xDA]);
            out.extend_from_slice(&8u16.to_be_bytes()); // Ls = 8 for Ns = 1
            out.push(1); // Ns = 1
            out.extend_from_slice(&[(c as u8) + 1, 0x00]); // Csj, mapping table 0
            out.extend_from_slice(&[0x00, 0x00, 0x00]); // NEAR = 0, ILV = 0, point transform 0
            let ctx = ContextModel::new(max_val, 0, DEFAULT_RESET);
            encode_scan_t87(&mut out, plane, w, h, ctx).unwrap();
        }
        out.extend_from_slice(&[0xFF, 0xD9]); // EOI
        out
    }

    #[test]
    fn decode_is_grayscale_only() {
        // A well-formed 3-component stream is rejected by the grayscale-only
        // decode() entry; decode_image() handles it (next test).
        let plane: Vec<u16> = (0..16).collect();
        let s = build_planar_rgb(&[plane.clone(), plane.clone(), plane], 4, 4, 8);
        assert!(matches!(decode(&s, 4, 4), Err(CodecError::Unsupported(_))));
    }

    #[test]
    fn decode_image_rgb_planar_roundtrip() {
        let (w, h) = (5usize, 3usize);
        let n = (w * h) as u16;
        let r: Vec<u16> = (0..n).map(|i| i * 3 % 256).collect();
        let g: Vec<u16> = (0..n).map(|i| (i * 7 + 1) % 256).collect();
        let b: Vec<u16> = (0..n).map(|i| 200u16.wrapping_sub(i * 5) % 256).collect();
        let s = build_planar_rgb(&[r.clone(), g.clone(), b.clone()], w, h, 8);
        let img = decode_image(&s).expect("decode_image");
        assert_eq!((img.width, img.height), (w as u32, h as u32));
        assert_eq!(img.components, 3);
        assert_eq!(img.interleave_mode, 0);
        assert_eq!((img.near, img.bits_per_sample), (0, 8));
        // Component-planar layout: R plane, then G, then B.
        let mut expected = r;
        expected.extend_from_slice(&g);
        expected.extend_from_slice(&b);
        assert_eq!(img.samples, expected);
    }

    #[test]
    fn decode_image_mono_matches_source() {
        let (w, h) = (7u32, 5u32);
        let pixels: Vec<u16> = (0..w * h).map(|i| (i * 3 % 256) as u16).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, w, h, &EncodeOptions::default(), &mut buf).unwrap();
        let img = decode_image(&buf).expect("decode_image mono");
        assert_eq!(img.components, 1);
        assert_eq!((img.width, img.height), (w, h));
        assert_eq!(img.samples, pixels);
    }

    #[test]
    fn encode_image_roundtrips_through_decode_image() {
        // Mono.
        let mono: Vec<u16> = (0..35u16).map(|i| i * 3 % 256).collect();
        let mut buf = Vec::new();
        encode_image(&mono, 7, 5, 1, &EncodeOptions::default(), &mut buf).unwrap();
        let img = decode_image(&buf).unwrap();
        assert_eq!((img.components, img.width, img.height), (1, 7, 5));
        assert_eq!(img.samples, mono);

        // 3-component planar (ILV 0).
        let (w, h) = (5u32, 3u32);
        let n = (w * h) as u16;
        let mut planar: Vec<u16> = Vec::new();
        planar.extend((0..n).map(|i| i * 3 % 256));
        planar.extend((0..n).map(|i| (i * 7 + 1) % 256));
        planar.extend((0..n).map(|i| 200u16.wrapping_sub(i * 5) % 256));
        let mut buf = Vec::new();
        encode_image(&planar, w, h, 3, &EncodeOptions::default(), &mut buf).unwrap();
        let img = decode_image(&buf).unwrap();
        assert_eq!((img.components, img.interleave_mode), (3, 0));
        assert_eq!(img.samples, planar);
    }

    #[test]
    fn precision_out_of_range_rejected() {
        for bad in [0u8, 1, 17] {
            let mut out: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xF7];
            out.extend_from_slice(&11u16.to_be_bytes());
            out.push(bad);
            out.extend_from_slice(&4u16.to_be_bytes());
            out.extend_from_slice(&4u16.to_be_bytes());
            out.extend_from_slice(&[1, 1, 0x11, 0x00]);
            assert!(
                matches!(
                    decode(&out, 4, 4),
                    Err(CodecError::InvalidParameter {
                        name: "precision",
                        ..
                    })
                ),
                "precision {bad} should be rejected"
            );
        }
    }

    #[test]
    fn truncated_scan_is_error_not_panic() {
        let pixels: Vec<u16> = (0..64).map(|i| (i * 4) as u16).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, 8, 8, &EncodeOptions::default(), &mut buf).unwrap();
        // Chop off the tail (scan + EOI) mid-stream.
        let truncated = &buf[..buf.len() - 3];
        let err = decode(truncated, 8, 8).unwrap_err();
        assert!(matches!(
            err,
            CodecError::Truncated { .. } | CodecError::InvalidData(_)
        ));
    }

    #[test]
    fn missing_eoi_rejected() {
        let pixels: Vec<u16> = (0..64).map(|i| (i * 4) as u16).collect();
        let mut buf = Vec::new();
        encode_with_options(&pixels, 8, 8, &EncodeOptions::default(), &mut buf).unwrap();
        // Replace the trailing EOI bytes with garbage.
        let n = buf.len();
        buf[n - 2] = 0x12;
        buf[n - 1] = 0x34;
        assert!(decode(&buf, 8, 8).is_err());
    }

    #[test]
    fn decode_rejects_bad_soi() {
        assert!(decode(&[0x00, 0x00, 0x00, 0x00], 1, 1).is_err());
    }

    #[test]
    fn t87_differs_from_legacy_bytes() {
        let pixels: Vec<u16> = (0..64).map(|i| (i * 4 % 256) as u16).collect();
        let mut t87 = Vec::new();
        encode_with_options(&pixels, 8, 8, &EncodeOptions::default(), &mut t87).unwrap();
        let mut legacy = Vec::new();
        let opts = EncodeOptions {
            near: 0,
            profile: Profile::LegacyGo,
            precision: None,
        };
        encode_with_options(&pixels, 8, 8, &opts, &mut legacy).unwrap();
        assert_ne!(t87, legacy, "default profile must differ from legacy bytes");
        // And each decodes under its own profile.
        let d1 = decode_with_options(&t87, 8, 8, &DecodeOptions::default())
            .unwrap()
            .0;
        let d2 = decode_with_options(
            &legacy,
            8,
            8,
            &DecodeOptions {
                profile: Profile::LegacyGo,
            },
        )
        .unwrap()
        .0;
        assert_eq!(d1, pixels);
        assert_eq!(d2, pixels);
    }
}
