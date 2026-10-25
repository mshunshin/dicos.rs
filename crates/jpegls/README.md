# JPEG-LS Codec

Pure Rust implementation of JPEG-LS (ITU-T T.87 / ISO/IEC 14495-1) encoder and decoder.

## Conformance scope

This crate emits **conformant ITU-T T.87 streams**. Supported: LSE ID=1 presets
(MAXVAL/T1/T2/T3/RESET) honored on decode, run mode, near-lossless, sample
precision `2..=16`, for:

- **Single-component** images (`Nf = 1`) — `encode` / `decode`, and
  `encode_image` / `decode_image`.
- **Multi-component** images (1 to 255 components, e.g. RGB, RGBA) —
  `decode_image`, in all three interleave modes: one scan per component
  (`ILV = 0`), line-interleaved scans (`ILV = 1`, DCMTK's default for colour)
  or sample-interleaved scans (`ILV = 2`), up to four components per scan and
  in any mix across the frame. Scans are placed by their component selectors
  (falling back to frame order when those are unusable, as CharLS ignores
  them). `encode_image` writes `ILV = 0`, for any component count.

Restart markers (`DRI` + `RSTm`) are decoded, every interval starting from
fresh statistics as CharLS does, and so are HP colour transforms HP1–HP3
(APP8 `"mrfx"`; three lossless 8- or 16-bit components in an interleaved scan,
CharLS 3.0's rule).

Rejected as `Unsupported`: LSE mapping tables, component subsampling, NEAR
differing between scans, and the HP lossy / matrix transforms (4, 5). Rejected
as `InvalidData`: a non-zero point transform, an impossible scan component
count, an interleave mode that does not fit it, an HP transform where it cannot
apply, and a missing or out-of-sequence `RSTm`.

Interop is verified bit-exactly against the **ITU-T T.87 conformance**
streams (HP encoder), **CharLS** test streams and CharLS-encoded images
(four components, SPIFF headers, presets), and **DCMTK** (`dcmcjpls`) output
— see `tests/fixtures/`.

### Compatibility notes

Like CharLS, the decoder accepts these deviations found in real files:

- `0x00` bytes, or `0xFF` fill bytes, between a scan and the next marker
  (DCMTK pads odd-length streams as `FF FF D9`; CharLS #365 cites `00 FF D9`
  in DICOM samples);
- LSE, COM, APPn and DRI segments between the scans of a planar image (an LSE
  there applies to the scans after it);
- an LSE `RESET` of 0, meaning the default (64), as for the other fields;
- a non-zero SOF `Tq` or SOS `Ah`, both of which JPEG-LS leaves unused.

APP8 segments other than `"mrfx"`, such as CharLS's SPIFF header, are
skipped.

Two CharLS codings differ from T.87, and files using them exist. A stream
that fails under T.87 is decoded again under each one its header allows; a
stream that decodes under T.87 is never reinterpreted:

- **CharLS 1.x default thresholds** (the copy bundled in DCMTK and others):
  for MAXVAL < 128 it used the MAXVAL >= 128 formula (CharLS #51) and wrote no
  LSE. DCMTK's `dcmcjpls +pc` produces such 6- and 7-bit streams.
- **CharLS's LSE MAXVAL**: CharLS derives `RANGE`, `LIMIT` and `qbpp` from the
  sample precision and uses an LSE `MAXVAL` only for the default thresholds,
  where T.87 derives all of them from `MAXVAL`. Any CharLS stream with an LSE
  `MAXVAL` below `2^P - 1` is coded this way. Decoded as CharLS does, a
  near-lossless sample may exceed `MAXVAL` by up to `NEAR`; a reading with any
  sample beyond that is rejected, since no CharLS encoder produces one.

Where CharLS departs from T.87 in a way that cannot be told apart, the decoder
follows T.87: a `RESET` above 255 is used as written (CharLS truncates it to 8
bits; DCMTK refuses to write one).

A caution for producers rather than a decoder rule: CharLS releases built before
commit 3790e10 (April 2023; e.g. the CharLS 2.4 inside pillow-jpls 1.3.2) are
not lossless for **four-component, sample-interleaved** images — a pixel that
differs from its left neighbour only in the 4th component (e.g. alpha) is coded
as part of a run, and that change is lost. Every decoder, this one included,
reproduces the stream as written.

### Profiles

- `Profile::T87` (default) — ITU-T T.87 conformant output.
- `Profile::LegacyGo` — reproduces the frozen 1.0.0 / Go-compatible bytes
  (T.81-style `0xFF00` stuffing, no run mode, uncapped Golomb); lossless only.

Near-lossless and explicit precision are set through `EncodeOptions` via
`encode_with_options`; `DecodeOptions` selects the decode profile.

## Features

- **Lossless and near-lossless** compression using the LOCO-I algorithm
- **Context-based adaptive prediction** with Median Edge Detection (MED)
- **Golomb-Rice entropy coding** with adaptive parameter selection
- **Run-length mode** for efficient coding of uniform regions
- **Grayscale and multi-component** (e.g. RGB, RGBA) images in every
  interleave mode, precision `2..=16`
- **DICOS/DICOM compatible**: Transfer Syntax `1.2.840.10008.1.2.4.80`
- **Pure Rust**: No external codec dependencies

## Transfer Syntax

DICOM UID: `1.2.840.10008.1.2.4.80` (JPEG-LS Lossless)

## Usage

Add the package under the plain import name:

```toml
[dependencies]
jpegls = { package = "pure_jpegls", version = "2.0" }
```

### Encoding

```rust
use jpegls::encode;

let pixels: Vec<u16> = (0..256).collect();
let mut compressed = Vec::new();
encode(&pixels, 16, 16, &mut compressed).unwrap();
```

### Decoding

```rust
use jpegls::decode;

let (decoded, width, height) = decode(&compressed, 16, 16).unwrap();
assert_eq!(decoded, pixels);
```

### Multi-component (e.g. RGB)

`encode_image` / `decode_image` take and return **component-planar** samples
(all of component 0, then component 1, …), whatever the interleave mode of the
stream. `decode_image` reads the geometry from the codestream and returns it
with the frame metadata, including the stream's `interleave_mode`.

```rust
use jpegls::{decode_image, encode_image, EncodeOptions};

let (w, h) = (16u32, 16u32);
let planar: Vec<u16> = (0..3 * 256).map(|i| (i % 256) as u16).collect(); // R, G, B planes
let mut rgb = Vec::new();
encode_image(&planar, w, h, 3, &EncodeOptions::default(), &mut rgb).unwrap();

let img = decode_image(&rgb).unwrap();
assert_eq!((img.width, img.height, img.components), (w, h, 3));
assert_eq!(img.samples, planar);
```

## JPEG-LS Algorithm

JPEG-LS uses the LOCO-I (LOw COmplexity LOssless COmpression for Images) algorithm:

1. **Edge detection** using local gradients to classify the coding context
2. **Context modeling** based on neighboring pixel relationships
3. **Adaptive prediction** using the Median Edge Detector (MED)
4. **Golomb-Rice coding** for prediction residuals with adaptive k parameter
5. **Run-length encoding** for uniform regions where all local gradients are zero

### Prediction Context

```
      c  b  d
      a  x
```

Where `x` is the current sample being encoded, and `a`, `b`, `c`, `d` are
neighboring samples used for prediction and context determination:

- **a** = sample immediately to the left
- **b** = sample immediately above
- **c** = sample diagonally above-left
- **d** = sample diagonally above-right

### Context Modeling

The encoder maintains 365 regular contexts plus 2 run-mode contexts, each
with adaptive A/B/C/N statistics that are updated after every sample. These
statistics drive both the bias cancellation and the Golomb-Rice parameter
selection, allowing the codec to adapt to local image characteristics.

## Supported Image Types

| Pixel Type | Precision | Components | Description |
|------------|-----------|------------|-------------|
| `u16` | `2..=16` bits | 1 (`Nf = 1`, `ILV = 0`) | single-component grayscale |
| `u16` | `2..=16` bits | 2-255 (`ILV = 0/1/2`, mixed) | multi-component, e.g. RGB, RGBA; decoded to planar samples (`encode_image` writes `ILV = 0`, any `Nf`) |

## JPEG-LS Stream Format

```
SOI   - Start of Image (0xFFD8)
SOF55 - Start of Frame, JPEG-LS (0xFFF7)
        Precision, height, width, components
LSE   - JPEG-LS Preset Parameters (0xFFF8, optional)
        Custom MAXVAL, T1, T2, T3, RESET thresholds
SOS   - Start of Scan (0xFFDA)
        Component selector(s), Near parameter (0 = lossless), ILV
[entropy-coded image data]
(multi-component: SOS + scan repeat until every component is covered —
 one component per ILV = 0 scan, up to four per ILV = 1/2 scan)
EOI   - End of Image (0xFFD9)
```

## Architecture

```
jpegls/src/
  lib.rs        # Public API (encode, decode, encode_image, decode_image, TRANSFER_SYNTAX_UID)
  encode.rs     # Encoder: prediction + context + Golomb-Rice output
  decode.rs     # Decoder: Golomb-Rice input + context + reconstruction
  predictor.rs  # MED predictor and gradient computation
  context.rs    # Context modeling with adaptive A/B/C/N statistics
  run_mode.rs   # Run-length mode for uniform regions
  bitstream.rs  # Bit-level I/O utilities
  error.rs      # Error types
```

## References

- ITU-T Rec. T.87 | ISO/IEC 14495-1 (JPEG-LS baseline)
- M. Weinberger, G. Seroussi, G. Sapiro, "The LOCO-I Lossless Image Compression Algorithm:
  Principles and Standardization into JPEG-LS", IEEE Trans. Image Processing, 2000
- DICOM Transfer Syntax: `1.2.840.10008.1.2.4.80` (JPEG-LS Lossless)
