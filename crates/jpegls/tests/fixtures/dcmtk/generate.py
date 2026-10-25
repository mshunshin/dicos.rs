#!/usr/bin/env python3
"""Regenerate the DCMTK interop fixtures in this directory.

Needs DCMTK's `dcmcjpls` and `dcmdjpls` on PATH; uses only the standard
library. For each case it synthesizes a native (uncompressed) DICOM image,
JPEG-LS encodes it with `dcmcjpls`, extracts the codestream as `<stem>.jls`,
decodes it again with `dcmdjpls` (DCMTK's bundled CharLS) and writes that
output as `<stem>.pgm` / `<stem>.ppm` -- the bit-exact expectation that
`tests/fixtures.rs` compares against. Lossless expectations are asserted equal
to the source pixels before anything is written.

    python3 -I crates/jpegls/tests/fixtures/dcmtk/generate.py

Stems: `{pattern}-{w}x{h}-{bits}-{rgb|mono}-ilv{N}[-near{N}][-lse]`, where
`bits` is the stored depth of the source and `ilvN` is the interleave mode
DCMTK wrote (colour: `+il` -> 1, `+is` -> 2; monochrome is always 0).
"""

import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent

# (pattern, width, height, samples per pixel, bits stored, dcmcjpls flags, stem suffix)
CASES = []
for pattern, w, h, bits in [
    ("mixed", 37, 24, 8),
    ("dots", 64, 48, 8),
    ("flat", 300, 40, 8),
    ("mixed", 61, 40, 16),
    ("dots", 64, 48, 12),
]:
    for flag, ilv in [("+il", 1), ("+is", 2)]:
        CASES.append((pattern, w, h, 3, bits, [flag], f"ilv{ilv}"))
        CASES.append((pattern, w, h, 3, bits, [flag, "+en"], f"ilv{ilv}-near2"))
for pattern, w, h in [("mixed", 1, 17), ("mixed", 17, 1)]:
    for flag, ilv in [("+il", 1), ("+is", 2)]:
        CASES.append((pattern, w, h, 3, 8, [flag], f"ilv{ilv}"))
CASES.append(("mixed", 37, 24, 3, 8, ["+il", "+t1", "5", "+t2", "9", "+t3", "30", "+rs", "32"], "ilv1-lse"))
for w, h in [(33, 21), (40, 29)]:
    CASES.append(("mixed", w, h, 1, 16, [], "ilv0"))
    CASES.append(("mixed", w, h, 1, 16, ["+en"], "ilv0-near2"))


def samples(pattern, w, h, spp, bits, seed):
    """Source samples, pixel-interleaved, row-major."""
    mx = (1 << bits) - 1
    rnd = random.Random(seed)
    out = []
    for y in range(h):
        for x in range(w):
            for c in range(spp):
                if pattern == "flat":
                    v = (mx * (c + 1)) // 4 if x < w - 7 else (mx * (3 - c)) // 4
                elif pattern == "dots":
                    # A flat field with isolated single-component dots: in ILV 2
                    # the run is broken by one component while the others keep
                    # Ra == Rb, which exercises the sign rule of the
                    # sample-interleaved run interruption.
                    v = (mx * (c + 1)) // 4
                    if (x * 7 + y * 13) % 23 == 0 and c == (x + y) % spp:
                        v = (v + mx // 3 + 1) % (mx + 1)
                else:
                    # Gradient band, flat band with vertical edges, noise band.
                    if y < h // 3:
                        v = ((x * mx) // max(w - 1, 1) + c * 37 + y) % (mx + 1)
                    elif y < 2 * h // 3:
                        v = ((x // 9) % 3 == c) * (mx * 3 // 4) + (mx // 8)
                    else:
                        v = rnd.randrange(mx + 1)
                out.append(v)
    return out


def element(group, elem, vr, value):
    if isinstance(value, str):
        value = value.encode()
        if len(value) % 2:
            value += b"\0" if vr == b"UI" else b" "
    if vr in (b"OB", b"OW"):
        return struct.pack("<HH2sHI", group, elem, vr, 0, len(value)) + value
    return struct.pack("<HH2sH", group, elem, vr, len(value)) + value


def native_dicom(w, h, spp, bits, pixels):
    """A minimal explicit-VR little-endian Secondary Capture image."""
    allocated = 8 if bits <= 8 else 16
    if allocated == 16:
        data = struct.pack(f"<{len(pixels)}H", *pixels)
    else:
        data = bytes(pixels)
    if len(data) % 2:
        data += b"\0"
    us = lambda v: struct.pack("<H", v)
    sop_class, sop_instance = "1.2.840.10008.5.1.4.1.1.7", "2.25.1234567890"
    meta_body = (
        element(2, 1, b"OB", b"\0\1")
        + element(2, 2, b"UI", sop_class)
        + element(2, 3, b"UI", sop_instance)
        + element(2, 0x10, b"UI", "1.2.840.10008.1.2.1")
        + element(2, 0x12, b"UI", "1.2.3.4")
    )
    meta = element(2, 0, b"UL", struct.pack("<I", len(meta_body))) + meta_body
    dataset = (
        element(8, 0x16, b"UI", sop_class)
        + element(8, 0x18, b"UI", sop_instance)
        + element(8, 0x60, b"CS", "OT")
        + element(0x20, 0x0D, b"UI", "2.25.1")
        + element(0x20, 0x0E, b"UI", "2.25.2")
        + element(0x28, 0x02, b"US", us(spp))
        + element(0x28, 0x04, b"CS", "MONOCHROME2" if spp == 1 else "RGB")
        + (element(0x28, 0x06, b"US", us(0)) if spp > 1 else b"")
        + element(0x28, 0x10, b"US", us(h))
        + element(0x28, 0x11, b"US", us(w))
        + element(0x28, 0x100, b"US", us(allocated))
        + element(0x28, 0x101, b"US", us(bits))
        + element(0x28, 0x102, b"US", us(bits - 1))
        + element(0x28, 0x103, b"US", us(0))
        + element(0x7FE0, 0x10, b"OW" if allocated == 16 else b"OB", data)
    )
    return b"\0" * 128 + b"DICM" + meta + dataset


def first_fragment(path):
    """The first frame's codestream from encapsulated pixel data."""
    d = path.read_bytes()
    tag = b"\xe0\x7f\x10\x00OB\x00\x00\xff\xff\xff\xff"
    i = d.rindex(tag) + len(tag)
    items = []
    while True:
        group, elem, length = struct.unpack_from("<HHI", d, i)
        i += 8
        if (group, elem) == (0xFFFE, 0xE0DD):
            break
        items.append(d[i : i + length])
        i += length
    return items[1]  # items[0] is the basic offset table


def native_pixels(path, count, allocated):
    """Samples of a native (explicit VR LE) DICOM image written by dcmdjpls."""
    d = path.read_bytes()
    for vr in (b"OW", b"OB"):
        i = d.rfind(b"\xe0\x7f\x10\x00" + vr + b"\x00\x00")
        if i >= 0:
            break
    data = d[i + 12 :]
    if allocated == 16:
        return list(struct.unpack_from(f"<{count}H", data))
    return list(data[:count])


def sof_precision(codestream):
    i = codestream.index(b"\xff\xf7")
    return codestream[i + 4]


def write_pnm(path, w, h, spp, maxval, pixels):
    magic = b"P5" if spp == 1 else b"P6"
    header = magic + f"\n{w} {h}\n{maxval}\n".encode()
    if maxval > 255:
        body = struct.pack(f">{len(pixels)}H", *pixels)
    else:
        body = bytes(pixels)
    path.write_bytes(header + body)


def run(*cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    for tool in ("dcmcjpls", "dcmdjpls"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH (install DCMTK)")
    version = subprocess.run(
        ["dcmcjpls", "--version"], capture_output=True, text=True
    ).stdout.splitlines()[0]

    for old in list(OUT.glob("*.jls")) + list(OUT.glob("*.pgm")) + list(OUT.glob("*.ppm")):
        old.unlink()

    stems, fill_padded = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for seed, (pattern, w, h, spp, bits, flags, suffix) in enumerate(CASES):
            kind = "mono" if spp == 1 else "rgb"
            stem = f"{pattern}-{w}x{h}-{bits}-{kind}-{suffix}"
            allocated = 8 if bits <= 8 else 16
            source = samples(pattern, w, h, spp, bits, seed)

            src, enc, dec = tmp / "src.dcm", tmp / "enc.dcm", tmp / "dec.dcm"
            src.write_bytes(native_dicom(w, h, spp, bits, source))
            run("dcmcjpls", *flags, str(src), str(enc))
            codestream = first_fragment(enc)
            run("dcmdjpls", str(enc), str(dec))
            expected = native_pixels(dec, w * h * spp, allocated)

            if "+en" not in flags:
                assert expected == source, f"{stem}: dcmdjpls output differs from the source"
            precision = sof_precision(codestream)
            (OUT / f"{stem}.jls").write_bytes(codestream)
            write_pnm(OUT / f"{stem}.{'pgm' if spp == 1 else 'ppm'}", w, h, spp, (1 << precision) - 1, expected)
            stems.append(stem)
            if codestream.endswith(b"\xff\xff\xd9"):
                fill_padded.append(stem)

    assert fill_padded, "no fixture ends in DCMTK's extended EOI (FF FF D9)"
    readme = f"""# DCMTK interop fixtures

Generated by `generate.py` with **{version}** (`dcmcjpls` encodes,
`dcmdjpls` decodes the expectation). Do not edit by hand; rerun the script.

Each `<stem>.jls` is a single-frame JPEG-LS codestream as DCMTK stores it in
DICOM. `<stem>.pgm` / `<stem>.ppm` holds what `dcmdjpls` (DCMTK's bundled
CharLS) decodes from it, which our decoder must match exactly. For lossless
streams the script checks that this equals the synthesized source.

Colour streams are line-interleaved (`ilv1`, DCMTK's default `+il`) or
sample-interleaved (`ilv2`, `+is`); DCMTK has no planar option for colour.
`near2` streams use `+en` (NEAR 2). Images stored with more than 8 bits are
encoded with P = 16 and an LSE segment, as DCMTK always does above 12 bits.

These end in DCMTK's extended EOI padding (`FF FF D9`):
{chr(10).join(f"- `{s}`" for s in fill_padded)}
"""
    (OUT / "README.md").write_text(readme)
    print(f"wrote {len(stems)} fixtures ({len(fill_padded)} with FF FF D9) using {version}")


if __name__ == "__main__":
    main()
