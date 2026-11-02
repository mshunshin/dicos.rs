#!/usr/bin/env python3
"""Regenerate the CharLS-encoded fixtures in this directory.

Encodes synthetic images with CharLS through the `pillow-jpls` Pillow plugin
and writes each codestream as `<stem>.jls`, with CharLS's own decode of it as
the exact expectation: `<stem>.pgm` / `.ppm`, or `.pam` (P7) for four
components. `tests/fixtures.rs` compares against these.

pillow-jpls ships wheels up to CPython 3.13, so run it with uv:

    uv run --no-project --python 3.12 --with pillow-jpls==1.3.2 --with numpy \\
        python -I crates/jpegls/tests/fixtures/charls-gen/generate.py

Stems: `{pattern}-{w}x{h}-{bits}-{gray|rgb|rgba}-ilv{N}[-near{N}][-spiff]
[-presets][-mx{MAXVAL}]`. `mx` streams carry an LSE MAXVAL below 2^P - 1,
which CharLS codes differently from T.87 (it derives RANGE, LIMIT and qbpp
from P); see the crate README.
"""

import importlib.metadata
import io
import random
from pathlib import Path

import numpy as np
import pillow_jpls  # noqa: F401  (registers the JPEG-LS plugin)
from PIL import Image

OUT = Path(__file__).resolve().parent
INTERLEAVE = {0: "none", 1: "line", 2: "sample"}

# Streams whose lossless encode is not lossless, through a CharLS encoder bug:
# before CharLS commit 3790e10 (2023-04-18) `quad` inherited `triplet`'s
# operator==, so in lossless sample-interleaved (ILV 2) four-component coding a
# pixel differing from Ra only in the 4th component continues the run, and that
# component's change is lost. The stream itself carries the wrong values, so
# every decoder (CharLS's and ours) reproduces them.
QUAD_RUN_BUG = []


def pattern_values(pattern, w, h, comps, mx, seed):
    rnd = random.Random(seed)
    a = np.zeros((h, w, comps), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            for c in range(comps):
                if pattern == "dots":
                    # Flat field with single-component dots: in ILV 2 one
                    # component ends the run while the others keep Ra == Rb.
                    v = (mx * (c + 1)) // (comps + 1)
                    if (x * 7 + y * 13) % 23 == 0 and c == (x + y) % comps:
                        v = (v + mx // 3 + 1) % (mx + 1)
                elif y < h // 3:
                    v = ((x * mx) // max(w - 1, 1) + c * 37 + y) % (mx + 1)
                elif y < 2 * h // 3:
                    v = ((x // 9) % comps == c) * (mx * 3 // 4) + (mx // 8)
                else:
                    v = rnd.randrange(mx + 1)
                a[y, x, c] = v
    return a


def write_reference(path_stem, arr, maxval):
    h, w, comps = arr.shape
    wide = maxval > 255
    body = arr.astype(">u2").tobytes() if wide else arr.astype(np.uint8).tobytes()
    if comps in (1, 3):
        ext, header = ("pgm", b"P5") if comps == 1 else ("ppm", b"P6")
        header += f"\n{w} {h}\n{maxval}\n".encode()
    else:
        ext = "pam"
        header = f"P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH {comps}\nMAXVAL {maxval}\nENDHDR\n".encode()
    (OUT / f"{path_stem}.{ext}").write_bytes(header + body)


def case(pattern, w, h, kind, bits, ilv, seed, near=0, spiff=False, presets=None, maxval=None):
    comps = {"gray": 1, "rgb": 3, "rgba": 4}[kind]
    mx = maxval if maxval is not None else (1 << bits) - 1
    vals = pattern_values(pattern, w, h, comps, mx, seed)
    if bits > 8:
        assert comps == 1, "Pillow has no 16-bit colour mode"
        img = Image.fromarray(vals[:, :, 0].astype(np.uint16))
    elif comps == 1:
        img = Image.fromarray(vals[:, :, 0].astype(np.uint8))
    else:
        img = Image.fromarray(vals.astype(np.uint8), {3: "RGB", 4: "RGBA"}[comps])

    opts = {"interleave": INTERLEAVE[ilv], "near_lossless": near}
    if not spiff:
        opts["spiff"] = None
    if presets:
        opts.update(presets)
    if maxval is not None:
        opts["maxval"] = maxval
    buf = io.BytesIO()
    img.save(buf, format="JPEG-LS", **opts)
    data = buf.getvalue()

    decoded = np.asarray(Image.open(io.BytesIO(data)), dtype=np.int64)
    decoded = decoded.reshape(h, w, comps)
    lost = int(np.count_nonzero(decoded != vals)) if near == 0 else 0
    if lost:
        assert comps == 4 and ilv == 2, f"CharLS lossless round-trip differs ({pattern} {kind} ILV {ilv})"

    stem = f"{pattern}-{w}x{h}-{bits}-{kind}-ilv{ilv}"
    stem += f"-near{near}" if near else ""
    stem += "-spiff" if spiff else ""
    stem += "-presets" if presets else ""
    stem += f"-mx{maxval}" if maxval is not None else ""
    (OUT / f"{stem}.jls").write_bytes(data)
    write_reference(stem, decoded, (1 << bits) - 1)
    if lost:
        QUAD_RUN_BUG.append((stem, lost))
    return stem


def main():
    for old in OUT.glob("*.*"):
        if old.suffix in (".jls", ".pgm", ".ppm", ".pam"):
            old.unlink()

    stems = []
    seed = 0
    # Four components in every interleave mode (DCMTK cannot write these).
    for pattern in ("mixed", "dots"):
        for ilv in (0, 1, 2):
            for near in (0, 2):
                seed += 1
                stems.append(case(pattern, 41, 30, "rgba", 8, ilv, seed, near=near))
    # CharLS's default SPIFF header (APP8) in front of the frame.
    for ilv in (0, 1, 2):
        seed += 1
        stems.append(case("mixed", 37, 24, "rgb", 8, ilv, seed, spiff=True))
    # Custom presets (RESET kept <= 255: CharLS truncates larger values).
    presets = {"t1": 30, "t2": 100, "t3": 400, "reset": 200}
    for near in (0, 1):
        seed += 1
        stems.append(case("mixed", 64, 48, "gray", 16, 0, seed, near=near, presets=presets))
    # An LSE MAXVAL below 2^P - 1 (CharLS codes it differently from T.87).
    for bits, mx, near in ((16, 4095, 0), (16, 4095, 2), (16, 4096, 0), (8, 200, 0), (8, 128, 1)):
        seed += 1
        stems.append(case("mixed", 64, 48, "gray", bits, 0, seed, near=near, maxval=mx))

    version = importlib.metadata.version("pillow-jpls")
    (OUT / "README.md").write_text(
        f"""# CharLS-encoded fixtures

Generated by `generate.py` with pillow-jpls {version} (CharLS). Do not edit by
hand; rerun the script (see its docstring for the `uv` command).

Each `<stem>.jls` was encoded by CharLS; `<stem>.pgm` / `.ppm` / `.pam` holds
CharLS's own decode of it, which our decoder must match exactly. Lossless
expectations were checked against the synthetic source.

- `rgba-*`: four components in each interleave mode (ILV 0 / 1 / 2).
- `-spiff`: CharLS's SPIFF header (APP8) precedes the frame.
- `-presets`: LSE T1/T2/T3/RESET = 30/100/400/200.
- `-mx{{MAXVAL}}`: an LSE MAXVAL below 2^P - 1. CharLS derives RANGE, LIMIT and
  qbpp from P rather than MAXVAL, unlike T.87, so a strictly T.87 decoder
  rejects these; this one decodes them by retrying under CharLS's rule.

The CharLS encoder in this pillow-jpls build is not lossless for four
components in sample-interleaved (ILV 2) mode: before CharLS commit 3790e10
(2023-04-18) a pixel differing from its left neighbour only in the 4th
component continued the run, dropping that change. These streams therefore
decode (in CharLS and here alike) to their expectation file, not to the
source image:
{chr(10).join(f"- `{s}` ({n} samples lost)" for s, n in QUAD_RUN_BUG) or "- (none)"}

{len(stems)} streams:
{chr(10).join(f"- `{s}`" for s in stems)}
"""
    )
    print(f"wrote {len(stems)} fixtures with pillow-jpls {version}")


if __name__ == "__main__":
    main()
