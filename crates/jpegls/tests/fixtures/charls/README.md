# Vendored JPEG-LS reference streams

Copied unchanged from [CharLS](https://github.com/team-charls/charls) commit
`24b4f99d4c976fe5c9ec35b2bec54994f67ea8f7` (2026-10-24): `test/data/conformance/`
and `test/data/`. Licensing and copyright are in `LICENSE` (BSD 3-Clause).

`tests/fixtures.rs` decodes each stream and checks the expected outcome:

| Stream | Coding | Expected |
|---|---|---|
| `t8c0e0` / `t8c1e0` / `t8c2e0` | 8-bit RGB, ILV 0 / 1 / 2, lossless (HP encoder) | `test8.ppm` exactly |
| `t8c0e3` / `t8c1e3` / `t8c2e3` | the same, NEAR 3 | `test8.ppm` within 3 |
| `t16e0` / `t16e3` | 12-bit-in-16 grayscale, lossless / NEAR 3 | `test16.pgm` (exact / within 3) |
| `t8nde0` / `t8nde3` | grayscale, non-default T1=T2=T3=9, RESET=31 | `test8bs2.pgm` (exact / within 3) |
| `t8sse0` | line-interleaved with subsampled components | `Unsupported` |
| `banny` | 8-bit RGB, ILV 1, CharLS encoder | `banny.ppm` exactly |
| `banny-hp1/2/3` | the same with an HP colour transform (APP8 `mrfx`) | `Unsupported` |
| `8bit-monochrome-2x2` | 2x2 grayscale | decodes |
| `fuzzy-input-*`, `fuzzy_input_golomb_16`, `no_start_byte_after_encoded_scan` | fuzz-found malformed input | an error, never a panic |

`CONFORMANCE-README.md` is the original description of the conformance set.
