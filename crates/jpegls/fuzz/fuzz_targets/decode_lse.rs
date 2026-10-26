#![no_main]

//! Fuzz target: decoding under LSE ID=1 coding-parameter presets.
//!
//! The encoder never writes an LSE segment, so no round-trip target reaches
//! the preset path, and blind byte fuzzing rarely assembles a valid SOF55 +
//! LSE + SOS header. This target builds that header from fuzz-chosen fields —
//! MAXVAL, T1–T3 and RESET either in a plausible range or as raw `u16`s, the
//! LSE before or after SOF55, an over-long LSE, an extra APPn and a DRI
//! segment — and feeds fuzz bytes as each component's scan. Any `Err` is an
//! acceptable rejection. A panic, or an `Ok` that breaks one of these, is a
//! finding:
//!
//! - the [`jpegls::DecodedImage`] contract, with no sample above the
//!   *effective* MAXVAL (the preset's, when it sets one) — plus its
//!   component's NEAR when that preset is below `2^P - 1`, where the
//!   CharLS-coding retry may apply;
//! - the frame and scan fields come back as written, each component's NEAR
//!   included — this and the sample bound only for the first component when
//!   the scan bytes hold an LSE or SOS segment of their own, which the
//!   decoder applies to the scans after it, as CharLS does, and the height
//!   only when no DNL in the scan bytes can stand in for the one written;
//! - the accepted preset obeys the rules the decoder documents: MAXVAL within
//!   the SOF precision, RESET 0 (the default) or `3..=max(255, MAXVAL)`,
//!   NEAR <= min(255, MAXVAL/2);
//! - a single-component stream decodes identically through `jpegls::decode`.
//!
//! Input layout (missing bytes read as 0):
//!
//! - byte 0: flags — bit 0 three components; bit 1 an APPn segment first;
//!   bit 2 a DRI segment; bit 3 the LSE after SOF55; bit 4 raw LSE fields;
//!   bit 5 an over-long LSE; bit 6 a NEAR per scan (`NEAR >> c` for
//!   component `c`, so the first scan's is the largest); bit 7 SOF55 height
//!   0, with a DNL segment after the first scan giving it;
//! - byte 1: precision `2 + (b % 15)`;
//! - bytes 2, 3: width and height, `1 + (b % 32)`;
//! - byte 4: NEAR — 0 unless the top bit is set, then the low 7 bits;
//! - bytes 5–14: MAXVAL, T1, T2, T3, RESET (`u16`, little-endian);
//! - bytes 15, 16: where to split the remainder between the component scans;
//! - the rest: scan bytes.

use jpegls_fuzz::{assert_decode_agrees, assert_image_contract, max_val, Input};
use libfuzzer_sys::fuzz_target;

fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&(payload.len() as u16 + 2).to_be_bytes());
    out.extend_from_slice(payload);
}

fuzz_target!(|data: &[u8]| {
    let mut input = Input::new(data);
    let flags = input.u8();
    let nf: u8 = if flags & 1 == 1 { 3 } else { 1 };
    let precision: u8 = 2 + input.u8() % 15;
    let width = 1 + u16::from(input.u8() % 32);
    let height = 1 + u16::from(input.u8() % 32);
    let b = input.u8();
    let near: u8 = if b & 0x80 == 0 { 0 } else { b & 0x7f };
    let component_near: Vec<u8> = (0..nf)
        .map(|c| if flags & 0x40 != 0 { near >> c } else { near })
        .collect();
    let raw = [
        input.u16(),
        input.u16(),
        input.u16(),
        input.u16(),
        input.u16(),
    ];
    let splits = [input.u8(), input.u8()];
    let body = input.rest();

    let sof_max = max_val(precision);
    let [maxval, t1, t2, t3, reset] = if flags & 0x10 != 0 {
        raw
    } else {
        // Plausible: MAXVAL within the precision (0 = default), thresholds
        // within MAXVAL (0 = default), RESET in 3..=max(255, MAXVAL).
        let maxval = (u32::from(raw[0]) % (sof_max + 1)) as u16;
        let m = if maxval == 0 {
            sof_max
        } else {
            u32::from(maxval)
        };
        let t = |r: u16| (u32::from(r) % (m + 1)) as u16;
        let reset = 3 + (u32::from(raw[4]) % (m.max(255) - 2)) as u16;
        [maxval, t(raw[1]), t(raw[2]), t(raw[3]), reset]
    };

    let mut lse = vec![1u8];
    for v in [maxval, t1, t2, t3, reset] {
        lse.extend_from_slice(&v.to_be_bytes());
    }
    if flags & 0x20 != 0 {
        lse.extend_from_slice(&[0xA5; 3]);
    }
    let dnl = flags & 0x80 != 0;
    let mut sof = vec![precision];
    sof.extend_from_slice(&(if dnl { 0 } else { height }).to_be_bytes());
    sof.extend_from_slice(&width.to_be_bytes());
    sof.push(nf);
    for c in 1..=nf {
        sof.extend_from_slice(&[c, 0x11, 0x00]);
    }

    let mut stream = vec![0xFF, 0xD8];
    if flags & 0x02 != 0 {
        segment(&mut stream, 0xE0, b"fuzz");
    }
    if flags & 0x08 == 0 {
        segment(&mut stream, 0xF8, &lse);
    }
    segment(&mut stream, 0xF7, &sof);
    if flags & 0x08 != 0 {
        segment(&mut stream, 0xF8, &lse);
    }
    if flags & 0x04 != 0 {
        segment(&mut stream, 0xDD, &[0x00, 0x00]);
    }

    // Split the body into one scan per component at fuzz-chosen points.
    let mut cuts = vec![0usize];
    for &s in &splits[..usize::from(nf) - 1] {
        let last = *cuts.last().unwrap();
        cuts.push(last + (body.len() - last) * usize::from(s) / 255);
    }
    cuts.push(body.len());
    for c in 0..usize::from(nf) {
        segment(&mut stream, 0xDA, &[1, c as u8 + 1, 0, component_near[c], 0, 0]);
        stream.extend_from_slice(&body[cuts[c]..cuts[c + 1]]);
        if dnl && c == 0 {
            segment(&mut stream, 0xDC, &height.to_be_bytes());
        }
    }
    stream.extend_from_slice(&[0xFF, 0xD9]);

    let Ok(img) = jpegls::decode_image(&stream) else {
        return;
    };

    let effective_max = if maxval == 0 {
        sof_max
    } else {
        u32::from(maxval)
    };
    assert!(
        effective_max <= sof_max,
        "accepted LSE MAXVAL {maxval} above 2^{precision} - 1"
    );
    // RESET 0 selects the default (T.87 C.2.4.1.1, as CharLS reads it);
    // otherwise it must lie in 3..=max(255, MAXVAL).
    assert!(
        reset == 0 || (3..=effective_max.max(255)).contains(&u32::from(reset)),
        "accepted LSE RESET {reset} outside 0 or 3..=max(255, MAXVAL {effective_max})"
    );
    assert!(
        u32::from(near) <= (effective_max / 2).min(255),
        "accepted NEAR {near} above min(255, MAXVAL/2) for MAXVAL {effective_max}"
    );
    // A preset MAXVAL below 2^P - 1 may decode through the CharLS-coding
    // retry, where samples can exceed MAXVAL by up to their component's NEAR,
    // exactly as CharLS decodes them (and never by more).
    let sample_max = |near: u8| {
        if effective_max < sof_max {
            (effective_max + u32::from(near)).min(sof_max)
        } else {
            effective_max
        }
    };
    // Fuzz bytes may themselves hold an LSE or SOS segment, and the scans
    // after it then run under its preset or NEAR: only the first scan
    // (component 0) is sure to run under the header's.
    let foreign = body
        .windows(2)
        .any(|p| p == [0xFF, 0xF8] || p == [0xFF, 0xDA]);
    let known = if foreign { 1 } else { usize::from(nf) };
    // With the height left to DNL, a DNL in the fuzz bytes may be the one the
    // decoder finds first.
    let height = if dnl && body.windows(2).any(|p| p == [0xFF, 0xDC]) {
        img.height
    } else {
        u32::from(height)
    };
    assert_image_contract(&img, Some(if foreign { sof_max } else { sample_max(near) }));
    let plane = usize::from(width) * height as usize;
    for (c, samples) in img.samples.chunks(plane).enumerate().take(known) {
        let max = sample_max(component_near[c]);
        assert!(
            samples.iter().all(|&s| u32::from(s) <= max),
            "component {c}: a sample exceeds {max}"
        );
    }
    assert_eq!(
        (
            img.width,
            img.height,
            img.components,
            img.bits_per_sample,
            &img.component_near[..known]
        ),
        (
            u32::from(width),
            height,
            nf,
            precision,
            &component_near[..known]
        ),
        "frame or scan fields changed"
    );
    if !foreign {
        assert_eq!(img.near, near, "near is not the first scan's NEAR");
    }
    assert_decode_agrees(&stream, &img);
});
