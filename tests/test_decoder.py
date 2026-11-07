# SPDX-FileCopyrightText: Copyright (c) 2026 Mikey Sklar for Adafruit Industries
# SPDX-License-Identifier: MIT
"""Check adafruit_flac against libFLAC's reference decode of every clip in
media/ (made by tools/make_test_media.py). Runs under CPython.

Two checks per clip:
* every channel of every frame from ``decode_frame`` matches ``flac -d``
  exactly at the source bit depth
* the 16-bit interleaved output from ``readinto`` matches the reference
  shifted to 16 bits (truncated)

usage: python3 tests/test_decoder.py [clip-name ...]
"""

import os
import struct
import sys
from array import array

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import adafruit_flac  # noqa: E402

MEDIA = os.path.join(HERE, "media")


def read_wav(path):
    """Return (channels, container bits, list of per-channel lists)."""
    data = open(path, "rb").read()
    pos = 12
    ch = bits = None
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            ch = struct.unpack("<H", body[2:4])[0]
            bits = struct.unpack("<H", body[14:16])[0]
        elif cid == b"data":
            step = bits // 8
            n = len(body) // step
            if step == 1:  # 8-bit WAV is unsigned
                vals = [b - 128 for b in body]
            elif step == 2:
                vals = array("h", body[: n * 2])
            elif step == 4:
                vals = array("i", body[: n * 4])
            else:
                vals = [
                    int.from_bytes(body[i : i + 3], "little", signed=True)
                    for i in range(0, n * 3, 3)
                ]
            return ch, bits, [list(vals[c::ch]) for c in range(ch)]
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def check(name, accel):
    flac = os.path.join(MEDIA, name + ".flac")
    ch, cont, ref = read_wav(os.path.join(MEDIA, name + ".ref.wav"))
    dec = adafruit_flac.FLACDecoder(flac, accel=accel)
    bps = dec.source_bits_per_sample
    just = cont - bps  # flac left-justifies 12/20-bit samples in the WAV
    ref = [[v >> just for v in r] for r in ref]
    nframes = len(ref[0])

    # 1: source-depth samples, frame by frame
    pos = 0
    while True:
        chans = dec.decode_frame()
        if chans is None:
            break
        bs = dec.block_size
        for c in range(ch):
            got = list(chans[c][:bs])
            want = ref[c][pos : pos + bs]
            if got != want:
                i = next(i for i in range(bs) if got[i] != want[i])
                return "channel %d frame at %d: sample %d got %d want %d" % (
                    c,
                    pos,
                    pos + i,
                    got[i],
                    want[i],
                )
        pos += bs
    if pos != nframes:
        return "decoded %d frames, reference has %d" % (pos, nframes)

    # 2: 16-bit interleaved output
    dec.rewind()
    out_ch = min(ch, 2)
    buf = array("h", bytes(2 * 4096 * out_ch))
    got = array("h")
    while True:
        n = dec.readinto(buf)
        if not n:
            break
        got.extend(buf[:n])
    want = array("h")
    sh = bps - 16
    for i in range(nframes):
        for c in range(out_ch):
            v = ref[c][i]
            want.append(v >> sh if sh >= 0 else v << -sh)
    if got != want:
        i = (
            next(i for i in range(min(len(got), len(want))) if got[i] != want[i])
            if len(got) == len(want)
            else -1
        )
        return "16-bit output differs at %d (len %d vs %d)" % (i, len(got), len(want))
    return "ok  bps=%d ch=%d modes=%s" % (bps, ch, list(dec.lpc_modes))


SEEK_TARGETS = (0, 1.5, 5.0, 9.2, 12.0)  # start, early, middle, last second, past the end


def check_seek(name):
    """seek() lands on a frame at or before the target, within half a second,
    and readinto then continues from exactly that sample."""
    ch, cont, ref = read_wav(os.path.join(MEDIA, name + ".ref.wav"))
    dec = adafruit_flac.FLACDecoder(os.path.join(MEDIA, name + ".flac"))
    bps = dec.source_bits_per_sample
    ref = [[v >> (cont - bps) for v in r] for r in ref]
    out_ch = min(ch, 2)
    sh = bps - 16
    buf = array("h", bytes(2 * 4096 * out_ch))
    for t in SEEK_TARGETS:
        got_t = dec.seek(t)
        if t >= dec.duration:
            if got_t != dec.duration or dec.readinto(buf):
                return "seek(%g) past the end: got %g and more audio" % (t, got_t)
            continue
        if not t - 0.5 < got_t <= t:
            return "seek(%g) landed at %g" % (t, got_t)
        first = round(got_t * dec.sample_rate)
        n = dec.readinto(buf)
        want = array("h")
        for i in range(first, first + n // out_ch):
            for c in range(out_ch):
                v = ref[c][i]
                want.append(v >> sh if sh >= 0 else v << -sh)
        if buf[:n] != want:
            return "seek(%g) to %g: audio does not match the reference there" % (t, got_t)
    return "ok"


def test_seek():
    """pytest entry: seek() on every clip in media/."""
    import pytest  # noqa: PLC0415

    names = _names()
    if not names:
        pytest.skip("no test media; run tools/make_test_media.py")
    bad = [(n, r) for n in names if (r := check_seek(n)) != "ok"]
    assert not bad, bad


def _names():
    if not os.path.isdir(MEDIA):
        return []
    return sorted(f[:-5] for f in os.listdir(MEDIA) if f.endswith(".flac"))


def test_decoder():
    """pytest entry: every clip in media/ (skipped when tools/make_test_media.py has not run)."""
    import pytest  # noqa: PLC0415

    names = _names()
    if not names:
        pytest.skip("no test media; run tools/make_test_media.py")
    bad = [(n, r) for n in names if not (r := check(n, False)).startswith("ok")]
    assert not bad, bad


def main():
    names = sys.argv[1:] or _names()
    accels = [False]
    if adafruit_flac._np is not None:
        accels.append(True)
    bad = 0
    for name in names:
        for accel in accels:
            r = check(name, accel)
            if not r.startswith("ok"):
                bad += 1
            print("%-18s accel=%-5s %s" % (name, accel, r))
    print("FAIL %d" % bad if bad else "all passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
