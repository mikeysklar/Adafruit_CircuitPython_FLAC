# SPDX-FileCopyrightText: Copyright (c) 2026 Mikey Sklar for Adafruit Industries
# SPDX-License-Identifier: MIT
"""
`adafruit_flac`
================================================================================

FLAC decoding and playback for CircuitPython.

`FLACDecoder` reads a FLAC file and returns 16-bit signed PCM. `FLACPlayer` streams it to an
audio output (``I2SOut``, ``PWMAudioOut`` or ``AudioOut``) through a double-buffered
``audiocore.RawSample``; call `FLACPlayer.update` often from your main loop.

The hot loops (Rice decoding, the predictors, stereo decorrelation and PCM conversion) run in
a turbo kernel, ``flac_kernel.<arch>.mpy``, prebuilt from ``src/flac_turbo.py`` with
``mpy-cross -march``. Without a kernel for the board, plain Python loops run instead
(correct, but several times slower than real time for music).

Written from RFC 9639, "Free Lossless Audio Codec (FLAC)" (section numbers in comments refer
to it). No third-party decoder code is included. Decoding is bit-exact. 20, 24 and 32-bit
sources are reduced to their top 16 bits.

* Author(s): Mikey Sklar

Implementation Notes
--------------------

**Hardware:**

* Tested on the Adafruit Fruit Jam (RP2350B): 16-bit/44.1 kHz and 24-bit/96 kHz at the
  stock 150 MHz.

**Software and Dependencies:**

* Adafruit CircuitPython firmware 11 or later, with native ``.mpy`` loading:
  https://circuitpython.org/downloads
"""

import struct
from array import array

try:
    from time import monotonic_ns as _now
except ImportError:  # unix port
    from time import time_ns as _now

try:
    from ulab import numpy as _np
except ImportError:
    try:
        import numpy as _np  # desktop testing with CPython
    except ImportError:
        _np = None

# The hot loops, prebuilt per chip: flac_native.<arch>.mpy (C, src/native), else
# flac_kernel.<arch>.mpy (turbo, src/flac_turbo.py). Same functions.
# Without either (CPython, or a chip with no build) the plain Python loops run.
try:
    from adafruit_flac import flac_native as _kernel
except (ImportError, ValueError):
    try:
        from adafruit_flac import flac_kernel as _kernel
    except (ImportError, ValueError):
        _kernel = None

# Largest sum the 32-bit turbo predictor can hold.
_INT32 = (1 << 31) - 1

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/adafruit/Adafruit_CircuitPython_FLAC.git"

# Largest value a small int holds on a 32-bit MicroPython build. Anything
# bigger becomes a heap-allocated long, which is several times slower.
_SMALL = (1 << 30) - 1


# Leading zero bits in a byte; 8 for zero.
def _lz_table():
    t = bytearray(256)
    t[0] = 8
    for i in range(1, 256):
        n = 0
        while not (i << n) & 0x80:
            n += 1
        t[i] = n
    return bytes(t)


_LZ = _lz_table()

# Section 9.1.1 / 9.1.2 lookup tables.
_BLOCK_SIZES = (
    0,
    192,
    576,
    1152,
    2304,
    4608,
    -1,
    -2,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
_RATES = (
    0,
    88200,
    176400,
    192000,
    8000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
    96000,
    -1,
    -2,
    -3,
    0,
)
_DEPTHS = (0, 8, 12, 0, 16, 20, 24, 32)

# Fixed predictor coefficients, section 9.2.5.
_FIXED = ((), (1,), (2, -1), (3, -3, 1), (4, -6, 4, -1))


class FLACError(ValueError):
    """Raised for files that are not FLAC or use unsupported features."""


class _Bits:
    """MSB-first bit reader over a bytearray. Used for headers; the hot
    residual loop reads the same buffer with its own inlined code."""

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos

    def read(self, n):
        if n == 0:
            return 0
        pos = self.pos
        d = self.data
        i = pos >> 3
        need = (pos & 7) + n
        v = 0
        for j in range((need + 7) >> 3):
            v = (v << 8) | d[i + j]
        self.pos = pos + n
        return (v >> (((need + 7) & ~7) - need)) & ((1 << n) - 1)

    def signed(self, n):
        v = self.read(n)
        if v >> (n - 1):
            v -= 1 << n
        return v

    def unary(self):
        # count zero bits before the next 1
        d = self.data
        pos = self.pos
        q = 0
        while not (d[pos >> 3] << (pos & 7)) & 0x80:
            q += 1
            pos += 1
        self.pos = pos + 1
        return q

    def align(self):
        self.pos = (self.pos + 7) & ~7


def _rice(d, pos, out, o, end, k):
    """Decode Rice codes with parameter ``k`` into ``out[o:end]`` from bit
    ``pos`` of ``d`` (section 9.2.7.2). Returns the new bit position.

    This is the hottest loop. ``for ... in range`` is MicroPython's fastest
    loop, and the common cases get their own copy so the loop body has no
    extra branches."""
    lz = _LZ
    if k == 0:
        for o in range(o, end):  # noqa: PLR1704, reuses the argument for speed
            i = pos >> 3
            b = (d[i] << (pos & 7)) & 0xFF
            if b:
                q = lz[b]
            else:
                q = 8 - (pos & 7)
                i += 1
                while not d[i]:
                    q += 8
                    i += 1
                q += lz[d[i]]
            pos += q + 1
            out[o] = (q >> 1) ^ -(q & 1)
        return pos
    mask = (1 << k) - 1
    sh = 24 - k
    if k <= 17:
        # the k remainder bits always fit in one 3-byte window
        for o in range(o, end):  # noqa: PLR1704, reuses the argument for speed
            i = pos >> 3
            b = (d[i] << (pos & 7)) & 0xFF
            if b:
                q = lz[b]
            else:
                q = 8 - (pos & 7)
                i += 1
                while not d[i]:
                    q += 8
                    i += 1
                q += lz[d[i]]
            pos += q + 1
            i = pos >> 3
            v = (q << k) | (((d[i] << 16 | d[i + 1] << 8 | d[i + 2]) >> (sh - (pos & 7))) & mask)
            pos += k
            out[o] = (v >> 1) ^ -(v & 1)
        return pos
    for o in range(o, end):
        i = pos >> 3
        b = (d[i] << (pos & 7)) & 0xFF
        if b:
            q = lz[b]
        else:
            q = 8 - (pos & 7)
            i += 1
            while not d[i]:
                q += 8
                i += 1
            q += lz[d[i]]
        pos += q + 1
        i = pos >> 3
        s = pos & 7
        if s <= sh:
            v = ((d[i] << 16 | d[i + 1] << 8 | d[i + 2]) >> (sh - s)) & mask
        else:
            # past 30 bits this is a long int on a 32-bit build; rare
            v = ((d[i] << 24 | d[i + 1] << 16 | d[i + 2] << 8 | d[i + 3]) >> (32 - s - k)) & mask
        pos += k
        v |= q << k
        out[o] = (v >> 1) ^ -(v & 1)
    return pos


_LPC_CACHE = {}

# Predictor variants, picked per subframe by `FLACDecoder._predict`.
_PLAIN = 0  # one dot product
_CSPLIT = 1  # coefficients split into 8-bit halves: two dot products
_XSPLIT = 2  # samples split into two limbs: two dot products
_TURBO = 3  # flac_kernel.lpc_restore, when the sum fits in 32 bits
_TURBO_SPLIT = 4  # flac_kernel.lpc_restore_split, sample limbs for 24-bit


# Orders up to this use the ring-unrolled loop; longer ones rotate locals.
_RING_MAX = 16


def _lpc_func(order, mode):
    """Build an unrolled predictor for one order (section 9.2.6).

    ``out`` holds warm-up samples followed by residuals, and is restored in
    place. The previous ``order`` samples live in local variables, which is
    much faster than indexing ``out`` on MicroPython. For orders up to
    `_RING_MAX` the loop body is unrolled ``order`` times and each step
    overwrites the variable holding the oldest sample, so no variables are
    shuffled between samples.

    On a 32-bit build a product or sum past 30 bits becomes a heap-allocated
    long int. The split modes keep every value small and stay exact:

    * ``_CSPLIT``: ``c = a * 256 + b`` with ``0 <= b < 256``. For a shift
      ``s >= 8``, ``(A * 256 + B) >> s == (A + (B >> 8)) >> (s - 8)``.
    * ``_XSPLIT``: ``x = h * 2 ** b + l`` with ``0 <= l < 2 ** b``. The
      prediction ``(H * 2 ** b + L) >> s`` equals
      ``(H + (L >> b)) >> (s - b)`` when ``s >= b`` and
      ``(H << (b - s)) + (L >> s)`` when ``s < b``. Both are
      ``((H << u) + (L >> v)) >> w``.
    """
    key = (order, mode)
    f = _LPC_CACHE.get(key)
    if f is not None:
        return f
    n = order
    r = range(n)
    tup = "," if n == 1 else ""
    lines = ["def f(out, end, coefs, shift, b):"]
    if mode == _CSPLIT:
        names_a = ", ".join("a%d" % j for j in r)
        names_b = ", ".join("b%d" % j for j in r)
        lines.append("    %s%s = [c >> 8 for c in coefs]" % (names_a, tup))
        lines.append("    %s%s = [c & 255 for c in coefs]" % (names_b, tup))
    else:
        lines.append("    %s%s = coefs" % (", ".join("c%d" % j for j in r), tup))
    if mode == _CSPLIT:
        lines.append("    s8 = shift - 8")
    elif mode == _XSPLIT:
        lines.append("    u = b - shift if b > shift else 0")
        lines.append("    v = shift if shift < b else b")
        lines.append("    w = shift - b if shift > b else 0")
        lines.append("    lm = (1 << b) - 1")

    def predict(var):
        # var(j) names the sample j + 1 back
        if mode == _PLAIN:
            p = " + ".join("c%d * %s" % (j, var("x", j)) for j in r)
            return "((%s) >> shift)" % p
        if mode == _CSPLIT:
            a = " + ".join("a%d * %s" % (j, var("x", j)) for j in r)
            b = " + ".join("b%d * %s" % (j, var("x", j)) for j in r)
            return "(((%s) + ((%s) >> 8)) >> s8)" % (a, b)
        h = " + ".join("c%d * %s" % (j, var("h", j)) for j in r)
        l = " + ".join("c%d * %s" % (j, var("l", j)) for j in r)
        return "((((%s) << u) + ((%s) >> v)) >> w)" % (h, l)

    def store(ind, name):
        # x is the new sample; keep it (or its limbs) in variable slot `name`
        if mode == _XSPLIT:
            return [ind + "h%s = x >> b" % name, ind + "l%s = x & lm" % name]
        return [ind + "x%s = x" % name]

    if n <= _RING_MAX:
        # slot t holds the sample whose index is t mod n
        for t in r:
            lines.append("    x = out[%d]" % t)
            lines.extend(store("    ", t))
        lines.append("    stop = end - (end - %d) %% %d" % (n, n))
        lines.append("    for i in range(%d, stop, %d):" % (n, n))
        for t in r:
            idx = "i + %d" % t if t else "i"
            lines.append(
                "        x = out[%s] + %s"
                % (idx, predict(lambda v, j, t=t: "%s%d" % (v, (t - 1 - j) % n)))
            )
            lines.append("        out[%s] = x" % idx)
            lines.extend(store("        ", t))
        # the last few samples, exactly and slowly
        lines.append("    for i in range(stop, end):")
        lines.append("        s = 0")
        lines.append("        for j in range(%d):" % n)
        lines.append("            s += coefs[j] * out[i - 1 - j]")
        lines.append("        out[i] += s >> shift")
    else:
        # slot j + 1 holds the sample j + 1 back; shuffle after each sample
        for j in r:
            lines.append("    x = out[%d]" % (n - 1 - j))
            lines.extend(store("    ", j + 1))
        lines.append("    for i in range(%d, end):" % n)
        lines.append("        x = out[i] + %s" % predict(lambda v, j: "%s%d" % (v, j + 1)))
        lines.append("        out[i] = x")
        names = ("h", "l") if mode == _XSPLIT else ("x",)
        for j in range(n, 1, -1):
            for v in names:
                lines.append("        %s%d = %s%d" % (v, j, v, j - 1))
        lines.extend(store("        ", 1))
    ns = {}
    exec("\n".join(lines), ns)
    f = ns["f"]
    _LPC_CACHE[key] = f
    return f


class FLACDecoder:
    """Decode a FLAC file to 16-bit signed PCM.

    :param file: path string or a file opened in ``"rb"`` mode
    :param bool accel: use ``ulab`` for interleaving when it applies
    :param bool turbo: use the prebuilt ``flac_kernel`` for the hot loops when
      this chip has one

    Playing a FLAC file::

        import audiobusio
        import board
        import adafruit_flac

        audio = audiobusio.I2SOut(board.I2S_BCLK, board.I2S_WS, board.I2S_DIN)
        decoder = adafruit_flac.FLACDecoder("song.flac")
        player = adafruit_flac.FLACPlayer(audio)
        player.play(decoder)
        while player.update():
            pass
    """

    def __init__(self, file, *, accel=True, turbo=True):
        self._owns = False
        self._file = None
        self._accel = bool(accel and _np is not None)
        self._kernel = _kernel if turbo else None
        self._buf = self._chan = self._pcm = None
        self._coefs = array("i", [0]) * 32  # scratch for flac_kernel.subframe
        self.lpc_modes = array("i", [0]) * 5
        """Subframes predicted with the plain, coefficient-split,
        sample-split, turbo and turbo-split loops (for benchmarking)."""
        self._open(file)

    # -- public API --

    @property
    def file(self):
        """The open file object. Assign a new file or path to switch songs."""
        return self._file

    @file.setter
    def file(self, file):
        self._open(file)

    def open(self, file):
        """Switch to a different file (same as assigning `file`)."""
        self._open(file)

    @property
    def sample_rate(self):
        """Sample rate in Hz."""
        return self._rate

    @property
    def channel_count(self):
        """Number of channels (output is limited to 2)."""
        return self._channels

    @property
    def bits_per_sample(self):
        """Always 16."""
        return 16

    @property
    def source_bits_per_sample(self):
        """Bit depth stored in the file (8 to 32)."""
        return self._bps

    @property
    def samples(self):
        """Total frames in the stream (0 if the encoder did not say)."""
        return self._total

    @property
    def duration(self):
        """Length in seconds (0 if unknown)."""
        return self._total / self._rate if self._rate else 0

    @property
    def samples_decoded(self):
        """Frames decoded since the file was opened or rewound."""
        return self._decoded

    def rewind(self):
        """Go back to the first audio frame."""
        self._file.seek(self._audio_start)
        self._len = 0
        self._off = 0
        self._eof = False
        self._decoded = 0
        self._pcm_len = 0
        self._pcm_pos = 0

    def seek(self, seconds):
        """Jump to the frame that holds about ``seconds`` (within half a
        second, usually less). Returns the time of that frame in seconds
        (`duration` if it is past the end)."""
        total = self._total
        rate = self._rate
        if seconds <= 0 or not total:
            self.rewind()
            return 0
        f = self._file
        f.seek(0, 2)
        size = f.tell()
        start = self._audio_start
        target = int(seconds * rate)
        self._len = self._off = self._pcm_len = self._pcm_pos = 0
        if target >= total:
            f.seek(size)
            self._eof = True
            self._decoded = total
            return total / rate
        # Guess the byte offset from the average bit rate, then correct it a
        # few times: the bit rate changes through a track.
        per = (size - start) / total  # bytes per sample
        pos = start + int(target * per)
        best = (start, 0)
        for _ in range(6):
            found = self._sync(max(start, pos), size)
            if found is None:
                pos -= self._chunk
                continue
            at, first = found
            if first <= target and first > best[1]:
                best = found
            if 0 <= target - first < rate // 2:
                break
            pos = at + int((target - first) * per) - (rate // 4 if first > target else 0) * int(per)
        f.seek(best[0])
        self._eof = False
        self._decoded = best[1]
        return best[1] / rate

    def _sync(self, pos, size):
        """(byte offset, first sample) of the first frame at or after pos."""
        f = self._file
        buf = self._buf
        while pos < size:
            f.seek(pos)
            got = f.readinto(memoryview(buf)[: self._chunk]) or 0
            # Section 9.1: a frame header is a sync code that passes CRC-8.
            # Audio data can hold the sync pattern too, so check each one.
            i = buf.find(b"\xff", 0, max(0, got - 16))
            while i >= 0:
                if (buf[i + 1] & 0xFE) == 0xF8:
                    first = self._frame_start(buf, i)
                    if first >= 0:
                        return pos + i, first
                i = buf.find(b"\xff", i + 1, got - 16)
            if got < self._chunk:
                return None
            pos += got - 16
        return None

    def _frame_start(self, buf, i):
        """The first sample of the frame whose header starts at buf[i], or -1
        if it is not a valid header."""
        bs_code = buf[i + 2] >> 4
        rate_code = buf[i + 2] & 15
        chan_code = buf[i + 3] >> 4
        if not bs_code or rate_code == 15 or chan_code > 10 or buf[i + 3] & 1:
            return -1
        # coded frame/sample number, UTF-8 style (section 9.1.5)
        p = i + 4
        n = buf[p]
        extra = 0
        while n & (0x80 >> extra) and extra < 7:
            extra += 1
        if extra == 1:
            return -1
        if extra:
            n &= 0x7F >> extra
            for k in range(1, extra):
                c = buf[p + k]
                if c & 0xC0 != 0x80:
                    return -1
                n = (n << 6) | (c & 0x3F)
            p += extra
        else:
            p += 1
        p += (bs_code == 6) + 2 * (bs_code == 7) + (rate_code == 12) + 2 * (rate_code in (13, 14))
        crc = 0
        for k in range(i, p):
            crc ^= buf[k]
            for _ in range(8):
                crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else crc << 1
        if crc != buf[p]:
            return -1
        if buf[i + 1] & 1:  # variable block size: n is the sample number
            return n
        return n * (self._maxblock or 4096)  # fixed: n is the frame number

    def readinto(self, buf):
        """Fill ``buf`` (an ``array('h')`` or memoryview of one) with
        interleaved samples. Returns the number of values written; less than
        ``len(buf)`` only at the end of the stream."""
        mv = memoryview(buf)
        total = len(mv)
        got = 0
        while got < total:
            n = self.fill(mv[got:])
            if n == 0:
                break
            got += n
        return got

    def fill(self, mv):
        """Copy decoded samples into memoryview ``mv``, decoding at most one
        frame. Returns the number of values written, 0 at end of stream."""
        if self._pcm_pos >= self._pcm_len:
            if not self._decode_pcm():
                return 0
        p = self._pcm_pos
        n = self._pcm_len - p
        n = min(n, len(mv))
        mv[:n] = self._pcm_mv[p : p + n]
        self._pcm_pos = p + n
        return n

    def decode_frame(self):
        """Decode the next frame and return its channels as a list of
        ``array('i')`` at the source bit depth (each may be longer than the
        block; use `block_size`). Returns None at the end of the stream."""
        if not self._decode():
            return None
        return self._chan

    @property
    def block_size(self):
        """Frames in the block most recently returned by `decode_frame`."""
        return self._bs

    def deinit(self):
        """Release the file (if this decoder opened it) and buffers."""
        if self._owns and self._file is not None:
            self._file.close()
        self._file = None
        self._buf = None
        self._chan = None
        self._pcm = None
        self._pcm_mv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.deinit()

    # -- internals --

    def _open(self, file):
        if self._owns and self._file is not None:
            self._file.close()
        if isinstance(file, str):
            file = open(file, "rb")
            self._owns = True
        else:
            self._owns = False
        self._file = file
        self._read_metadata()
        maxblock = self._maxblock or 4096
        # Worst case frame is a verbatim block plus headers (section 10.1).
        maxframe = self._maxframe or (maxblock * self._channels_coded * (self._bps + 1) // 8 + 64)
        self._chunk = max(maxframe, 4096)
        # Keep the buffers from the previous file when they are big enough: where they
        # landed (internal SRAM or PSRAM) matters several times over for decode speed,
        # and the first allocation is the one most likely to get SRAM.
        # Hottest first: the channel arrays are read and written many times per sample,
        # the PCM frame twice, the input buffer once. If internal RAM runs short, the
        # input buffer is the one that should spill.
        chan = self._chan or []
        if len(chan) < self._channels_coded or len(chan[0]) < maxblock:
            chan = [array("i", [0]) * maxblock for _ in range(self._channels_coded)]
        self._chan = chan
        if self._pcm is None or len(self._pcm) < maxblock * self._channels:
            self._pcm = array("h", [0]) * (maxblock * self._channels)
            self._pcm_mv = memoryview(self._pcm)
        if self._buf is None or len(self._buf) < 2 * self._chunk + 8:
            self._buf = bytearray(2 * self._chunk + 8)  # +8: guard bytes for the bit reader
        self._audio_start = file.tell()
        self.rewind()

    def _read_metadata(self):
        # Section 8: "fLaC" then metadata blocks, the first being STREAMINFO.
        f = self._file
        head = f.read(4)
        if head == b"ID3\x03" or head[:3] == b"ID3":
            # skip an ID3v2 tag some taggers prepend
            rest = f.read(6)
            size = (rest[2] << 21) | (rest[3] << 14) | (rest[4] << 7) | rest[5]
            f.seek(10 + size)
            head = f.read(4)
        if head != b"fLaC":
            raise FLACError("not a FLAC file")
        self._rate = 0
        last = False
        while not last:
            h = f.read(4)
            if len(h) < 4:
                raise FLACError("truncated metadata")
            last = h[0] & 0x80
            kind = h[0] & 0x7F
            size = (h[1] << 16) | (h[2] << 8) | h[3]
            if kind == 0:  # STREAMINFO, section 8.2
                si = f.read(size)
                self._minblock, self._maxblock = struct.unpack(">HH", si[0:4])
                self._maxframe = (si[7] << 16) | (si[8] << 8) | si[9]
                b = _Bits(si, 80)
                self._rate = b.read(20)
                self._channels_coded = b.read(3) + 1
                self._bps = b.read(5) + 1
                hi = b.read(4)
                self._total = (hi << 32) | struct.unpack(">I", si[14:18])[0]
            else:
                f.seek(size, 1)
        if not self._rate:
            raise FLACError("no STREAMINFO")
        if self._channels_coded > 2:
            # only the first two channels are played
            self._channels = 2
        else:
            self._channels = self._channels_coded

    def _refill(self):
        # Keep at least one maximum frame of data after self._off.
        buf = self._buf
        rem = self._len - self._off
        if rem >= self._chunk or self._eof:
            return
        buf[:rem] = buf[self._off : self._len]
        mv = memoryview(buf)
        want = 2 * self._chunk - rem
        got = self._file.readinto(mv[rem : rem + want]) or 0
        if got < want:
            self._eof = True
        self._len = rem + got
        self._off = 0
        # zero guard bytes so the bit reader never sees stale data
        for i in range(self._len, self._len + 8):
            buf[i] = 0

    def _decode(self):
        self._refill()
        buf = self._buf
        # Section 9.1: find the frame sync code.
        off = self._off
        end = self._len - 1
        while off < end and not (buf[off] == 0xFF and (buf[off + 1] & 0xFE) == 0xF8):
            off += 1
        if off >= end or self._len - off < 6:
            return False
        b = _Bits(buf, off * 8 + 16)
        bs_code = b.read(4)
        rate_code = b.read(4)
        chan_code = b.read(4)
        depth_code = b.read(3)
        b.read(1)
        # coded frame/sample number, UTF-8 style (section 9.1.5)
        first = b.read(8)
        extra = 0
        while first & (0x80 >> extra) and extra < 7:
            extra += 1
        if extra:
            b.read(8 * (extra - 1))
        bs = _BLOCK_SIZES[bs_code]
        if bs == -1:
            bs = b.read(8) + 1
        elif bs == -2:
            bs = b.read(16) + 1
        elif bs == 0:
            raise FLACError("reserved block size")
        rate = _RATES[rate_code]
        if rate == -1:
            b.read(8)
        elif rate < -1:
            b.read(16)
        b.read(8)  # CRC-8, not checked
        bps = _DEPTHS[depth_code] or self._bps
        if bs > len(self._chan[0]):
            self._chan = [array("i", [0]) * bs for _ in self._chan]
            self._pcm = array("h", [0]) * (bs * self._channels)
            self._pcm_mv = memoryview(self._pcm)
        chan = self._chan
        if chan_code < 8:
            nch = chan_code + 1
            for c in range(nch):
                self._subframe(b, chan[c], bs, bps)
        elif chan_code <= 10:
            # section 4.2: the side channel carries one extra bit
            self._subframe(b, chan[0], bs, bps + (chan_code == 9))
            self._subframe(b, chan[1], bs, bps + (chan_code != 9))
            self._decorrelate(chan_code, bs, bps)
        else:
            raise FLACError("reserved channel assignment")
        b.align()
        if (b.pos >> 3) + 2 > self._len:
            return False  # truncated last frame
        self._off = (b.pos >> 3) + 2  # skip CRC-16
        self._bs = bs
        self._frame_bps = bps
        self._decoded += bs
        return True

    def _subframe(self, b, out, bs, bps):
        # Section 9.2.
        kernel = self._kernel
        if kernel is not None:
            pos = kernel.subframe(b.data, _LZ, out, b.pos, bs, bps, self._coefs, self.lpc_modes)
            if pos >= 0:
                b.pos = pos
                return
            if pos == -1:
                raise FLACError("bad subframe")
            # -2: past what 32-bit viper math holds; decode this one in Python
        h = b.read(8)
        kind = (h >> 1) & 0x3F
        wasted = 0
        if h & 1:
            wasted = b.unary() + 1
            bps -= wasted
        if kind == 0:  # constant
            v = b.signed(bps)
            for i in range(bs):
                out[i] = v
        elif kind == 1:  # verbatim
            for i in range(bs):
                out[i] = b.signed(bps)
        elif 8 <= kind <= 12:  # fixed
            order = kind - 8
            for i in range(order):
                out[i] = b.signed(bps)
            self._residual(b, out, bs, order)
            if order:
                self._predict(out, bs, _FIXED[order], 0, bps)
        elif kind >= 32:  # LPC
            order = kind - 31
            for i in range(order):
                out[i] = b.signed(bps)
            prec = b.read(4) + 1
            if prec == 16:
                raise FLACError("bad LPC precision")
            shift = b.signed(5)
            if shift < 0:
                raise FLACError("negative LPC shift")
            coefs = tuple(b.signed(prec) for _ in range(order))
            self._residual(b, out, bs, order)
            self._predict(out, bs, coefs, shift, bps)
        else:
            raise FLACError("reserved subframe type")
        if wasted:
            for i in range(bs):
                out[i] <<= wasted

    def _predict(self, out, bs, coefs, shift, bps):
        # Pick the cheapest predictor whose products and sums provably stay
        # in small ints: |x| < 2 ** (bps - 1), so |sum(c * x)| is at most
        # sum(|c|) << (bps - 1).
        order = len(coefs)
        mode = _PLAIN
        b = bps >> 1  # limb split point for _XSPLIT
        ca = 0
        for c in coefs:
            ca += c if c >= 0 else -c
        k = self._kernel
        if k is not None:
            if ca << (bps - 1) <= _INT32:
                self.lpc_modes[_TURBO] += 1
                k.lpc_restore(out, array("i", coefs), order, shift, bs)
                return
            # 24-bit: split samples into two limbs so each dot product fits 32 bits
            if ca << max(b, bps - 1 - b) <= _INT32 and (
                shift >= b or ca << (bps - 1 - shift) <= _INT32
            ):
                self.lpc_modes[_TURBO_SPLIT] += 1
                k.lpc_restore_split(out, array("i", coefs), order, shift, bs, b)
                return
        if ca << (bps - 1) > _SMALL:
            hi = 0
            for c in coefs:
                ch = c >> 8
                hi += ch if ch >= 0 else -ch
            if shift >= 8 and (hi << (bps - 1)) <= _SMALL and (255 * order) << (bps - 1) <= _SMALL:
                mode = _CSPLIT
            elif ca << max(b, bps - 1 - b) <= _SMALL:
                mode = _XSPLIT
            # else: nothing keeps it small; plain with long ints is still exact
        self.lpc_modes[mode] += 1
        _lpc_func(order, mode)(out, bs, coefs, shift, b)

    def _residual(self, b, out, bs, order):
        # Section 9.2.7.
        kernel = self._kernel
        if kernel is not None:
            pos = kernel.residual(b.data, _LZ, out, b.pos, bs, order)
            if pos < 0:
                raise FLACError("reserved residual coding method")
            b.pos = pos
            return
        self._residual_py(b, out, bs, order)

    def _residual_py(self, b, out, bs, order):
        method = b.read(2)
        if method > 1:
            raise FLACError("reserved residual coding method")
        pbits = 4 + method
        escape = (1 << pbits) - 1
        porder = b.read(4)
        psize = bs >> porder
        o = order
        d = b.data
        kernel = self._kernel
        for p in range(1 << porder):
            end = psize * (p + 1)
            k = b.read(pbits)
            if k == escape:
                n = b.read(5)
                if n:
                    for i in range(o, end):
                        out[i] = b.signed(n)
                else:
                    for i in range(o, end):
                        out[i] = 0
            elif kernel is not None:
                b.pos = kernel.rice_decode(d, _LZ, out, b.pos, o, end, k)
            else:
                b.pos = _rice(d, b.pos, out, o, end, k)
            o = end

    def _decorrelate(self, code, bs, bps):
        a, s = self._chan[0], self._chan[1]
        # mid/side holds up to bps + 1 bits; << 1 stays in 32 bits up to 24-bit
        if self._kernel is not None and bps <= 24:
            self._kernel.decorrelate(a, s, bs, code)
            return
        if code == 8:  # left/side: right = left - side
            for i in range(bs):
                s[i] = a[i] - s[i]
        elif code == 9:  # side/right: left = side + right
            for i in range(bs):
                a[i] += s[i]
        else:  # mid/side
            for i in range(bs):
                side = s[i]
                mid = (a[i] << 1) | (side & 1)
                a[i] = (mid + side) >> 1
                s[i] = (mid - side) >> 1

    def _decode_pcm(self):
        if not self._decode():
            return False
        bs = self._bs
        bps = self._frame_bps
        nch = self._channels
        pcm = self._pcm
        chan = self._chan
        kernel = self._kernel
        if kernel is not None:
            for c in range(nch):
                kernel.to_pcm16(pcm, chan[c], bs, nch, c, bps - 16)
        elif self._accel and bps in (16, 24, 32):
            # Bytes 0-1 (16-bit), 1-2 (24-bit) or 2-3 (32-bit) of each
            # little-endian int32 are exactly the 16-bit output. Let ulab copy
            # them byte by byte into place; it has no int32 type.
            off = (bps - 16) >> 3
            step = 2 * nch
            dst = _np.frombuffer(self._pcm_mv[: bs * nch], dtype=_np.uint8)
            for c in range(nch):
                src = _np.frombuffer(chan[c], dtype=_np.uint8, count=4 * bs)
                dst[2 * c :: step] = src[off::4]
                dst[2 * c + 1 :: step] = src[off + 1 :: 4]
        elif nch == 2:
            l, r = chan[0], chan[1]
            if bps > 16:
                sh = bps - 16
                j = 0
                for i in range(bs):
                    pcm[j] = l[i] >> sh
                    pcm[j + 1] = r[i] >> sh
                    j += 2
            else:
                sh = 16 - bps
                j = 0
                for i in range(bs):
                    pcm[j] = l[i] << sh
                    pcm[j + 1] = r[i] << sh
                    j += 2
        else:
            m = chan[0]
            if bps > 16:
                sh = bps - 16
                for i in range(bs):
                    pcm[i] = m[i] >> sh
            else:
                sh = 16 - bps
                for i in range(bs):
                    pcm[i] = m[i] << sh
        self._pcm_len = bs * nch
        self._pcm_pos = 0
        return True


def _gain(pcm, n, q):
    # Scale n samples of pcm in place by q / 32768.
    if _kernel is not None:
        _kernel.gain(pcm, n, q)
        return
    for i in range(n):
        pcm[i] = (pcm[i] * q) >> 15


class FLACPlayer:
    """Stream a `FLACDecoder` to an audio output from Python.

    Decoded PCM goes straight into a double-buffered `audiocore.RawSample` that
    loops on the output, so playback keeps going while your code runs. Call
    `update` often (several times per ``buffer_ms``); each call does at most one
    FLAC frame of decoding.

    The output must copy each half when it starts playing it, as ``I2SOut``,
    ``PWMAudioOut`` and ``AudioOut`` do (into their DMA buffers): a half is
    rewritten as soon as the output has taken it. Not for ``audiomixer``, which
    reads the sample in place.

    Keep the buffer in internal RAM where there is a choice: on an RP2350 with
    PSRAM, writing decoded audio to PSRAM nearly doubled the decode time at
    96 kHz. Pass ``buffer`` to allocate it early, before other code takes the
    internal RAM.

    :param audio: an ``audiobusio.I2SOut``, ``audiopwmio.PWMAudioOut`` or
        ``audioio.AudioOut``-like object with ``play(sample, loop=True)``,
        ``stop()`` and optionally ``pause()``
    :param int buffer_ms: length of each half of the output buffer. The output's DMA
        needs internal RAM for twice the whole buffer; on a Fruit Jam with the display
        running, 120 ms is about the most that fits at 48 kHz stereo
    :param buffer: optional ``array('h')`` to use for both halves (its length
        sets the half size; ``buffer_ms`` is then ignored)
    :param float level: output level from 0.0 (silent) to 1.0 (unchanged), see `level`
    """

    def __init__(self, audio, *, buffer_ms=64, buffer=None, level=1.0):
        self._audio = audio
        self.level = level
        self._buffer_ms = buffer_ms
        self._fixed = buffer is not None
        self._buf = buffer
        self._mv = memoryview(buffer) if buffer is not None else None
        self._zero = array("h", [0]) * 256
        self._decoder = None
        self._playing = False
        self._paused = False
        self.underruns = 0
        """Count of buffer halves that were not ready in time."""

    @property
    def playing(self):
        """True while a file is playing (including while paused)."""
        return self._playing

    @property
    def paused(self):
        """True while paused."""
        return self._paused

    @property
    def decoder(self):
        """The `FLACDecoder` being played."""
        return self._decoder

    @property
    def level(self):
        """Output level from 0.0 (silent) to 1.0 (unchanged, the default), like
        ``audiomixer.MixerVoice.level``. Samples are scaled as they are decoded, so a
        change is heard after the buffered audio (up to two ``buffer_ms`` halves)."""
        return self._q / 32768

    @level.setter
    def level(self, value):
        self._q = min(32768, max(0, int(value * 32768 + 0.5)))

    def play(self, decoder, *, loop=False):
        """Start playing ``decoder`` from its current position."""
        import audiocore  # noqa: PLC0415, so the decoder also imports on desktop Python

        self.stop()
        self._decoder = decoder
        self._loop = loop
        ch = decoder.channel_count
        if self._fixed:
            half = len(self._buf) // 2 // ch * ch
        else:
            half = max(256, decoder.sample_rate * self._buffer_ms // 1000) * ch
            if self._buf is None or len(self._buf) != 2 * half:
                self._buf = array("h", [0]) * (2 * half)
                self._mv = memoryview(self._buf)
        self._half = half
        self._eof = False
        self._last_real = -1
        # prime both halves before starting the output
        for w in range(2):
            self._w = w
            self._filled = 0
            while self._filled < half:
                self._fill_once()
            if self._eof and self._last_real < 0:
                self._last_real = w
        self._w = 2
        self._filled = 0
        self._half_ns = half // ch * 1_000_000_000 // decoder.sample_rate
        self._sample = audiocore.RawSample(
            self._mv[: 2 * half],
            channel_count=ch,
            sample_rate=decoder.sample_rate,
            single_buffer=False,
        )
        self._audio.play(self._sample, loop=True)
        self._t0 = _now()
        self._playing = True
        self._paused = False

    def _fill_once(self):
        # Decode up to one frame into the half being written; pad with silence at the end.
        half = self._half
        a = (self._w & 1) * half + self._filled
        room = half - self._filled
        if not self._eof:
            n = self._decoder.fill(self._mv[a : a + room])
            if not n and self._loop:
                self._decoder.rewind()
                n = self._decoder.fill(self._mv[a : a + room])
            if n:
                if self._q < 32768:
                    _gain(self._mv[a : a + n], n, self._q)
                self._filled += n
                return
            self._eof = True
        z = self._zero
        while room > 0:
            k = room if room < len(z) else len(z)
            self._mv[a : a + k] = z[:k]
            a += k
            room -= k
        self._filled = half

    def update(self):
        """Do a slice of decoding work. Returns True while still playing."""
        if not self._playing:
            return False
        if self._paused:
            return True
        elapsed = _now() - self._t0
        hd = self._half_ns
        # halves the output has taken so far: two at start, then one per half
        fetched = 2 + elapsed // hd
        if self._last_real >= 0 and fetched >= self._last_real + 3:
            # the half after the last real one (silence) is now playing
            self.stop()
            return False
        w = self._w
        if w < fetched:
            # fell behind: half w was taken before it was ready; skip ahead
            self.underruns += 1
            w = self._w = fetched + 1
            self._filled = 0
        # Half w reuses the slot of half w - 2, taken at (w - 3) * hd. Leave a quarter
        # half of margin for the output's background task to copy it.
        if elapsed < (w - 3) * hd + (hd >> 2):
            return True
        self._fill_once()
        if self._filled >= self._half:
            if self._eof and self._last_real < 0:
                self._last_real = w
            self._w = w + 1
            self._filled = 0
        return True

    def pause(self):
        """Pause output, if the audio object supports it."""
        if self._playing and not self._paused:
            self._audio.pause()
            self._pause_t = _now()
            self._paused = True

    def resume(self):
        """Resume after `pause`."""
        if self._paused:
            self._t0 += _now() - self._pause_t
            self._audio.resume()
            self._paused = False

    def stop(self):
        """Stop playback."""
        if self._playing:
            self._audio.stop()
        self._playing = False
        self._paused = False

    def deinit(self):
        """Stop and release buffers."""
        self.stop()
        self._buf = None
        self._mv = None
        self._fixed = False
        self._sample = None
