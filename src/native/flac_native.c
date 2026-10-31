// SPDX-FileCopyrightText: Copyright (c) 2026 Mikey Sklar for Adafruit Industries
// SPDX-License-Identifier: MIT
//
// flac_native: the adafruit_flac kernel (src/flac_turbo.py) as a native C module. Same functions
// and arguments as the turbo kernel, and the same samples: FLAC decoding is exact.
//
// Build: make -C src/native MPY_DIR=path/to/circuitpython ARCH=armv7emsp

#include "py/dynruntime.h"

static inline void *buf_w(mp_obj_t o) {
    mp_buffer_info_t bi;
    mp_get_buffer_raise(o, &bi, MP_BUFFER_RW);
    return bi.buf;
}

static inline const void *buf_r(mp_obj_t o) {
    mp_buffer_info_t bi;
    mp_get_buffer_raise(o, &bi, MP_BUFFER_READ);
    return bi.buf;
}

// The unary quotient of a Rice code at bit *pos. Moves *pos past it and its stop bit.
static inline int rice_q(const uint8_t *d, const uint8_t *lz, int *pos) {
    int p = *pos;
    int i = p >> 3;
    int b = (d[i] << (p & 7)) & 0xFF;
    int q;
    if (b) {
        q = lz[b];
    } else {
        q = 8 - (p & 7);
        i++;
        while (!d[i]) {
            q += 8;
            i++;
        }
        q += lz[d[i]];
    }
    *pos = p + q + 1;
    return q;
}

// n bits (1 to 32) at bit pos, in the low bits of the result.
static inline uint32_t bits_at(const uint8_t *d, int pos, int n) {
    const uint8_t *p = d + (pos >> 3);
    int s = pos & 7;
    uint32_t w = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    uint32_t v = (w << s) >> (32 - n);
    if (s + n > 32) {
        v |= (uint32_t)p[4] >> (40 - s - n);
    }
    return v;
}

// Rice codes with parameter k into out[o:end], from bit pos of d. Returns the new bit position.
static int rice_run(const uint8_t *d, const uint8_t *lz, int32_t *out, int pos, int o, int end,
    int k) {
    if (k == 0) {
        while (o < end) {
            int q = rice_q(d, lz, &pos);
            out[o++] = (q >> 1) ^ -(q & 1);
        }
        return pos;
    }
    while (o < end) {
        int q = rice_q(d, lz, &pos);
        uint32_t v = bits_at(d, pos, k) | ((uint32_t)q << k);
        pos += k;
        out[o++] = (int32_t)(v >> 1) ^ -(int32_t)(v & 1);
    }
    return pos;
}

// rice_decode(d, lz, out, pos, o, end, k): see flac_turbo.py. Returns the new bit position.
static mp_obj_t rice_decode(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    return mp_obj_new_int(rice_run((const uint8_t *)buf_r(args[0]), (const uint8_t *)buf_r(args[1]),
        (int32_t *)buf_w(args[2]), mp_obj_get_int(args[3]), mp_obj_get_int(args[4]),
        mp_obj_get_int(args[5]), mp_obj_get_int(args[6])));
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(rice_decode_obj, 7, 7, rice_decode);

// A whole subframe residual (RFC 9639 section 9.2.7) into out[order:bs]. Returns the new bit
// position, or -1 for a reserved coding method.
static int residual_run(const uint8_t *d, const uint8_t *lz, int32_t *out, int pos, int bs,
    int order) {
    int v = bits_at(d, pos, 6); // method (2 bits), partition order (4)
    pos += 6;
    int method = v >> 4;
    if (method > 1) {
        return -1;
    }
    int porder = v & 15;
    int pbits = 4 + method;
    int escape = (1 << pbits) - 1;
    int psize = bs >> porder;
    int parts = 1 << porder;
    int o = order;
    for (int p = 1; p <= parts; p++) {
        int end = psize * p;
        int k = bits_at(d, pos, pbits);
        pos += pbits;
        if (k != escape) {
            pos = rice_run(d, lz, out, pos, o, end, k);
            if (o < end) {
                o = end;
            }
            continue;
        }
        // escape: n-bit signed values, or zeros when n == 0
        int n = bits_at(d, pos, 5);
        pos += 5;
        if (n == 0) {
            while (o < end) {
                out[o++] = 0;
            }
            continue;
        }
        int rn = 32 - n;
        while (o < end) {
            out[o++] = (int32_t)(bits_at(d, pos, n) << rn) >> rn; // sign-extend n bits
            pos += n;
        }
    }
    return pos;
}

// residual(d, lz, out, pos, bs, order): see flac_turbo.py.
static mp_obj_t residual(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    return mp_obj_new_int(residual_run((const uint8_t *)buf_r(args[0]),
        (const uint8_t *)buf_r(args[1]), (int32_t *)buf_w(args[2]), mp_obj_get_int(args[3]),
        mp_obj_get_int(args[4]), mp_obj_get_int(args[5])));
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(residual_obj, 6, 6, residual);

// count n-bit signed values (1 <= n <= 32) from bit pos of d into dst. Returns the new position.
static int read_signed_run(const uint8_t *d, int32_t *dst, int pos, int count, int n) {
    int rn = 32 - n;
    for (int j = 0; j < count; j++) {
        dst[j] = (int32_t)(bits_at(d, pos, n) << rn) >> rn;
        pos += n;
    }
    return pos;
}

// Prediction terms, q pointing at the sample being restored.
#define T32(j) (c[j] * q[-1 - (j)])
#define T64(j) ((int64_t)c[j] * q[-1 - (j)])
#define SUM1(T) T(0)
#define SUM2(T) SUM1(T) + T(1)
#define SUM3(T) SUM2(T) + T(2)
#define SUM4(T) SUM3(T) + T(3)
#define SUM5(T) SUM4(T) + T(4)
#define SUM6(T) SUM5(T) + T(5)
#define SUM7(T) SUM6(T) + T(6)
#define SUM8(T) SUM7(T) + T(7)
#define SUM9(T) SUM8(T) + T(8)
#define SUM10(T) SUM9(T) + T(9)
#define SUM11(T) SUM10(T) + T(10)
#define SUM12(T) SUM11(T) + T(11)
#define PREDICT(N, T) case N: \
    for (int32_t *q = out + N; q < end; q++) { \
        *q += (int32_t)((SUM##N(T)) >> shift); \
    } \
    return;

// Restore out[order:n] in place: out[i] += sum(c[j] * out[i - 1 - j]) >> shift, in 32 bits. The
// caller checks that the sum fits.
static void lpc32(int32_t *out, const int32_t *c, int order, int shift, int n) {
    int32_t *end = out + n;
    switch (order) {
        PREDICT(1, T32)
        PREDICT(2, T32)
        PREDICT(3, T32)
        PREDICT(4, T32)
        PREDICT(5, T32)
        PREDICT(6, T32)
        PREDICT(7, T32)
        PREDICT(8, T32)
        PREDICT(9, T32)
        PREDICT(10, T32)
        PREDICT(11, T32)
        PREDICT(12, T32)
    }
    for (int32_t *q = out + order; q < end; q++) {
        int32_t s = 0;
        for (int j = 0; j < order; j++) {
            s += T32(j);
        }
        *q += s >> shift;
    }
}

// lpc32 with a 64-bit sum, which holds any legal FLAC predictor up to 32-bit samples.
static void lpc64(int32_t *out, const int32_t *c, int order, int shift, int n) {
    int32_t *end = out + n;
    switch (order) {
        PREDICT(1, T64)
        PREDICT(2, T64)
        PREDICT(3, T64)
        PREDICT(4, T64)
        PREDICT(5, T64)
        PREDICT(6, T64)
        PREDICT(7, T64)
        PREDICT(8, T64)
        PREDICT(9, T64)
        PREDICT(10, T64)
        PREDICT(11, T64)
        PREDICT(12, T64)
    }
    for (int32_t *q = out + order; q < end; q++) {
        int64_t s = 0;
        for (int j = 0; j < order; j++) {
            s += T64(j);
        }
        *q += (int32_t)(s >> shift);
    }
}

// subframe(d, lz, out, pos, bs, bps, co, stats): one whole subframe (RFC 9639 section 9.2) into
// out[0:bs], see flac_turbo.py. Returns the new bit position, -1 for a malformed or reserved
// subframe, or -2 over 25 bits, with nothing consumed, for the Python path. stats[3] counts 32-bit
// predictions and stats[4] the 64-bit ones. Viper also returns -2 for predictor sums it can't hold
// in 32 bits; the 64-bit sum here decodes those.
static mp_obj_t subframe(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    const uint8_t *d = (const uint8_t *)buf_r(args[0]);
    const uint8_t *lz = (const uint8_t *)buf_r(args[1]);
    int32_t *out = (int32_t *)buf_w(args[2]);
    int pos = mp_obj_get_int(args[3]);
    int bs = mp_obj_get_int(args[4]);
    int bps = mp_obj_get_int(args[5]);
    int32_t *co = (int32_t *)buf_w(args[6]);
    int32_t *stats = (int32_t *)buf_w(args[7]);
    if (bps > 25) {
        return mp_obj_new_int(-2);
    }
    int h = bits_at(d, pos, 8);
    pos += 8;
    int kind = (h >> 1) & 0x3F;
    int wasted = 0;
    if (h & 1) {
        // wasted bits: unary count of zeros, then a 1 (section 9.2.2)
        wasted = rice_q(d, lz, &pos) + 1;
        bps -= wasted;
        if (bps <= 0) {
            return mp_obj_new_int(-1);
        }
    }
    if (kind == 0) { // constant
        pos = read_signed_run(d, out, pos, 1, bps);
        for (int j = 1; j < bs; j++) {
            out[j] = out[0];
        }
    } else if (kind == 1) { // verbatim
        pos = read_signed_run(d, out, pos, bs, bps);
    } else if (kind >= 8 && kind <= 12) { // fixed, section 9.2.5
        static const int8_t fixed[5][4] = {
            {0, 0, 0, 0}, {1, 0, 0, 0}, {2, -1, 0, 0}, {3, -3, 1, 0}, {4, -6, 4, -1},
        };
        int order = kind - 8;
        pos = read_signed_run(d, out, pos, order, bps);
        pos = residual_run(d, lz, out, pos, bs, order);
        if (pos < 0) {
            return mp_obj_new_int(-1);
        }
        if (order) {
            for (int j = 0; j < order; j++) {
                co[j] = fixed[order][j];
            }
            // sum(|c|) <= 15, so one 32-bit sum holds it up to 27-bit samples
            lpc32(out, co, order, 0, bs);
            stats[3]++;
        }
    } else if (kind >= 32) { // LPC, section 9.2.6
        int order = kind - 31;
        pos = read_signed_run(d, out, pos, order, bps);
        int v = bits_at(d, pos, 9); // precision (4), shift (5)
        pos += 9;
        int prec = (v >> 5) + 1;
        if (prec == 16) {
            return mp_obj_new_int(-1);
        }
        int shift = v & 31;
        if (shift & 16) {
            return mp_obj_new_int(-1); // negative shift
        }
        pos = read_signed_run(d, co, pos, order, prec);
        uint32_t ca = 0;
        for (int j = 0; j < order; j++) {
            ca += co[j] < 0 ? -co[j] : co[j];
        }
        pos = residual_run(d, lz, out, pos, bs, order);
        if (pos < 0) {
            return mp_obj_new_int(-1);
        }
        if (ca <= (0x7FFFFFFFu >> (bps - 1))) { // |x| < 2**(bps - 1), so the sum fits in 32 bits
            lpc32(out, co, order, shift, bs);
            stats[3]++;
        } else {
            lpc64(out, co, order, shift, bs);
            stats[4]++;
        }
    } else {
        return mp_obj_new_int(-1);
    }
    if (wasted) {
        for (int j = 0; j < bs; j++) {
            out[j] = (int32_t)((uint32_t)out[j] << wasted);
        }
    }
    return mp_obj_new_int(pos);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(subframe_obj, 8, 8, subframe);

// lpc_restore(out, coefs, order, shift, n): see flac_turbo.py. The sum must fit in 32 bits.
static mp_obj_t lpc_restore(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    lpc32((int32_t *)buf_w(args[0]), (const int32_t *)buf_r(args[1]), mp_obj_get_int(args[2]),
        mp_obj_get_int(args[3]), mp_obj_get_int(args[4]));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(lpc_restore_obj, 5, 5, lpc_restore);

// lpc_restore_split(out, coefs, order, shift, n, b): the same result as the turbo kernel's split
// sums, from one 64-bit sum, so b is not used.
static mp_obj_t lpc_restore_split(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    lpc64((int32_t *)buf_w(args[0]), (const int32_t *)buf_r(args[1]), mp_obj_get_int(args[2]),
        mp_obj_get_int(args[3]), mp_obj_get_int(args[4]));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(lpc_restore_split_obj, 6, 6, lpc_restore_split);

// decorrelate(a, s, n, code): undo stereo decorrelation in place (RFC 9639 section 4.2).
// code 8: left/side, 9: side/right, 10: mid/side. a and s are channels 0 and 1.
static mp_obj_t decorrelate(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    int32_t *a = (int32_t *)buf_w(args[0]);
    int32_t *s = (int32_t *)buf_w(args[1]);
    int n = mp_obj_get_int(args[2]);
    int code = mp_obj_get_int(args[3]);
    if (code == 8) { // right = left - side
        for (int i = 0; i < n; i++) {
            s[i] = a[i] - s[i];
        }
    } else if (code == 9) { // left = side + right
        for (int i = 0; i < n; i++) {
            a[i] += s[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            int32_t side = s[i];
            int32_t mid = (int32_t)((uint32_t)a[i] << 1) | (side & 1);
            a[i] = (mid + side) >> 1;
            s[i] = (mid - side) >> 1;
        }
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(decorrelate_obj, 4, 4, decorrelate);

// to_pcm16(pcm, ch, n, nch, c, sh): channel ch (n samples) into interleaved 16-bit pcm at slot c
// of nch: ch[i] >> sh for sh >= 0 (truncate deeper sources), else ch[i] << -sh.
static mp_obj_t to_pcm16(size_t n_args, const mp_obj_t *args) {
    (void)n_args;
    int16_t *pcm = (int16_t *)buf_w(args[0]) + mp_obj_get_int(args[4]);
    const int32_t *ch = (const int32_t *)buf_r(args[1]);
    int n = mp_obj_get_int(args[2]);
    int nch = mp_obj_get_int(args[3]);
    int sh = mp_obj_get_int(args[5]);
    if (sh >= 0) {
        for (int i = 0; i < n; i++) {
            *pcm = (int16_t)(ch[i] >> sh);
            pcm += nch;
        }
    } else {
        for (int i = 0; i < n; i++) {
            *pcm = (int16_t)((uint32_t)ch[i] << -sh);
            pcm += nch;
        }
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(to_pcm16_obj, 6, 6, to_pcm16);

// gain(pcm, n, q): scale n 16-bit samples in place by q / 32768, for 0 <= q <= 32768.
static mp_obj_t gain(mp_obj_t pcm_in, mp_obj_t n_in, mp_obj_t q_in) {
    int16_t *pcm = (int16_t *)buf_w(pcm_in);
    int n = mp_obj_get_int(n_in);
    int q = mp_obj_get_int(q_in);
    for (int i = 0; i < n; i++) {
        pcm[i] = (int16_t)((pcm[i] * q) >> 15);
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(gain_obj, gain);

// address(buf): where a buffer sits, internal SRAM (0x2...) or PSRAM (0x11...).
static mp_obj_t address(mp_obj_t buf) {
    return mp_obj_new_int_from_uint((uintptr_t)buf_r(buf));
}
static MP_DEFINE_CONST_FUN_OBJ_1(address_obj, address);

mp_obj_t mpy_init(mp_obj_fun_bc_t *self, size_t n_args, size_t n_kw, mp_obj_t *args) {
    MP_DYNRUNTIME_INIT_ENTRY
    mp_store_global(MP_QSTR_rice_decode, MP_OBJ_FROM_PTR(&rice_decode_obj));
    mp_store_global(MP_QSTR_residual, MP_OBJ_FROM_PTR(&residual_obj));
    mp_store_global(MP_QSTR_subframe, MP_OBJ_FROM_PTR(&subframe_obj));
    mp_store_global(MP_QSTR_lpc_restore, MP_OBJ_FROM_PTR(&lpc_restore_obj));
    mp_store_global(MP_QSTR_lpc_restore_split, MP_OBJ_FROM_PTR(&lpc_restore_split_obj));
    mp_store_global(MP_QSTR_decorrelate, MP_OBJ_FROM_PTR(&decorrelate_obj));
    mp_store_global(MP_QSTR_to_pcm16, MP_OBJ_FROM_PTR(&to_pcm16_obj));
    mp_store_global(MP_QSTR_gain, MP_OBJ_FROM_PTR(&gain_obj));
    mp_store_global(MP_QSTR_address, MP_OBJ_FROM_PTR(&address_obj));
    MP_DYNRUNTIME_INIT_EXIT
}
