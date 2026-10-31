# SPDX-FileCopyrightText: Copyright (c) 2026 Mikey Sklar for Adafruit Industries
# SPDX-License-Identifier: MIT
# flac_native, the C build of the kernel, as a native module:
#   make MPY_DIR=path/to/circuitpython ARCH=armv7emsp
# Use the CircuitPython tree whose firmware will load it (the .mpy version must match).
MPY_DIR ?= ../../../../circuitpython
MOD = flac_native
SRC = flac_native.c
ARCH ?= armv7emsp
# libgcc for 64-bit multiplies and division
LINK_RUNTIME = 1
CFLAGS_EXTRA = -O2
# mpy_ld's symbol order depends on Python's hash seed; pin it so builds are repeatable
export PYTHONHASHSEED = 0

include $(MPY_DIR)/py/dynruntime.mk
