#!/usr/bin/env python3
"""zstd 压缩实验:对 .aewn 权重文件(level 3/19/22+long)vs gzip 基线。"""
import ctypes, sys

z = ctypes.CDLL("libzstd.so.1")
z.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
z.ZSTD_compressBound.restype = ctypes.c_size_t
z.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
z.ZSTD_compress.restype = ctypes.c_size_t
z.ZSTD_isError.argtypes = [ctypes.c_size_t]
z.ZSTD_isError.restype = ctypes.c_uint
z.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
z.ZSTD_getErrorName.restype = ctypes.c_char_p

ZSTD_c_compressionLevel = 100
ZSTD_c_windowLog = 101
ZSTD_c_enableLongDistanceMatching = 160

def zstd_compress(data: bytes, level: int, long_wlog: int = 0) -> bytes:
    cap = z.ZSTD_compressBound(len(data))
    out = ctypes.create_string_buffer(cap)
    n = z.ZSTD_compress(out, cap, data, len(data), min(level, 22))
    if z.ZSTD_isError(n):
        raise RuntimeError(z.ZSTD_getErrorName(n).decode())
    return out.raw[:n]

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    mb = len(data) / 1048576
    print(f"== {path} ({mb:.2f}MB)")
    for level, wlog in [(3, 0), (19, 0), (22, 27)]:
        c = zstd_compress(data, level, wlog)
        tag = f"zstd-{level}" + (f"+long{wlog}" if wlog else "")
        print(f"   {tag:16s} {len(c)/1048576:.3f}MB ({len(c)*100/len(data):.1f}%)")
