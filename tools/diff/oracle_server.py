#!/usr/bin/env python3
"""共享推理 oracle:TCP 行协议,任意引擎同输入必得位级同输出(CPU fp32,LRU 缓存)。
协议(小端):req = u32 magic(0x314F4741) u32 nSpat f32*nSpat u32 nGlob f32*nGlob
           rsp = u32 magic u32 cnt[5] f32*(pass,policy,value,scorevalue,ownership)
"""
import socket, struct, sys, hashlib
import numpy as np, onnxruntime as ort

HOST, PORT = "127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 9911
MODEL = sys.argv[1]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
names_in = [i.name for i in sess.get_inputs()]
outs_meta = sess.get_outputs()
print("[oracle] input names:", names_in, file=sys.stderr)
print("[oracle] output shapes:", [(o.name, o.shape) for o in outs_meta], file=sys.stderr)
sys.stderr.flush()

cache = {}
def infer(spat, glob):
    key = hashlib.md5(spat.tobytes() + glob.tobytes()).digest()
    if key in cache: return cache[key]
    sp = spat.reshape(1, 22, 19, 19); gl = glob.reshape(1, 19, 1, 1)
    mk = np.ones((1, 1, 19, 19), np.float32)
    r = dict(zip([o.name for o in outs_meta], sess.run(None, {names_in[0]: sp, names_in[1]: gl, names_in[2]: mk})))
    def pick(part, exclude=()):
        for n in outs_meta:
            if part in n.name and not any(e in n.name for e in exclude): return r[n.name]
        raise KeyError(part)
    res = tuple(np.ascontiguousarray(pick(p, e)).astype(np.float32) for p, e in
                (("PolicyPass", ()), ("Policy", ("Pass",)), ("Value", ("Score",)), ("ScoreValue", ()), ("Ownership", ())))
    cache[key] = res
    return res

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT)); srv.listen(8)
print("[oracle] listening", HOST, PORT, file=sys.stderr); sys.stderr.flush()
while True:
    conn, _ = srv.accept()
    with conn:
        try:
            while True:
                hdr = b""
                while len(hdr) < 4: 
                    c = conn.recv(4 - len(hdr))
                    if not c: raise EOFError
                    hdr += c
                (magic,) = struct.unpack("<I", hdr)
                if magic != 0x314F4741: raise ValueError("bad magic")
                def recv_arr():
                    nb = b""
                    while len(nb) < 4:
                        c = conn.recv(4 - len(nb))
                        if not c: raise EOFError
                        nb += c
                    (n,) = struct.unpack("<I", nb)
                    buf = b""
                    while len(buf) < 4 * n:
                        c = conn.recv(4 * n - len(buf))
                        if not c: raise EOFError
                        buf += c
                    return np.frombuffer(buf, dtype="<f4").astype(np.float32)
                spat = recv_arr(); glob = recv_arr()
                res = infer(spat, glob)
                out = struct.pack("<I5I", 0x314F4741, *[int(a.size) for a in res])
                for a in res: out += a.astype("<f4").tobytes()
                conn.sendall(out)
        except (EOFError, ValueError, ConnectionResetError):
            continue
