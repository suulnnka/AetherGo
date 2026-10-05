#!/usr/bin/env python3
"""差分调试 KataGo 侧:oracle 推理、单线程、批 1、64 visits,kata-analyze 采根分布。"""
import subprocess, sys, time, re, os

KATAGO = "/home/a/go/KataGo/build-onnx/katago"
MODEL = "/home/a/go/AetherGo/models/b8c96h3tfrs_19.onnx"
CFG = "/home/a/go/trainrun/gtp_b8c96.cfg"
RULES_CFG = "ae_rules.cfg"
VISITS = int(os.environ.get("VISITS", "64"))

env = dict(os.environ)
env["KATA_NNSERVER"] = "127.0.0.1:9911"
env["LD_LIBRARY_PATH"] = "/home/a/go/tools/onnxruntime-linux-x64-1.30.0/lib"

p = subprocess.Popen(
    [KATAGO, "gtp", "-config", CFG, "-config", RULES_CFG, "-model", MODEL,
     "-override-config", "numSearchThreads=1,maxVisits=%d,nnMaxBatchSize=1,ponderingEnabled=false,"
     "allowResignation=false,logAllGTPCommunication=false,logDir=/tmp/kbench/logs,onnxProvider=cpu" % VISITS],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)

def wait_ok(cmd):
    p.stdin.write(cmd + "\n"); p.stdin.flush()
    while True:
        l = p.stdout.readline()
        if not l: raise RuntimeError("eof")
        l = l.strip()
        if l.startswith("=") or l.startswith("?"):
            while True:
                l2 = p.stdout.readline()
                if not l2 or l2.strip() == "": break
            return l

def analyze():
    p.stdin.write("kata-analyze 50 maxmoves 40\n"); p.stdin.flush()
    last = ""
    deadline = time.time() + 60
    while time.time() < deadline:
        l = p.stdout.readline()
        if not l: break
        l = l.rstrip("\n")
        if "info move" in l:
            last = l
            s = sum(int(m) for m in re.findall(r"visits (\d+)", l))
            if s >= VISITS - 2: break
    return last

COLS = "ABCDEFGHJKLMNOPQRST"
def gtp_text(idx): return COLS[idx % 19] + str(19 - idx // 19)

POSITIONS = [
  ("P1_empty", [], "b"), ("P2_q16", [72], "w"), ("P3", [72, 288], "b"),
  ("P6_k10", [180], "w"),
  ("P7", [72, 288, 300, 60, 111, 313, 249, 43], "b"),
  ("P8", [72, 288, 300, 60, 111, 313, 249, 43, 270, 35], "w"),
  ("M1_ply40", [72, 288, 300, 60, 111, 313, 249, 43, 270, 35, 53, 41, 301, 97, 63, 308,
                74, 296, 46, 271, 319, 192, 51, 286, 225, 212, 100, 292, 67, 318], "b"),
]

wait_ok("boardsize 19"); wait_ok("komi 7.5")
for tag, moves, side in POSITIONS:
    wait_ok("clear_board")
    for i, m in enumerate(moves):
        assert wait_ok("play %s %s" % ("b" if i % 2 == 0 else "w", gtp_text(m))).startswith("=")
    line = analyze()
    cands = []
    for seg in line.split("info ")[1:]:
        mv = re.match(r"move (\S+)", seg)
        v = re.search(r"visits (\d+)", seg)
        sl = re.search(r"scoreLead (-?[\d.]+)", seg)
        wr = re.search(r"winrate ([\d.]+)", seg)
        if mv and v:
            cands.append({"mv": mv.group(1), "v": int(v.group(1)),
                          "sl": float(sl.group(1)) if sl else None,
                          "wr": float(wr.group(1)) if wr else None})
    cands.sort(key=lambda c: -c["v"])
    import json
    print(json.dumps({"tag": tag, "stm": side, "children": cands}, ensure_ascii=False))
p.stdin.write("quit\n"); p.stdin.flush()
