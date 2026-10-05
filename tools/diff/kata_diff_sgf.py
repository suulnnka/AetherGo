#!/usr/bin/env python3
"""真实棋谱差分 KataGo 侧:同批 SGF 同步取样(局面 = 前 i 手,与 JS 侧一致),
kata-analyze(oracle + 规则对齐)64 visits。"""
import subprocess, sys, time, re, os, json

KATAGO = "/home/a/go/KataGo/build-onnx/katago"
MODEL = "/home/a/go/AetherGo/models/b8c96h3tfrs_19.onnx"
CFG = "/home/a/go/trainrun/gtp_b8c96.cfg"
VISITS = int(os.environ.get("VISITS", "64"))
STEP = int(os.environ.get("STEP", "40"))

env = dict(os.environ)
env["KATA_NNSERVER"] = "127.0.0.1:9911"
env["LD_LIBRARY_PATH"] = "/home/a/go/tools/onnxruntime-linux-x64-1.30.0/lib"

p = subprocess.Popen(
    [KATAGO, "gtp", "-config", CFG, "-config", "ae_rules.cfg", "-model", MODEL,
     "-override-config", "numSearchThreads=1,maxVisits=%d,nnMaxBatchSize=1,ponderingEnabled=false,wideRootNoise=0,"
     "allowResignation=false,logAllGTPCommunication=false,logDir=/tmp/kbench/logs,onnxProvider=cpu" % VISITS],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
    cwd="/home/a/go/trainrun")

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

FIRST_VISITS_RE = re.compile(r"(?:^| )move \S+ visits (\d+)")

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
            s = sum(int(m.group(1)) for m in FIRST_VISITS_RE.finditer(l))
            if s >= VISITS - 2: break
    return last

COLS = "ABCDEFGHJKLMNOPQRST"
def gtp_text(idx): return COLS[idx % 19] + str(19 - idx // 19)

wait_ok("boardsize 19"); wait_ok("komi 7.5")
for path in sys.argv[1:]:
    t = open(path).read()
    mvre = re.findall(r";([BW])\[([a-z]{0,2})\]", t)
    moves = []
    for c, coord in mvre:
        moves.append(None if coord == "" else (ord(coord[1]) - 97) * 19 + (ord(coord[0]) - 97))
    g = os.path.basename(path)
    wait_ok("clear_board")
    for i, mv in enumerate(moves):
        color = "b" if i % 2 == 0 else "w"
        if i >= 20 and (i - 20) % STEP == 0 and i + 30 < len(moves):
            line = analyze()
            cands = []
            for seg in line.split("info ")[1:]:
                mmv = re.match(r"move (\S+)", seg)
                v = re.search(r"visits (\d+)", seg)
                sl = re.search(r"scoreLead (-?[\d.]+)", seg)
                wr = re.search(r"winrate (-?[\d.]+)", seg)
                if mmv and v:
                    wrv = float(wr.group(1)) if wr else None
                    if wrv is not None and wrv > 1.0: wrv = 1.0   # 终局怪值钳位
                    cands.append({"mv": mmv.group(1), "v": int(v.group(1)),
                                  "sl": float(sl.group(1)) if sl else None, "wr": wrv})
            cands.sort(key=lambda c: -c["v"])
            print(json.dumps({"game": g, "ply": i, "stm": "B" if i % 2 == 0 else "W",
                              "children": cands}, ensure_ascii=False), flush=True)
        if mv is None:
            wait_ok("play %s pass" % color)
        else:
            wait_ok("play %s %s" % (color, gtp_text(mv)))
p.stdin.write("quit\n"); p.stdin.flush()
