import subprocess, json, math, os
import numpy as np

D = json.load(open(__import__("os").environ.get("PARITY_JSON", "/tmp/kbench/nn_parity_browser.json")))
KATAGO = os.environ.get("KATAGO", "/home/a/go/KataGo/build-cuda/katago")
COLS = "ABCDEFGHJKLMNOPQRST"
def gtp_text(idx):
    return COLS[idx % 19] + str(19 - idx // 19)

GTP_MOVES = {
  "P1": [], "P2": [72], "P3": [72, 288], "P6": [180],
  "P7": [72,288,300,60,111,313,249,43],
  "P8": [72,288,300,60,111,313,249,43,270,35],
}

extra = "" if "eigen" in KATAGO else ("" if os.environ.get("FP16") == "1" else ",cudaUseFP16=false")
p = subprocess.Popen(
  [KATAGO, "gtp",
   "-config", "/home/a/go/trainrun/gtp_b8c96.cfg", "-config", "ae_rules.cfg",
   "-model", "/home/a/go/trainrun/export_bin/b8c96h3tfrs-s68320512.bin.gz",
   "-override-config", "numSearchThreads=1,ponderingEnabled=false,logAllGTPCommunication=false,logDir=/tmp/kbench/logs" + extra],
  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

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

def raw_nn():
    p.stdin.write("kata-raw-nn 0\n"); p.stdin.flush()
    lines = []
    while True:
        l = p.stdout.readline()
        if not l: raise RuntimeError("eof")
        l = l.rstrip("\n")
        if l.strip() == "": break
        lines.append(l)
    return lines

wait_ok("boardsize 19"); wait_ok("komi 7.5")
def softplus(x): return x if x > 30 else math.log1p(math.exp(x))

worst_pol = worst_own = worst_val = 0.0
for tag, moves in GTP_MOVES.items():
    wait_ok("clear_board")
    for i, m in enumerate(moves):
        assert wait_ok("play %s %s" % ("b" if i%2==0 else "w", gtp_text(m))).startswith("=")
    lines = raw_nn()
    kv, grid, own_grid, mode = {}, [], [], None
    for l in lines:
        t = l.split()
        if not t: continue
        if t[0] in ("whiteWin","whiteLoss","noResult","whiteLead","shorttermWinlossError","shorttermScoreError"):
            kv[t[0]] = float(t[1])
        elif t[0] == "policy": mode = "policy"
        elif t[0] == "policyPass": kv["policyPass"] = float(t[1]); mode = None
        elif t[0] == "whiteOwnership": mode = "own"
        elif mode == "policy" and len(t) == 19: grid.append([float(x) for x in t])
        elif mode == "own" and len(t) == 19: own_grid.append([float(x) for x in t])
    kata_pol = np.array(grid).flatten()
    kata_own = np.array(own_grid).flatten()

    d = D[tag]; side = d["side"]
    probs = np.array(d["probs362"])
    val = np.array(d["val3"], dtype=np.float64)
    e = np.exp(val - val.max()); pv = e / e.sum()
    stmW, stmL = pv[0], pv[1]
    br_whiteWin = stmW if side == 1 else stmL
    sv = d["sv6"]
    br_lead = (sv[2] * 20) * (1 if side == 1 else -1)
    br_own = np.tanh(np.array(d["own361"], dtype=np.float64)) * (1 if side == 1 else -1)

    mask = ~np.isnan(kata_pol)
    pd = np.abs(kata_pol[mask] - probs[:361][mask])
    od = np.abs(kata_own - br_own)
    vd = abs(kv["whiteWin"] - br_whiteWin)
    ld = abs(kv["whiteLead"] - br_lead)
    ka = gtp_text(int(np.nanargmax(kata_pol)))
    ba = gtp_text(int(np.argmax(np.where(mask, probs[:361], -1))))
    topk = np.argsort(np.where(mask, probs[:361], -1))[::-1][:3]
    print("%s polMax=%.2e polMean=%.2e ownMax=%.2e winD=%.2e leadD=%.2e argmax kata=%s browser=%s top3br=%s" % (
      tag, pd.max(), pd.mean(), od.max(), vd, ld, ka, ba,
      [gtp_text(int(i)) + ":" + format(probs[i], ".4f") for i in topk]))
    worst_pol = max(worst_pol, pd.max()); worst_own = max(worst_own, od.max()); worst_val = max(worst_val, vd)

p.stdin.write("quit\n"); p.stdin.flush()
p.wait(timeout=15)
print("WORST policy %.2e ownership %.2e whiteWin %.2e" % (worst_pol, worst_own, worst_val))
