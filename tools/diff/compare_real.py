#!/usr/bin/env python3
"""真实对战局面差分汇总:top1/top2 一致性、访问相关性、分歧局面的细节。"""
import json, math

js = [json.loads(l) for l in open("/tmp/kbench/js_real.jsonl")]
ka = [json.loads(l) for l in open("/tmp/kbench/kata_real.jsonl")]
kmap = {(k["game"], k["ply"]): k for k in ka}

def wl_from_wr(wr, stm):
    """kata winrate(行棋方视角)→ 行棋方 wl;我方 q 是走进子那方(=行棋方)视角效用,
    近似换算 wl ≈ q(效用含目差项,仅方向可比)"""
    return wr * 2 - 1 if stm == "B" or True else 0

stats = {"n": 0, "top1": 0, "top2set": 0, "top2order": 0, "corr": [], "diverge": []}
for j in js:
    k = kmap.get((j["game"], j["ply"])) or kmap.get((j["game"], j["ply"] + 1))
    if not k: continue
    stats["n"] += 1
    jc = j["children"]; kc = k["children"]
    jtop2 = [c["mv"] for c in jc[:2]]
    ktop2 = [c["mv"] for c in kc[:2]]
    if jc[0]["mv"] == kc[0]["mv"]: stats["top1"] += 1
    if set(jtop2) == set(ktop2): stats["top2set"] += 1
    if jtop2 == ktop2: stats["top2order"] += 1
    jmap = {c["mv"]: c for c in jc}; kmap2 = {c["mv"]: c for c in kc}
    pairs = [(jmap.get(m, {}).get("v", 0), kmap2.get(m, {}).get("v", 0)) for m in set(jmap) | set(kmap2)]
    n = len(pairs)
    if n > 2:
        mj = sum(a for a, b in pairs) / n; mk = sum(b for a, b in pairs) / n
        sj = math.sqrt(sum((a - mj) ** 2 for a, b in pairs))
        sk = math.sqrt(sum((b - mk) ** 2 for a, b in pairs))
        if sj > 0 and sk > 0:
            cov = sum((a - mj) * (b - mk) for a, b in pairs)
            stats["corr"].append(cov / (sj * sk))
    if jc[0]["mv"] != kc[0]["mv"]:
        stats["diverge"].append({
            "game": j["game"], "ply": j["ply"], "stm": j["stm"],
            "jsMove": j["move"], "jsTop": jc[0]["mv"], "kaTop": kc[0]["mv"],
            "jsTopV": jc[0]["v"], "kaTopV": kc[0]["v"],
            "detail": " | ".join(
                "%s js(v%d,q%+.3f) ka(v%d,wr%.3f)" % (
                    m, jmap.get(m, {}).get("v", 0), jmap.get(m, {}).get("q", 0) or 0,
                    kmap2.get(m, {}).get("v", 0), (kmap2.get(m, {}).get("wr") or 0.5))
                for m in list(dict.fromkeys(jtop2 + ktop2)))
        })

n = stats["n"]
print("样本 n=%d  top1一致 %d(%.0f%%)  top2集合一致 %d(%.0f%%)  top2排序一致 %d(%.0f%%)  访问相关中位 %.3f" % (
    n, stats["top1"], 100 * stats["top1"] / n, stats["top2set"], 100 * stats["top2set"] / n,
    stats["top2order"], 100 * stats["top2order"] / n,
    sorted(stats["corr"])[len(stats["corr"]) // 2]))
print()
print("=== top1 分歧局面(%d) ===" % len(stats["diverge"]))
for d in stats["diverge"]:
    print("%s ply%d stm%s  我方选 %s(vis首 %s v%d)  kata首 %s(v%d)" % (
        d["game"], d["ply"], d["stm"], d["jsMove"], d["jsTop"], d["jsTopV"], d["kaTop"], d["kaTopV"]))
    print("   " + d["detail"])
