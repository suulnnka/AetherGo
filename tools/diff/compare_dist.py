#!/usr/bin/env python3
"""对比两引擎根访问分布:不对称局面为主,输出每局面 top5 对齐与差异统计。"""
import json

js = {r["tag"]: r for r in map(json.loads, open("/tmp/kbench/js_dist.jsonl"))}
ka = {r["tag"]: r for r in map(json.loads, open("/tmp/kbench/kata_dist.jsonl"))}

# 空盘/强对称局面单列(对称摊访问 vs 剪枝收敛,天然不同)
for tag in ("P3", "P6_k10", "P7", "P8", "M1_ply40"):
    j, k = js[tag], ka[tag]
    jmap = {c["mv"]: c for c in j["children"]}
    kmap = {c["mv"]: c for c in k["children"]}
    jtop = j["children"][:3]
    ktop = k["children"][:3]
    common = sorted(set(jmap) | set(kmap),
                    key=lambda m: -max(jmap.get(m, {}).get("v", 0), kmap.get(m, {}).get("v", 0)))[:10]
    print("== %s (stm %s)  JS move=%s  KA top=%s" % (tag, j["stm"], j["move"], ktop[0]["mv"] if ktop else "-"))
    print("   JS top3: " + ", ".join("%s:%d(q%.3f)" % (c["mv"], c["v"], c["q"]) for c in jtop))
    print("   KA top3: " + ", ".join("%s:%d(wr%.3f,sl%.2f)" % (c["mv"], c["v"], c["wr"] or 0, c["sl"] or 0) for c in ktop))
    diffs = []
    for m in common:
        jv = jmap.get(m, {}).get("v", 0)
        kv = kmap.get(m, {}).get("v", 0)
        if jv or kv:
            diffs.append("%s js%d/ka%d" % (m, jv, kv))
    print("   visits: " + "  ".join(diffs[:10]))
    # 访问分布相关性
    import math
    pairs = [(jmap.get(m, {}).get("v", 0), kmap.get(m, {}).get("v", 0)) for m in set(jmap) | set(kmap)]
    n = len(pairs)
    if n > 2:
        mj = sum(a for a, b in pairs) / n
        mk = sum(b for a, b in pairs) / n
        cov = sum((a - mj) * (b - mk) for a, b in pairs)
        sj = math.sqrt(sum((a - mj) ** 2 for a, b in pairs))
        sk = math.sqrt(sum((b - mk) ** 2 for a, b in pairs))
        corr = cov / (sj * sk) if sj > 0 and sk > 0 else 0
        print("   访问数相关系数: %.3f (n=%d)" % (corr, n))
    print()
