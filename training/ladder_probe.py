#!/usr/bin/env python3
"""Board::searchIsLadderCaptured 的逐行复刻(供 JS 对拍调试)。
用法:python3 ladder_probe.py  (内嵌争议局面)"""
import sys

MAXP = 81
NB = []
for p in range(81):
    r, c = divmod(p, 9)
    NB.append([q for q in (p-9 if r > 0 else -1, p+9 if r < 8 else -1, p-1 if c > 0 else -1, p+1 if c < 8 else -1) if q >= 0])

def flood(bd, seed):
    v = bd[seed]
    seen = {seed}; libs = set(); st = [seed]
    while st:
        x = st.pop()
        for y in NB[x]:
            if bd[y] == 0: libs.add(y)
            elif bd[y] == v and y not in seen: seen.add(y); st.append(y)
    return seen, libs

def immediate_libs(bd, p):
    return sum(1 for q in NB[p] if bd[q] == 0)

def would_be_ko(bd, p, v):
    if bd[p] != 0: return False
    his = 3 - v
    cap = None
    for q in NB[p]:
        if bd[q] != his: return False
        _, l = flood(bd, q)
        if len(l) == 1:
            if cap is not None: return False
            cap = q
    if cap is None: return False
    st, _ = flood(bd, cap)
    return len(st) == 1

def libs_after_play(bd, p, v, mx):
    """getNumLibertiesAfterPlay:落子后新链气数(封顶 mx)"""
    his = 3 - v
    b = list(bd); b[p] = v
    capheads = set()
    for q in NB[p]:
        if b[q] == his:
            st, l = flood(b, q)
            if len(l) == 0: capheads |= st
    # 新链
    seen = {p}; st = [p]; libs = set()
    while st:
        x = st.pop()
        for y in NB[x]:
            if b[y] == 0 or y in capheads:
                libs.add(y)
                if len(libs) >= mx: return mx
            elif b[y] == v and y not in seen: seen.add(y); st.append(y)
    return len(libs)

def bound_libs_after_play(bd, p, v):
    his = 3 - v
    imm = caps = capstones = connlibs = maxconn = 0
    heads = set()
    for q in NB[p]:
        if bd[q] == 0: imm += 1
        elif bd[q] == his:
            st, l = flood(bd, q)
            h = min(st)
            if len(l) == 1 and h not in heads:
                heads.add(h); caps += 1; capstones += len(st)
        else:
            st, l = flood(bd, q)
            connlibs += len(l) - 1
            maxconn = max(maxconn, len(l) - 1)
    return caps + max(maxconn, imm), imm + capstones + connlibs

def liberty_gaining_captures(bd, seed):
    his = 3 - bd[seed]
    st, _ = flood(bd, seed)
    out = []; checked = set()
    for s in st:
        for q in NB[s]:
            if bd[q] == his:
                st2, l2 = flood(bd, q)
                h = min(st2)
                if len(l2) == 1 and h not in checked:
                    checked.add(h); out += list(l2)
    return out

def has_liberty_gaining_captures(bd, seed):
    his = 3 - bd[seed]
    st, _ = flood(bd, seed)
    for s in st:
        for q in NB[s]:
            if bd[q] == his:
                _, l2 = flood(bd, q)
                if len(l2) == 1: return True
    return False

def play(bd, p, v):
    """返回 (newbd, caplist) 或 None 非法;更新全局 ko 由调用方处理"""
    his = 3 - v
    b = list(bd)
    b[p] = v
    caps = []
    for q in NB[p]:
        if b[q] == his:
            st, l = flood(b, q)
            if len(l) == 0: caps += list(st)
    for c in caps: b[c] = 0
    st, l = flood(b, p)
    if len(l) == 0: return None
    return b, caps

class Ladder:
    def __init__(self, bd, ko):
        self.bd = list(bd); self.ko = ko; self.nodes = 0
    def search(self, seed, defender_first):
        bd = self.bd
        v = bd[seed]
        if v not in (1, 2): return False
        st, l = flood(bd, seed)
        if len(l) > 2 or (defender_first and len(l) > 1): return False
        his = 3 - v
        ko_saved = self.ko
        if defender_first: self.ko = -1
        r = self.rec(seed, v, his, defender_first, 0)
        self.ko = ko_saved
        return r
    def rec(self, seed, v, his, defender_first, depth):
        if depth >= 121: return True
        if self.nodes >= 25000: raise RuntimeError('budget')
        bd = self.bd
        is_def = (defender_first and depth % 2 == 0) or (not defender_first and depth % 2 == 1)
        st, l = flood(bd, seed)
        libs = len(l)
        if not is_def:
            if libs <= 1: return True
            if libs >= 3: return False
            m0, m1 = sorted(l)[0], sorted(l)[1]
            libs0, libs1 = immediate_libs(bd, m0), immediate_libs(bd, m1)
            if libs0 == 0 and libs1 == 0 and would_be_ko(bd, m0, his) and would_be_ko(bd, m1, his):
                if libs_after_play(bd, m0, v, 3) <= 2 and libs_after_play(bd, m1, v, 3) <= 2:
                    if not has_liberty_gaining_captures(bd, seed): return True
            moves = [m0, m1]
            adj2 = m1 in NB[m0]
            if not adj2:
                if libs0 >= 3 and libs1 >= 3: return False
                elif libs0 >= 3: moves = [m0]
                elif libs1 >= 3: moves = [m1]
            if len(moves) > 1:
                k0 = libs0 * 2 + sum((len(flood(bd, q)[1]) * 2 - 3) for q in NB[m0] if bd[q] == v and len(flood(bd, q)[1]) > 1)
                k1 = libs1 * 2 + sum((len(flood(bd, q)[1]) * 2 - 3) for q in NB[m1] if bd[q] == v and len(flood(bd, q)[1]) > 1)
                if k1 > k0: moves = [m1, m0]
            for mv in moves:
                ko_before = self.ko
                r = play(bd, mv, his)
                if r is None: continue
                self.bd, caps = r
                st2, l2 = flood(self.bd, mv)
                self.ko = -1
                if len(caps) == 1 and len(st2) == 1 and len(l2) == 1 and l2 == set(caps):
                    self.ko = caps[0]
                self.nodes += 1
                try:
                    res = self.rec(seed, v, his, defender_first, depth + 1)
                finally:
                    # undo
                    self.bd[mv] = 0
                    for c in caps: self.bd[c] = his
                    self.ko = ko_before
                if res: return True
            return False
        else:
            if libs >= 2: return False
            if self.ko >= 0: return False
            caps_moves = liberty_gaining_captures(bd, seed)
            escape = sorted(l)[0]
            moves = caps_moves + [escape]
            lb, ub = bound_libs_after_play(bd, escape, v)
            if lb >= 3: return False
            if len(moves) == 1 and ub <= 1: return True
            for mv in moves:
                ko_before = self.ko
                r = play(bd, mv, v)
                if r is None: continue
                self.bd, caps = r
                st2, l2 = flood(self.bd, mv)
                self.ko = -1
                if len(caps) == 1 and len(st2) == 1 and len(l2) == 1 and l2 == set(caps):
                    self.ko = caps[0]
                self.nodes += 1
                try:
                    res = self.rec(seed, v, his, defender_first, depth + 1)
                finally:
                    self.bd[mv] = 0
                    for c in caps: self.bd[c] = his
                    self.ko = ko_before
                if not res: return False
            return True

def ladder_at(bd, ko, seed):
    L = Ladder(bd, ko)
    st, l = flood(bd, seed)
    if len(l) == 1:
        return L.search(seed, True)
    if len(l) == 2:
        # AttackerFirst2Libs
        v = bd[seed]; his = 3 - v
        for mv in sorted(l):
            ko_before = L.ko
            r = play(bd, mv, his)
            if r is None: continue
            L.bd, caps = r
            st2, l2 = flood(L.bd, mv)
            L.ko = -1
            if len(caps) == 1 and len(st2) == 1 and len(l2) == 1 and l2 == set(caps):
                L.ko = caps[0]
            try:
                works = L.search(seed, True)
            finally:
                L.bd[mv] = 0
                for c in caps: L.bd[c] = his
                L.ko = ko_before
            if works: return True
        return False
    return False

if __name__ == '__main__':
    # 从 JSON 参数跑:node 侧把盘面/种子传进来
    import json
    req = json.loads(sys.stdin.read())
    bd = req['bd']; ko = req.get('ko', -1)
    out = []
    for seed in req['seeds']:
        if bd[seed] in (1, 2):
            st, l = flood(bd, seed)
            if len(l) in (1, 2):
                out.append({'seed': seed, 'libs': len(l), 'laddered': ladder_at(bd, ko, seed)})
    print(json.dumps(out))
