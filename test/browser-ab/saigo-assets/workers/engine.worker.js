(async ()=>{
    function C(h) {
        return h === "b" ? "w" : "b";
    }
    function H(h, t) {
        return h === null || t === null ? h === t : h.x === t.x && h.y === t.y;
    }
    let M = new Int32Array(0), P = 0;
    const L = (()=>{
        const h = new Int32Array(722);
        let t = 2654435769;
        for(let s = 0; s < h.length; s++)t = Math.imul(t, 1664525) + 1013904223 >>> 0, h[s] = t | 0;
        return h;
    })();
    class S {
        size;
        grid;
        koPoint = null;
        positionHistory;
        hash = 0;
        constructor(t = 19, s = !1){
            this.size = t, this.grid = new Array(t * t).fill(null), this.positionHistory = new Set, s || this.positionHistory.add(this.hash);
        }
        idx(t, s) {
            return s * this.size + t;
        }
        inBounds(t, s) {
            return t >= 0 && s >= 0 && t < this.size && s < this.size;
        }
        get(t, s) {
            return this.grid[this.idx(t, s)];
        }
        set(t, s, o) {
            const i = this.idx(t, s), r = this.grid[i];
            r !== null && (this.hash ^= L[i * 2 + (r === "b" ? 0 : 1)]), this.grid[i] = o, o !== null && (this.hash ^= L[i * 2 + (o === "b" ? 0 : 1)]);
        }
        removeStone(t, s) {
            this.inBounds(t, s) && this.set(t, s, null);
        }
        setupStones(t) {
            for (const s of t)this.inBounds(s.x, s.y) && this.set(s.x, s.y, s.color);
        }
        get ko() {
            return this.koPoint;
        }
        clone() {
            const t = new S(this.size, !0);
            return t.grid = this.grid.slice(), t.koPoint = this.koPoint ? {
                ...this.koPoint
            } : null, t.positionHistory = new Set(this.positionHistory), t.hash = this.hash, t;
        }
        scratchClone() {
            const t = new S(this.size, !0);
            return t.grid = this.grid.slice(), t.koPoint = this.koPoint ? {
                ...this.koPoint
            } : null, t.positionHistory = new Set, t.hash = this.hash, t;
        }
        neighbors(t, s) {
            const o = [];
            return t > 0 && o.push({
                x: t - 1,
                y: s
            }), t < this.size - 1 && o.push({
                x: t + 1,
                y: s
            }), s > 0 && o.push({
                x: t,
                y: s - 1
            }), s < this.size - 1 && o.push({
                x: t,
                y: s + 1
            }), o;
        }
        groupAt(t, s) {
            const o = this.size, i = this.grid, r = s * o + t, p = i[r];
            if (p === null) return null;
            const u = [], n = [], g = o * o;
            (M.length < g || P >= 2147483647) && (M = new Int32Array(Math.max(g, M.length)), P = 0);
            const a = M, l = ++P, f = [
                r
            ];
            for(a[r] = l; f.length > 0;){
                const y = f.pop(), m = y % o, c = (y - m) / o;
                if (u.push({
                    x: m,
                    y: c
                }), m > 0) {
                    const e = y - 1, d = i[e];
                    d === null ? a[e] !== l && (a[e] = l, n.push({
                        x: m - 1,
                        y: c
                    })) : d === p && a[e] !== l && (a[e] = l, f.push(e));
                }
                if (m < o - 1) {
                    const e = y + 1, d = i[e];
                    d === null ? a[e] !== l && (a[e] = l, n.push({
                        x: m + 1,
                        y: c
                    })) : d === p && a[e] !== l && (a[e] = l, f.push(e));
                }
                if (c > 0) {
                    const e = y - o, d = i[e];
                    d === null ? a[e] !== l && (a[e] = l, n.push({
                        x: m,
                        y: c - 1
                    })) : d === p && a[e] !== l && (a[e] = l, f.push(e));
                }
                if (c < o - 1) {
                    const e = y + o, d = i[e];
                    d === null ? a[e] !== l && (a[e] = l, n.push({
                        x: m,
                        y: c + 1
                    })) : d === p && a[e] !== l && (a[e] = l, f.push(e));
                }
            }
            return {
                color: p,
                stones: u,
                liberties: n
            };
        }
        libertyCount(t, s) {
            return this.libCountFast(t, s);
        }
        libCountFast(t, s) {
            const o = this.size, i = this.grid, r = s * o + t, p = i[r];
            if (p === null) return 0;
            const u = o * o;
            (M.length < u || P >= 2147483646) && (M = new Int32Array(Math.max(u, M.length)), P = 0);
            const n = M, g = ++P, a = [
                r
            ];
            n[r] = g;
            let l = 0;
            for(; a.length > 0;){
                const f = a.pop(), y = f % o, m = (f - y) / o;
                if (y > 0) {
                    const c = f - 1, e = i[c];
                    n[c] !== g && (n[c] = g, e === null ? l++ : e === p && a.push(c));
                }
                if (y < o - 1) {
                    const c = f + 1, e = i[c];
                    n[c] !== g && (n[c] = g, e === null ? l++ : e === p && a.push(c));
                }
                if (m > 0) {
                    const c = f - o, e = i[c];
                    n[c] !== g && (n[c] = g, e === null ? l++ : e === p && a.push(c));
                }
                if (m < o - 1) {
                    const c = f + o, e = i[c];
                    n[c] !== g && (n[c] = g, e === null ? l++ : e === p && a.push(c));
                }
            }
            return l;
        }
        play(t) {
            if (t.point === null) return this.koPoint = null, this.positionHistory.add(this.hash), {
                ok: !0,
                captured: []
            };
            const { x: s, y: o } = t.point;
            if (!this.inBounds(s, o)) return {
                ok: !1,
                reason: "off-board"
            };
            if (this.get(s, o) !== null) return {
                ok: !1,
                reason: "occupied"
            };
            if (this.koPoint && H(this.koPoint, t.point)) return {
                ok: !1,
                reason: "ko"
            };
            const i = t.color, r = C(i);
            this.set(s, o, i);
            const p = [];
            for (const u of this.neighbors(s, o))if (this.get(u.x, u.y) === r && this.libCountFast(u.x, u.y) === 0) for (const n of this.groupAt(u.x, u.y).stones)this.set(n.x, n.y, null), p.push(n);
            if (this.libCountFast(s, o) === 0) return this.set(s, o, null), {
                ok: !1,
                reason: "suicide"
            };
            if (this.positionHistory.has(this.hash)) {
                this.set(s, o, null);
                for (const u of p)this.set(u.x, u.y, r);
                return {
                    ok: !1,
                    reason: "superko"
                };
            }
            if (this.koPoint = null, p.length === 1) {
                const u = this.groupAt(s, o);
                u.stones.length === 1 && u.liberties.length === 1 && (this.koPoint = p[0]);
            }
            return this.positionHistory.add(this.hash), {
                ok: !0,
                captured: p
            };
        }
        isLegal(t) {
            return t.point === null ? !0 : this.clone().play(t).ok;
        }
        key() {
            return this.serialize();
        }
        serialize() {
            let t = "";
            for (const s of this.grid)t += s === null ? "." : s;
            return t;
        }
        stones() {
            const t = [];
            for(let s = 0; s < this.size; s++)for(let o = 0; o < this.size; o++){
                const i = this.get(o, s);
                i !== null && t.push({
                    point: {
                        x: o,
                        y: s
                    },
                    color: i
                });
            }
            return t;
        }
    }
    function I(h, t, s, o, i, r, p, u) {
        const n = h.size, g = t === "b" ? o : 1 - o, a = t === "b" ? 1 : -1, l = new Array(n * n);
        let f = 0;
        for(let d = 0; d < n * n; d++){
            const z = i[d] * a;
            l[d] = z, f += z;
        }
        const y = f - s, m = (d, z)=>{
            const x = [];
            for(let k = 0; k < n; k++)for(let v = 0; v < n; v++)h.get(v, k) === null && x.push({
                point: {
                    x: v,
                    y: k
                },
                logit: d[k * n + v]
            });
            x.push({
                point: null,
                logit: z
            });
            const R = x.reduce((k, v)=>v.logit > k ? v.logit : k, -1 / 0), B = x.map((k)=>Math.exp(k.logit - R)), q = B.reduce((k, v)=>k + v, 0) || 1;
            return x.map((k, v)=>({
                    point: k.point,
                    prior: B[v] / q
                })).sort((k, v)=>v.prior - k.prior);
        }, c = m(r, p), e = {
            winrateBlack: g,
            scoreLeadBlack: y,
            ownership: l,
            topMoves: c
        };
        return u && (e.netScoreMeanBlack = u.scoreMeanMover * a, e.netScoreLeadBlack = u.leadMover * a, e.scoreStdev = u.scoreStdev, e.valueUncertainty = u.valueUncertainty, e.scoreUncertainty = u.scoreUncertainty, e.optimisticTopMoves = m(u.optPolicyLogits, u.optPassLogit)), e;
    }
    async function U(h) {
        const t = await import("./chunks/CDUQ8qGC.js").then(async (m)=>{
            await m.__tla;
            return m;
        }), s = new t.KataGoNet(h);
        return new E(s);
    }
    class E {
        name;
        runtime = "wasm";
        supportsRank;
        net;
        rank = null;
        visitCap = 24;
        constructor(t){
            this.net = t, this.supportsRank = t.hasMetaEncoder(), this.name = `saigo · ${t.summary()}`;
        }
        setRank(t) {
            this.rank = t;
        }
        glassesJson(t, s, o, i) {
            return this.net.glassesJson(t, s, o, i);
        }
        toArgs(t) {
            const s = t.board.size, o = Int32Array.from(t.history.map((r)=>r.point ? r.point.y * s + r.point.x : -1)), i = t.history.length > 0 ? t.history[0].color === "b" ? 1 : 2 : t.toMove === "b" ? 1 : 2;
            return {
                n: s,
                moveLocs: o,
                firstPlayer: i
            };
        }
        async evaluate(t) {
            const { n: s, moveLocs: o, firstPlayer: i } = this.toArgs(t), r = this.rank && this.supportsRank ? this.net.evaluateRanked(s, o, i, t.komi, this.rank) : this.net.evaluate(s, o, i, t.komi);
            return I(t.board, t.toMove, t.komi, r.winrate, r.ownership(), r.policy(), r.pass, {
                scoreMeanMover: r.scoreMean,
                scoreStdev: r.scoreStdev,
                leadMover: r.lead,
                valueUncertainty: r.valueUncertainty,
                scoreUncertainty: r.scoreUncertainty,
                optPolicyLogits: r.optimisticPolicy(),
                optPassLogit: r.optimisticPass
            });
        }
        async genMove(t, s = {}) {
            const o = await this.evaluate(t), i = (n)=>t.toMove === "b" ? n : 1 - n;
            if (s.resignThreshold !== void 0 && i(o.winrateBlack) < s.resignThreshold) return {
                move: {
                    color: t.toMove,
                    point: null
                },
                resign: !0,
                evaluation: o
            };
            const r = s.temperature ?? 0;
            let p;
            if (r <= 0) p = o.topMoves[0]?.point ?? null;
            else {
                const n = o.topMoves.slice(0, 12), g = n.map((f)=>Math.pow(f.prior, 1 / Math.max(r, .1))), a = g.reduce((f, y)=>f + y, 0);
                let l = G(g) % 1e3 / 1e3 * a;
                p = n[0]?.point ?? null;
                for(let f = 0; f < n.length; f++)if (l -= g[f], l <= 0) {
                    p = n[f].point;
                    break;
                }
            }
            return {
                move: {
                    color: t.toMove,
                    point: p
                },
                resign: !1,
                evaluation: o
            };
        }
    }
    function G(h) {
        let t = 2166136261;
        for (const s of h)t ^= Math.floor(s * 1e6), t = Math.imul(t, 16777619);
        return Math.abs(t);
    }
    function A(h) {
        const t = new S(h.size);
        return t.setupStones(h.stones), {
            board: t,
            toMove: h.toMove,
            komi: h.komi,
            history: h.history
        };
    }
    const w = self;
    let b = null;
    w.onmessage = async (h)=>{
        const t = h.data;
        try {
            switch(t.type){
                case "init":
                    {
                        b = U(t.bytes);
                        const s = await b;
                        w.postMessage({
                            id: t.id,
                            ok: !0,
                            result: {
                                name: s.name,
                                supportsRank: s.supportsRank,
                                visitCap: s.visitCap
                            }
                        });
                        break;
                    }
                case "evaluate":
                    {
                        const s = await b;
                        w.postMessage({
                            id: t.id,
                            ok: !0,
                            result: await s.evaluate(A(t.pos))
                        });
                        break;
                    }
                case "evaluateBatch":
                    {
                        const s = await b, o = [];
                        for (const i of t.positions)o.push(await s.evaluate(A(i)));
                        w.postMessage({
                            id: t.id,
                            ok: !0,
                            result: o
                        });
                        break;
                    }
                case "genMove":
                    {
                        const s = await b;
                        w.postMessage({
                            id: t.id,
                            ok: !0,
                            result: await s.genMove(A(t.pos), t.opts)
                        });
                        break;
                    }
                case "setRank":
                    {
                        (await b).setRank(t.rank), w.postMessage({
                            id: t.id,
                            ok: !0,
                            result: null
                        });
                        break;
                    }
                case "glasses":
                    {
                        const s = await b;
                        w.postMessage({
                            id: t.id,
                            ok: !0,
                            result: s.glassesJson(t.n, Int32Array.from(t.moveLocs), t.firstPlayer, t.komi)
                        });
                        break;
                    }
            }
        } catch (s) {
            w.postMessage({
                id: t.id,
                ok: !1,
                error: s instanceof Error ? s.message : String(s)
            });
        }
    };
})();
