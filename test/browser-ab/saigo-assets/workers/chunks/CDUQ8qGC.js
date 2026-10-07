let k, F, W, V, J, q;
let __tla = (async ()=>{
    var S = "" + new URL("../assets/saigo_net_bg-B0LxIosR.wasm", import.meta.url).href, B = async (n = {}, e)=>{
        let r;
        if (e.startsWith("data:")) {
            const _ = e.replace(/^data:.*?base64,/, "");
            let a;
            if (typeof Buffer == "function" && typeof Buffer.from == "function") a = Buffer.from(_, "base64");
            else if (typeof atob == "function") {
                const i = atob(_);
                a = new Uint8Array(i.length);
                for(let o = 0; o < i.length; o++)a[o] = i.charCodeAt(o);
            } else throw new Error("Cannot decode base64-encoded data URL");
            r = await WebAssembly.instantiate(a, n);
        } else {
            const _ = await fetch(e), a = _.headers.get("Content-Type") || "";
            if ("instantiateStreaming" in WebAssembly && a.startsWith("application/wasm")) r = await WebAssembly.instantiateStreaming(_, n);
            else {
                const i = await _.arrayBuffer();
                r = await WebAssembly.instantiate(i, n);
            }
        }
        return r.instance.exports;
    };
    let t;
    W = function(n) {
        t = n;
    };
    const P = typeof TextDecoder > "u" ? (0, module.require)("util").TextDecoder : TextDecoder;
    let M = new P("utf-8", {
        ignoreBOM: !0,
        fatal: !0
    });
    M.decode();
    let b = null;
    function u() {
        return (b === null || b.byteLength === 0) && (b = new Uint8Array(t.memory.buffer)), b;
    }
    function d(n, e) {
        return n = n >>> 0, M.decode(u().subarray(n, n + e));
    }
    let f = null;
    function L() {
        return (f === null || f.byteLength === 0) && (f = new Float32Array(t.memory.buffer)), f;
    }
    function m(n, e) {
        return n = n >>> 0, L().subarray(n / 4, n / 4 + e);
    }
    let p = null;
    function z() {
        return (p === null || p.byteLength === 0) && (p = new Uint32Array(t.memory.buffer)), p;
    }
    let l = 0;
    function h(n, e) {
        const r = e(n.length * 4, 4) >>> 0;
        return z().set(n, r / 4), l = n.length, r;
    }
    function R(n, e) {
        const r = e(n.length * 1, 1) >>> 0;
        return u().set(n, r / 1), l = n.length, r;
    }
    const G = typeof TextEncoder > "u" ? (0, module.require)("util").TextEncoder : TextEncoder;
    let v = new G("utf-8");
    const I = typeof v.encodeInto == "function" ? function(n, e) {
        return v.encodeInto(n, e);
    } : function(n, e) {
        const r = v.encode(n);
        return e.set(r), {
            read: n.length,
            written: r.length
        };
    };
    function A(n, e, r) {
        if (r === void 0) {
            const s = v.encode(n), c = e(s.length, 1) >>> 0;
            return u().subarray(c, c + s.length).set(s), l = s.length, c;
        }
        let _ = n.length, a = e(_, 1) >>> 0;
        const i = u();
        let o = 0;
        for(; o < _; o++){
            const s = n.charCodeAt(o);
            if (s > 127) break;
            i[a + o] = s;
        }
        if (o !== _) {
            o !== 0 && (n = n.slice(o)), a = r(a, _, _ = o + n.length * 3, 1) >>> 0;
            const s = u().subarray(a + o, a + _), c = I(n, s);
            o += c.written, a = r(a, _, o, 1) >>> 0;
        }
        return l = o, a;
    }
    function U(n) {
        const e = t.__wbindgen_export_0.get(n);
        return t.__externref_table_dealloc(n), e;
    }
    let y = null;
    function D() {
        return (y === null || y.byteLength === 0) && (y = new Float64Array(t.memory.buffer)), y;
    }
    function O(n, e) {
        return n = n >>> 0, D().subarray(n / 8, n / 8 + e);
    }
    typeof FinalizationRegistry > "u" || new FinalizationRegistry((n)=>t.__wbg_encodedtensors_free(n >>> 0, 1));
    const E = typeof FinalizationRegistry > "u" ? {
        register: ()=>{},
        unregister: ()=>{}
    } : new FinalizationRegistry((n)=>t.__wbg_evalresult_free(n >>> 0, 1));
    k = class {
        static __wrap(e) {
            e = e >>> 0;
            const r = Object.create(k.prototype);
            return r.__wbg_ptr = e, E.register(r, r.__wbg_ptr, r), r;
        }
        __destroy_into_raw() {
            const e = this.__wbg_ptr;
            return this.__wbg_ptr = 0, E.unregister(this), e;
        }
        free() {
            const e = this.__destroy_into_raw();
            t.__wbg_evalresult_free(e, 0);
        }
        get scoreMean() {
            return t.evalresult_scoreMean(this.__wbg_ptr);
        }
        get scoreStdev() {
            return t.evalresult_scoreStdev(this.__wbg_ptr);
        }
        get optimisticPass() {
            return t.evalresult_optimisticPass(this.__wbg_ptr);
        }
        optimisticPolicy() {
            const e = t.evalresult_optimisticPolicy(this.__wbg_ptr);
            var r = m(e[0], e[1]).slice();
            return t.__wbindgen_free(e[0], e[1] * 4, 4), r;
        }
        get scoreUncertainty() {
            return t.evalresult_scoreUncertainty(this.__wbg_ptr);
        }
        get valueUncertainty() {
            return t.evalresult_valueUncertainty(this.__wbg_ptr);
        }
        get lead() {
            return t.evalresult_lead(this.__wbg_ptr);
        }
        get pass() {
            return t.evalresult_pass(this.__wbg_ptr);
        }
        seki() {
            const e = t.evalresult_seki(this.__wbg_ptr);
            var r = m(e[0], e[1]).slice();
            return t.__wbindgen_free(e[0], e[1] * 4, 4), r;
        }
        policy() {
            const e = t.evalresult_policy(this.__wbg_ptr);
            var r = m(e[0], e[1]).slice();
            return t.__wbindgen_free(e[0], e[1] * 4, 4), r;
        }
        get winrate() {
            return t.evalresult_winrate(this.__wbg_ptr);
        }
        ownership() {
            const e = t.evalresult_ownership(this.__wbg_ptr);
            var r = m(e[0], e[1]).slice();
            return t.__wbindgen_free(e[0], e[1] * 4, 4), r;
        }
    };
    const x = typeof FinalizationRegistry > "u" ? {
        register: ()=>{},
        unregister: ()=>{}
    } : new FinalizationRegistry((n)=>t.__wbg_katagonet_free(n >>> 0, 1));
    F = class {
        static __wrap(e) {
            e = e >>> 0;
            const r = Object.create(F.prototype);
            return r.__wbg_ptr = e, x.register(r, r.__wbg_ptr, r), r;
        }
        __destroy_into_raw() {
            const e = this.__wbg_ptr;
            return this.__wbg_ptr = 0, x.unregister(this), e;
        }
        free() {
            const e = this.__destroy_into_raw();
            t.__wbg_katagonet_free(e, 0);
        }
        static fromSaigo(e, r) {
            const _ = A(e, t.__wbindgen_malloc, t.__wbindgen_realloc), a = l, i = R(r, t.__wbindgen_malloc), o = l, s = t.katagonet_fromSaigo(_, a, i, o);
            if (s[2]) throw U(s[1]);
            return F.__wrap(s[0]);
        }
        inferRank(e, r, _, a, i, o) {
            const s = h(r, t.__wbindgen_malloc), c = l, g = A(o, t.__wbindgen_malloc, t.__wbindgen_realloc), j = l, w = t.katagonet_inferRank(this.__wbg_ptr, e, s, c, _, a, i, g, j);
            var T = O(w[0], w[1]).slice();
            return t.__wbindgen_free(w[0], w[1] * 8, 8), T;
        }
        get numBlocks() {
            return t.katagonet_numBlocks(this.__wbg_ptr) >>> 0;
        }
        get reachedEof() {
            return t.katagonet_reachedEof(this.__wbg_ptr) !== 0;
        }
        glassesJson(e, r, _, a) {
            let i, o;
            try {
                const s = h(r, t.__wbindgen_malloc), c = l, g = t.katagonet_glassesJson(this.__wbg_ptr, e, s, c, _, a);
                return i = g[0], o = g[1], d(g[0], g[1]);
            } finally{
                t.__wbindgen_free(i, o, 1);
            }
        }
        get totalParams() {
            return t.katagonet_totalParams(this.__wbg_ptr);
        }
        get modelVersion() {
            return t.katagonet_modelVersion(this.__wbg_ptr);
        }
        get trunkChannels() {
            return t.katagonet_trunkChannels(this.__wbg_ptr) >>> 0;
        }
        evaluateRanked(e, r, _, a, i) {
            const o = h(r, t.__wbindgen_malloc), s = l, c = A(i, t.__wbindgen_malloc, t.__wbindgen_realloc), g = l, j = t.katagonet_evaluateRanked(this.__wbg_ptr, e, o, s, _, a, c, g);
            return k.__wrap(j);
        }
        hasMetaEncoder() {
            return t.katagonet_hasMetaEncoder(this.__wbg_ptr) !== 0;
        }
        get numGpoolBlocks() {
            return t.katagonet_numGpoolBlocks(this.__wbg_ptr) >>> 0;
        }
        get numInputChannels() {
            return t.katagonet_numInputChannels(this.__wbg_ptr) >>> 0;
        }
        get numGlobalChannels() {
            return t.katagonet_numGlobalChannels(this.__wbg_ptr) >>> 0;
        }
        constructor(e){
            const r = R(e, t.__wbindgen_malloc), _ = l, a = t.katagonet_new(r, _);
            if (a[2]) throw U(a[1]);
            return this.__wbg_ptr = a[0] >>> 0, x.register(this, this.__wbg_ptr, this), this;
        }
        get name() {
            let e, r;
            try {
                const _ = t.katagonet_name(this.__wbg_ptr);
                return e = _[0], r = _[1], d(_[0], _[1]);
            } finally{
                t.__wbindgen_free(e, r, 1);
            }
        }
        summary() {
            let e, r;
            try {
                const _ = t.katagonet_summary(this.__wbg_ptr);
                return e = _[0], r = _[1], d(_[0], _[1]);
            } finally{
                t.__wbindgen_free(e, r, 1);
            }
        }
        evaluate(e, r, _, a) {
            const i = h(r, t.__wbindgen_malloc), o = l, s = t.katagonet_evaluate(this.__wbg_ptr, e, i, o, _, a);
            return k.__wrap(s);
        }
    };
    V = function() {
        const n = t.__wbindgen_export_0, e = n.grow(4);
        n.set(0, void 0), n.set(e + 0, void 0), n.set(e + 1, null), n.set(e + 2, !0), n.set(e + 3, !1);
    };
    J = function(n, e) {
        return d(n, e);
    };
    q = function(n, e) {
        throw new Error(d(n, e));
    };
    URL = globalThis.URL;
    const N = await B({
        "./saigo_net_bg.js": {
            __wbindgen_string_new: J,
            __wbindgen_throw: q,
            __wbindgen_init_externref_table: V
        }
    }, S), { memory: $, __wbg_encodedtensors_free: H, __wbg_evalresult_free: K, __wbg_katagonet_free: Q, encode_position: X, encodedtensors_global: Y, encodedtensors_spatial: Z, evalresult_lead: ee, evalresult_optimisticPass: te, evalresult_optimisticPolicy: ne, evalresult_ownership: re, evalresult_pass: _e, evalresult_policy: ae, evalresult_scoreMean: oe, evalresult_scoreStdev: se, evalresult_scoreUncertainty: ie, evalresult_seki: le, evalresult_valueUncertainty: ce, evalresult_winrate: ge, katagonet_evaluate: ue, katagonet_evaluateRanked: de, katagonet_fromSaigo: we, katagonet_glassesJson: be, katagonet_hasMetaEncoder: fe, katagonet_inferRank: me, katagonet_modelVersion: pe, katagonet_name: he, katagonet_new: ye, katagonet_numBlocks: ve, katagonet_numGlobalChannels: ke, katagonet_numGpoolBlocks: je, katagonet_numInputChannels: Ae, katagonet_reachedEof: xe, katagonet_summary: Fe, katagonet_totalParams: Re, katagonet_trunkChannels: Ue, move_shape_game_json: Ee, move_tesuji_game_json: Me, net_read_reliability: Ce, oracle_contingency_json: Te, oracle_enclosure_json: Se, oracle_group_graph_json: Be, oracle_move_attribution_json: We, oracle_ownership: Pe, oracle_ownership_board: Le, oracle_ownership_net_vetoed: ze, oracle_potential: Ge, position_facts_board_json: Ie, position_facts_json: De, tsumego_solve_board_json: Oe, saigo_ownership_ld: Ve, saigo_urgent_moves: Je, saigo_vital_moves: qe, __wbindgen_export_0: Ne, __wbindgen_free: $e, __wbindgen_malloc: He, __wbindgen_realloc: Ke, __externref_table_dealloc: Qe, __wbindgen_start: C } = N;
    var Xe = Object.freeze({
        __proto__: null,
        __externref_table_dealloc: Qe,
        __wbg_encodedtensors_free: H,
        __wbg_evalresult_free: K,
        __wbg_katagonet_free: Q,
        __wbindgen_export_0: Ne,
        __wbindgen_free: $e,
        __wbindgen_malloc: He,
        __wbindgen_realloc: Ke,
        __wbindgen_start: C,
        encode_position: X,
        encodedtensors_global: Y,
        encodedtensors_spatial: Z,
        evalresult_lead: ee,
        evalresult_optimisticPass: te,
        evalresult_optimisticPolicy: ne,
        evalresult_ownership: re,
        evalresult_pass: _e,
        evalresult_policy: ae,
        evalresult_scoreMean: oe,
        evalresult_scoreStdev: se,
        evalresult_scoreUncertainty: ie,
        evalresult_seki: le,
        evalresult_valueUncertainty: ce,
        evalresult_winrate: ge,
        katagonet_evaluate: ue,
        katagonet_evaluateRanked: de,
        katagonet_fromSaigo: we,
        katagonet_glassesJson: be,
        katagonet_hasMetaEncoder: fe,
        katagonet_inferRank: me,
        katagonet_modelVersion: pe,
        katagonet_name: he,
        katagonet_new: ye,
        katagonet_numBlocks: ve,
        katagonet_numGlobalChannels: ke,
        katagonet_numGpoolBlocks: je,
        katagonet_numInputChannels: Ae,
        katagonet_reachedEof: xe,
        katagonet_summary: Fe,
        katagonet_totalParams: Re,
        katagonet_trunkChannels: Ue,
        memory: $,
        move_shape_game_json: Ee,
        move_tesuji_game_json: Me,
        net_read_reliability: Ce,
        oracle_contingency_json: Te,
        oracle_enclosure_json: Se,
        oracle_group_graph_json: Be,
        oracle_move_attribution_json: We,
        oracle_ownership: Pe,
        oracle_ownership_board: Le,
        oracle_ownership_net_vetoed: ze,
        oracle_potential: Ge,
        position_facts_board_json: Ie,
        position_facts_json: De,
        saigo_ownership_ld: Ve,
        saigo_urgent_moves: Je,
        saigo_vital_moves: qe,
        tsumego_solve_board_json: Oe
    });
    W(Xe);
    C();
})();
export { k as EvalResult, F as KataGoNet, W as __wbg_set_wasm, V as __wbindgen_init_externref_table, J as __wbindgen_string_new, q as __wbindgen_throw, __tla };
