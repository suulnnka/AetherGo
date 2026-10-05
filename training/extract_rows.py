#!/usr/bin/env python3
"""AetherGo 特征对拍数据提取:katago selfplay 的 sgf.gz + 训练 npz → JSON。

用法:python3 extract_rows.py <sgf.gz 或 sgf 目录> <npz 文件或目录> <out.json>

输出 JSON:{ games: [ { komi, moves: [点 0..80 或 81=pass], rows: N,
                         spatial: [每行 22*81 个 0/1], global: [每行 19 个 float] } ] }
行序 = 着序(selfplay_featdiff.cfg 保证一行一手)。
"""
import json
import glob
import gzip
import os
import re
import sys

import numpy as np

PASS = 81


def parse_sgf(path):
    """解析 katago 自对弈 SGF:返回 (komi, moves)。坐标 (x=列, y=行, a=0,顶行)。"""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    komi = 5.5
    m = re.search(r'KM\[([^\]]*)\]', text)
    if m:
        komi = float(m.group(1))
    moves = []
    # 逐个取 ;B[..] / ;W[..]
    for mm in re.finditer(r';([BW])\[([a-zA-Z]{0,2})\]', text):
        coord = mm.group(2)
        if coord == '' or coord == 'tt':
            moves.append(PASS)
        else:
            x = ord(coord[0]) - ord('a')
            y = ord(coord[1]) - ord('a')
            moves.append(y * 9 + x)
    return komi, moves


def collect_sgfs(sgf_arg):
    """收集 SGF 文本:.sgf(.gz) 单局,或 .sgfs(katago selfplay,每行一局)。"""
    texts = []
    if os.path.isdir(sgf_arg):
        files = sorted(glob.glob(os.path.join(sgf_arg, '**', '*.sgf*'), recursive=True))
    else:
        files = [sgf_arg]
    for path in files:
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
            t = f.read()
        if path.endswith('.sgfs') or t.count('(;GM') > 1 or t.count('\n(;') > 0:
            texts.extend(line.strip() for line in t.splitlines() if line.strip().startswith('('))
        else:
            texts.append(t)
    return texts


def parse_sgf_text(text):
    komi = 5.5
    m = re.search(r'KM\[([^\]]*)\]', text)
    if m:
        komi = float(m.group(1))
    moves = []
    for mm in re.finditer(r';([BW])\[([a-zA-Z]{0,2})\]', text):
        coord = mm.group(2)
        if coord == '' or coord == 'tt':
            moves.append(PASS)
        else:
            x = ord(coord[0]) - ord('a')
            y = ord(coord[1]) - ord('a')
            moves.append(y * 9 + x)
    return komi, moves


def main():
    sgf_arg, npz_arg, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    texts = collect_sgfs(sgf_arg)
    npzs = sorted(glob.glob(os.path.join(npz_arg, '**', '*.npz'), recursive=True)) if os.path.isdir(npz_arg) else [npz_arg]

    games = []
    pending_rows = 0
    for npz_path in npzs:
        z = np.load(npz_path)
        # binaryInputNCHWPacked:(N, 22, (81+7)/8) uint8 —— 每通道 81 bit,
        # 每 bit MSB-first(bit i = byte[i/8] >> (7-i%8) & 1)
        packed = z['binaryInputNCHWPacked']
        n, nch, nbytes = packed.shape
        bits = np.unpackbits(packed, axis=2, bitorder='big')[:, :, :81]  # (N,22,81)
        binary = bits.astype(np.uint8)
        globalin = z['globalInputNC']      # (N, 19, 1, 1) float32
        n = binary.shape[0]
        pending_rows += n
        games.append({
            'nrows': int(n),
            'spatial': binary.reshape(n, -1).astype(np.uint8).tolist(),
            'global': globalin.reshape(n, -1).astype(np.float64).tolist(),
        })

    for g in games:
        g.pop('npz', None)

    # SGF 按序配对:npz 块按「局完成序」混装各行,逐局顺序切分
    meta = []
    for text in texts:
        komi, moves = parse_sgf_text(text)
        meta.append({'komi': komi, 'moves': moves})
    out = {'games': [], 'totalRows': pending_rows, 'totalMoves': sum(len(m['moves']) for m in meta)}
    rowPool = []
    for g in games:
        for r in range(g['nrows']):
            rowPool.append((g['spatial'][r], g['global'][r]))
    cursor = 0
    for m in meta:
        n = len(m['moves'])
        if cursor + n > len(rowPool):
            print(f'! 警告:行池不够(需 {cursor + n},有 {len(rowPool)})')
            break
        out['games'].append({
            'komi': m['komi'],
            'moves': m['moves'],
            'rows': n,
            'spatial': [rowPool[cursor + i][0] for i in range(n)],
            'global': [rowPool[cursor + i][1] for i in range(n)],
        })
        cursor += n
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f'sgfs={len(texts)} npzs={len(npzs)} moves={out["totalMoves"]} rows={pending_rows} → {out_path}')


if __name__ == '__main__':
    main()
