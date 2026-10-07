# KataGo CUDA(b8c96h3tfrs-s68320512)vs AetherGo WebGPU(aewnn)1024v 吞吐
日期:2026-10-07 | 环境:同一块 NVIDIA dGPU(WSL CUDA vs Chrome WebGPU/D3D12)
命令:`katago benchmark -model export_bin/b8c96h3tfrs-s68320512.bin.gz -config gtp_b8c96.cfg -config ae_rules.cfg -visits 1024 -t 8,16,32,64 -numpositions 3`
AetherGo 口径:nnSearchBest visits=1024, maxBatch=T, reuseTree:false, 3 局面(空/中/官),对称默认开。

## KataGo CUDA(numSearchThreads = T)
| T | visits/s | nnEvals/s | avgBatchSize |
|---|---|---|---|
| 8 | 2528.59 | 2015.18 | 3.86 |
| 16 | 4467.64 | 3723.75 | 9.38 |
| 32 | 4848.29 | 4456.14 | 22.04 |
| 64 | 4904.75 | 4743.71 | 44.39 |

## AetherGo WebGPU(visits/s)
| maxBatch | onnx(ORT-Web) | aewnn f32 | aewnn i8 |
|---|---|---|---|
| 8 | 245.0 | 443.4 | 456.3 |
| 16 | 255.5 | 516.3 | 483.7 |
| 32 | 293.1 | 554.6 | 492.7 |
| 64 | 346.6 | 557.3 | 555.4 |
