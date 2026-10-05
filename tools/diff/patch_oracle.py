#!/usr/bin/env python3
"""KataGo onnxbackend.cpp 插桩:KATA_NNSERVER=host:port 时推理转发共享 oracle。"""
src = open("/home/a/go/KataGo/cpp/neuralnet/onnxbackend.cpp").read()

anchor = '#include "../neuralnet/onnxmodelbuilder.h"'
if anchor not in src:
    raise SystemExit("anchor include not found")
guard = anchor + """
#if !defined(_WIN32)
// AetherGo 差分调试:KATA_NNSERVER 共享推理 oracle 所需
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
"""
if "KATA_NNSERVER 共享推理 oracle 所需" not in src:
    src = src.replace(anchor, guard, 1)

old_run = """  auto outputTensors = gpuHandle->session->Run(
    Ort::RunOptions{nullptr},
    gpuHandle->inputNamePtrs.data(),
    inputTensors.data(),
    inputTensors.size(),
    gpuHandle->outputNamePtrs.data(),
    gpuHandle->outputNamePtrs.size());
"""
new_run = """  // ===== AetherGo 差分调试:KATA_NNSERVER=host:port 时推理转发共享 oracle =====
  // (同输入位级同输出,与 JS 侧同一进程;不设则走本地 ORT)
  vector<Ort::Value> outputTensors;
  std::vector<std::vector<float>> oracleStore(5);
  bool usedOracle = false;
#if !defined(_WIN32)
  do {
    const char* nnServerEnv = std::getenv("KATA_NNSERVER");
    if(nnServerEnv == nullptr || runBatchSize <= 0) break;
    std::string nnServer(nnServerEnv);
    size_t colon = nnServer.rfind((char)58);
    if(colon == std::string::npos) throw StringError("KATA_NNSERVER expects host:port");
    std::string host = nnServer.substr(0, colon);
    int port = std::atoi(nnServer.c_str() + colon + 1);
    static int sockFd = -1;
    if(sockFd < 0) {
      sockFd = socket(AF_INET, SOCK_STREAM, 0);
      if(sockFd < 0) throw StringError("oracle socket create fail");
      sockaddr_in addr;
      std::memset(&addr, 0, sizeof(addr));
      addr.sin_family = AF_INET;
      addr.sin_port = htons((uint16_t)port);
      addr.sin_addr.s_addr = inet_addr(host.empty() || host == "localhost" ? "127.0.0.1" : host.c_str());
      if(connect(sockFd, (sockaddr*)&addr, sizeof(addr)) != 0)
        throw StringError("oracle connect fail " + nnServer);
    }
    auto sendAll = [&](const void* p, size_t n) -> void {
      const char* c = (const char*)p; size_t done = 0;
      while(done < n) { ssize_t k = send(sockFd, c + done, n - done, 0); if(k <= 0) throw StringError("oracle send fail"); done += (size_t)k; }
    };
    auto recvAll = [&](void* p, size_t n) -> void {
      char* c = (char*)p; size_t done = 0;
      while(done < n) { ssize_t k = recv(sockFd, c + done, n - done, 0); if(k <= 0) throw StringError("oracle recv fail"); done += (size_t)k; }
    };
    const float* spatBase = inputBuffers->spatialInput.data();
    const float* globBase = inputBuffers->globalInput.data();
    const int64_t nS = inputBuffers->singleInputElts;
    const int64_t nG = inputBuffers->singleInputGlobalElts;
    for(int row = 0; row < runBatchSize; row++) {
      uint32_t magic = 0x314F4741;
      sendAll(&magic, 4);
      uint32_t cs = (uint32_t)nS, cg = (uint32_t)nG;
      sendAll(&cs, 4);
      sendAll(spatBase + (size_t)row * nS, 4 * (size_t)nS);
      sendAll(&cg, 4);
      sendAll(globBase + (size_t)row * nG, 4 * (size_t)nG);
      uint32_t rmagic; uint32_t cnt[5];
      recvAll(&rmagic, 4);
      if(rmagic != magic) throw StringError("oracle bad magic");
      recvAll(cnt, 20);
      for(int k = 0; k < 5; k++) {
        size_t oldLen = oracleStore[k].size();
        oracleStore[k].resize(oldLen + cnt[k]);
        if(cnt[k] > 0) recvAll(oracleStore[k].data() + oldLen, 4 * (size_t)cnt[k]);
      }
    }
    // 按输出名对位包装为 Ort::Value(形状与原图一致)
    const int64_t B = runBatchSize;
    const int64_t XY = (int64_t)nnXLen * nnYLen;
    auto mkVal = [&](int k, std::array<int64_t,4> shape) -> Ort::Value {
      return Ort::Value::CreateTensor<float>(memInfo, oracleStore[k].data(), oracleStore[k].size(), shape.data(), shape.size());
    };
    for(size_t i = 0; i < gpuHandle->outputNames.size(); i++) {
      const std::string& nm = gpuHandle->outputNames[i];
      if(nm.find("PolicyPass") != std::string::npos)
        outputTensors.push_back(mkVal(0, {B, (int64_t)(oracleStore[0].size() / B), 1, 1}));
      else if(nm.find("ScoreValue") != std::string::npos)
        outputTensors.push_back(mkVal(3, {B, (int64_t)(oracleStore[3].size() / B), 1, 1}));
      else if(nm.find("Policy") != std::string::npos)
        outputTensors.push_back(mkVal(1, {B, (int64_t)(oracleStore[1].size() / (B * XY)), nnYLen, nnXLen}));
      else if(nm.find("Value") != std::string::npos)
        outputTensors.push_back(mkVal(2, {B, (int64_t)(oracleStore[2].size() / B), 1, 1}));
      else if(nm.find("Ownership") != std::string::npos)
        outputTensors.push_back(mkVal(4, {B, (int64_t)(oracleStore[4].size() / (B * XY)), nnYLen, nnXLen}));
      else
        throw StringError("oracle: unexpected output name " + nm);
    }
    usedOracle = true;
  } while(false);
#endif
  if(!usedOracle)
    outputTensors = gpuHandle->session->Run(
      Ort::RunOptions{nullptr},
      gpuHandle->inputNamePtrs.data(),
      inputTensors.data(),
      inputTensors.size(),
      gpuHandle->outputNamePtrs.data(),
      gpuHandle->outputNamePtrs.size());
"""
if old_run not in src:
    if "KATA_NNSERVER=host:port" not in src:
        raise SystemExit("run block not found and not already patched")
    print("already patched")
else:
    src = src.replace(old_run, new_run, 1)

open("/home/a/go/KataGo/cpp/neuralnet/onnxbackend.cpp", "w").write(src)
print("patched onnxbackend.cpp")
