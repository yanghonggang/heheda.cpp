# 硬件环境分析与优化方案

## 一、硬件环境

| 组件 | 配置 | 状态 |
|------|------|------|
| CPU | Intel i7-14700KF (20核/28线程, 5.6GHz) | ✅ 足够 |
| GPU | NVIDIA RTX 4070 Ti SUPER (16GB VRAM) | ✅ 足够 |
| 内存 | 64GB DDR5 | ✅ 足够 |
| 磁盘 | NVMe SSD，/home 剩余 98GB | ✅ 充足 |

---

## 二、模型信息

- **模型**: qwen3:14b-q4_K_M
- **大小**: 9.3 GB
- **量化**: Q4_K_M (4-bit quantization, K-quant Medium)
- **参数量**: 14B

---

## 三、存储空间计算

```
模型文件:          9.3 GB
KV Cache (默认):  ~2-4 GB (取决于上下文长度和量化)
临时空间:          ~1 GB
总计需求:         ~13-15 GB

/home 可用:       98 GB
剩余空间:         ~83 GB (充足)
```

---

## 四、优化配置

### 1. GPU 全量加载（推荐）

```bash
# 14B模型 + 16GB显存，可以完全放入GPU
llama-cli -m qwen3-14b-q4_k_m.gguf \
  -ngl 99 \          # 全部层放入GPU
  -c 8192 \          # 上下文长度
  -t 8 \             # CPU线程数（留一半给系统）
  --mlock \          # 锁定内存防止换出
  --no-mmap          # 禁用mmap，全量加载到GPU
```

### 2. 服务器模式（最佳性能）

```bash
llama-server -m qwen3-14b-q4_k_m.gguf \
  -ngl 99 \              # GPU层数
  --flash-attn \         # FlashAttention2，减少显存占用
  -c 16384 \             # 上下文长度
  --cache-type-k q8_0 \  # K cache 量化
  --cache-type-v q8_0 \  # V cache 量化
  -t 8 \                 # CPU线程数
  --host 0.0.0.0 \
  --port 8080
```

---

## 五、用户当前配置分析

### 当前启动命令

```bash
./build/bin/llama-server \
  -m qwen3.gguf \
  -ngl 99 \
  -c 32768 \
  --jinja \
  --host 0.0.0.0 \
  --port 9999
```

### 配置问题

| 问题 | 说明 |
|------|------|
| ❌ 上下文过大 | `-c 32768` 导致 KV cache 占用过多显存 |
| ❌ 无 FlashAttention | 缺少 `--flash-attn`，显存效率低 |
| ❌ 无 KV cache 量化 | 缺少 `--cache-type-k/v`，显存浪费 |
| ❌ 无采样参数 | 影响生成质量 |

### 显存占用计算

```
Q4_K_M 模型:         9.3 GB
KV Cache (c=32768):  ~5-7 GB  ⚠️ 过大
系统开销:            ~1 GB
总计:                ~15-17 GB  ⚠️ 接近或超出显存
```

---

## 六、显存优化方案

### 显存规格

- **GPU**: RTX 4070 Ti SUPER
- **VRAM**: 16GB

### 方案对比

| 方案 | 模型大小 | KV Cache | 总显存 | 质量 | 推荐 |
|------|---------|----------|--------|------|------|
| Q4_K_M (当前) | 9.3GB | 5-7GB | 15-17GB | ⭐⭐ | ⚠️ |
| Q4_K_M + 优化 | 9.3GB | 3GB | 13GB | ⭐⭐ | ✅ |
| Q5_K_M | 11GB | 4GB | 15GB | ⭐⭐⭐ | ✅ |
| Q8_0 + 部分卸载 | 18GB(部分) | 3GB | 16GB | ⭐⭐⭐⭐ | ⭐ |

### 方案1：当前模型 + KV cache 优化（立即可用）

```bash
./build/bin/llama-server \
  -m qwen3.gguf \
  -ngl 99 \
  -c 16384 \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --host 0.0.0.0 \
  --port 9999
```

**显存占用**: ~13GB ✅

### 方案2：Q5_K_M（平衡方案）

```bash
./build/bin/llama-server \
  -m qwen3-14b-q5_k_m.gguf \
  -ngl 99 \
  -c 16384 \
  --flash-attn \
  --jinja \
  --host 0.0.0.0 \
  --port 9999
```

**显存占用**: ~15GB ✅

### 方案3：Q8_0 + 部分卸载（最佳质量）

```bash
./build/bin/llama-server \
  -m qwen3-14b-q8_0.gguf \
  -ngl 40 \
  -c 8192 \
  --flash-attn \
  --jinja \
  --host 0.0.0.0 \
  --port 9999
```

**显存占用**: ~16GB ✅（需要调试 -ngl 参数）

---

## 七、关键优化参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `-ngl` | 99 | GPU层数，14B模型建议全放GPU |
| `-c` | 8192-16384 | 上下文长度，影响KV cache大小 |
| `-t` | 8-12 | CPU线程数 |
| `--flash-attn` | 启用 | FlashAttention2，减少显存占用 |
| `--cache-type-k` | q8_0 | KV cache量化，节省显存 |
| `--cache-type-v` | q8_0 | V cache量化，节省显存 |
| `--mlock` | 启用 | 防止模型被换出内存 |
| `--no-mmap` | 启用 | 全量加载，避免mmap开销 |

---

## 八、性能预期

| 指标 | 预期值 |
|------|--------|
| 首token延迟 | 100-200ms |
| 生成速度 | 30-50 tokens/s |
| 显存占用 | 12-14 GB |
| 上下文长度 | 8K-16K |

---

## 九、与 MiMo V2.5 Free 对比

| 维度 | Qwen3-14B (本地) | MiMo V2.5 Free (云端) |
|------|------------------|----------------------|
| 参数量 | 14B | 7B |
| 量化精度 | Q4_K_M | FP16/BF16 |
| 基座能力 | ✅ 更强 | 一般 |
| RL优化 | ❌ 无 | ✅ 有 |
| 延迟 | ✅ 本地，低 | ⚠️ 网络延迟 |
| 隐私 | ✅ 本地 | ❌ 云端 |
| 成本 | ✅ 一次性 | ⚠️ 按量付费 |

**结论**：Qwen3-14B 基座能力更强，但缺少 RL 训练。通过优化可接近但难完全超越 MiMo V2.5 Free 的服务质量。

---

## 十、启动脚本

```bash
#!/bin/bash
# start-qwen3.sh

MODEL_PATH="./models/qwen3-14b-q4_k_m.gguf"

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 启动服务器
llama-server \
  -m "$MODEL_PATH" \
  -ngl 99 \
  --flash-attn \
  -c 16384 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -t 8 \
  --host 0.0.0.0 \
  --port 8080
```

---

## 十一、监控与调优

### 监控显存使用

```bash
nvidia-smi -l 1
```

### 监控性能

```bash
# 查看实时性能
curl http://localhost:8080/health
```

### 调优建议

1. **如果显存不足**：降低 `-c` 上下文长度，或使用 `--cache-type-k q4_0 --cache-type-v q4_0`
2. **如果延迟高**：减少 `-c` 上下文长度
3. **如果生成质量差**：增加 `-c` 上下文长度，或使用更大量化精度
