# 下一步学习计划

## 已完成

- [x] `src/llama-mmap.h/.cpp` — 文件 I/O 和内存映射（Direct I/O、mmap、mlock）
- [x] `src/llama-model-loader.h/.cpp` — GGUF 解析、weights_map 构建、mmap 零拷贝加载、split 文件
- [x] GGUF 格式、模型权重、Session 概念理解

---

## Phase 1 收尾

### 1. `llama.h` load_mode 枚举（~206-212 行）
- 理解 MMAP / MMAP_MLOCK / DIRECT_IO / AUTO 四种模式的适用场景
- 关注 `llama_model_loader` 构造函数中 load_mode 如何决定 `use_mmap` 和 `use_direct_io`

---

## Phase 2：KV Cache 深度分析

### 2. `src/llama-memory.h` — 内存接口抽象
- `llama_memory_i` 虚接口定义了哪些操作
- `llama_memory_context_i` 上下文接口
- 存储类比：内存接口 ≈ 存储引擎的抽象层

### 3. `src/llama-kv-cells.h` — 细粒度位置管理
- `used` set：追踪非空 cell（O(logN) 查找）
- `pos[]`：每个 cell 的位置
- `seq[]`：每个 cell 所属的 sequence bitset
- `seq_pos[seq_id]`：per-sequence 位置映射（map<pos, count>）
- 位置 shift / div 操作
- 存储类比：cell 管理 ≈ LSM-tree 的 cell/slot 管理

### 4. `src/llama-kv-cache.h` — 整体 cache 架构
- `layers[kv_layer]`：每层独立 K/V tensor
- `type_k / type_v`：量化类型选择（Q4/Q8/F16/F32）
- `v_trans`：V cache 转置存储优化
- `n_pad`：cache line 对齐填充
- `find_slot()`：环形 buffer slot 查找
- `state_write / state_read`：持久化
- 存储类比：ring buffer ≈ Circular log buffer

### 5. `src/llama-batch.h` — 批处理与 cache 访问
- `llama_ubatch`：物理批处理单元
- token/embd/pos/seq_id 的紧凑存储
- batch 如何影响 cache 的访问模式

### 6. `src/llama-kv-cache.cpp` — 具体实现
- `seq_rm / seq_cp / seq_keep / seq_add / seq_div` 序列操作
- `find_slot()` 环形 buffer 查找逻辑
- `state_write_data / state_read_data` 序列化

---

## Phase 3：高级特性

### 7. `src/llama-kv-cache-iswa.h` — 混合 SWA
- Sliding Window Attention 与 Full Attention 的混合

### 8. `src/llama-kv-cache-dsa.h` — DeepSeek Attention
- DSA 的特殊 cache 管理

### 9. `src/llama-kv-cache-dsv4.h` — 多 GPU 支持
- 分布式 KV cache 的设计

---

## 存储类比速查

| llama.cpp 概念 | 存储系统类比 |
|--------------|------------|
| GGUF 文件格式 | LSM-tree SSTable |
| mmap + prefetch | Page cache + 预读 |
| KV cache ring buffer | Circular log buffer |
| `seq_pos` map | Inverted index |
| `type_k/type_v` 量化 | 压缩存储 |
| `state_write/read` | Checkpoint/Snapshot |
| `n_pad` 对齐 | Block alignment |
