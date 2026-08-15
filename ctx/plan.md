# llama.cpp 学习计划（面向分布式存储工程师）

## 学习目标
作为分布式存储工程师，重点关注 llama.cpp 中的 **数据加载**、**缓存管理**、**存储布局** 等核心环节，理解其如何实现高效的数据访问模式。

---

## 一、核心架构理解（1-2天）

### 1. 数据加载流程
- **`src/llama-mmap.h/.cpp`** - 文件I/O核心（mmap、mlock、direct I/O）
  - `llama_file`: 文件读写抽象层，支持分块读取（64MB chunks）
  - `llama_mmap`: 内存映射，支持prefetch和NUMA优化
  - `llama_mlock`: 防止页面被换出（关键：模型权重常驻内存）

### 2. 模型加载器
- **`src/llama-model-loader.h`** - GGUF格式解析
  - `weights_map`: 按层排序的权重索引表（`blk.%d.*`）
  - `init_mappings()`: 建立文件到内存的映射
  - `load_all_data()`: 支持多buffer类型分配（CPU/GPU）

### 3. 关键数据结构
- **`src/llama-batch.h`** - 批处理数据布局
  - token/embd/pos/seq_id 的紧凑存储设计
  - `llama_ubatch`: 物理批处理单元（影响cache访问模式）

---

## 二、KV Cache深度分析（3-5天）

### 1. 缓存架构
```
llama_kv_cache
  ├── layers[kv_layer]          # 每层独立的K/V tensor
  │   ├── k                     # [n_embd_head_k, n_head_k, kv_size]
  │   └── v                     # [kv_size, n_embd_head_v, n_head_v] (可能转置)
  ├── v_cells                   # 细粒度位置管理
  │   ├── pos[]                 # 每个cell的位置
  │   ├── seq[]                 # 每个cell所属的sequence bitset
  │   └── seq_pos[seq_id]       # per-sequence的位置索引（map<pos, count>）
  └── v_heads                   # 环形buffer搜索起始点
```

### 2. 存储工程师关注点

**内存布局优化：**
- `type_k/type_v`: 支持Q4/Q8/F16/F32等量化类型（`src/llama-memory.h:19-21`）
- `v_trans`: V cache转置存储，优化attention计算（`llama-kv-cache.h:238`）
- `n_pad`: 对齐填充，考虑cache line对齐（`llama-kv-cache.h:244`）

**位置管理：**
- `llama_kv_cells` 实现（`src/llama-kv-cells.h`）
  - `used`: std::set追踪已用cell（支持O(logN)查找）
  - `seq_pos[LLAMA_MAX_SEQ]`: per-sequence位置映射
  - 支持位置偏移（shift）和除法（div）操作

**Cache操作：**
- `seq_rm/seq_cp/seq_keep/seq_add/seq_div`: 序列级操作
- `find_slot()`: 环形buffer slot查找（`llama-kv-cache.h:196`）
- `state_write/state_read`: 持久化序列状态

---

## 三、推荐阅读顺序

### 第一阶段：数据流理解
1. `llama-mmap.cpp` - 理解模型如何从磁盘加载
2. `llama-model-loader.h/.cpp` - GGUF格式和tensor映射
3. `llama.h:206-212` - load_mode枚举（MMAP/MLOCK/DIRECT_IO）

### 第二阶段：Cache机制
4. `llama-memory.h` - 内存接口抽象
5. `llama-kv-cells.h` - 细粒度位置管理（类似LSM-tree的cell管理）
6. `llama-kv-cache.h` - 整体cache架构
7. `llama-batch.h` - 批处理如何影响cache访问

### 第三阶段：高级特性
8. `llama-kv-cache-iswa.h` - 混合SWA（Sliding Window Attention）
9. `llama-kv-cache-dsa.h` - DSA（DeepSeek Attention）
10. `src/llama-kv-cache.cpp` - 具体实现

---

## 四、与存储系统类比

| llama.cpp 概念 | 存储系统类比 |
|--------------|------------|
| GGUF文件格式 | LSM-tree SSTable |
| mmap + prefetch | Page cache + 预读 |
| KV cache ring buffer | Circular log buffer |
| `seq_pos` map | Inverted index |
| `type_k/type_v` 量化 | 压缩存储 |
| `state_write/read` | Checkpoint/Snapshot |
| `n_pad` 对齐 | Block alignment |

---

## 五、关键代码片段

### Cache内存占用计算
```cpp
// llama-kv-cache.h:293-295
size_t total_size() const;
size_t size_k_bytes() const;  // n_layers * kv_size * n_embd_head_k * n_head_k * type_size
size_t size_v_bytes() const;  // 注意v_trans可能改变布局
```

### Slot查找逻辑
```cpp
// ll:196 - 环形buffer中查找可用slot
slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;
```

---

## 六、核心代码文件索引

| 文件 | 用途 | 关键类/结构 |
|------|------|------------|
| `src/llama-mmap.h/.cpp` | 文件I/O和内存映射 | `llama_file`, `llama_mmap`, `llama_mlock` |
| `src/llama-model-loader.h` | GGUF模型加载 | `llama_model_loader` |
| `src/llama-memory.h` | 内存管理接口 | `llama_memory_i`, `llama_memory_context_i` |
| `src/llama-kv-cells.h` | KV Cell位置管理 | `llama_kv_cells` |
| `src/llama-kv-cache.h` | KV Cache实现 | `llama_kv_cache`, `llama_kv_cache_context` |
| `src/llama-batch.h` | 批处理数据 | `llama_ubatch`, `llama_batch_allocr` |

---

## 七、扩展学习

- **多GPU支持**: `llama-kv-cache-dsv4.h`, `multi-gpu.md`
- **量化优化**: `type_k/type_v` 参数，理解不同量化格式的存储效率
- **持久化**: `state_write/state_read` 实现session保存/恢复
