# FlagGems `torch.linalg.qr` 昇腾私有实现总结

> 适用范围：Ascend 910B4 / CANN 8.5.0 / triton-ascend 3.2（aiv 向量核路径）。
> 对应实现：`src/flag_gems/runtime/backend/_ascend/ops/linalg_qr.py`。
> 通用实现（对照基准）：`src/flag_gems/ops/linalg_qr.py`。

## 一、背景

- **通用实现**（约 1900 行）：算法源自 Michael Lutz 的 GPU MODE qr_v2 竞赛总结
  （<https://ml-mike.com/writing/qr_v2/>，B200 第 5 名），核心是
  **blocked Householder + compact-WY（Gram-solve）+ 两级 TSQR 树 + SRAM 融合
  kernel + tl.dot GEMM 更新**，依赖 tensor core 形状的矩阵乘与较大片上驻留 tile。
- **问题**：在昇腾 910B 上运行测试大量用例报 MLIR/BiShengIR 编译失败
  （`ub overflow` / `Failed to run BiShengIR pipeline`），无法运行。
- **方案**：昇腾后端私有实现，在 `_ascend/ops/__init__.py` 注册覆盖；
  纯 Triton kernel + `torch.empty` 声明张量，不借助任何 torch 计算算子。

## 二、通用实现编不过的根因（逐 kernel 编译实测）

| 通用实现 kernel | 关键模式 | 昇腾编译结果 |
| --- | --- | --- |
| `_tsqr_apply`（TSQR Q 组装） | tl.dot ×3 | **UB overflow 失败**（报错日志定位处） |
| `_qr_fused`（融合 kernel） | SRAM 驻留大 tile | **UB overflow 失败** |
| `_geqrt_sram`（SRAM panel） | loop-carried 2-D tile | **UB overflow 失败** |
| `_larft`（WY-T 构建，INVERT=True） | tl.dot Neumann 5 次平乘求逆 | **编译约 25 分钟后失败** |
| `_larft`（INVERT=False，纯 Gram 存储） | 无 tl.dot | 编译通过 |
| `_larfb`（WY 应用） | tl.dot GEMM | 3 分钟未编完（至少极慢） |

**结论**：编不过的不是"WY 这个算法"，而是一类共性模式——
**`tl.dot`（MMA/mix 路径）+ 大 loop-carried 2-D 寄存器 tile + 多层循环嵌套**。
编译器对循环做多缓冲（multi-buffer）后 UB（192KB）需求被放大 30× 以上直接溢出。
"WY 有问题就整体绕开"的诊断属于部分正确、归因过宽。

## 三、昇腾路径架构

### 路由决策树

```text
输入 A (B, m, n), mode ∈ {reduced, complete, r}
│
├─ m==0 或 n==0 ──────────────► 退化路径（空因子，complete+n==0 时 Q=I）
│
├─ 高瘦: m>128 且 n≤64 且 m≥4n 且 reduced/r
│      └─► 多级递归 TSQR：
│           L1: 128 行块并行独立分解（寄存器驻留，末级 R 直写列主序栈）
│           栈 P·n > 512 行 → 递归再切一级
│           L2: ≤512 行栈走单 CTA panel
│           Q : 面板因子作用于恒等阵，按级链式 L3 组装（内层先算）
│
├─ 少行: m≤128（列任意）
│      └─► 寄存器分块 kernel（单 launch 全流程）：
│           (RM,TN) 列块驻留寄存器跨反射器循环；跨块反射器在寄存器内作用；
│           R 每块写一次；Q = 寄存器 Qt 应用于全部反射器
│           TN = min(行宽上限, next_pow2(n))，上限: m≤16→256, m≤32→128, m≤64→64
│
└─ 其余（方阵/一般）
       └─► blocked panel 路径：
           panel: mcta 多 CTA（64 行一带，2≤带数≤16 且 B·带数≤40）
                  —— 原子自旋栅栏 + 槽位部分和，替代单 CTA 串行 panel
                  条件不满足时回退单 CTA panel
           trailing: 每 panel 一次 launch，CTA 按列块并行，逐反射器两趟
           Q 组装: 单 launch（恒等 + 全部反射器，按列块并行）
           全程列主序 padded 暂存（ld 为 RM 的倍数，访存对齐）
```

### kernel 清单（12 个）

| kernel | 职责 | 关键模式 |
| --- | --- | --- |
| `_copy_a_to_wc` | A→列主序暂存 | 每列一 CTA，Wc 侧连续向量写；3-D grid（axis-2 行 tile，上限 `_CPY_TILES=8`，grid-stride 循环兜底） |
| `_panel_kernel` | 单 CTA panel 分解 | 仅标量与 (PW,) 累加器跨反射器循环，tile 每趟从 global 重载 |
| `_panel_mcta_kernel` | 多 CTA panel | 64 行一带；每反射器 3 次原子自旋栅栏；部分和槽位存储（确定性，无浮点原子）；栅栏计数单调递增，host 镜像传目标值 |
| `_trailing_apply_kernel` | 尾随更新 | CTA 按列块并行，`coff` 偏移分片（守 40-block 上限） |
| `_q_apply_kernel` | Q 组装 | 恒等 + 全反射器单 launch，Q 侧步长参数化（列主序暂存/行主序直写两用） |
| `_qr_reg_kernel` | 少行矩阵全流程 | loop-carried 2-D 寄存器 tile（≤16KB）+ 算术掩码，禁 tl.where 存储 |
| `_tsqr_l1_kernel` | TSQR 行块分解 | 寄存器驻留块 QR，R 存储步长参数化（行主序栈/直写 Wc2） |
| `_tsqr_q_kernel` | TSQR 分块 Q 组装 | Q2 块补零到块高 + 本块反射器寄存器内作用 |
| `_copy_qc_to_q` / `_triu_copy_cm` | 结果拷出（列主序→行主序转置） | 每 CTA 一个 (64,TN) 2-D tile：源侧沿行连续、目的侧沿列连续，双侧向量化（行 gather 旧写法在大矩阵上慢 ~100x）；非连续 out= 有标量回退版本 |

退化路径（m==0 或 n==0）不再使用专用 kernel：与通用实现相同，直接用
`torch.eye` + `copy_` 处理 complete+n==0 的 Q=I。

### 工程层

- **普通 `kernel[grid](...)` 启动**：无任何自定义启动器/缓存（v11 曾用
  `_fast_launcher` 缓存 `CompiledKernel` 将单次启动从 0.4ms 压到 ~65µs，
  后应"保持简单、不要自定义启动机制"的要求移除）。注意代价：
  triton-ascend 的 `JITFunction.run` 每次启动都调用
  `get_current_device/stream/target`（含 NPUUtils 构造，实测 ~0.4ms/次），
  小矩阵 QR 只有 1~3 次 launch，启动开销占主导。
- **无任何设备探测代码**：aiv 核数不再运行时查询，硬编码 `_MAX_BLOCKS = 40`
  （910B 向量核数；>40-block 损坏 bug 在 2026-08-17 复测仍存在，分片防护保留）；
  也不存在 `torch.npu.current_stream()` 之类的 stream 获取调用。
- **无全局缓存**：工作区每次调用 `torch.empty`（torch 分配器自带缓存，~9µs）；
  mcta 栅栏计数器每次调用 `torch.zeros` 全新分配（起始恒为 0，宿主镜像 `cbase`
  只在单次调用内累加，天然规避了旧计数器池被其他 shape 工作区覆盖的问题）。
- **batch/axis 分片**：`_batch_chunks`（batch 轴 ≤40）与 `_axis1_chunks`（列块轴，
  总 block ≤40）+ `_bslice` 张量切片，规避 >40-block 损坏 bug。
- 参数校验复用通用实现的 `_validate_mode`。

## 四、与通用实现的差别对照

| 维度 | 通用实现 | 昇腾私有实现 |
| --- | --- | --- |
| 面板分解 | SRAM 驻留 / 多 CTA（atomic+栅栏） | 单 CTA（仅标量+窄累加器跨循环）或 mcta（槽位部分和 + 自旋栅栏，无 tl.dot） |
| 尾随更新 | compact-WY：`C -= V·T·(VᵀC)`，2 趟/panel，tl.dot GEMM | 逐反射器两趟（WY 已试过：数学正确、gram kernel 可编译，但寄存器内三角解的变体触发编译器 crash，已回退） |
| WY-T 构建 | `_larft` Gram-solve + Neumann 乘方求逆（tl.dot） | 不使用 |
| TSQR | 两级树 + 折叠小栈归约 + T-合成 | 多级递归行块 + ≤512 行栈走 panel + 链式 Q 组装；无 tl.dot |
| Q 组装 | 0/1 选择矩阵 tl.dot 提取 | 单 launch 逐反射器两趟 / 寄存器 Qt |
| 矩阵乘 | tl.dot（tensor core） | 全 vector 单元：1-D 连续向量 + 2-D 仿射 tile + tl.sum |
| 小矩阵 | SRAM 融合单 kernel | 寄存器分块单 kernel（同思路，tile 受 16KB 限制） |
| 数据布局 | 行列混合 | 列主序 padded 暂存（ld=RM 倍数）为主，用户张量侧只做对齐访存 |
| 掩码写 | tl.where 选择存储 | 算术掩码 `(mask)*value`（tl.where 向量存储会 miscompile） |
| 启动工程 | torch 原生 | 普通 `kernel[grid](...)` 启动；工作区/计数器每次调用现分配 |
| 并行上限 | 无特殊限制 | 单 launch 总 block ≤ 40（超限损坏 batch 0） |

## 五、已做的适配

### 1. 正确性适配

- 临时关闭 `empty.memory_format` 注册（use_gems 下零元素 grid 启动崩溃；
  主线已修，`src/flag_gems/__init__.py` 留 TODO）。
- **>40-block 损坏 bug**：复杂 kernel 单次 launch 总 block 超过 aiv 核数（40）时
  确定性损坏 batch 0（原实现就有，测试集 B≤32 未覆盖）→ batch 切片 + `coff`
  列偏移分片；补 `(64, 8, 8)` 回归测试。
- **TSQR 路径 B>40 未分片（v9 修复）**：TSQR 的 launch grid axis 0 直接是 B
  且未做 batch 分片，B>40 的高瘦批量（如 `(64,512,64)`）触发同一个
  >40-block 损坏 bug，输出 NaN（存量问题，测试集未覆盖）→ TSQR 门控加
  `B <= _MAX_BLOCKS`，超限回落到有完整分片的 blocked 路径；补
  `(64,512,64)` 回归测试。op 层复现另记：解除分片后 `(48,256,256)` 的
  blocked 路径 batch 0 误差 ~1e9，仅 batch 0 受损，与结论一致。
- 融合 kernel R 写出缺 `rows<rrows` 掩码，高瘦矩阵越界写穿下一 batch → 修复。
- 寄存器路径门控：非列连续输入的 2-D gather 会 UB 溢出 → 回退暂存路径；
  `put_q` 类开关必须 constexpr（runtime-0 分支不被消除反而更大）。

### 2. 性能适配（按 msprof 证据链）

- **msprof 三瓶颈定位**（首轮）：标量拷贝 kernel 占设备时间 74%
  （(512,512) 拷一次 34.5ms）→ 向量化+多 CTA；O(k) 次 Python 启动 0.4ms/次
  → panel 阻塞 + Q 单 launch + **直连 launcher**。
- **寄存器分块 kernel**（少行）：发现 loop-carried 2-D tile + 算术掩码在 ≤16KB
  时可编译（推翻"必挂"假设）。
- **多级递归 TSQR**（高瘦）：(16384,32) 1.55→21x、(4096,64) 0.62→4.6x；
  末级 R 直写列主序栈、Q 直写行主序，省 2 次 launch。
- **宽列块**：TN 按 `min(行宽上限, next_pow2(n))`（不 min(n) 会通道浪费 4 倍）；
  (8,8)/(8,32) kernel 进一步 22→15 / 21→17µs。
- **mcta panel**（方阵）：panel 占方阵 58%（单 CTA 带宽瓶颈）→ 64 行带并行
  + 栅栏；(512,512) 39.8→22.8ms。
- 自适应 tile 宽度（PW）；fp64 同路径（dtype 泛化，本机 `support_fp64=False`
  测试被跳过、未实测）。

### 3. 平台硬约束（踩坑记录，均已验证）

1. 单 launch 总 block > aiv 核数（40）→ 复杂 kernel 损坏 batch 0，
   简单流式 kernel 不受影响；
2. loop-carried 寄存器 tile 上限约 16KB（64×64 / 32×128 / 8×256 可过，
   64×128 / 32×256 溢出）；多重循环 kernel 更小（32×32）；
3. **单 lane 掩码标量存储崩向量核**（`store(向量地址, 标量, mask≈全假)`）
   → 整列写回原值；
4. **constexpr 大展开 trap**：NC=16 展开 16 路后定点单元 trap 挂死 16 block
   → 改 runtime 参数；
5. tile 维度禁用 1（退化维度爆 UB），下限 8；
6. 非列连续输入的 2-D gather 会 UB 溢出 → 门控回退；
7. msprof 设备侧采集在部分卡上会失效（device sqlite 空、"Failed to connect
   database"）→ `ASCEND_RT_VISIBLE_DEVICES=<空闲卡>` 恢复；kernel 级数据在
   `op_summary_*.csv`（Op Name / Task Duration / Input Shapes 列）。

## 六、性能结果

- **benchmark operator 模式**（v9 全量 100 行，linalg_qr/linalg_qr_out ×
  reduced/complete/r，普通 `kernel[grid]` 启动）：mean **3.7x** /
  median ~1.0x；**46/100 行 ≥ 1.2x**；torch 耗时 ≥ 6ms 的行除
  (256,256) out-reduced（torch 5.97ms，1.13x）外全部达标。
- **方阵**（reduced/r）：(256,256) ~1.2x/1.37x、(512,512) 2.84x/3.69x、
  (1024,1024) 3.49x/5.53x、(4096,4096) 5.39x/7.89x。
- **complete 大 Q**（tiled 转置拷出的收益不受启动机制影响）：
  (8192,8) **47x**、(4096,4) **34-42x**（逐轮波动）、(4096,4096) 3.9x。
- **高瘦**（v9 WRITE_R 融合后）：r 模式 (512,64) 1.53x、(8192,8) 1.05x；
  reduced (512,64) 0.94x、(8192,8) 0.59x——仍受启动地板限制，见下表。
- **批量**：(128,32,32) 2.1-3.0x、(32,128,128) 6.8-10.0x、
  (4,1024,1024) 3.8-7.1x。
- **回归**：`tests/test_linalg_qr.py` 126 passed（含 (64,8,8)、(128,64)、
  (64,512,64) 回归 shape；fp64 因 `support_fp64=False` 跳过 109 项）。

### <1.2x 行分析（54 行，全部落在 host 启动地板模型内）

host 侧硬地板（v8 实测，noop kernel）：每次调用 Python wrapper+校验+分配
≈ 110µs；普通 `kernel[grid]` launch ≈ **423µs/次**（triton-ascend
`JITFunction.run` 每次启动都做 device/stream/target 解析）。
地板 ≈ 0.11 + 0.42 × launch 数（ms）：1 次 ≈0.53，2 次 ≈0.95，
3 次 ≈1.4，5 次 ≈2.2，10 次 ≈4.3。torch 侧小矩阵是单次 C++/ACLNN
调用，无此开销，故小 shape 在普通启动下物理不可达 1.2x。

| 类别 | 行数 | 路径 launch 数 → 地板 | torch 耗时 → 达标所需 | 结论 |
| --- | --- | --- | --- | --- |
| 微型 reg 单 launch：(8,8)/(64,64)/(128,32) ×6 mode，(8,32)/(16,64)/(32,128)/(64,256) ×4 mode | 34 | 1 → 0.53ms | 0.03-0.50ms → 需 ≤0.03-0.42ms | 低于纯启动地板，豁免 |
| (64,8,8) ×6 mode | 6 | B=64>40 强制分片 2 → 0.95ms | 0.23-0.30ms | 豁免 |
| (4096,4) reduced/r ×4 | 4 | TSQR 2-3 → 0.95-1.4ms | 0.35ms | 豁免 |
| (8192,8) reduced/r ×4 | 4 | TSQR 3-4（L1 因 P=64>40 分 2 次）→ 1.4-1.8ms | 1.8ms → 需 ≤1.5ms | r 1.05x 已贴地板；豁免 |
| (512,64) reduced ×2 | 2 | TSQR 4 → 1.8ms | 2.1ms → 需 ≤1.76ms | 0.94x 贴地板；豁免（r 1.53x 达标） |
| (256,256) complete/out ×3 + reduced（边界波动） | 4 | blocked 10 → 4.3ms+计算 | 4.8-6.2ms → 需 ≤4.0-5.1ms | 边界，豁免（r 1.37x 达标） |

恢复这些行只有两条路：launch 结果缓存（v7 的 `_fast_launcher`，
~65µs/次，已应"保持简单、不要自定义启动机制"的要求移除）或压缩
launch 数的 kernel 融合（q_apply 并入 tsqr_q、L1 grid-stride 等；估算
最多把 (8192,8) reduced 提到 ~0.95x，仍不达 1.2x，复杂度不值得）。
当前取舍为代码简单优先：
所有 kernel 调用都是普通 `kernel[grid](...)`，大 shape 全部达标。

注：torch ≤ 5ms 的行在共享主机上 benchmark 逐行波动明显（同一代码
(4096,4) complete 在 33-43x 间摆动、(512,64) complete 在 1.1-2.0x 间
摆动），上述归类以多轮下界为准。

## 七、遗留问题攻关记录（2026-08-17）

### v11：工程简化 + 方阵小尺寸优化（2026-08-17 同日）

- **启动机制变迁**：初版用 `torch.npu.current_stream()` + 直调
  `kernel.run`（16.5µs/次）→ 应"零 stream/设备查询代码"改为 stock
  runner `kernel[(g0,g1,g2)](*args)`（~65µs/次）→ 最终应"不要自定义
  启动机制"**整体删除 `_fast_launcher`**，回归普通 `kernel[grid](...)`
  （实测 423µs/次，小 shape 性能让位于代码简单；最终 v8 全量结果与
  <1.2x 行逐项分析见第六节）。
- **`_par_tile_width(cols, batches)`**：trailing 与 Q 组装的 tile 宽从
  `_tile_width` 改为按并行度折半（64→32→16，直到列块数填满
  `40//batches` 个核，下限 16）。小方阵列数少、原 PW 只用满几个核，
  收窄后并行度翻倍：实测 (256,256) complete 6.36→3.99ms（当时含
  `_fast_launcher`，对 torch 1.22x）、(512,512) 22.9→14.1ms、
  (1024,1024) 93→74ms。panel（串行 kernel）仍用 `_tile_width`。
- **拷出 kernel 改 2-D tile 转置（大收益）**：旧写法按行 gather 列主序源
  （每元素一条 cache line），(8192,8) complete 的 8192×8192 Q 拷出独占
  138/155ms。改为一 CTA 一个 (64,TN) tile（源沿行连续、目的沿列连续，
  双侧向量化）后：(8192,8) complete 155→20.2ms（47.4x）、(4096,4)
  complete 37.6→3.4ms（42.7x）、(256,256) complete 3.99→3.78ms。
  （中间形态的 3-D grid 单 tile CTA 在巨大拷贝上 CTA 调度开销反超，
  已弃用；`_copy_a_to_wc` 保留 3-D grid + `_CPY_TILES=8` grid-stride。）
- **32 行带实验已回退**：mcta panel 行带 64→32 在 (256,256)/(512,512)
  上反而变慢（栅栏次数翻倍得不偿失），保持 64 行带。
- **TSQR 小栈直分解（l2_reg）**：末级栈 ≤128 行时（如 (4096,4)→128×4
  栈）改用寄存器 kernel 单次 launch 直写 Q2/R，省掉 panel + Q2 组装 +
  R 拷出 3 次 launch（每次 runner 启动实测 ~65µs，窄矩阵是 host-bound）。
- **每次调用 Python 开销精简**：`_bslice` 单分片直通（不再逐张量切片）、
  reg 路径 Vc/tau 直接分配（省一次大 buffer 的切片+view）、l2_reg 时
  跳过 Wc2/Vc2/tau2 视图构建。
- **reg kernel 门控修正（正确性）**：RM=128 且列块数 >1（即 m∈(64,128]
  且 n>32，如 (128,64)）时跨块反射器应用结果错误（原实现即存在，测试
  集未覆盖；重构误差 ~5.0 量级）。`_reg_tile_cfg` 对该区间只在 n≤32
  （单块）时返回配置，其余落到 staged 路径；补 (128,64) 回归测试。

### v9：TSQR 末级直写 R + B>40 门控（2026-08-18）

- **背景**：删除 `_fast_launcher` 后普通启动 423µs/次，高瘦 TSQR 路径
  （3-5 次串行 launch）性能衰退最严重：(8192,8) reduced 2.49x→0.51x。
- **WRITE_R 融合**：TSQR 末级的 `_panel_kernel` / `_panel_mcta_kernel`
  加 constexpr `WRITE_R`——分解完顺手把 n×n 上三角按 `_triu_copy_cm`
  的算术掩码模式直写 R，省掉单独的 R 拷出 launch。mcta 版由 band 0
  写（n≤64，R 全部行都在 band 0 自有的 64 行切片内，无需额外栅栏）。
  blocked 路径调用传 `WRITE_R=False`（constexpr 分支被消除）。
- **效果**：每行省 ~0.4-0.5ms。(512,64) r 1.23x→1.53x、(8192,8) r
  0.88x→1.05x、reduced 各行 +0.07-0.14x。reduced 类仍 <1.2x：
  地板刚性（见第六节表），继续压缩 launch 数也到不了 1.2x。
- **TSQR B>40 门控**：见第五节正确性适配。

### mcta"死锁/竞态"的真正根因：计数器被数据布局覆盖（已修复）

mcta 曾被误判为"release 原子不能排序跨 CTA 普通 store"而禁用。实际根因：
**栅栏计数器放在共享工作区 buffer 内**，不同 shape 的暂存布局共 用同一块
ws，一个 shape 的 Wc/Vc 数据区会覆盖另一个 shape 的计数器位置。宿主镜像
领先于被清零/污染的设备计数器时：

- 设备计数器被清 0 → 栅栏目标永远达不到 → **死锁**（vector core timeout）；
- 设备计数器被写成偏大的垃圾值 → 栅栏瞬间通过 → **竞态、错误结果**。

这解释了全部症状：单 shape 独跑必对、全量套件按顺序必挂、死锁与错值
交替、以及"acquire 正确但偶发死锁 / volatile 不死锁但竞态"的假象
（两种等待方式只是改变了症状出现的概率）。

**修复**：计数器搬进专用独立分配 `_CTR_POOL`（int32，64 槽 × 64 batch，
每布局一个槽位，槽位耗尽时整体清零重启——调用是流序的，安全），
宿主镜像按布局键多槽管理。修复后 blocked 路径与 TSQR L2 面板的 mcta
全部恢复启用，全量测试 ×2、8 轮交错 shape 压力、完整 benchmark 均稳定。

**后续简化（同日 v10）**：`_CTR_POOL` 已删除，改为每次 QR 调用现分配
`torch.zeros(B)` 计数器（起始恒为 0，`base`/宿主镜像只在单次调用内
累加），从机制上杜绝跨 shape 覆盖；代价是每次调用一次小 memset
（~19µs，仅 mcta 路径用到，相对 ms 级 panel 时间可忽略）。

### L2-mcta 的 n 门控

TSQR 末级栈的 mcta 面板要求 `n >= 16`：极窄栈（(4096,4)、(8192,8)）
panel 只有几个反射器，栅栏开销超过并行收益（实测 (8192,8) 0.579ms
带门控 vs 0.674ms 不带；(4096,4) 0.456 vs 0.569）。

### 仍未解决（真实遗留）

- m>1024 的 panel 仍单 CTA：多带版（每 CTA 若干 64 行带）当初失败
  也是计数器覆盖问题，理论上现在可用全新计数器（每次调用零初始化）重启
  验证——未做，
  潜在收益 (4096,4096) 5.4→8x、(2048,1030) 2.3→4.2x（当时实测）；
- (64,256) r/reduced 0.87-0.97x：m=64 单 CTA 寄存器路径 4 列块串行，
  需 blocked 化重构（见第六节豁免表）；
- reg kernel RM=128 多块结果错误的根因未定位（已门控回避）；
- WY 尾随更新：数学已验证，编译器对寄存器内三角解变体 crash——
  重启时建议 W 落 global 或 Neumann 展开避开串行回代。

### 平台结论（修订）

- ~~release 原子不能排序跨 CTA 普通 store~~ → **撤销该结论**：
  acquire 自旋的排序语义正确；此前异常全部源于计数器被覆盖。
  跨 CTA 栅栏同步在本后端可用，前提是同步变量放在专用内存；
- 模块以 `_ascend.ops.*`（顶层别名，**服务实例**）与完整路径双实例
  加载——运行期 monkeypatch/探测务必取 `sys.modules["_ascend.ops.linalg_qr"]`；
- 单 lane 掩码标量存储崩向量核、constexpr 大展开 trap、>40-block 损坏、
  UB 16KB loop-carried tile 上限等结论不变（见第五节）。

### 旧实现存档（v9，已被 v11 重写取代）

mean 4.41x / median 1.50x（当时含 RM=128 多块错误路径的虚高批量数字，
如 (32,128,128) 30x 实为错误结果；修正后门控到 staged 路径为 7-10x）。
