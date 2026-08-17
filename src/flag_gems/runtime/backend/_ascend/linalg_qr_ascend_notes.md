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

### kernel 清单（10 个）

| kernel | 职责 | 关键模式 |
| --- | --- | --- |
| `_copy_a_to_wc` | A→列主序暂存 | 每列一 CTA，Wc 侧连续向量写 |
| `_panel_kernel` | 单 CTA panel 分解 | 仅标量与 (PW,) 累加器跨反射器循环，tile 每趟从 global 重载 |
| `_panel_mcta_kernel` | 多 CTA panel | 64 行一带；每反射器 3 次原子自旋栅栏；部分和槽位存储（确定性，无浮点原子）；栅栏计数单调递增，host 镜像传目标值 |
| `_trailing_apply_kernel` | 尾随更新 | CTA 按列块并行，`coff` 偏移分片（守 40-block 上限） |
| `_q_apply_kernel` | Q 组装 | 恒等 + 全反射器单 launch，Q 侧步长参数化（列主序暂存/行主序直写两用） |
| `_qr_reg_kernel` | 少行矩阵全流程 | loop-carried 2-D 寄存器 tile（≤16KB）+ 算术掩码，禁 tl.where 存储 |
| `_tsqr_l1_kernel` | TSQR 行块分解 | 寄存器驻留块 QR，R 存储步长参数化（行主序栈/直写 Wc2） |
| `_tsqr_q_kernel` | TSQR 分块 Q 组装 | Q2 块补零到块高 + 本块反射器寄存器内作用 |
| `_copy_qc_to_q` / `_triu_copy_cm` | 结果拷出 | 按行 CTA，目的侧连续向量写；非连续 out= 有标量回退版本 |
| `_identity_rm_kernel` | 退化路径 Q=I | 标量循环 |

### 工程层

- **`_CachedLauncher`**：缓存 `CompiledKernel` 直调 C launcher（`kernel.run(...)`），
  复刻 triton 特化键（dtype / 指针 16 对齐 / int ==1 与 %16 / constexpr），
  单次启动 0.4ms→约 70µs；参数序从 `kernel.src.signature` 恢复。
- **`_get_ws` 工作区缓存**：每 (dtype, device) 一块按需增长的扁平 buffer，所有暂存
  视图切自它（视图必须显式闭区间切片——缓存 buffer 比当前请求大）。
- **batch/axis 分片**：`_batch_chunks`（batch 轴 ≤40）与 `_axis1_chunks`（列块轴，
  总 block ≤40）+ `_bslice` 张量切片，规避 >40-block 损坏 bug。

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
| 启动工程 | torch 原生 | CompiledKernel 直连 launcher + 工作区缓存 |
| 并行上限 | 无特殊限制 | 单 launch 总 block ≤ 40（超限损坏 batch 0） |

## 五、已做的适配

### 1. 正确性适配

- 临时关闭 `empty.memory_format` 注册（use_gems 下零元素 grid 启动崩溃；
  主线已修，`src/flag_gems/__init__.py` 留 TODO）。
- **>40-block 损坏 bug**：复杂 kernel 单次 launch 总 block 超过 aiv 核数（40）时
  确定性损坏 batch 0（原实现就有，测试集 B≤32 未覆盖）→ batch 切片 + `coff`
  列偏移分片；补 `(64, 8, 8)` 回归测试。
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

- **benchmark operator 模式**（v4 全量 100 行）：mean **4.30x** / median **1.08x**；
  v5（含 mcta 方阵，68 行超时截断）：mean 4.13x / median 1.03x。
- **方阵**（v5）：(512,512) reduced **1.66x** / r **2.53x**；(1024,1024) reduced
  **2.63x** / r **4.15x**；(256,256) 0.91x / 1.43x。
- **高瘦**：(16384,32) 21x、(4096,64) 4.6x、(8192,8) 3.7x、(4096,4) 0.81x。
- **kernel 级**（msprof）：小/宽矩阵全部快于 torch（torch 小矩阵走 `QrAiCPU`
  协处理器路径）：(8,8) 6.3x、(8,32) 2.5x、(16,64) 1.9x、(32,128) 1.2x。
- **批量**：(128,32,32) 14-15x、(32,128,128) 27-35x、(4,1024,1024) 3.5-6.6x。
- **回归**：`tests/test_linalg_qr.py` 120 passed（含新增 B>40 回归 shape）。

## 七、遗留问题攻关记录（2026-08-17）

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

### L2-mcta 的 n 门控

TSQR 末级栈的 mcta 面板要求 `n >= 16`：极窄栈（(4096,4)、(8192,8)）
panel 只有几个反射器，栅栏开销超过并行收益（实测 (8192,8) 0.579ms
带门控 vs 0.674ms 不带；(4096,4) 0.456 vs 0.569）。

### 仍未解决（真实遗留）

- m>1024 的 panel 仍单 CTA：多带版（每 CTA 若干 64 行带）当初失败
  也是计数器覆盖问题，理论上现在可用计数器池重启验证——未做，
  潜在收益 (4096,4096) 5.1→8.2x、(2048,1030) 2.3→4.2x（当时实测）；
- (256,256) reduced 0.94x：剩余 gap 在 trailing / q_apply（可同 mcta 化）；
- 小矩阵端到端 0.2ms Python wrapper 地板（约定不再单独调）；
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

### 最终性能（v9，mcta 全启，全量 100 行）

mean **4.41x** / median **1.50x**；reduced 代表值：
(256,256) 0.94、(512,512) 1.66、(1024,1024) 2.62、(4096,4096) 5.10、
(512,64) 1.65、(4096,4) 0.73、(8192,8) 3.16、(16384,32) ~26（自测）、
批量 (128,32,32) 14.3、(32,128,128) 30.3、(4,1024,1024) 4.53。
回归 120 passed ×2 + 8 轮交错压力。
