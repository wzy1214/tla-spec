# TLA 接口规格

> 规格草案 · 评审中

[Cube](#cube-index) · [SIMD](#simd-index) · [SIMT](#simt-index) · [DFX](#dfx-index) · [DMA](#dma-index) · [标量](#scalar-index) · [内存](#memory-index) · [同步](#sync-index) · [调度](#schedule-index) · [语言](#language-index) · [数据类型](#dtype-index) · [主机接口](#host-index)

本仓按接口组维护 TLA 规格草案。点击接口名查看参数与行为，点击评审查看待决问题，点击源码查看 triton-ascend 的实现。

规格梳理基线为 Triton-Ascend `394068acd` 与 AscendNPU-IR `28fb0b0b3`。源码和小测试链接固定到提交，默认使用 [triton-ascend 的参考提交](https://github.com/triton-lang/triton-ascend/tree/074cf62907f815694a01e5cdc4ac78a90610ae82)。个别入口的链接指向已包含它的开发提交。可执行范围以具体的前端、IR 校验及目标 lowering 为准。

状态取值为待转测、不转测、转测中。已命名的条目初始标记待转测，其中提案或预留入口须先补齐实现。已有小测试表示存在参考用例，正式转测状态仍以首页为准。

「不需要」表示已确认无需提供的接口，接口名保留删除线供追溯。计划中的签名和新入口以规格草案为准，源码列标明实现进度。

日常更新只改对应位置：进度和分工在首页，接口约束在规格页，待决项与结论在跟踪页。评审确认后更新规格，并在问题页写明结论。

<a name="cube-index"></a>

## Cube

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.copy_ub_to_l1](spec/cube.md#api-tla-copy-ub-to-l1) | 待&#8288;转&#8288;测 | 将数据从 UB 搬到 L1，保持已有物理布局 | [待&#8288;转&#8288;测](tracking/cube.md#cube-001) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L629) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [tla.copy_l1_to_ub](spec/cube.md#api-tla-copy-l1-to-ub) | 待&#8288;转&#8288;测 | 将 L1 的 NZ 数据搬到 UB 并转换为 ND | [待&#8288;转&#8288;测](tracking/cube.md#cube-002) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L663) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [~~tla.copy_l1_to_bt~~](spec/cube.md#api-tla-copy-l1-to-bt) | 不&#8288;需&#8288;要 | 将偏置数据从 L1 搬到 Bias Table | [已&#8288;关&#8288;闭](tracking/cube.md#cube-003) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L687) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [~~tla.copy_l1_to_fb~~](spec/cube.md#api-tla-copy-l1-to-fb) | 不&#8288;需&#8288;要 | 预留的 L1 到 FixBUF 搬运入口 | [已&#8288;关&#8288;闭](tracking/cube.md#cube-004) | [预&#8288;留&#8288;入&#8288;口](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L714) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [tla.copy_l0c_to_ub](spec/cube.md#api-tla-copy-l0c-to-ub) | 待&#8288;转&#8288;测 | 通过 Fixpipe 将 L0C 累加结果写入 UB | [待&#8288;转&#8288;测](tracking/cube.md#cube-005) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L828) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [tla.copy_l0c_to_l1](spec/cube.md#api-tla-copy-l0c-to-l1) | 待&#8288;转&#8288;测 | 通过 Fixpipe 将 L0C 累加结果写入 L1 | [待&#8288;转&#8288;测](tracking/cube.md#cube-006) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L855) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [tla.store_l0c_to_gm](spec/cube.md#api-tla-store-l0c-to-gm) | 待&#8288;转&#8288;测 | 通过 Fixpipe 将 L0C 累加结果写入 GM | [待&#8288;转&#8288;测](tracking/cube.md#cube-007) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L801) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [tla.cube.mmad_mx](spec/cube.md#api-tla-cube-mmad-mx) | 待&#8288;转&#8288;测 | 对带 E8M0 块缩放因子的矩阵做乘累加并更新 L0C | 未&#8288;登&#8288;记 | [cube.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/cube.py#L257) | 陈&#8288;相&#8288;廷 | — |
| [tla.scope.cube](spec/cube.md#api-tla-scope-cube) | 待&#8288;转&#8288;测 | 在 Cube 核执行域中构建编译期区域。以无参数 with 区域形式使用。跨核数据交接须显式同步 | 未&#8288;登&#8288;记 | [scope.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scope.py#L94) | 陈&#8288;相&#8288;廷 | 姜&#8288;正 |
| [tla.cube.mmad](spec/cube.md#api-tla-cube-mmad) | 待&#8288;转&#8288;测 | 使用 L1 矩阵执行矩阵乘累加并更新 L0C | 未&#8288;登&#8288;记 | [cube.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/cube.py#L115) | 陈&#8288;相&#8288;廷 | 姜&#8288;正 |
| [tla.set_ctrl](spec/cube.md#api-tla-set-ctrl) | 不&#8288;转&#8288;测 | 设置或清除一个硬件 CTRL bit | 未&#8288;登&#8288;记 | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L880) | 陈&#8288;相&#8288;廷 | 姜&#8288;正 |

<a name="simd-index"></a>

## SIMD

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.scope.vector](spec/simd.md#api-tla-scope-vector) | 待&#8288;转&#8288;测 | 构建普通 Vector 核区域，可内嵌 SIMD。以无参数 with 区域形式使用 | 未&#8288;登&#8288;记 | [scope.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scope.py#L56) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.scope.simd](spec/simd.md#api-tla-scope-simd) | 待&#8288;转&#8288;测 | 构建 AIV SIMD 向量函数区域。以无参数 with 区域形式使用，后端可提取为独立 VF | 未&#8288;登&#8288;记 | [scope.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scope.py#L70) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.membar](spec/simd.md#api-tla-simd-membar) | 待&#8288;转&#8288;测 | 设置 SIMD VF 内访存次序屏障 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-081) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L3036) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.pand](spec/simd.md#api-tla-simd-pand) | 待&#8288;转&#8288;测 | 逐 lane 求谓词与 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2058) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pnot](spec/simd.md#api-tla-simd-pnot) | 待&#8288;转&#8288;测 | 逐 lane 取反谓词 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2094) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pset](spec/simd.md#api-tla-simd-pset) | 待&#8288;转&#8288;测 | 按固定图样生成谓词 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2248) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pxor](spec/simd.md#api-tla-simd-pxor) | 待&#8288;转&#8288;测 | 逐 lane 求谓词异或 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2082) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vabs](spec/simd.md#api-tla-simd-vabs) | 待&#8288;转&#8288;测 | 逐 lane 求绝对值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2543) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vadd](spec/simd.md#api-tla-simd-vadd) | 待&#8288;转&#8288;测 | 逐 lane 相加 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1122) | 蒋&#8288;继&#8288;极 | 姚&#8288;季 |
| [tla.simd.vaddc](spec/simd.md#api-tla-simd-vaddc) | 待&#8288;转&#8288;测 | 逐 lane 相加并返回无符号进位 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1398) | 蒋&#8288;继&#8288;极 | 姚&#8288;季 |
| [tla.simd.vadds](spec/simd.md#api-tla-simd-vadds) | 待&#8288;转&#8288;测 | 每个 lane 加同一个标量 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-011) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1171) | 蒋&#8288;继&#8288;极 | 姚&#8288;季 |
| [tla.simd.vand](spec/simd.md#api-tla-simd-vand) | 待&#8288;转&#8288;测 | 逐 lane 对位模式执行 AND | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1929) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vaxpy](spec/simd.md#api-tla-simd-vaxpy) | 待&#8288;转&#8288;测 | 先计算 src × scalar，再与 acc 相加 | [待&#8288;修&#8288;复](tracking/simd.md#simd-013) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1995) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vcadd](spec/simd.md#api-tla-simd-vcadd) | 待&#8288;转&#8288;测 | 对活跃元素求和，结果位于 lane 0 | [待&#8288;决](tracking/simd.md#simd-014) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L2645) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vcgadd](spec/simd.md#api-tla-simd-vcgadd) | 待&#8288;转&#8288;测 | 950 上按八个 32 字节组求和，结果尾部按 mode 处理 | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-015) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3087) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vcgmax](spec/simd.md#api-tla-simd-vcgmax) | 待&#8288;转&#8288;测 | 950 上按八个 32 字节组取最大值，结果尾部按 mode 处理 | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-016) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3101) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vcgmin](spec/simd.md#api-tla-simd-vcgmin) | 待&#8288;转&#8288;测 | 950 上按八个 32 字节组取最小值，结果尾部按 mode 处理 | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-017) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3112) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vci](spec/simd.md#api-tla-simd-vci) | 待&#8288;转&#8288;测 | 生成 base+i 的 64 lane int32 序列 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2989) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vcmax](spec/simd.md#api-tla-simd-vcmax) | 待&#8288;转&#8288;测 | 规约求最大值并写入 lane 0 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2735) | 蒋&#8288;继&#8288;极 | 姚&#8288;季 |
| [tla.simd.vcmin](spec/simd.md#api-tla-simd-vcmin) | 待&#8288;转&#8288;测 | 规约求最小值并写入 lane 0 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2748) | 蒋&#8288;继&#8288;极 | 姚&#8288;季 |
| [tla.simd.vcmp](spec/simd.md#api-tla-simd-vcmp) | 待&#8288;转&#8288;测 | 逐 lane 比较两个向量并返回谓词 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2769) | 蒋&#8288;继&#8288;极 | 姚&#8288;季 |
| [tla.simd.vcmps](spec/simd.md#api-tla-simd-vcmps) | 待&#8288;转&#8288;测 | 广播标量后逐 lane 比较并返回谓词 | [待&#8288;修&#8288;复](tracking/simd.md#simd-022) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2133) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vcvt](spec/simd.md#api-tla-simd-vcvt) | 待&#8288;转&#8288;测 | 按元素转换向量类型并调整 lane 数 | [待&#8288;确&#8288;认](tracking/simd.md#simd-023) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2317) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vdintlv](spec/simd.md#api-tla-simd-vdintlv) | 待&#8288;转&#8288;测 | 交错操作的逆变换，返回两个去交错向量 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1028) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vdiv](spec/simd.md#api-tla-simd-vdiv) | 待&#8288;转&#8288;测 | 逐 lane 相除 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1463) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vdups](spec/simd.md#api-tla-simd-vdups) | 待&#8288;转&#8288;测 | 把标量广播到满寄存器 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1714) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vdupv](spec/simd.md#api-tla-simd-vdupv) | 待&#8288;转&#8288;测 | 把向量首个或末个 lane 广播到整个向量 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1725) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vexp](spec/simd.md#api-tla-simd-vexp) | 待&#8288;转&#8288;测 | 逐 lane 求指数 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1642) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vexpdif](spec/simd.md#api-tla-simd-vexpdif) | 待&#8288;转&#8288;测 | 逐 lane 求 exp(lhs−rhs) | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1859) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vgather](spec/simd.md#api-tla-simd-vgather) | 待&#8288;转&#8288;测 | 按向量索引从 UB 收集元素 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L862) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vintlv](spec/simd.md#api-tla-simd-vintlv) | 待&#8288;转&#8288;测 | 交错两个向量的元素并返回两个向量 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1019) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vld](spec/simd.md#api-tla-simd-vld) | 待&#8288;转&#8288;测 | 按向量索引从 UB 收集元素 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-032) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L903) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vlds](spec/simd.md#api-tla-simd-vlds) | 待&#8288;转&#8288;测 | 从 UB 连续加载向量或加载一个元素并广播 | [待&#8288;确&#8288;认](tracking/simd.md#simd-vlds) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L914) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vld2](spec/simd.md#api-tla-simd-vld2) | 待&#8288;转&#8288;测 | 连续读取两个寄存器并返回偶、奇位置向量 | [待&#8288;决](tracking/simd.md#simd-033) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L936) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vln](spec/simd.md#api-tla-simd-vln) | 待&#8288;转&#8288;测 | 逐 lane 求自然对数 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2021) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vmadd](spec/simd.md#api-tla-simd-vmadd) | 待&#8288;转&#8288;测 | 计算 acc + src0 × src1 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1982) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vmax](spec/simd.md#api-tla-simd-vmax) | 待&#8288;转&#8288;测 | 逐 lane 取较大值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1556) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vmaxs](spec/simd.md#api-tla-simd-vmaxs) | 待&#8288;转&#8288;测 | 逐 lane 与标量取较大值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1598) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vmin](spec/simd.md#api-tla-simd-vmin) | 待&#8288;转&#8288;测 | 逐 lane 取较小值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1577) | 蒋&#8288;继&#8288;极 | 丁&#8288;泓 |
| [tla.simd.vmins](spec/simd.md#api-tla-simd-vmins) | 待&#8288;转&#8288;测 | 逐 lane 与标量取较小值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2383) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vmul](spec/simd.md#api-tla-simd-vmul) | 待&#8288;转&#8288;测 | 逐 lane 相乘 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1087) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vmula](spec/simd.md#api-tla-simd-vmula) | 待&#8288;转&#8288;测 | 计算 acc + src0 × src1 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1965) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vmull](spec/simd.md#api-tla-simd-vmull) | 待&#8288;转&#8288;测 | 32 位 × 32 位得到 64 位乘积，返回低 32 位和高 32 位 | [待&#8288;修&#8288;复](tracking/simd.md#simd-042) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2169) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vmuls](spec/simd.md#api-tla-simd-vmuls) | 待&#8288;转&#8288;测 | 每个 lane 乘同一个标量 | [待&#8288;修&#8288;复](tracking/simd.md#simd-043) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1945) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vneg](spec/simd.md#api-tla-simd-vneg) | 待&#8288;转&#8288;测 | 逐 lane 取负 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2576) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vor](spec/simd.md#api-tla-simd-vor) | 待&#8288;转&#8288;测 | 逐 lane 对位模式执行 OR | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1895) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vpack](spec/simd.md#api-tla-simd-vpack) | 待&#8288;转&#8288;测 | 把整数元素窄化到指定奇偶槽位 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2861) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vrelu](spec/simd.md#api-tla-simd-vrelu) | 待&#8288;转&#8288;测 | 逐 lane 执行 ReLU | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2637) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vscatter](spec/simd.md#api-tla-simd-vscatter) | 待&#8288;转&#8288;测 | 按向量索引将活跃元素写入 UB | 未&#8288;登&#8288;记 | [fork&nbsp;实&#8288;现](https://github.com/patchliu/triton-ascend/blob/54d11d3e3cd1e7f985cce85f87ef1053530508be/third_party/ascend/language/tla/simd.py#L2754) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vsel](spec/simd.md#api-tla-simd-vsel) | 待&#8288;转&#8288;测 | 按谓词逐 lane 选择两个向量的元素 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2509) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vselr](spec/simd.md#api-tla-simd-vselr) | 待&#8288;转&#8288;测 | 按 lane 索引重排 uint16 向量 | 未&#8288;登&#8288;记 | [fork&nbsp;实&#8288;现](https://github.com/patchliu/triton-ascend/blob/54d11d3e3cd1e7f985cce85f87ef1053530508be/third_party/ascend/language/tla/simd.py#L2742) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vshl](spec/simd.md#api-tla-simd-vshl) | 待&#8288;转&#8288;测 | 逐 lane 左移 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1207) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vshls](spec/simd.md#api-tla-simd-vshls) | 待&#8288;转&#8288;测 | 按同一个标量位数逐 lane 左移 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1291) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vshr](spec/simd.md#api-tla-simd-vshr) | 待&#8288;转&#8288;测 | 逐 lane 右移，符号行为由源 dtype 决定 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1248) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vsqrt](spec/simd.md#api-tla-simd-vsqrt) | 待&#8288;转&#8288;测 | 逐 lane 求平方根 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2668) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vsstb](spec/simd.md#api-tla-simd-vsstb) | 待&#8288;转&#8288;测 | 按给定步长分块存储向量 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2426) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vsts](spec/simd.md#api-tla-simd-vsts) | 待&#8288;转&#8288;测 | 按分布模式把向量写入缓冲区 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-056) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L944) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vsub](spec/simd.md#api-tla-simd-vsub) | 待&#8288;转&#8288;测 | 逐 lane 相减 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1793) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vxor](spec/simd.md#api-tla-simd-vxor) | 待&#8288;转&#8288;测 | 逐 lane 对位模式执行 XOR | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1759) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.tail_mask](spec/simd.md#api-tla-simd-tail-mask) | 待&#8288;转&#8288;测 | 用 int32 计数构造 64 lane 尾块谓词 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2149) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.por](spec/simd.md#api-tla-simd-por) | 待&#8288;转&#8288;测 | 逐 lane 求谓词或 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2070) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.psel](spec/simd.md#api-tla-simd-psel) | 待&#8288;转&#8288;测 | 按控制谓词逐 lane 选择两个谓词值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2111) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.ppack](spec/simd.md#api-tla-simd-ppack) | 待&#8288;转&#8288;测 | 将谓词放入更长谓词的连续半区，另一半补 false | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-063) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3001) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.punpack](spec/simd.md#api-tla-simd-punpack) | 待&#8288;转&#8288;测 | 取出谓词连续的下半区或上半区 | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-064) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3011) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pintlv](spec/simd.md#api-tla-simd-pintlv) | 待&#8288;转&#8288;测 | 交错两个逻辑谓词，返回前后两个结果 | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-065) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3030) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pdintlv](spec/simd.md#api-tla-simd-pdintlv) | 待&#8288;转&#8288;测 | 对拼接谓词取偶、奇位置，返回两个结果 | [已&#8288;记&#8288;录&#8288;结&#8288;论](tracking/simd.md#simd-066) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3040) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pld](spec/simd.md#api-tla-simd-pld) | 待&#8288;转&#8288;测 | 从 UB 字节流加载按位打包的谓词 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L790) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.pst](spec/simd.md#api-tla-simd-pst) | 待&#8288;转&#8288;测 | 把谓词按位打包后写入 UB | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L834) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vsubc](spec/simd.md#api-tla-simd-vsubc) | 待&#8288;转&#8288;测 | 逐 lane 相减并返回无符号借位 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1421) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vaddcs](spec/simd.md#api-tla-simd-vaddcs) | 待&#8288;转&#8288;测 | 带输入进位的向量加法，返回和与进位。vaddc 不接收输入进位 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-070) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/language/tla/simd.py#L1595) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vsubcs](spec/simd.md#api-tla-simd-vsubcs) | 待&#8288;转&#8288;测 | 带输入借位的向量减法，返回差与借位。vsubc 不接收输入借位 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-071) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/language/tla/simd.py#L1613) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vlrelu](spec/simd.md#api-tla-simd-vlrelu) | 待&#8288;转&#8288;测 | 逐 lane 执行带标量斜率的 Leaky ReLU | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1333) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vprelu](spec/simd.md#api-tla-simd-vprelu) | 待&#8288;转&#8288;测 | 逐 lane 执行带向量斜率的 PReLU | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1365) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vcpadd](spec/simd.md#api-tla-simd-vcpadd) | 待&#8288;转&#8288;测 | 相邻 lane 配对求和，结果写入下半部 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-074) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/language/tla/simd.py#L1631) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vsqz](spec/simd.md#api-tla-simd-vsqz) | 待&#8288;转&#8288;测 | 按原顺序压紧活跃 lane，尾部按 mode 处理 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/language/tla/simd.py#L1660) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vusqz](spec/simd.md#api-tla-simd-vusqz) | 待&#8288;转&#8288;测 | 返回谓词活跃 lane 的排他前缀计数 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-076) | [simd.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/language/tla/simd.py#L1691) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.vunpack](spec/simd.md#api-tla-simd-vunpack) | 待&#8288;转&#8288;测 | 从选中槽位取整数并做零扩展或符号扩展 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L2270) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vgatherb](spec/simd.md#api-tla-simd-vgatherb) | 待&#8288;转&#8288;测 | 按字节偏移收集八个 32 字节块，逐输出元素应用 mask | [待&#8288;决](tracking/simd.md#simd-078) | [开&#8288;发&#8288;实&#8288;现／待&#8288;合&#8288;入](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L967) | 蒋&#8288;继&#8288;极 | 魏&#8288;子&#8288;尧 |
| [tla.simd.vabsdif](spec/simd.md#api-tla-simd-vabsdif) | 待&#8288;转&#8288;测 | 逐 lane 求差的绝对值 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/simd.py#L1829) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.dhistv2](spec/simd.md#api-tla-simd-dhistv2) | 待&#8288;转&#8288;测 | 拟生成直方图。现有 chistv2 的输出为前缀计数 | [1&nbsp;项&#8288;待&#8288;决](tracking/simd.md#simd-080) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |
| [tla.simd.chistv2](spec/simd.md#api-tla-simd-chistv2) | 待&#8288;转&#8288;测 | 向 uint16 累加器累加活跃 uint8 元素的累积直方图 | 未&#8288;登&#8288;记 | [simd.py](https://github.com/patchliu/triton-ascend/blob/54d11d3e3cd1e7f985cce85f87ef1053530508be/third_party/ascend/language/tla/simd.py#L2711) | 蒋&#8288;继&#8288;极 | 奚&#8288;伟 |

<a name="simt-index"></a>

## SIMT

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.scope.simt](spec/simt.md#api-tla-scope-simt) | 不&#8288;转&#8288;测 | 拟构造 SIMT VF 执行域 | [1&nbsp;项&#8288;待&#8288;决](tracking/simt.md#simt-001) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 宋&#8288;尧&#8288;飞 | 宋&#8288;尧&#8288;飞 |
| [tla.simt.to_ptr](spec/simt.md#api-tla-simt-to-ptr) | 不&#8288;转&#8288;测 | 拟将 tla.buffer 转为 Triton pointer | [1&nbsp;项&#8288;待&#8288;决](tracking/simt.md#simt-007) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 宋&#8288;尧&#8288;飞 | 宋&#8288;尧&#8288;飞 |

<a name="dfx-index"></a>

## DFX

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.device_assert](spec/dfx.md#api-tla-device-assert) | 不&#8288;转&#8288;测 | 在启用调试时生成设备端断言 | [2&nbsp;项&#8288;待&#8288;决](tracking/dfx.md#dfx-001) | [debug.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/debug.py#L115) | 高&#8288;威 | 高&#8288;威 |
| [tla.device_print](spec/dfx.md#api-tla-device-print) | 待&#8288;转&#8288;测 | 打印设备端标量或缓冲区 | 未&#8288;登&#8288;记 | [debug.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/debug.py#L98) | 高&#8288;威 | 高&#8288;威 |

<a name="dma-index"></a>

## DMA

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.load_gm_to_l1](spec/dma.md#api-tla-load-gm-to-l1) | 待&#8288;转&#8288;测 | 从 GM 加载 ND 数据到 L1 并转为 NZ | 待&#8288;转&#8288;测 | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L563) | 陈&#8288;相&#8288;廷 | 姜&#8288;正/Ricky |
| [tla.load_gm_to_ub](spec/dma.md#api-tla-load-gm-to-ub) | 待&#8288;转&#8288;测 | 从 GM 加载数据到 UB，可指定缓存策略 | [3&nbsp;项&#8288;待&#8288;决](tracking/dma.md#dma-002) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L499) | 宋&#8288;尧&#8288;飞 | 姜&#8288;正 |
| [tla.store_ub_to_gm](spec/dma.md#api-tla-store-ub-to-gm) | 待&#8288;转&#8288;测 | 将 UB 数据写入 GM | [2&nbsp;项&#8288;待&#8288;决](tracking/dma.md#dma-003) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L537) | 宋&#8288;尧&#8288;飞 | 姜&#8288;正 |
| [tla.copy_l1_to_l0](spec/dma.md#api-tla-copy-l1-to-l0) | 待&#8288;转&#8288;测 | 从 L1 向 L0A 或 L0B 搬运矩阵分块 | [待&#8288;转&#8288;测](tracking/dma.md#dma-004) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L737) | Ricky | Ricky |
| [~~tla.make_partition_view~~](spec/dma.md#api-tla-make-partition-view) | 不&#8288;需&#8288;要 | 拟构造逻辑分块访问描述符 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-005) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | Ricky | Ricky |
| [~~tla.make_strided_view~~](spec/dma.md#api-tla-make-strided-view) | 不&#8288;需&#8288;要 | 拟构造跨步分块访问描述符 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-006) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | Ricky | Ricky |
| [~~tla.make_gather_scatter_view~~](spec/dma.md#api-tla-make-gather-scatter-view) | 不&#8288;需&#8288;要 | 拟构造稀疏 gather/scatter 访问描述符 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-007) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | Ricky | Ricky |
| [~~tla.load_view~~](spec/dma.md#api-tla-load-view) | 不&#8288;需&#8288;要 | 拟从 tensor_view 加载分块 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-008) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | Ricky | Ricky |
| [~~tla.store_view~~](spec/dma.md#api-tla-store-view) | 不&#8288;需&#8288;要 | 拟写入 tensor_view 的指定分块 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-009) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | Ricky | Ricky |
| [~~tla.set_pad_value~~](spec/dma.md#api-tla-set-pad-value) | 不&#8288;需&#8288;要 | 拟单独设置 DMA 填充值。当前 GM→UB 接口不提供显式 padding | [已&#8288;关&#8288;闭](tracking/dma.md#dma-010) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 陈&#8288;相&#8288;廷 | Ricky |
| [~~tla.set_post_copy~~](spec/dma.md#api-tla-set-post-copy) | 不&#8288;需&#8288;要 | 拟设置搬运后处理 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-011) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 陈&#8288;相&#8288;廷 | Ricky |
| [~~tla.stride_store~~](spec/dma.md#api-tla-stride-store) | 不&#8288;需&#8288;要 | 拟对缓冲区执行带步长的读写。当前可用 view/subview 表达部分切片，需按具体用法组合 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-012) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 刘&#8288;鹏&#8288;程 | Ricky |
| [~~tla.stride_load~~](spec/dma.md#api-tla-stride-load) | 不&#8288;需&#8288;要 | 拟对缓冲区执行带步长的读写。当前可用 view/subview 表达部分切片，需按具体用法组合 | [已&#8288;关&#8288;闭](tracking/dma.md#dma-013) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 刘&#8288;鹏&#8288;程 | Ricky |

<a name="scalar-index"></a>

## 标量

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.set_atomic](spec/scalar.md#api-tla-set-atomic) | 待&#8288;转&#8288;测 | 设置后续 GM store 的原子模式 | [已&#8288;修&#8288;订](tracking/scalar.md#scalar-001) | [dma.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L900) | 陈&#8288;相&#8288;廷 | 宋&#8288;尧&#8288;飞 |
| [tla.select](spec/scalar.md#api-tla-select) | 待&#8288;转&#8288;测 | 根据标量条件选择一个值 | [已&#8288;修&#8288;订](tracking/scalar.md#scalar-002) | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L43) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.abs](spec/scalar.md#api-tla-abs) | 待&#8288;转&#8288;测 | 求标量绝对值 | [已&#8288;修&#8288;订](tracking/scalar.md#scalar-003) | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/33819a5b6a9bb503e46da2a6e5962f0cd0c44393/third_party/ascend/language/tla/scalar.py#L59) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [^](spec/scalar.md#api-xor) | 待&#8288;转&#8288;测 | 对整数执行按位异或 | 未&#8288;登&#8288;记 | [core.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/python/triton/language/core.py#L967) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.cast](spec/scalar.md#api-tla-cast) | 待&#8288;转&#8288;测 | 对标量作数值转换或等宽位重解释 | [已&#8288;修&#8288;订](tracking/scalar.md#scalar-005) | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L111) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.cdiv](spec/scalar.md#api-tla-cdiv) | 待&#8288;转&#8288;测 | 按 (x+div−1)//div 计算非负整数分块数 | 未&#8288;登&#8288;记 | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L90) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.clamp](spec/scalar.md#api-tla-clamp) | 待&#8288;转&#8288;测 | 将标量限制在上下界间 | [已&#8288;修&#8288;订](tracking/scalar.md#scalar-007) | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L175) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [~~tl.constexpr~~](spec/scalar.md#api-tl-constexpr) | 不&#8288;需&#8288;要 | 标注编译期可读取的内核参数 | [已&#8288;关&#8288;闭](tracking/scalar.md#scalar-016) | [core.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/python/triton/language/core.py#L200) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.infinity](spec/scalar.md#api-tla-infinity) | 待&#8288;转&#8288;测 | 构造指定浮点类型的正无穷 | 未&#8288;登&#8288;记 | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L158) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.maximum](spec/scalar.md#api-tla-maximum) | 待&#8288;转&#8288;测 | 取两个标量的较大值 | [1&nbsp;项&#8288;待&#8288;决](tracking/scalar.md#scalar-012) | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L58) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.max_value](spec/scalar.md#api-tla-max-value) | 待&#8288;转&#8288;测 | 构造指定类型的最大有限值 | 未&#8288;登&#8288;记 | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L127) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |
| [tla.minimum](spec/scalar.md#api-tla-minimum) | 待&#8288;转&#8288;测 | 取两个标量的较小值 | [1&nbsp;项&#8288;待&#8288;决](tracking/scalar.md#scalar-014) | [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L74) | 刘&#8288;鹏&#8288;程 | 宋&#8288;尧&#8288;飞 |

<a name="memory-index"></a>

## 内存

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.alloc](spec/memory.md#api-tla-alloc) | 待&#8288;转&#8288;测 | 分配未初始化的片上连续缓冲区 | [1&nbsp;项&#8288;待&#8288;决](tracking/memory.md#memory-001) | [签&#8288;名&#8288;待&#8288;更&#8288;新](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/alloc.py#L27) | 刘&#8288;鹏&#8288;程 | 刘&#8288;鹏&#8288;程 |
| [tla.reshape](spec/memory.md#api-tla-reshape) | 待&#8288;转&#8288;测 | 对连续缓冲区做零拷贝 reshape | [1&nbsp;项&#8288;待&#8288;决](tracking/memory.md#memory-004) | [view.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/view.py#L184) | 刘&#8288;鹏&#8288;程 | 刘&#8288;鹏&#8288;程 |
| [tla.view](spec/memory.md#api-tla-view) | 待&#8288;转&#8288;测 | 从 GM 指针构造共享底层存储的视图，保留指针来源 | [1&nbsp;项&#8288;待&#8288;决](tracking/memory.md#memory-004) | [签&#8288;名&#8288;待&#8288;更&#8288;新](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/view.py#L130) | 刘&#8288;鹏&#8288;程 | 刘&#8288;鹏&#8288;程 |
| [tla.subview](spec/memory.md#api-tla-subview) | 待&#8288;转&#8288;测 | 在已有缓冲区上构造同秩子视图 | [1&nbsp;项&#8288;待&#8288;决](tracking/memory.md#memory-004) | [view.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/view.py#L105) | 刘&#8288;鹏&#8288;程 | 刘&#8288;鹏&#8288;程 |
| [tla.fill](spec/memory.md#api-tla-fill) | 待&#8288;转&#8288;测 | 将同一个标量写入 UB 缓冲区的各元素 | [待&#8288;转&#8288;测](tracking/memory.md#memory-005) | [fill.py](https://github.com/triton-lang/triton-ascend/blob/33819a5b6a9bb503e46da2a6e5962f0cd0c44393/third_party/ascend/language/tla/fill.py#L94) | 刘&#8288;鹏&#8288;程 | Ricky |

<a name="sync-index"></a>

## 同步

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.sync.block_set_flag](spec/sync.md#api-tla-sync-block-set-flag) | 待&#8288;转&#8288;测 | 设置跨核逻辑标志或物理块内计数器 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L159) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.block_wait_flag](spec/sync.md#api-tla-sync-block-wait-flag) | 待&#8288;转&#8288;测 | 等待对应的跨核逻辑标志或物理块内计数器 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L192) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.mutex_lock](spec/sync.md#api-tla-sync-mutex-lock) | 待&#8288;转&#8288;测 | 获取流水线缓冲区同步令牌 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L231) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.barrier](spec/sync.md#api-tla-sync-barrier) | 待&#8288;转&#8288;测 | 等待指定流水线的先前操作 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L125) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.mutex_unlock](spec/sync.md#api-tla-sync-mutex-unlock) | 待&#8288;转&#8288;测 | 释放流水线缓冲区同步令牌 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L260) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.set_flag](spec/sync.md#api-tla-sync-set-flag) | 待&#8288;转&#8288;测 | 从生产流水线发出事件 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L131) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.wait_flag](spec/sync.md#api-tla-sync-wait-flag) | 待&#8288;转&#8288;测 | 从消费流水线等待配对事件 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L138) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |
| [tla.sync.sync_all](spec/sync.md#api-tla-sync-sync-all) | 待&#8288;转&#8288;测 | 执行 AIV 或 AIC/AIV 全局同步 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L145) | 谭&#8288;子&#8288;睿 | 姚&#8288;季 |

<a name="schedule-index"></a>

## 调度

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.program_id](spec/schedule.md#api-tla-program-id) | 待&#8288;转&#8288;测 | 返回指定 launch 网格轴的当前 program ID | [1&nbsp;项&#8288;待&#8288;决](tracking/schedule.md#schedule-001) | 待&#8288;实&#8288;现 | 刘&#8288;鹏&#8288;程 | 丁&#8288;泓 |
| [tla.annotate_multi_buffer](spec/schedule.md#api-tla-annotate-multi-buffer) | 待&#8288;转&#8288;测 | 给缓冲区添加多缓冲版本数注解 | [1&nbsp;项&#8288;待&#8288;决](tracking/schedule.md#schedule-002) | 待&#8288;实&#8288;现 | 刘&#8288;鹏&#8288;程 | — |
| [tla.sub_block_id](spec/schedule.md#api-tla-sub-block-id) | 待&#8288;转&#8288;测 | 读取当前 AIV 子块在所属 Cube 块内的索引 | 未&#8288;登&#8288;记 | [sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/sync.py#L284) | 刘&#8288;鹏&#8288;程 | 刘&#8288;鹏&#8288;程 |

<a name="language-index"></a>

## 语言

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.sqrt](spec/language.md#api-tla-sqrt) | 待&#8288;转&#8288;测 | 求标量平方根 | [已&#8288;修&#8288;订](tracking/language.md#language-001) | [math.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/math.py#L79) | 刘&#8288;鹏&#8288;程 | 丁&#8288;泓 |
| [~~tla.symbolic~~](spec/language.md#api-tla-symbolic) | 不&#8288;需&#8288;要 | 拟构造带名称的符号表达式。动态标量可由内核形参提供 | [已&#8288;关&#8288;闭](tracking/language.md#language-002) | 未&#8288;找&#8288;到&#8288;实&#8288;现 | 刘&#8288;鹏&#8288;程 | — |
| [tla.truncmod](spec/language.md#api-tla-truncmod) | 待&#8288;转&#8288;测 | 计算向零截断除法的余数 | 未&#8288;登&#8288;记 | [math.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/math.py#L49) | 刘&#8288;鹏&#8288;程 | 丁&#8288;泓 |
| [tl.assume](spec/language.md#api-tl-assume) | 不&#8288;转&#8288;测 | 向编译器承诺条件成立 | 未&#8288;登&#8288;记 | [core.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/python/triton/language/core.py#L2950) | 刘&#8288;鹏&#8288;程 | 丁&#8288;泓 |
| [~~tla.where~~](spec/language.md#api-tla-where) | 不&#8288;需&#8288;要 | 根据标量条件或向量谓词做逐元素选择 | [已&#8288;关&#8288;闭](tracking/language.md#language-003) | [selection.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/selection.py#L89) | 刘&#8288;鹏&#8288;程 | 丁&#8288;泓 |

<a name="dtype-index"></a>

## 数据类型

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.bfloat16](spec/dtype.md#api-tla-bfloat16) | 待&#8288;转&#8288;测 | 表示 bfloat16 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.dtype](spec/dtype.md#api-tla-dtype) | 待&#8288;转&#8288;测 | 以 Triton canonical 名称构造 dtype 对象 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.float32](spec/dtype.md#api-tla-float32) | 待&#8288;转&#8288;测 | 表示 float32 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.int32](spec/dtype.md#api-tla-int32) | 待&#8288;转&#8288;测 | 表示 int32 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.int64](spec/dtype.md#api-tla-int64) | 待&#8288;转&#8288;测 | 表示 int64 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.uint16](spec/dtype.md#api-tla-uint16) | 待&#8288;转&#8288;测 | 表示 uint16 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.uint32](spec/dtype.md#api-tla-uint32) | 待&#8288;转&#8288;测 | 表示 uint32 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.uint8](spec/dtype.md#api-tla-uint8) | 待&#8288;转&#8288;测 | 表示 uint8 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.float16](spec/dtype.md#api-tla-float16) | 待&#8288;转&#8288;测 | 表示 float16 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.int8](spec/dtype.md#api-tla-int8) | 待&#8288;转&#8288;测 | 表示 int8 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.int16](spec/dtype.md#api-tla-int16) | 待&#8288;转&#8288;测 | 表示 int16 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
| [tla.uint64](spec/dtype.md#api-tla-uint64) | 待&#8288;转&#8288;测 | 表示 uint64 元素类型 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L54) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |

<a name="host-index"></a>

## 主机接口

| 接口 | 状态 | 用途 | 评审 | 源码 | 设计 | 实现 |
| --- | :---: | --- | :---: | :---: | :---: | :---: |
| [tla.jit](spec/host.md#api-tla-jit) | 待&#8288;转&#8288;测 | 装饰使用 TLA 前端编译的内核或辅助函数 | 未&#8288;登&#8288;记 | [__init__.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/__init__.py#L134) | 刘&#8288;鹏&#8288;程 | 姚&#8288;季 |
