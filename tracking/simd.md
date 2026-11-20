# SIMD评审跟踪

[返回接口目录](../README.md)

<a name="simd-baseline"></a>

## 魏子尧负责的新增接口实现基线

这 10 个接口的实现负责人均为魏子尧，补充范围为：vcadd、vcgadd / vcgmax / vcgmin、vld2、ppack / punpack、pintlv / pdintlv、vgatherb。

规格依据为 [TLA d54604c1133d](https://github.com/wzy1214/triton-ascend/tree/d54604c1133df2631c6b32d8649eea79c6b2c975) 与其配套 [AscendNPU-IR bd88eea996b4](https://gitcode.com/Ascend/AscendNPU-IR/tree/bd88eea996b49aee9898465b77815bb32f762bb6)，核对了公开入口、builder、AVE 类型约束及 lowering。各小节提供固定提交的源码和用例链接；开发分支已实现不等于主线已合入或完成转测。

在本次核对的 [main-dev 0227cdebc794](https://github.com/triton-lang/triton-ascend/tree/0227cdebc7940f38cbe67007110a44fe2434ec94) 中，这 10 个接口尚无公开入口；不能以开发分支实现认定该主线基线可执行。该主线的 SIMD 前端与上一轮 `ff7b02c72795` 相同，配套 AscendNPU-IR 仍为 `bd228faec3b5`。

本次仅作源码与文档核对，未重新编译或运行真卡。引用用例不等于本次执行通过；既有联调记录限于 Ascend950 / Process，不能外推至其他目标、后端或最新主线合并版本。

<a name="simd-enum-parameters"></a>

## SIMD 枚举参数

规格已将 PAT、ROUND、PART、POS 定为枚举，并为 vlds / vld2 / vsts / pld 的 dist、vunpack.pp、vdiv / vexp / vsqrt 的 precision 分别约定 LOAD_DIST / STORE_DIST / PRED_DIST、PP、PRECISION；membar.scope 使用 SCOPE 枚举。各接口仅接受参数表列出的成员；vsts.dist 和 precision 保留 None。

实现基线 `0227cde` / `d54604c` 中，PAT / ROUND / PART / POS 仍是字符串常量集合，membar.scope 仍接收字符串，新增枚举尚未提供。待前端补枚举定义、成员校验及字符串拒绝用例；本次仅更新规格，未修改代码或执行设备验证。

<a name="simd-mode-contract"></a>

## SIMD mode 语义

规格统一为 ZEROING 填 0、MERGING 保留原值。逐元素算子保留第一个数据向量，累加算子保留 acc；转换、加载、标量广播及归约通过新增的可选 merge 参数提供保留值。merge 须与结果同型，MERGING 时必填，ZEROING 时为 None。

mask 决定参与计算的输入；归约的结果位置始终写入，mode 控制其余位置。vcpadd 的结果写入前半部分，vsqz 的结果写入压缩后的前缀，MERGING 只保留各自尾部，不能按原输入 mask 覆盖有效结果。

实现待跟进：通用 builder 的 MERGING 补零行为需改为保留输入；11 个接口需增加 merge；归约需补齐尾部处理，vcpadd / vsqz 需调整合并位置。此处是规格设计，不能视为代码或设备已支持。核对基线：TLA 0227cde、d54604c，以及 vcpadd / vsqz 所在的 aa927de；本次未运行编译或设备测试。

<a name="simd-review-followup"></a>

## SIMD 复查待办

本轮补充规格、示例和测试引用；未修改实现或执行编译、设备测试。核对基线：TLA 0227cde / d54604c / aa927de，AVE bd228fae / bd88eea；公共测试链接固定到 074cf629。

| 接口 | 尚需确认或补测 |
| --- | --- |
| pset | 固定图样统一使用 pset。旧前端 pset(64) 生成 32 个谓词元素，B64 数据向量却有 64 个；64 元素全真谓词使用 pset(32, PAT.ALL)。需直接验证谓词长度和 true/false 分布，并补充 M3 / M4 的目标语义。 |
| vcmp / vcmps | 保留现有 EQ / LT 默认值；“修改默认”的评审未给出目标值，待确认。补浮点特殊值与无符号比较边界。 |
| vci | 现有引用主要作为其他测试的辅助操作；需直接验证递增序列。 |
| vcmax / vcmin | add_cmp.py 只核对 lane 0；新增 merge 和尾部规则尚无对应测试。 |
| vmin / vmuls | 已改为公共专用测试；vmin.py 的 MERGING 参考值仍为补零，需按新规格更新。vmuls 的 float32 用例不能覆盖 B64 常量和标量拒绝规则。 |
| vaddc / vsubc / vaddcs / vsubcs | 保留当前长度差异：vaddc 至多 64 元素，后三者要求满 64 元素；统一限制前先确认短向量需求。 |
| vabs / vdiv | 最小负整数的绝对值、整数除零及最小负数除以 −1 的结果需按目标确认，不新增饱和或异常承诺。 |
| vscatter | 仅补充写入及同步责任；readonly 声明、别名和写后重读问题继续跟踪，不能凭补屏障就认定修复。 |
| vselr / dhistv2 / chistv2 | vselr 保留现有 uint16 范围；扩展类型和三者的最终主线入口、公开测试仍需确认。 |

<a name="simd-vlds"></a>

## tla.simd.vlds

[查看规格](../spec/simd.md#api-tla-simd-vlds)

按 TLA 0227cde 的 vlds → create_simd_vlds → AVE VFLoadOp 补充入口；单次返回一个向量。LOAD_DIST 枚举沿用规格设计，旧前端仍接收字符串。

规格先列 NORM 和 BRC；UNPK、拆通道等分布的读取范围、填充位置和 elem 类型组合待单独定义。DINTLV 的双结果使用 vld2，谓词使用 pld。B64 返回 64 元素沿用 TLA 类型约定，但所查 AVE 直接 lowering 使用 v32i64，完整读写需专项验证。新增规格不代表这些路径已通过设备验收。

结论：NORM / BRC 的 dtype、对齐和 B64 矩阵待补测；未对齐加载及 elem 与 buffer 不同型暂不纳入公开规格。

<a name="simd-dsa-migration"></a>

## DSA 接口替换

| 原调用 | 对应接口 | 迁移时核对 |
| --- | --- | --- |
| vfull / vsplat | vdups | 输入是标量，不能直接改成接收向量的 vdupv；任意长度和 B64 类型覆盖须另核对。 |
| vreduce_max | vcmax + vdupv | 多向量输入先逐元素 vmax 合并，再归约和广播。 |
| vreduce_sum | vcadd | 核对 dtype；当前 vcadd 规格仅列 int32 / float32。 |
| vbroadcast | vdupv | low 参数映射为 POS.LOWEST / HIGHEST。 |
| vtruncf | vcvt | 核对 round、sat、part；FP8 不在当前 vcvt 支持范围。 |
| vfma | vmula | 保留融合乘加的舍入语义。 |

这些别名已从规格目录移除；DSA 调用迁移仍需在对应代码仓完成，本次未修改 DSA。

<a name="simd-011"></a>

## SIMD-011 · tla.simd.vadds

[查看规格](../spec/simd.md#api-tla-simd-vadds)

- `scalar`：考虑前端提前拒绝单 lane 向量。

结论：待决。

<a name="simd-013"></a>

## SIMD-013 · tla.simd.vaxpy

[查看规格](../spec/simd.md#api-tla-simd-vaxpy)

- `scalar`：单 lane vector 会进入 AVE 标量操作数，类型契约不符，见 [_scalar_handle](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/simd.py#L1136) 与 [AVE_VecScalarOp](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L111)。正例保留同 dtype 的秩 0 tensor；单 lane vector 的前端拒绝及诊断用例待补。

结论：已确认当前限制；实现修复与回归待完成。

<a name="simd-014"></a>

## SIMD-014 · tla.simd.vcadd

[vcadd 规格](../spec/simd.md#api-tla-simd-vcadd)

入口已实现，复用既有 AVE ReductionOp。[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L2645) 与 [HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L767) 在该基线中返回同型向量，只有第 0 个 lane 保证为归约结果。新规格要求尾部按 [mode](#simd-mode-contract) 处理，仍待实现；内部扩宽不改变公开返回类型。

[memory_completion.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L68) 目前覆盖 int32、float32；前端未限制其他 dtype，但不能据此声明完整执行支持。后续需补逐类型、溢出及浮点特殊值矩阵。

多后端：AVE [Ascend960 分支](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp#L2122) 已组合带 mask 的 `vcgaddv2` 与不带 mask 的 `vcaddv2`，由 AVE 隐藏指令差异。检视中的 `vcgadd2` 在此代码中拼作 `vcgaddv2`。这允许保留 TLA 签名，但需验证 lane 0、输出 dtype、全假 / 非前缀 mask、溢出及浮点顺序；不能只发汇总阶段的 vcaddv2。

A6 需确认：具体产品对应的 target 枚举、CANN / 编译器版本，以及是否采用上述 Ascend960 / dav-920r1 路径。源码中的目标名称不能单独证明全部 A6 型号的支持；本次无 A6 编译或真卡结果。

结论：vcadd 的公开语义可保留，AVE 已有多目标适配分支；设备覆盖及其他 dtype 待补，TLA 新入口主线合入仍待确认。

<a name="simd-015"></a>

## SIMD-015 · tla.simd.vcgadd

[vcgadd 规格](../spec/simd.md#api-tla-simd-vcgadd)

开发基线 d54604c / bd88eea 的 AVE GroupReductionOp 使用 Ascend950 的 8 个连续 32 字节组，前 8 个结果有效、尾部补零，mask 选择输入元素。新规格的 [MERGING 尾部保留](#simd-mode-contract) 仍待实现。vcgadd 的 int16 / uint16 扩宽为 int32 / uint32；max / min 保持输入类型。

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3087)、[HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L514) 与 [group_reduce.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33) 相互对应。verifier 拒绝浮点输入配无符号 cast；Ascend960 当前明确不支持。浮点顺序、NaN、正负零及空 min/max 组遵循目标指令，未另行承诺跨硬件统一值。

多后端：当前 [GroupReductionOp 目标检查](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp#L5028) 仅接受 Ascend950。A6 / V2 需逐项核对每组源 lane、组数、结果位置、整数扩宽、mask 及空组值；若 AVE 能组合保持本接口八组语义，可复用签名。若要直接暴露新的原生分组，需新参数或新接口，不能静默改变现有结果布局。

结论：950 开发实现边界已确认；A6 group 语义与适配待确认，当前不放开其他目标。

<a name="simd-016"></a>

## SIMD-016 · tla.simd.vcgmax

[vcgmax 规格](../spec/simd.md#api-tla-simd-vcgmax)

开发基线 d54604c / bd88eea 的 AVE GroupReductionOp 使用 Ascend950 的 8 个连续 32 字节组，前 8 个结果有效、尾部补零，mask 选择输入元素。新规格的 [MERGING 尾部保留](#simd-mode-contract) 仍待实现。vcgadd 的 int16 / uint16 扩宽为 int32 / uint32；max / min 保持输入类型。

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3101)、[HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L514) 与 [group_reduce.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33) 相互对应。verifier 拒绝浮点输入配无符号 cast；Ascend960 当前明确不支持。浮点顺序、NaN、正负零及空 min/max 组遵循目标指令，未另行承诺跨硬件统一值。

多后端：沿用 [SIMD-015](#simd-015) 的八组契约审查；另核对 V2 max 的组内成员、输出位置、全假组初值、NaN 和正负零选择。底层存在 vcgmaxv2 wrapper 不能证明它与本接口等价。

结论：950 开发实现边界已确认；A6 group 语义与适配待确认，当前不放开其他目标。

<a name="simd-017"></a>

## SIMD-017 · tla.simd.vcgmin

[vcgmin 规格](../spec/simd.md#api-tla-simd-vcgmin)

开发基线 d54604c / bd88eea 的 AVE GroupReductionOp 使用 Ascend950 的 8 个连续 32 字节组，前 8 个结果有效、尾部补零，mask 选择输入元素。新规格的 [MERGING 尾部保留](#simd-mode-contract) 仍待实现。vcgadd 的 int16 / uint16 扩宽为 int32 / uint32；max / min 保持输入类型。

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3112)、[HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L514) 与 [group_reduce.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33) 相互对应。verifier 拒绝浮点输入配无符号 cast；Ascend960 当前明确不支持。浮点顺序、NaN、正负零及空 min/max 组遵循目标指令，未另行承诺跨硬件统一值。

多后端：沿用 [SIMD-015](#simd-015) 的八组契约审查；另核对 V2 min 的组内成员、输出位置、全假组初值、NaN 和正负零选择。底层存在 vcgminv2 wrapper 不能证明它与本接口等价。

结论：950 开发实现边界已确认；A6 group 语义与适配待确认，当前不放开其他目标。

<a name="simd-022"></a>

## SIMD-022 · tla.simd.vcmps

[查看规格](../spec/simd.md#api-tla-simd-vcmps)

严格同型要求尚未完全实现：[vcmps](https://github.com/triton-lang/triton-ascend/blob/0227cdebc7940f38cbe67007110a44fe2434ec94/third_party/ascend/language/tla/simd.py#L2133) 先调用 vdup，[_make_scalar](https://github.com/triton-lang/triton-ascend/blob/0227cdebc7940f38cbe67007110a44fe2434ec94/third_party/ascend/language/tla/simd.py#L741) 遇到不同 dtype 的整数 tensor 会调用 cast。它不会像 vaxpy / vmuls 一样直接报 dtype 不一致；其他不匹配类型及非标量输入会抛 TypeError。

结论：规格要求运行时标量与源向量 dtype 一致，不一致抛 TypeError，当前使用同型输入；将不同整数 dtype 改为拒绝需后续前端修改与回归，本次未修代码。

<a name="simd-023"></a>

## SIMD-023 · tla.simd.vcvt

[查看规格](../spec/simd.md#api-tla-simd-vcvt)

类型对与属性表已写回规格，依据 TLA `0227cdebc794` 与配套 AscendNPU-IR `bd228faec3b5`。主要实现边界：

- 不支持的类型对未统一在 TLA 前端拒绝。整数→整数直接抛 ValueError；同型浮点、float16↔bfloat16 等可能进入没有匹配的 AVE lowering。后续应按已公布白名单提前诊断，不应出现后端 fatal error。
- round / sat / part 按类型对使用；例如 float16→int32 不传 sat，int32→float32 不传 part。ROUND 是 str 常量集合，当前参数名为 round；六个名称被前端接受，不代表每种指令都实现了全部舍入方式。sat 当前通过 bool 转换而非严格检查类型，非 bool 拒绝用例也需前端补齐。
- FP8↔float32 有转换分支，但四路槽位与 EVEN / ODD 的关系、mask 和各目标舍入未完成验收；暂不列设备正例。
- 64 位路径走 [AVE 库转换](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/lib/Conversion/HIVMAVEToStandard/HIVMAVEToStandard.cpp#L858)，不能仅凭直接 intrinsic 分支判定支持。[VecOpI64.cpp](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/lib/Template/lib/RegBase/Vector/VecOpI64.cpp#L24) 的 int64→float32 和 float32→int64 包装未使用调用者 mask，后者固定 RS_ENABLE；ROUND.O 未见对应分支。TLA 按总位数计算的输出 lane 与库的 2×VL 表示也需对齐。uint64↔float32 同样有库函数，仍需独立核对精度、lane、mask 与饱和，暂不列设备正例。

现有专用测试只覆盖 int32↔float32 的部分 round / sat / mask，以及 float16↔float32 的 EVEN / ODD 与源 mask。其他类型对及 NaN / Inf / 非规格化数、边界溢出、ROUND.A / F / C / O 须按目标补测；不能由这几个用例外推整个矩阵。本次未执行编译或设备测试。

结论：类型对范围已明确；前置诊断、FP8 / 64 位完整语义及各目标设备矩阵仍待完成。

<a name="simd-032"></a>

## SIMD-032 · tla.simd.vld

[查看规格](../spec/simd.md#api-tla-simd-vld)

- 原需求是按分布加载连续数据，当前 vld 接收索引并委托 vgather。当前连续加载使用 vlds；原始连续 vld 的命名和迁移入口仍待确认。

结论：待决。

<a name="simd-033"></a>

## SIMD-033 · tla.simd.vld2

[vld2 规格](../spec/simd.md#api-tla-simd-vld2)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L936) 已通过两次 NORM vload 和一次 vdintlv 组合实现，完整读取两个寄存器，再返回偶 / 奇位置数据；off 按元素计。它没有依赖旧的双结果分布 load lowering。

[memory_completion.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L68) 包含 int16、float16、int32、float32。uint16、uint32、bfloat16 虽被入口接受，尚需专用执行覆盖。

结论：开发分支已提供入口，主线合入仍待确认；剩余 dtype 的验证待补。

<a name="simd-042"></a>

## SIMD-042 · tla.simd.vmull

[查看规格](../spec/simd.md#api-tla-simd-vmull)

问题单报告的 950PR 实测：`uint32(0xFFFFFFFF) × uint32(2)` 返回低半部 `0xFFFFFFFE`、高半部 `0xFFFFFFFF`，正确高半部应为 `0x00000001`；相同位型按 `int32` 计算符合有符号语义。本次未重跑该设备用例。

源码复核：[前端](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/simd.py#L2169) 接受 uint32，但 [builder](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/cpp/tla_builder.cc#L913) 创建 VFVMULLOp 未传 cast；[ODS](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L76) 默认 cast_signed，[库调用类型推导](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/lib/Dialect/HIVMAVE/IR/HIVMAVEOps.cpp#L56) 使用此属性。无符号信息未传递到 AVE。

转测处理：保留 int32 用例；uint32 暂不列设备正例，待传递无符号 cast 后回归最高位为 1、普通值、零值及 mask。不能仅凭低半部正确判断完整乘积正确。

结论：已确认当前缺陷；本次仅修正规格，代码尚未修复。

<a name="simd-043"></a>

## SIMD-043 · tla.simd.vmuls

[查看规格](../spec/simd.md#api-tla-simd-vmuls)

- `scalar`：单 lane vector 会进入 AVE 标量操作数，类型契约不符，见 [_scalar_handle](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/simd.py#L1136) 与 [AVE_VecScalarOp](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L111)。正例保留同 dtype 的秩 0 tensor；单 lane vector 的前端拒绝及诊断用例待补。

- B64 的 Python int 常量经 float 和接收 double 的 [create_simd_scalar](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/cpp/tla_builder.cc#L1048) 构造，`2**53+1` 在建 IR 前已变为 `2**53`。保留同 dtype 的秩 0 tensor 路径；常量、tensor、最高位与 mask 组合需分别回归。

结论：已确认当前限制；实现修复与回归待完成。

<a name="simd-056"></a>

## SIMD-056 · tla.simd.vsts

[查看规格](../spec/simd.md#api-tla-simd-vsts)

- 规格明确 NORM 按元素写入、mask=false 保留内存；None 固定为 NORM_B32。ONEPT 仅规定 mask=None 时写 val[0]，其他 mask 行为待确认。
- 公开数值存储规格移除 int1；按位存储使用 pst。不能仅凭 AVE 的 I1 类型约束推断 vsts 已定义谓词密度和写入范围。
- 本轮规定 NORM 起点按 32 字节、ONEPT 按元素大小对齐；未对齐路径、B64 的 64 元素完整存储仍须按目标回归。
- packing、交错、通道合并未列入本节支持范围；保留 dist × dtype × 对齐 × mask 的独立验收。scope/simd_only.py 只覆盖构建/编译，add_cmp.py 含 NORM_B32 和 ONEPT_B32 调用，不能代表完整矩阵。

结论：待决。

<a name="simd-063"></a>

## SIMD-063 · tla.simd.ppack

[ppack 规格](../spec/simd.md#api-tla-simd-ppack)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3001) 已通过独立 AVE 接口实现。逻辑 pack / unpack 使用 LOWER / HIGHER 连续半区；pack 的另一半补 false，unpack 只选连续的一半。不能把它替换成旧 preg.cast。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-064"></a>

## SIMD-064 · tla.simd.punpack

[punpack 规格](../spec/simd.md#api-tla-simd-punpack)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3011) 已通过独立 AVE 接口实现。逻辑 pack / unpack 使用 LOWER / HIGHER 连续半区；pack 的另一半补 false，unpack 只选连续的一半。不能把它替换成旧 preg.cast。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-065"></a>

## SIMD-065 · tla.simd.pintlv

[pintlv 规格](../spec/simd.md#api-tla-simd-pintlv)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3030) 已通过独立 AVE 接口实现。交错 / 去交错接口返回两个逻辑谓词，输入密度相同；宽度从 lane 数推导，不需要 width 参数。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-066"></a>

## SIMD-066 · tla.simd.pdintlv

[pdintlv 规格](../spec/simd.md#api-tla-simd-pdintlv)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3040) 已通过独立 AVE 接口实现。交错 / 去交错接口返回两个逻辑谓词，输入密度相同；宽度从 lane 数推导，不需要 width 参数。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-070"></a>

## SIMD-070 · tla.simd.vaddcs

[查看规格](../spec/simd.md#api-tla-simd-vaddcs)

- 实现已存在：`vaddcs(src0, src1, carry, mask=None)`，返回 `(sum, carry_out)`；64 lane int32/uint32，`carry` 必填。

结论：待决。规格已按实现补齐，请确认签名后关闭。

<a name="simd-071"></a>

## SIMD-071 · tla.simd.vsubcs

[查看规格](../spec/simd.md#api-tla-simd-vsubcs)

- 实现已存在：`vsubcs(src0, src1, borrow, mask=None)`，返回 `(diff, borrow_out)`；约束同 `vaddcs`。
- 注意极性：本接口的 `borrow` 与 AVE 侧 no-borrow 的 `carry_in` 相反，前端已取反。

结论：待决。规格已按实现补齐，请确认签名与极性描述后关闭。

<a name="simd-074"></a>

## SIMD-074 · tla.simd.vcpadd

[查看规格](../spec/simd.md#api-tla-simd-vcpadd)

- 实现已存在：`vcpadd(src, mask=None, mode=...)`，返回 `tla.vector`（原提案的 `-> tla,tensor` 是笔误）。
- 旧基线 aa927de 的语义为相邻 lane 配对求和：结果 lane `j < N/2` 为 `src[2j] + src[2j+1]`，其余 lane 为 0；只支持 f16/f32 满寄存器。新规格的尾部处理见 [mode](#simd-mode-contract)，尚不能按旧实现关闭评审。

结论：待决。需补齐 mode 实现及对应测试。

<a name="simd-076"></a>

## SIMD-076 · tla.simd.vusqz

[查看规格](../spec/simd.md#api-tla-simd-vusqz)

- 原描述"拟将谓词展开为索引向量"与实现不符：实现返回排他前缀计数（结果 lane `i` 为索引小于 `i` 的活跃 lane 数）。
- aa927de 的默认 dtype 为字符串 u32；规格统一为 tl.uint32。前端已接受 tl.dtype，但默认值和字符串入口仍需统一。
- 前端文档明确有符号结果使用相同位模式，规格据此说明 int8 的 128～255 计数显示为 −128～−1；设备需补 127/128/255 边界用例。

结论：待决。规格已按实现更正，请确认后关闭。

<a name="simd-078"></a>

## SIMD-078 · tla.simd.vgatherb

[vgatherb 规格](../spec/simd.md#api-tla-simd-vgatherb)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L967) 与 [HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L192) 已定义为八个 32 字节块的 gather。数据为 int32 / uint32 / float32；64 个 uint32 索引中使用前 8 个，索引单位为字节，mask 则逐输出元素生效。

AVE 通过现有寄存器操作展开索引并调用 masked gather，因此旧的原生无 mask block-gather 限制不等于当前接口没有 mask。索引只要求 4 字节对齐；[memory_completion.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L108) 的现有块起点均为 32 字节对齐，4 字节对齐但非 32 字节对齐的地址仍需独立执行用例。

结论：开发分支已提供入口，主线合入仍待确认；非 32 字节对齐块起点及最终目标版本回归待补。

<a name="simd-080"></a>

## SIMD-080 · tla.simd.dhistv2

[查看规格](../spec/simd.md#api-tla-simd-dhistv2)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-081"></a>

## SIMD-081 · tla.simd.membar

[查看规格](../spec/simd.md#api-tla-simd-membar)

- 规格已把 `scope` 定为枚举 `tla.simd.SCOPE`（16 个成员，与 `ave.hir.membar` 的 i32 编码 0–15 一一对应），不再按字符串描述。
- 实现现状：`tla.simd.membar(scope)` 仍只接受字符串（内部 `str(scope).upper()` 后交给 C++ 白名单校验），没有 `SCOPE` 常量类。
- 待办：前端补 `class SCOPE` 及取值校验（非法值报错）；补完后规格无需再改。

结论：待决。
