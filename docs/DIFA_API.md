# DiFA 通用推理接口

DiFA 分成三个层次。模型和求解器只出现在适配层，核心没有 SiT 类、类别数、CFG通道或时间步数。

| 层 | 接口 | 输入/职责 |
|---|---|---|
| 通用算法 | `DiFA.refine(...)` | 已对齐的分组预测/参考张量；输出修正guidance |
| 共识构造 | `SpatialReference` | BCHW预测的参考候选、原始历史、空间变换和分组 |
| 模型接入 | `SiTG03Guidance` | SiT坐标、CFG、结构响应、固定Heun时序 |

仅导入 `DiFA` 不会导入 SiT 适配模块或模型依赖。通用算法只依赖 PyTorch。

## 通用算法

```python
from difa import DiFA

algorithm = DiFA()
result = algorithm.refine(
    base_guidance=g0,
    contributions=components,
    branch_responses=branches,
    state_response=response,
)
updated_guidance = result.guidance
diagnostics = result.diagnostics
```

| 参数 | 形状 | 准确语义 |
|---|---|---|
| `base_guidance` | `[B,G,D]` | 原始DiFA参考产生的实际guidance增量，尚未经过结构近端修正 |
| `contributions` | `[B,K,G,D]`或`None` | 各加权参考对比 `D_current[w_i(z-a_i)]`；无历史时用None |
| `branch_responses` | `[B,R,G,D]` | 已转换到当前参考预测坐标的结构响应；R≥1；原G03没有再对它们应用D_current |
| `state_response` | `[B,G,D]`或`None` | `D_current[n]`，n为有效同时间状态响应；没有有效响应时用None |

`B`为batch，`K`为参考候选数，`R`为响应分支数，`G`为任意分组数，`D`为每组维度。
这些轴不固定为SiT的patch、通道或两条分支。张量须共享浮点dtype/device及对应坐标。
如果不使用二维空间参考构造器，可以从其他预测布局直接构造这些张量。

`base_guidance`与`contributions.sum(1)`在实数算术中可对应，但原实现可能有浮点分解差异。
请传入实际基准guidance，不要为了简写而强行替换成sum，否则不再保证继承基线的算术。
核心保留如下运算顺序：

```text
结构基准 + (逐贡献分支近端 − 解析总量分支近端)
         + (联合近端之和 − 逐贡献分支近端之和)
         → 继承的分组范数匹配
```

核心是无状态的，不提交历史、不调用模型、不产生随机数、不训练参数。
`state_response=None`保留M04贡献处理；并不代表把整个DiFA关闭。
零分支向量、零状态响应和零guidance均有数值处理；不要用不存在的质量/置信度标签代替响应。

## 模型无关的空间参考构造器

```python
from difa import DiFA, SpatialReference

algorithm = DiFA()
reference = SpatialReference(
    guide_scale=1.75,
    window_size=3,
    group_size=2,
)
```

该具体构造器处理 `[B,C,H,W]`，支持任意通道和可被group_size整除的高宽。
保留原DiFA的局部相似度、正交投影和空间滤波。它不假装知道其他布局的空间邻接关系。
默认时间映射是本实验使用的归一化线性路径；其他路径可提供
`log_snr=lambda t: ...`，并由调用方确保raw预测坐标正确。

下面函数不依赖任何网络或求解器，可放入自己的推理适配器：

```python
import torch

@torch.no_grad()
def refine_prediction(raw_prediction, time, branch_directions,
                      state_response, reference, algorithm, history):
    # raw_prediction: [B,C,H,W]，适配器已完成自身预测参数化的转换。
    # branch_directions: [B,R,C,H,W]，同一预测坐标中的结构响应。
    # state_response: [B,C,H,W]或None，不是误差估计。
    raw_record = reference.record(raw_prediction, time)
    original = reference.evaluate(raw_prediction, time, history)
    g0 = reference.to_groups(original - raw_prediction)
    components = reference.contributions(raw_prediction, time, history)
    branches = torch.stack([
        reference.to_groups(branch_directions[:, index])
        for index in range(branch_directions.shape[1])
    ], dim=1)
    response = reference.map_response(raw_prediction, time, state_response)
    result = algorithm(g0, components, branches, response)
    improved = reference.add_guidance(raw_prediction, result.guidance)
    return improved, raw_record
```

历史生命周期由调用方显式控制：

```python
history = reference.snapshot()  # 冻结一次读取
# 使用history完成需要共享参考的预测评估，收集record_1、record_2等raw记录。
# 到求解器/适配器定义的提交点：
reference.commit(record_1, record_2)
# 新轨迹/新条件前：
reference.reset()
```

`evaluate`不持久提交；`record`保存raw张量的独立副本。只提交原始模型观察，
不要提交改进后的预测。模型定义、噪声/clean/velocity参数化转换、CFG规则、
时间配对与提交时点均是适配器职责。

## SiT 现成适配

```python
from difa.adapters import SiTG03Guidance

guidance = SiTG03Guidance(model)
prediction = guidance(x, t, y, cfg_scale=1.5)
```

可直接作为原SiT `Sampler.sample_ode` 的model callable。
`from difa import G03Guidance`保留同一类的别名，供根目录两个生成脚本使用。

SiT适配负责：

1. 观察既有末层前向，计算attention/MLP响应，不改 `models.py`。
2. 保留SiT前三通道CFG，并转为 `z=x−t·velocity`。
3. 将分支响应转换到z坐标并分组；将raw状态响应通过D_current映射。
4. 在固定Heun interval内冻结历史，结束后提交两次raw记录。
5. 调用通用DiFA，再转换回velocity交给原求解器。

`SiTG03Guidance(model, guide_scale=1.75, window_size=3, chunk_size=None)`提供接口参数；
默认值对应实测G03。变更这些参数不自动继承已报告的FID。
通用API不保证任意输入构造或其他模型取得同样收益；目前完整质量测量来自SiT配置。
