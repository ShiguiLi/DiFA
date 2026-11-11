# SiT + DiFA：推理阶段无需训练的自我改善

这是**完整 SiT 项目与 DiFA 推理算法的集成目录**。模型、训练、transport、权重下载、
原采样入口均来自 [SiT 官方仓库](https://github.com/willisma/SiT)。
DiFA 在推理时利用模型自身的预测历史与结构响应修正预测，再交给原求解器。

本版本使用 **G03 Joint Metric Reference**。它在当前固定协议实验中取得 FID **2.0406080**。
无需训练、微调或额外模型，保持原 Heun2 / 20 个时间节点 / 38 次完整条件-null配对调用。
尾部导数与参考算术仍有额外计算成本。

原 SiT 文档：[README_SiT.md](README_SiT.md)。
代码基于官方 commit `cbde832a40b153ccc79603412409da9c9b0c568c`。

## 1. 安装

下载本仓库或解压后，在**包含 models.py 的项目根目录**执行。
推理推荐 NVIDIA CUDA GPU。以下环境与原实验的 Python/PyTorch 版本对齐：

```bash
conda create -n sit-difa python=3.12 -y
conda activate sit-difa

python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-difa.txt
```

以上多行命令使用 Linux / Bash 语法。CUDA wheel 需与机器驱动相容。
已有可用 SiT 环境可直接安装 `requirements-difa.txt`；
不需要安装独立的 DiFA 包，也不需要另外克隆 SiT。

原 `environment.yml` 保留，完整训练环境与原 SiT 使用方式见 `README_SiT.md`。
**使用 DiFA 生成图片不需要训练，也不需要 ImageNet 训练集或 wandb。**

## 2. 直接生成图片

首次可自动获取官方 SiT-XL/2-256 权重和 EMA VAE：

```bash
python sample_difa.py ODE --seed 1 --output sample_difa.png
```

也可以使用已经准备好的本地资产：

```bash
python sample_difa.py ODE \
  --ckpt /path/to/SiT-XL-2-256x256.pt \
  --vae /path/to/sd-vae-ft-ema \
  --seed 1 \
  --labels 207 360 387 974 \
  --output sample_difa.png
```

输出是生成图片拼图。`--labels` 为 ImageNet 类别索引0–999，属于类别条件生成。
显存较小时减少一次提供的类别数。SiT权重可通过根目录 `download.py` 所使用的官方链接获取；
VAE默认 `stabilityai/sd-vae-ft-ema`，也接受本地目录。

### 提前下载权重与离线运行

在联网机器的项目根目录、激活上述环境后执行。这里只下载并保存模型，不生成图片：

```bash
python -c "from download import find_model; find_model('SiT-XL-2-256x256.pt')"
python -c "from diffusers.models import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').save_pretrained('pretrained_models/sd-vae-ft-ema')"
```

准备完成后的本地文件布局为：

```text
pretrained_models/
├── SiT-XL-2-256x256.pt
└── sd-vae-ft-ema/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

VAE 权重文件后缀可能随 diffusers 版本不同，搬运时复制整个 VAE 目录。
SiT 权重保存在当前项目的 `pretrained_models/`；直接使用 Hugging Face 模型 ID 时，
VAE 还会使用 Hugging Face 缓存（默认 `~/.cache/huggingface/hub`，可由 `HF_HOME` 配置）。
上述 `save_pretrained` 会额外保存一份可独立加载的本地 VAE 目录。

离线机器提前安装好依赖，并复制项目与上述 `pretrained_models/`，然后执行：

```bash
HF_HUB_OFFLINE=1 python sample_difa.py ODE \
  --ckpt pretrained_models/SiT-XL-2-256x256.pt \
  --vae pretrained_models/sd-vae-ft-ema \
  --seed 1 --labels 207 360 387 974 \
  --output samples/sample_difa.png
```

入口通过 `find_model` 加载 SiT state dict，再调用 `model.load_state_dict`；
VAE 通过 `AutoencoderKL.from_pretrained` 加载本地配置与权重。无需训练或权重转换。
模型文件与生成结果已在 `.gitignore` 中排除，不应提交到源码仓库。

推理入口固定使用实测配置：SiT-XL/2、256、CFG1.5、DiFA1.75、window3、
原 `transport.Sampler` 的 Heun2、20节点。FP32模型和TF32 matmul，不改变权重。

## 3. 多卡批量生成

使用 SiT 熟悉的 `torchrun` 方式。下例与已报告50k实验的样本数、seed和批量一致：

```bash
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp_difa.py ODE \
  --ckpt /path/to/SiT-XL-2-256x256.pt \
  --vae /path/to/sd-vae-ft-ema \
  --global-seed 1 \
  --per-proc-batch-size 3125 \
  --num-fid-samples 50000 \
  --sample-dir samples/difa_g03
```

生成PNG及相邻NPZ，供标准图像评估工具使用；此入口只生成数据，不包含评价调度器。
`3125` 是原双卡实验的每卡batch，显存要求高。日常生成可降低batch或样本数，
但改变GPU数/batch会改变随机数分配，不能将少量示例输出当作同一50k指标复现。
固定 global seed1、双卡时 rank seeds 为2、3，与原实验一致。

若只想使用原 SiT，原入口仍完整保留：

```bash
python sample.py ODE --image-size 256 --seed 1
```

原 SiT 的默认求解器设置与 DiFA 的上述固定实验设置不同，不应直接将两个默认命令当作同预算质量对照。

## 4. DiFA 如何接入 SiT

`models.py`、`train.py`、`transport/` 都保持官方原样。DiFA 的核心接法只有：

```python
import torch
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from difa.adapters import SiTG03Guidance

model = SiT_models["SiT-XL/2"](
    input_size=32, num_classes=1000, learn_sigma=True
).cuda().eval()
model.load_state_dict(find_model("SiT-XL-2-256x256.pt"))
torch.backends.cuda.matmul.allow_tf32 = True

sampler = Sampler(create_transport("Linear", "velocity", None, None, None))
sample_fn = sampler.sample_ode(
    sampling_method="heun2", num_steps=20,
    atol=1e-6, rtol=1e-3, reverse=False,
)

with torch.no_grad():
    noise = torch.randn(2, 4, 32, 32, device="cuda")
    labels = torch.tensor([207, 360], dtype=torch.long, device="cuda")
    x = torch.cat([noise, noise], dim=0)
    y = torch.cat([labels, torch.full_like(labels, 1000)], dim=0)

    guidance = SiTG03Guidance(model)
    trajectory = sample_fn(x, guidance, y=y, cfg_scale=1.5)
    latents = trajectory[-1].chunk(2, dim=0)[0]
```

传给求解器的 callable 从 `model.forward_with_cfg` 换成 `SiTG03Guidance(model)`。
DiFA内部利用临时推理钩子读取已有前向中的末层张量，既不修改模型文件，也不增加模型参数。

- 每个新采样批次创建新wrapper，或调用 `guidance.reset()`。
- 保留SiT前三latent通道的CFG约定和条件/null双半批。
- 当前接口使用固定正向Heun的调用顺序；从t=0开始，不接自适应拒步/SDE/反向采样。
- 使用eager、eval、no_grad推理。同一模型实例同时处理一条采样请求；并发服务应使用独立worker。

## 5. 通用 DiFA 接口

算法核心与模型接入分离：

```python
from difa import DiFA, SpatialReference

algorithm = DiFA()  # 不接收模型，不包含SiT或求解器逻辑
result = algorithm.refine(
    base_guidance=g0,             # [B,G,D]
    contributions=components,     # [B,K,G,D]
    branch_responses=branches,     # [B,R,G,D]，分支数R不固定
    state_response=response,      # [B,G,D]，没有有效响应时用None
)
updated_guidance = result.guidance
```

`G/D`为通用分组和特征维度，核心不写死图片分辨率、通道数、类别或模型层。
`SpatialReference`另行提供BCHW预测的共识构造、分组和显式raw历史管理。
`SiTG03Guidance`是现成的SiT接入层，负责模型前向、CFG、坐标变换和Heun提交时点。

**完整参数语义与自有模型接入示例见 [通用接口文档](docs/DIFA_API.md)。**
不同来源的张量必须先对齐预测坐标；不能仅凭形状相同替换响应含义。

## 6. 原理与通用性

模型当前raw表示为 `z = x − t·v`。沿SiT线性路径，它对应噪声端估计；
DiFA从三个历史raw观察生成参考候选及权重。两次Heun interval内评估读取同一历史快照，
结束后提交两次原始观察，避免把已修正的预测当成新的原始证据。

将各个加权对比映射到实际guidance坐标，得到 `g_i = D[w_i(z − a_i)]`。
G03联合使用末层attention/MLP响应方向 `u`，以及已有同时间预测对的映射响应 `m`，
求每个参考贡献：

$$
h_i^*=\arg\min_h\frac12\|h-g_i\|^2
+\frac12(h^\top u)_+^2
+\frac{(h^\top m)^2}{2(\|\sum_i g_i\|^2+\|m\|^2)}.
$$

零分母按零响应处理。这是解析rank-two近端计算，无训练、无迭代优化器。
随后沿M04的继承路径应用增量和每组guidance范数匹配，再转换回velocity。
同时间响应是状态敏感性，不是预测误差或置信度；局部目标下降不保证生成指标提升。

**通用的部分是推理时自参考构造思想与张量参考算子；当前完整接入和实测范围是SiT。**
其他模型需要提供自己的预测坐标、结构响应及历史时序接口，不能仅换模型名称便声称效果已验证。
`from difa import joint_reference` 可单独使用参考算子，输入为
`values[B,K,T,D]` 和 `total/response/u[B,T,D]`。

## 7. 已报告性能

SiT-XL/2、ImageNet256，N=50000，global seed1，2 GPU×3125，CFG1.5，DiFA1.75，
window3，原Heun2/20节点/38次完整配对调用。数值来自提供的开发机完整报告。

| 方法 | FID ↓ | IS ↑ | sFID ↓ | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| **DiFA G03** | **2.0406080** | 262.16718 | 4.443002 | 0.81094 | 0.60140 |
| C04 | 2.0409008 | 262.17563 | 4.442841 | 0.81050 | 0.60020 |
| M04 | 2.0424686 | 262.05154 | 4.441535 | 0.81156 | 0.60180 |
| 原DiFA | 2.0576795 | 261.57193 | 4.559277 | 0.80968 | 0.60310 |

G03为这批结果中的最低FID，但相对C04的增量仅约0.000293，且不是所有指标最优。
这不构成跨模型通用最优结论。目录重整保留原算法；推理钩子接入已在CPU模型夹具上
与原实测实现逐位比较通过，真实GPU上的此目录重整版尚未重新进行50k评价。

## 8. 完整目录

```text
SiT-DiFA/
├── models.py                 # 官方SiT模型，原样
├── transport/                # 官方路径、求解器与Sampler，原样
│   ├── __init__.py
│   ├── transport.py
│   ├── integrators.py
│   ├── path.py
│   └── utils.py
├── download.py               # 官方权重下载/加载
├── train.py                  # 官方训练入口，原样
├── train_utils.py
├── wandb_utils.py
├── sample.py                 # 官方单卡采样，原样
├── sample_ddp.py              # 官方多卡采样，原样
├── sample_difa.py             # 新增：DiFA单卡生成
├── sample_ddp_difa.py         # 新增：DiFA多卡生成
├── difa/                     # 模型无关的算法＋独立模型适配
│   ├── __init__.py
│   ├── core.py               # DiFA.refine：通用分组张量接口
│   ├── ops.py                # 联合近端、范数匹配等纯张量计算
│   ├── reference.py          # SpatialReference：显式raw历史与空间参考
│   ├── consensus.py          # DiFA候选与权重
│   ├── geometry.py           # BCHW空间变换/分组
│   └── adapters/
│       ├── __init__.py
│       ├── sit.py            # SiT调用和固定Heun生命周期
│       ├── observer.py       # 临时推理观察接口
│       ├── sit_response.py   # SiT末层响应
│       └── pairs.py          # 已有同时间预测配对
├── docs/DIFA_API.md           # 通用接口与模型接入契约
├── requirements-difa.txt
├── environment.yml           # 官方完整环境
├── run_SiT.ipynb              # 官方notebook
├── visuals/                  # 官方说明图片
├── README.md
├── README_SiT.md              # 官方原README
├── DIFA_CHANGES.md
├── .gitignore                # 排除权重、样本和本地缓存
└── LICENSE.txt               # 保留上游许可
```

权重、VAE和生成样本按需获取，不随源码目录打包。原SiT引用与许可见
[README_SiT.md](README_SiT.md) / [LICENSE.txt](LICENSE.txt)，新增推理代码范围见
[DIFA_CHANGES.md](DIFA_CHANGES.md)。
