# M3 策略层精确设计与实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不越过 M2 门禁、不引入 privileged 信息的前提下，实现可复现的
BC-1 与 action-chunk 策略训练、checkpoint 和闭环执行链。

**Architecture:** BC-1 是本仓库自有的最小双视角基线；两路 RGB 共享一个
ResNet-18，8D state 经小型 MLP 后与视觉特征融合并预测下一步 8D action。chunked
policy 首选固定版本 LeRobot `ACTPolicy` 的薄适配层，但只有隔离环境兼容性 smoke
通过才正式采用；只有完整、白名单化的 deterministic incompatibility evidence 产生
`failed` receipt 时，才切换到本文精确定义、诚实标记为 `act_like_minimal_v1` 的本地
fallback。资源/网络/中断/缺证据只会 blocked，不能触发 fallback。

**Tech Stack:** Python 3.12、PyTorch/torchvision、HDF5/h5py、MuJoCo 3.11、
LeRobot 0.6.1（首选且隔离）、safetensors、uv 锁定环境。

**Spec:** [`README.md`](../README.md) 是唯一项目规格；本设计消费当前
[`src/env/pick_place.py`](../src/env/pick_place.py)、
[`src/expert/scripted.py`](../src/expert/scripted.py) 和
[`src/data/hdf5.py`](../src/data/hdf5.py) 的已实现契约。

## Global Constraints

- M2 未达到 `200 train successes + 40 validation successes + 0 schema errors +
  deterministic replay >= 18/20`，且 collection ledger、manifest、validation、data
  report、replay plan/trials/summary 的逐项 hash/身份对账未通过时，任何真实数据
  optimizer step（包括 8-episode overfit）都必须 fail closed；更不能开始正式训练。
- 策略只能读取 `observation.images.front`、`observation.images.wrist`、
  `observation.state`。`stage`、timestamp、seed、success、failure metadata 以及任何
  `privileged/*` 均不得进入模型 batch。
- split 的最小单位是 episode seed；同一 episode 的帧不得分到不同 split。
- normalization 统计只从 frozen train episode 估计；validation 和在线评测只能复用
  train 统计。
- 所有闭环动作都通过 `PickPlace.step()`；不得直接调用 `mujoco.mj_step()` 判分。
- M3 本轮只定义实现和验收契约，不声称任何 loss、闭环成功率或 IID 指标已经产生。
- M2/M3 运行 artifacts 统一写入已由仓库根 `.gitignore` 忽略的 `$EMB/runs/m2/...` 与
  `$EMB/runs/m3/...`，每个 run dir no-clobber；禁止把 checkpoint、receipt、normalization
  或 trial records 写到任何未完整忽略的其他 output tree 后污染 status/provenance 观测。
- 禁止对 LeRobot 使用 `--no-deps` 后宣称正式兼容；官方 ACT adoption 必须来自完整
  resolver、锁文件和 smoke receipt。
- 实施者不得在未获得单独授权时 commit、push、上传 checkpoint 或发布数据。

---

## 1. 当前上游契约与 M3 边界

### 1.1 当前 HDF5 v1

一个文件对应一个完整 episode。`T = num_steps`，时间对齐为 pre-action，控制频率为
20 Hz（`control_dt_s=0.05`）：

```text
observation[t] -> expert action[t] -> append row t -> PickPlace.step(action[t])
```

| HDF5 键 | 磁盘 dtype / shape | M3 用途 |
|---|---|---|
| `observation.images.front` | `uint8 [T,128,128,3]` RGB/HWC | allowlisted input |
| `observation.images.wrist` | `uint8 [T,128,128,3]` RGB/HWC | allowlisted input |
| `observation.state` | `float32 [T,8]` | HDF5 raw input；经唯一 adapter 后仍为 `float32` |
| `action` | `float32 [T,8]` | supervised target |
| `timestamp` | `float64 [T]` | audit/alignment only |
| `stage` | UTF-8 `[T]` | audit only；禁止输入策略 |

根属性至少包含 `schema_version=1`、episode `seed`、`success`、`num_steps`、
`time_alignment=pre_action`、动作上下界和下面的固定动作语义：

```text
absolute_joint_position_targets_rad[7]+normalized_gripper_open[1]
```

M3 首先调用 `validate_episode()` 和 `HDF5EpisodeReader.metadata` 验证契约。为避免每个
sample 构造 Python `Transition` 造成额外 I/O，训练 Dataset 可以在验证后用只读 h5py
slice 取连续 action chunk；它不得绕开同一组 schema 常量和元数据检查。

### 1.2 策略输入 allowlist 与唯一 dtype 适配边界

唯一 allowlist 是：

```python
POLICY_OBSERVATION_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
    "observation.state",
)
```

训练和 rollout 都必须调用同一个 `preprocess_policy_observation(raw, source_kind)`，
先构造一个新 mapping，只复制这三个键。边界验证采用 exact-set 语义：缺键、额外的
`observation.*` 键、错误 shape、非有限数或不符合已声明 raw-source contract 的 dtype
都直接报错；绝不以“模型没用到”为由把额外字段留在 batch 中。监督 batch 可以再包含
`action` 和 `action_is_pad`，但它们不是观测。

这里冻结两种真实 raw-source contract，而不是假设磁盘与环境 dtype 相同：

| `source_kind` | front / wrist raw | state raw | 验证后的唯一转换 |
|---|---|---|---|
| `hdf5_v1` | exact `uint8 [128,128,3]` | exact `float32 [8]` | state 显式转/保持 `float32` |
| `pick_place_v1` | exact `uint8 [128,128,3]` | exact `float64 [8]` | **先检查 shape 与 finite，再显式 cast 到 `float32`** |

图像在同一函数中从 HWC 转 CHW、显式转 `float32` 并除以 255。state 则先按上表验证
raw dtype/shape/finite，再执行 `np.asarray(state).astype(np.float32, copy=False)`；禁止在
`PickPlace.observe()`、Dataset、collate、模型 `forward` 或 rollout runner 中出现第二处
隐式/显式 state cast。两种 source 从这里开始进入完全相同的 train-only normalization、
batching 和模型输入链，model-input contract 固定为 finite `float32 [B,8]`。测试必须把
真实 `PickPlace.observe()` 的 `float64 [8]` 送过该边界，并证明输出逐元素等于该 raw
值的明确 float32 cast；另用同数值 HDF5 row 证明训练与 rollout 产生一致的 model input。

checkpoint 分开保存 `raw_source_contracts={hdf5_v1,pick_place_v1}` 和
`model_input_contract`，不能用一个含混的 `observation.state=float32` 同时描述 raw env
与模型张量。

模型 `forward`/`predict` 不接收 `PickPlace`、`EpisodeMetadata`、stage、seed 或任意
原始 HDF5 handle。在线策略唯一入口是 `env.observe()` 的返回值。

### 1.3 split 与锁定评测：`m3-m4-seed-protocol.v2`

M3 不重新生成 M2 split，也不接受帧级 `random_split`。本文把
`docs/DATA_SCHEMA.md` 中原先未分配的 `>=10_050` namespace 以版本化协议
`m3-m4-seed-protocol.v2` 明确升级；M2 HDF5 schema 仍为 v1，M2 collection 仍只能使用
train/validation 两个 candidate namespace。四个集合冻结为：

| 用途 | 精确 seed | 是否写入 M2 | 可见时机 |
|---|---:|---|---|
| M2 train collection | `0..999` | 可以；只收 eligible success 训练 | collection |
| M2 validation collection | `1000..9999` | 可以；只用于选择/诊断 | collection |
| M4 locked test | `10000..10049`（50 个） | **禁止** | M4 全部决定冻结后 |
| M3 locked IID acceptance | `10050..10149`（100 个） | **禁止** | 正式训练、超参和 checkpoint 全冻结后一次 |

`seed-protocol.json` 必须在第一次正式训练前按第 9.2 节 canonical JSON 算法发布，固定
`schema_version`、上述完整显式列表、用途、环境随机化范围版本和四集合不交规则；M3
acceptance plan 与 M4 plan 都引用其 `seed_protocol_id`。不得用 M2 collection/validation
seed 补足 100 IID，不得把 M3 IID seed 用于 M4，也不得把 M4 locked seed 用于 overfit、
checkpoint 选择、阈值调参或 M3 acceptance。

M3 消费 M2 manifest 中冻结的 episode seed 列表并验证：

```text
len(train successful episodes)      == 200
len(validation successful episodes) == 40
all train attempted/accepted seeds are in 0..999
all validation attempted/accepted seeds are in 1000..9999
train_seeds ∩ validation_seeds      == ∅
train/validation ∩ M4 locked seeds    == ∅
train/validation ∩ M3 IID seeds       == ∅
M4 locked seeds ∩ M3 IID seeds        == ∅
all accepted files have unique seed and SHA-256
all accepted files are schema v1, complete, successful and pre-action aligned
```

M3 100-IID plan 必须逐项列出 `10050..10149`，不能只保存首尾范围；M4 继续消费当前
`src/evaluation/protocol.py` 已锁定的 `10000..10049`。实现不得另造隐式范围，所有 runner
只消费带 hash 的显式 plan，并在启动前对 M2 manifest、M3 plan、M4 plan 四集合重算。

validation 只用于 checkpoint 选择和诊断。100-seed IID acceptance 只在网络、
normalization、checkpoint 和超参数全部冻结后运行；其结果不得反向用于调参。
若 `>=60/100` 未通过，M3 就是 FAIL；不得根据这 100 个结果改配置后复用同一 seed 重测。
未来若要再开发，必须先发布新的、不重叠 test namespace/protocol version，原 v2 结果仍按
test exposure 保留。

## 2. 训练 sample 与 streaming Dataset

### 2.1 Dataset 索引

`HDF5PolicyDataset(manifest, split, h_pred)` 在初始化时只做以下轻量工作：

1. 读取 manifest 和每个 episode metadata；
2. 验证 seed/split/hash/schema/action bounds；
3. 建立 `(relative_path, local_t)` 的全局样本索引；
4. 不读取、拼接或缓存整套图像。

每个 DataLoader worker 在第一次 `__getitem__` 时独立打开只读 HDF5；父进程不得在
fork 前持有 h5py handle。worker 使用最多 8 个文件的 LRU handle cache，并在 worker
退出时关闭。正式文件已经原子 finalize，不与 writer 并发，因此不需要读取 partial
文件；manifest 中出现 partial 路径必须报错。

### 2.2 BC-1 sample

对 episode 内时刻 `t`：

```text
front = front[t]   # uint8 [128,128,3]
wrist = wrist[t]   # uint8 [128,128,3]
state = state[t]   # float32 [8]
target = action[t] # float32 [8]
```

这里的 HDF5 row 仍必须经过第 1.2 节唯一 adapter；Dataset 不自行 normalize/cast，
collate 也不得再实现另一套 state 转换。这样训练的 `float32 [B,8]` 与在线 env 的
`float64 [8] -> validate -> explicit float32` 使用同一后半段预处理。

collate/preprocess 后的 batch：

| 张量 | dtype | shape |
|---|---|---|
| front | `float32`，AMP forward 可 cast bf16 | `[B,3,128,128]` |
| wrist | 同上 | `[B,3,128,128]` |
| state | `float32` | `[B,8]` |
| action target | `float32` | `[B,8]` |

### 2.3 Chunked-policy sample 与 episode 末尾 padding

固定 `H_pred=16`。对时刻 `t`，有效 target 为同一 episode 内：

```text
action[t : min(t + 16, T)]
```

不得越过 episode 边界。剩余位置在 normalized action 空间填 0，同时
`action_is_pad=True`；有效位置为 `False`。padding 值本身不参与 loss。

| 张量 | dtype | shape |
|---|---|---|
| front | `float32` / bf16 forward | `[B,3,128,128]` |
| wrist | `float32` / bf16 forward | `[B,3,128,128]` |
| state | `float32` | `[B,8]` |
| action chunk | `float32` | `[B,16,8]` |
| `action_is_pad` | `bool` | `[B,16]` |

训练 sample 是 transition-uniform：长 episode 按其有效 transition 数量自然提供更多
sample。不做 stage 重采样、未来观测输入、数据增强或跨 episode 拼接；这些都不是
M3 最小对照的一部分。

### 2.4 DataLoader 默认值

| 模式 | batch | workers | shuffle | 其他 |
|---|---:|---:|---|---|
| BC-1 CUDA | 16 | 2 | train only | `pin_memory=True`, `persistent_workers=True` |
| selected chunk backend CUDA | 4 | 2 | train only | 同上，bf16 AMP |
| 任一 CPU | 1 | 0 | train only | `pin_memory=False`, AMP off |
| val/gate-loss | 对应设备 batch | 0 | false | 固定全量顺序 |

OOM 时只按预注册的 `batch 4 -> 2 -> 1`（BC 对应 `16 -> 8 -> 4 -> 2 -> 1`）下降，并用
gradient accumulation 保持记录的 effective batch；每一档是独立 run/receipt，不得在
同一 run 中悄悄改变网络、`H_pred` 或图像尺寸。official ACT adoption 的相同阶梯和
“何时才算兼容”由第 6.4 节唯一规定，batch 4 单独 OOM 不等于选择 fallback。

## 3. Train-only normalization

### 3.1 统计算法

单独的 `compute-normalization` 阶段按 manifest 的 train 文件顺序 streaming 扫描，
用 float64 Welford accumulator 计算：

- front/wrist 各自 RGB channel 的 mean/std/count；输入先除以 255；
- 8D state 的逐维 mean/std/min/max/count；
- 8D action 的逐维 mean/std/min/max/count。

任何 validation/evaluation 文件被打开都视为泄漏并失败。统计输入的 episode seed 列表
和 manifest SHA-256 写入 `normalization.json`；其 `normalization_id` 严格使用第 9.2 节
“排除唯一 identity 字段后再 canonicalize/hash”的算法，避免递归自哈希定义。

两种策略使用同一份 train-only stats：

```text
x_norm = (x - mean) / max(std, 1e-6)
action = action_norm * scale + mean
```

`std < 1e-6` 的维度在 `constant_mask` 中标记并使用 `scale=1`；加载时必须完全复用
保存的 scale，不重新估计。image、state、action 的统计彼此独立。state 统计只看第
1.2 节 adapter 已产生的 float32 model-input 值，因此和 rollout 的显式 float64-to-float32
路径一致，不允许一边用 raw float64 统计、一边用 float32 推理。

### 3.2 与预训练 ResNet 的关系

两种模型都使用 `ResNet18_Weights.IMAGENET1K_V1` 初始化，但输入 normalization 使用
本项目 train-only RGB stats，而不是从 validation/test 估计或临时改回另一套常数。
checkpoint 记录 weight enum、torchvision 版本、缓存权重文件 SHA-256 和 normalization
SHA-256。若权重未缓存，下载到 `$EMB/cache/torch`；无法取得或 hash 不一致时 preflight
失败，不静默改为随机初始化。

## 4. BC-1：最小双视角共享编码器

### 4.1 精确网络

只实例化一个 torchvision ResNet-18，两路图像按顺序分别通过同一个对象，权重完全
共享。用 `FrozenBatchNorm2d` 避免小 batch 的运行统计漂移，移除分类 fc，保留 global
average pool：

```text
front [B,3,128,128] -> shared ResNet18 -> front_feat [B,512]
wrist [B,3,128,128] -> shared ResNet18 -> wrist_feat [B,512]

state [B,8]
  -> Linear(8,64) -> LayerNorm(64) -> GELU
  -> Linear(64,64) -> GELU
  -> state_feat [B,64]

concat(front_feat, wrist_feat, state_feat) -> [B,1088]
  -> Linear(1088,512) -> LayerNorm(512) -> GELU
  -> Linear(512,256) -> GELU
  -> Linear(256,8)
  -> normalized next action [B,8]
```

M3 基线不加 dropout、attention、历史帧、文本、stage embedding 或独立相机 encoder。
测试必须用 module identity/parameter id 证明两路调用共享同一 backbone，而不是两个
结构相同但参数独立的 ResNet。

### 4.2 loss 和优化器

- loss：8 个 normalized action 维度上的 elementwise MSE 后取全局均值；
- AdamW：backbone `lr=1e-5`，state/fusion/head `lr=1e-4`，
  `weight_decay=1e-4`；
- gradient clip：global norm `1.0`；
- scheduler：无；任何后续 scheduler 都是新实验配置，不能覆盖此 baseline；
- CUDA：bf16 autocast；loss accumulation 和 optimizer state 保持 float32；
- CPU：float32。

输出反归一化后逐维 clip 到 checkpoint 保存的 action bounds，再传给环境。每次 clip
的维数和幅度都写日志；runtime env bounds 与 checkpoint bounds 不一致时直接拒绝加载，
不以 clip 掩盖接口错配。

## 5. Chunked policy：官方 ACT 优先、smoke-gated fallback

### 5.1 决策树

**首选：固定 LeRobot 0.6.1 的官方 `ACTPolicy`，写薄适配层。正式采用前必须先通过
第 6 节的完整依赖兼容性 smoke；smoke 未通过时，不得以 `--no-deps`、手工复制若干
上游文件或忽略 CUDA 错误来宣称 LeRobot 兼容。**

理由：

1. README 明确要求优先官方 LeRobot ACT；官方实现已经覆盖 CVAE、padding mask、
   transformer、共享多相机 ResNet-18 和 action chunk。
2. 本项目真正需要自有控制的是 HDF5 streaming、禁止泄漏、train-only stats、
   `H_pred/K_exec` 分离和 MuJoCo 闭环；这些适合放在薄 adapter 中。
3. 但当前本机 PyTorch/CUDA 与 LeRobot stable 的依赖上限有真实冲突，是否能在
   sm_120 GPU 上运行必须由锁定环境的 forward/backward/save-load 证据决定，而不是
   由 import 成功推断。

receipt 只能落到以下三个互斥状态；`blocked` 不构成 backend 决策：

```text
status == "passed" and passed == true
  -> chunk_policy_backend = "lerobot_act_0_6_1"

status == "failed" and passed == false
  -> 保存失败 stage/error/package/device 信息
  -> chunk_policy_backend = "act_like_minimal_v1"
  -> 报告中禁止简称为 ACT

status == "blocked" and passed == null
  -> 只证明空间/ignore 等前置尚未满足
  -> chunk_policy_backend 不得取值，禁止 overfit/training/fallback
```

参考上游契约：

- [LeRobot ACTConfig v0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/configuration_act.py)
- [LeRobot ACTPolicy v0.6.1](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/act/modeling_act.py)
- [LeRobot 0.6.1 dependency contract](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml)

### 5.2 官方 backend 的固定 ACTConfig

```text
input_features:
  observation.images.front -> VISUAL (3,128,128)
  observation.images.wrist -> VISUAL (3,128,128)
  observation.state        -> STATE  (8,)
output_features:
  action                   -> ACTION (8,)

n_obs_steps=1
chunk_size=16                 # H_pred
n_action_steps=1              # 仅防止上游内部 queue 偷渡 K_exec
vision_backbone=resnet18
pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
dim_model=512
n_heads=8
dim_feedforward=3200
n_encoder_layers=4
n_decoder_layers=1
use_vae=true
latent_dim=32
n_vae_encoder_layers=4
dropout=0.1
kl_weight=10.0
temporal_ensemble_coeff=null
optimizer_lr=1e-5
optimizer_lr_backbone=1e-5
optimizer_weight_decay=1e-4
normalization_mapping=IDENTITY for VISUAL/STATE/ACTION
```

`IDENTITY` 是故意的：本地 adapter 已使用带 hash 的 train-only stats，禁止 LeRobot
processor 再做一遍 normalization。adapter 直接调用 `ACTPolicy.forward()` 训练，调用
`predict_action_chunk()` 推理；不得调用会维护隐藏 action queue 的 `select_action()`。

训练 loss 使用上游 masked L1 reconstruction 加 `10 * KL`；日志分别记录 `loss`、
`l1_loss` 和 `kld_loss`，padding 必须被 mask。

### 5.3 兼容的最小 fallback

只有 `status="failed",passed=false` 的 `lerobot_smoke_receipt.json` 存在时才能启用
fallback；`blocked` receipt 必须拒绝。它不是 ACT 的等价复刻，不包含 CVAE、KL loss
或 temporal ensemble；artifact、CLI、图表和简历都必须用
完整名称 **`ACT-like minimal chunk transformer`** 或 id `act_like_minimal_v1`。

精确网络如下，禁止在 fallback 时临时扩大：

```text
front [B,3,128,128] -> shared ResNet18 + FrozenBN + GAP -> [B,512]
wrist [B,3,128,128] -> same exact backbone             -> [B,512]
state [B,8] -> Linear(8,64) -> LayerNorm -> GELU
             -> Linear(64,64) -> GELU                   -> [B,64]

front token: Linear(512,256) -> [B,1,256]
wrist token: Linear(512,256) -> [B,1,256]
state token: Linear(64,256)   -> [B,1,256]
memory = concat -> [B,3,256]

16 learned query embeddings -> [B,16,256]
TransformerDecoder(
  num_layers=2, d_model=256, nhead=4,
  dim_feedforward=1024, dropout=0.1,
  activation="gelu", norm_first=True, batch_first=True
)
Linear(256,8) -> normalized action chunk [B,16,8]
```

loss 是 `action_is_pad` mask 后的 normalized L1；AdamW 对 backbone 用 `1e-5`、其他
参数用 `1e-4`、`weight_decay=1e-4`，gradient clip 1.0，无 scheduler。数据、stats、
H_pred、K_exec、checkpoint 和闭环接口与官方 backend 完全一致，使 action-chunk 对照
仍可执行；但模型名称和结论不得越级成“官方 ACT”。fallback 只运行在第 6.1–6.2 节
`train` group sync 且 project receipt passed 后的 locked `.venv`，不依赖 LeRobot。当前
`.venv` 已精确同步 torch 2.13.0+cu130、torchvision 0.28.0+cu130 与 safetensors 0.8.0，
development CUDA smoke 已通过；但候选 source 尚未提交到固定 HEAD，所以 formal project
receipt 仍是 BLOCKED，BC/fallback 训练不得提前启动。任何其他解释器中的 PyTorch/CUDA
版本只属诊断事实，明确禁止成为正式训练栈。

### 5.4 H_pred 与 K_exec 的唯一语义

```text
H_pred = checkpoint/model architecture 的预测长度，固定为 16
K_exec = rollout runtime 参数，baseline 为 4，允许 1 <= K_exec <= H_pred
```

每次 observation 只推理一次得到 `[1,16,8]`，反归一化后顺序执行前 `K_exec` 个动作；
成功或控制步预算耗尽时提前停止，否则执行完才重新调用 `env.observe()`。余下
`16-K_exec` 个预测丢弃。

因此 `H_pred=16,K_exec=4` 的含义是“看一次当前观测，预测未来 16 步，只执行前 4 步，
然后用新观测重算”，绝不是预测长度 4。改变 K_exec 不改变 checkpoint、权重 hash、
normalization 或 H_pred；M4 才在同一个已选 backend 的同一 checkpoint 上比较
`{1,4,8,16}`。若 backend 是 fallback，M4 也必须继续标记为 ACT-like，不能改名。

## 6. LeRobot 依赖隔离与 8 GB/CPU 可复现性

### 6.1 项目环境与 LeRobot 环境的职责边界

项目唯一 uv 环境是 `$EMB/.venv`。2026-08-24 已在非默认 `train` dependency group 中
精确锁定 `torch==2.13.0`、`torchvision==0.28.0`、`safetensors==0.8.0`，并由同一个
`uv.lock` 固定 CUDA 13 wheel source/index、完整 URL 与 hashes。受控同步后，development
smoke 在 RTX 5070 Laptop / sm_120 上完成 bf16 forward/backward、双视角 ResNet18 forward
与 safetensors 跨进程加载；`policy` 也从 locked editable checkout 导入。第一次同步因系统盘
瞬时跌破 3 GiB 被终止并留下 `resource_drift` blocked receipt，第二次 superseding 同步才通过。

候选 source/lock 进入提交 `0063226` 后，formal verifier 已在无 `PYTHONPATH`/`VIRTUAL_ENV`
的新进程中通过全部 source、lock、editable package、空间和真实 CUDA smoke 检查。正式
receipt id 为 `sha256:e31baea7dc02a965dcf165c534fac275110af1bdd63337d746efcd7deb5cc373`；
这只证明项目训练环境准入，不证明 BC、fallback、training rollout、checkpoint 或闭环指标
已经完成。常规 CI 的 test/video 命令不安装 train group；训练主机只允许用第 10.1 节冻结
的绝对 uv 命令执行
`uv sync --locked --group test --group video --group train --project "$EMB"
--python "$EMB/.venv/bin/python"`。

实现 `src/policy/` 时必须同步扩展根项目 package discovery；冻结目标为：

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["data*", "env*", "evaluation*", "expert*", "policy*", "robotics*"]
```

上述 `uv sync` 保持 uv 默认的 editable root install（禁止 `--no-install-project` 与
`--no-editable`）。post-sync gate 必须在未设置 `PYTHONPATH` 的新进程中成功
`import policy`，断言 `policy.__file__` 位于 `$EMB/src/policy/`，并从安装 metadata/
`direct_url.json` 证明 distribution `panda-reactive-il` 的 editable root 为 `$EMB`；否则
project receipt blocked。这样所有 `python -m policy...` 都来自 locked editable install，
不是 shell-local source-path 注入。

sync 前后的 `pyproject.toml`、`uv.lock` 必须 tracked-clean 且内容 hash 不变；完成后还要
以绝对解释器校验 prefix、精确包来源、`uv sync --check`，并在 sm_120 上完成 bf16
forward/backward 与 safetensors save/new-process load smoke。
在这些检查形成 atomic `project-train-env-receipt.v1` passed receipt 之前，训练 CLI
fail closed。当前 fixed-HEAD formal environment receipt 已通过；后续训练仍必须显式消费
并复核该 receipt，且不得用它替代 M2 数据、policy 实现、checkpoint 与闭环验收证据。

LeRobot 0.6.1 的 NumPy/PyTorch 约束与根项目依赖合同不同，official ACT 仍只能进入
第 6.3 节 `$EMB/.venv-lerobot` isolate；不得把 LeRobot 直接装入项目 `.venv`。

### 6.2 BC 和 fallback：只使用 locked `.venv`

BC-1 与 `act_like_minimal_v1` 的唯一解释器是 `$EMB/.venv/bin/python`。CPU fallback
使用同一解释器、`--device cpu`、batch 1、workers 0、float32；CUDA 路径只能在
`project-train-env-receipt.v1` passed 后启用。checkpoint provenance 记录
`pyproject.toml`/`uv.lock` hashes、`train` group 的实际包版本/wheel/index、绝对解释器、
`sys.prefix` 和 project training environment receipt hash。

M3 不通过 `env.sh` 猜测环境。第 10.1 节冻结唯一 `PROJECT_ENV_CMD`，以绝对
`$PROJECT_PY` 执行所有 M2 preflight、normalization、BC、fallback 和对应 rollout，并
统一注入数据盘 cache、EGL 与 determinism 环境变量。`env.sh` 当前只激活 locked
项目 `.venv`；official LeRobot isolate 仍禁止 source 它，因为这会把项目环境的
`PATH/VIRTUAL_ENV` 带进 isolate。

若第 6.4 节 official smoke 产生可审计的兼容失败 receipt，fallback 仍必须先通过上述
project training environment gate；BC 或 fallback 的完成不构成 LeRobot 兼容证据。

### 6.3 官方 ACT：隔离的锁定环境

官方 backend 的**唯一**环境路径是仓库数据盘内 `$EMB/.venv-lerobot`；禁止创建 Conda
环境或任何第二份 LeRobot venv，也不改根项目的 `uv.lock`。当前已生成
`requirements.lerobot-act.in` 和 development-blocked
`requirements.lerobot-act.lock.txt`，但它们没有获得可执行环境准入：官方 v0.6.1/main
同时把 datasets、setuptools、torch 限制在已知漏洞修复版本以下，正常 resolver 证明三个
安全 floor 均不可满足；cu128 还需要未来协议冻结 package-specific 官方 source。因而
`.venv-lerobot` 未创建、未同步，receipt 明确是 `dependency_security` BLOCKED，禁止把
这份 lock 当作可安装交付物。只有上游或冻结协议产生可审阅的安全更新后，才重新生成包含
全部 transitive versions、wheel URL/hash 的候选 lock。
解释器合同不属于 pip requirement：launcher 固定 `LR_BOOTSTRAP_PY=/usr/bin/python3.12`，
在创建 venv 前断言其 `sys.version_info[:2] == (3, 12)`，并在 receipt 记录实际 patch
version、realpath 与 executable SHA-256。`requirements.lerobot-act.in` 只含可由 resolver
解析的 package anchors：

```text
lerobot[training] @ git+https://github.com/huggingface/lerobot.git@7e241bd630a3719a56157a497ce5d08f244784f1
torch==2.11.0+cu128        # official PyTorch cu128 index
torchvision==0.26.0+cu128
numpy==2.2.6
h5py==3.16.0
mujoco==3.11.0
imageio==2.37.4
safetensors==0.8.0
```

源代码不以 editable project 安装，因为根项目的 `numpy==2.5.2` 会造成 resolver 冲突；
official backend **禁止 `source ./env.sh`**，因为当前脚本只激活项目 locked `.venv`，
这会污染 LeRobot isolate 的 `PATH/VIRTUAL_ENV`。专用
launcher 必须 `env -u VIRTUAL_ENV`，把 `PATH` 设为
`$EMB/.venv-lerobot/bin:/usr/local/bin:/usr/bin:/bin`，并始终调用绝对解释器。receipt
必须断言：`realpath(sys.executable)==realpath("$EMB/.venv-lerobot/bin/python")`、
`realpath(sys.prefix)==realpath("$EMB/.venv-lerobot")`、`VIRTUAL_ENV` 未设置，且
LeRobot/torch 等 module path 都位于该 prefix（本仓库 `src/` 除外）。

isolate 只为本仓库 wrapper/adapter 保留显式 `PYTHONPATH=$EMB/src`；这不是安装来源，也
不能成为绕过 provenance 的捷径。launcher 必须按第 9.3/10.1 节在首次 import 前、smoke
前后以及任何 terminal receipt 发布前，用 trusted HEAD snapshot helper 重算
`lerobot-source-input-manifest.v1`。official train/rollout 的每次进程启动也必须在 import
训练栈前重算并与 adoption receipt 精确对账；任一 commit/path/index/content 漂移均
BLOCKED。project `PROJECT_ENV_CMD` 不使用这一例外，它只能从 locked editable root install
导入 `policy`。

所有会膨胀的路径都必须先创建在数据盘并由 launcher 显式传入：

```text
UV_CACHE_DIR=$EMB/cache/uv-lerobot
PIP_CACHE_DIR=$EMB/cache/pip-lerobot
TMPDIR=$EMB/cache/tmp-lerobot
XDG_CACHE_HOME=$EMB/cache/xdg-lerobot
HF_HOME=$EMB/hf
TORCH_HOME=$EMB/cache/torch
CUDA_CACHE_PATH=$EMB/cache/cuda-lerobot
CCACHE_DIR=$EMB/cache/ccache-lerobot
PYTHONPYCACHEPREFIX=$EMB/cache/pycache-lerobot
```

sync 前必须用 `df -Pk -- "$EMB" /` 分别解析 data/root 可用空间：data 起始至少
20 GiB、root 至少 3 GiB；venv creation 后/sync 前、sync 后/smoke 前以及 smoke 后都
再次检查 data 至少 15 GiB、root 至少 3 GiB。任一不足时必须按第 10.1 节先原子发布
`status="blocked",passed=null` receipt 再退出，不得由 bare `set -e` 无证据中止；不安装、不训练，
也不能把“空间不足、网络/中断/I/O 不确定、证据不完整”当成 official incompatibility
而启用 fallback。每个 resolver/install/smoke command 的 stdout/stderr 都写 no-clobber
文件并保存 hash；nonzero 必须先重查空间和错误类别，ENOSPC、I/O、network/HTTP timeout、
signal/interrupt 或未知失败一律 blocked/indeterminate。只有 sync 日志完整且命中预注册的
lock-resolution/platform-wheel incompatibility 白名单，或 smoke evidence 完整且声明允许的
compatibility failure class，才能发布 `failed` 并允许 fallback。preflight 还要求
`git check-ignore -q "$EMB/.venv-lerobot/"` 以及上述 cache/temp 目录（均以 trailing
slash 按目录语义探测）全部为真；若 ignore
规则未就绪，先在实施任务中补规则，禁止生成会污染 formal clean-worktree 的环境。

禁止 `uv pip install --no-deps lerobot`；lock 必须由正常 resolver 生成并通过
`uv pip check --python "$EMB/.venv-lerobot/bin/python"`。receipt 记录 LeRobot tag、commit
`7e241bd630a3719a56157a497ce5d08f244784f1`、upstream/wrapper lock hash、wheel URL/hash、
绝对解释器和真实 `sys.prefix`。实现测试还必须拒绝把解释器伪装成 package requirement；
`requirements.lerobot-act.in` 的每个非注释行都必须是 resolver 可识别的 package anchor。

同一锁定 CUDA wheel 可以用 `--device cpu` 执行 official backend 的 CPU 路径；CPU 配置固定为
batch 1、workers 0、float32。CUDA 从 microbatch 4、bf16、workers 2 开始，并只允许
第 6.4 节冻结的 4→2→1 阶梯。preflight 记录 Python/package versions、CUDA runtime、
cuDNN、GPU 名称/显存和 CPU 型号；这只证明运行环境，不证明模型指标。

### 6.4 adoption smoke receipt

在读取正式 M2 数据或开始 overfit 前，隔离环境必须顺序完成：

1. 第 6.3 节空间、ignore、interpreter/prefix/module-path、lock 和 resolver 检查全部通过；
   `uv pip check` 为 0 error，打印精确 package/wheel 来源。
2. CPU batch 1 构造第 5.2 节 ACTConfig。train mode 单独调用
   `loss, loss_dict = ACTPolicy.forward(batch)`，断言 `loss` 为 finite scalar、loss dict
   finite，再完成 `backward()` 和 optimizer step；eval mode 另行调用
   `predict_action_chunk(observation)`，断言 finite `[1,16,8]`。保存
   `model.safetensors+config+normalization`，由新进程加载后重复 predict 并按
   `atol=1e-6,rtol=1e-5` 对账。
3. CUDA sm_120 只按预注册 microbatch ladder `4 -> 2 -> 1` 尝试，对应
   `gradient_accumulation_steps=1 -> 2 -> 4`，effective batch 恒为 4。每一档在**全新
   进程/全新 model+optimizer** 中，从固定 synthetic batch 开始，依次完成：
   train `forward -> finite scalar loss/loss_dict -> backward -> accumulated optimizer step`；
   eval `predict_action_chunk -> [B,16,8]`；保存；新进程加载；同一输入 predict 数值对账。
   任一步 OOM 都保留该档 failure stage、stdout/stderr hash 和 peak allocated/reserved，
   退出该进程后才进入下一档；不能只做一个小 forward 就把该档判为通过。
4. batch 4 OOM 但 batch 2 或 1 完整通过时，receipt 仍是 `status="passed",passed=true`，采用 official
   backend，并冻结首个完整通过档的 `selected_microbatch`、accumulation 和 effective
   batch。只有 batch 1 在上述完整链路中仍 OOM，才可因显存写 `status="failed",passed=false` 并启用
   fallback。resolver、kernel、non-finite 或 save-load 等非 OOM 兼容错误不走降 batch
   阶梯，而是记录真实 failure 并 `status="failed",passed=false`；空间/ignore 前置不足则是 `blocked`，
   不是 fallback admission。
5. 用已选 CUDA checkpoint 让第 1.2 节 adapter 接收真实 `PickPlace.observe()`：先验证
   raw state 为 finite `float64 [8]`，显式 cast 后调用 `predict_action_chunk()` 得到
   `[1,16,8]`，反归一化首个动作通过 `PickPlace.step()` 接口。这只是接口 smoke，不是
   成功率。
6. 相同 checkpoint 在 `--device cpu` 再执行一次 `predict_action_chunk()`，shape/finite
   检查通过。

所有子项及每个 ladder attempt 的 stdout/stderr、package freeze、device info、config/hash
先写入不可覆盖的 `smoke-evidence.json`；launcher 另外捕获整个 smoke process 的 no-clobber
stdout/stderr 并计算 hashes。只有所需项全部通过时 `status="passed",passed=true`。
nonzero 只有同时满足“process 正常退出而非 signal/interrupt、资源门槛仍满足、stdout/stderr
完整可 hash、evidence schema/attempt reconciliation 完整、`failure_class` 属于预注册
compatibility allowlist”时，才可发布 `status="failed",passed=false` 并允许第 5.3 节
fallback。missing/partial evidence、空日志证据、I/O/resource/network 错误或未知 class 都是
`blocked/indeterminate`，不得用空 SHA 冒充 compatibility failure。

不得删除失败 receipt 后重复安装直到“看起来成功”而不保留 lineage；若 formal run 后来
需要更低 microbatch，必须生成 superseding receipt 并完整重跑对应剩余阶梯。venv creation
后/sync 前、sync 后/smoke 前与 smoke 后必须各自重验 space、ignore；后两处还重验
interpreter、exact packages/lock。smoke runner 只能写 evidence，最终 passed terminal
receipt 必须由 launcher 在最后一次复检成功后原子发布，不能提前出现。

默认 reproducibility 开关必须由 shell launcher 在 Python 解释器启动**之前**注入：

```text
PYTHONHASHSEED=0                 # baseline train_seed；其他 run 在启动前写其精确整数
CUBLAS_WORKSPACE_CONFIG=:4096:8
random / numpy / torch / torch.cuda seeds 全部设置
torch.use_deterministic_algorithms(True)
cudnn.benchmark=False
cudnn.deterministic=True
DataLoader generator 和 worker_init_fn 从 train_seed 派生
```

Python 内只读取并断言 `os.environ["PYTHONHASHSEED"] == str(train_seed)`；在进程内补设
该变量无效且必须报错。run/receipt 保存 launcher command 与实际环境值，所有 worker 和
新进程 save-load check 必须继承同一值。

若某算子不支持 deterministic mode，run 失败并记录算子；不得悄悄关闭确定性继续
生成 acceptance 数字。开发者可以启动明确标记 `deterministic=false` 的诊断 run，
但它不能成为 M3 acceptance artifact。

## 7. M2 replay gate：所有训练的硬前置

M3 不信任一个自报通过的 summary。`policy.preflight` 必须消费并逐级重算以下版本化
M2 证据；在 M2 实现真正发布这些 artifacts 之前，M3 gate 保持 PENDING：

```text
M2_RUN_DIR/
  collection-ledger.jsonl       # canonical JSONL；每次尝试一行
  collection-manifest.json      # m2-collection-manifest.v1
  validation-report.json        # m2-validation-report.v1
  data-report.json              # m2-data-report.v1
  manual-review-trials.jsonl    # m2-manual-review-trial.v1；20 条真实人工记录
  replay/plan.json              # m2-replay-plan.v1
  replay/trials.jsonl           # m2-replay-trial.v1；每个计划 trial 一行
  replay/summary.json           # m2-replay-summary.v1
```

### 7.1 collection ledger、manifest 与 validation schema

`collection-ledger.jsonl` 是 append-only 事实源，每行至少包含：

```text
schema_version="m2-collection-ledger-entry.v1"
run_id, attempt_index, split, seed, started_at_utc, finished_at_utc
success, failure_stage, eligible, rejection_reason
relative_hdf5_path, episode_sha256, num_steps, episode_schema_version
controller_config_id, environment_config_id, source_provenance_id
```

formal ledger 的 `attempt_index` 从 0 连续递增；train seed 必须从 0 连续尝试且全部位于
`0..999`，达到第 200 个 success 立即停止；validation 从 1000 连续尝试且全部位于
`1000..9999`，达到第 40 个 success 立即停止。成功和失败都必须有 no-clobber HDF5 与
ledger row；`eligible=true` 当且仅当文件 schema 完整、`success=true` 且 split 合法。
不得删失败、跳过难 seed、重复 seed 或在到达目标后继续挑更短 episode。JSONL 每行按第
9.2 节 canonical bytes 加一个 LF，整个文件的 SHA-256 作为 `ledger_sha256`。

`collection-manifest.json` 至少包含：`schema_version`、唯一 `manifest_id`、run/config/
source provenance、`ledger={path,sha256,row_count}`、第 1.3 节 seed contract、每个 split
的 target/attempt/success/eligible 计数和 exact seed 列表，以及与 ledger 一一对应的
episode entries（split、seed、outcome、eligible、relative path、file hash、HDF5 metadata）。
manifest 不反向写入 ledger，避免循环 hash；其 `manifest_id` 按第 9.2 节只排除自身字段
计算。正式 manifest 必须拒绝任何 `seed>=10000` 的 collection entry。

`relative_hdf5_path` 是不可信输入，必须在读取、hash 或打开 HDF5 **之前**完成 containment
检查。唯一合法表示是 canonical POSIX relative path：拒绝空字符串、absolute path、NUL、
反斜杠、空 segment、`.`/`..` segment、重复分隔符和 normalization 后不等于原字符串的
alias。先对 `data_root` 执行 `resolve(strict=True)` 并确认是目录；随后从 resolved root 到
leaf 对每一级 `lstat`，任何 symlink 都拒绝。leaf 必须存在且是 regular file，
`leaf.resolve(strict=True)` 必须满足 `is_relative_to(resolved_data_root)`；canonical path、
resolved target 以及 `(st_dev,st_ino)` 在本 run 内都必须唯一，避免 hard-link/文本 alias
重复计数。ledger 与 manifest 的 path 集合、顺序身份、seed、hash 必须逐项完全相等。
实现最终 open 时还要使用 no-follow 语义并对打开后的 fd `fstat`，防止检查与读取之间被
替换；任何 containment/uniqueness 错误先失败，禁止触碰 root 外文件内容。

`validation-report.json` 至少包含 `schema_version="m2-validation-report.v1"`、自身 id、
manifest id/file hash、ledger hash、validator code/provenance、逐 entry 的 file/hash/schema
结果、集合级 namespace/唯一性/计数结果、`error_count` 和 `passed`。M3 只接受全部 240+
失败尝试都完成身份对账、所有文件 hash 匹配、所有 HDF5 `validate_episode().valid`、
`error_count=0` 的 report；不能只校验 eligible 240 个文件。

`data-report.json` 必须从同一 manifest/raw files 重算 attempted/success、长度和 action
分布，并引用 `manual-review-trials.jsonl` 的 path/hash/row_count。人工证据正好 20 个
不同 attempt，每行含 source split/seed/path/hash、reviewer、review timestamp、
`failure|anomaly` 分类、可定位 frame/media 引用和非空 finding；没有逐条记录时不能用
summary 中的 `manual_review_count=20` 自证 README 门禁。

### 7.2 确定性随机 20 replay 与 trial-level 契约

replay 候选集是最终 manifest 中 train+validation 的 240 个 `eligible=true` success。
选择规则冻结为 `selection_algorithm="sha256-rank-v1"`、
`selection_seed=20260824`。对每个候选构造以下对象，按第 9.2 节 canonical bytes 求
`rank_digest`：

```json
{
  "algorithm": "sha256-rank-v1",
  "manifest_id": "<exact manifest_id>",
  "selection_seed": 20260824,
  "split": "train|validation",
  "seed": 0,
  "relative_hdf5_path": "<manifest path>",
  "episode_sha256": "<64 lowercase hex>"
}
```

按 `(rank_digest, split, seed, relative_hdf5_path)` 升序取前 20 个；这是由冻结 manifest
和 seed 决定的伪随机样本，不调用 Python `random.sample`，不存在版本差异，也不允许
人工替换。`replay/plan.json` 保存 `schema_version="m2-replay-plan.v1"`、plan id、上述
算法/seed/manifest id+file hash、240 个候选的集合 hash，以及所选 20 个按 rank 排序的
`trial_id/rank/split/seed/path/file_sha256/num_steps`。preflight 必须从 manifest 重算排名
和完整 exact 20；plan 中任何换样都失败。

runner 对计划中每个 trial 重新用同一 seed reset，按原顺序读取并经
`PickPlace.step()` 执行 HDF5 的全部 `T` 个 action。即使异常/失败也必须向
`replay/trials.jsonl` 写一行：

```text
schema_version="m2-replay-trial.v1"
trial_id, rank, plan_id, manifest_id, split, seed
source_relative_path, source_file_sha256, source_num_steps, action_dataset_sha256
runner_config_id, reset_seed, expected_steps, executed_steps
success, failure_stage, exception_type, final_hold_steps
started_at_utc, finished_at_utc
```

trial identity 是 `(trial_id,rank,split,seed,source path,source hash)`；必须与 plan 一一
匹配，恰好 20 个唯一 row，无 missing/unexpected/duplicate。`executed_steps` 必须等于
`expected_steps`；中途 runner 异常保留原 row 但该 trial 失败，不能换 seed 补分母。
`replay/summary.json` 保存 `schema_version="m2-replay-summary.v1"`、自身 id、plan id/hash、
trials path/hash、identity reconciliation、`successes`、固定 `trials=20`、`passed` 和所有
失败 trial id；只有逐条对账完整且 `successes>=18` 才能 `passed=true`。

### 7.3 M3 preflight 输入与判定

规范 CLI 必须显式接收 data root、manifest、ledger、validation、data report、manual
review trials、replay plan、replay trials、replay summary 和第 1.3 节 seed protocol；不
允许约定隐藏路径或自动选择“最新”。它重算并检查：

- 200 条 train 与 40 条 validation eligible success，且 ledger 中全部成功/失败尝试与
  manifest/HDF5 一一对应；
- 所有 ledger/manifest HDF5 路径先通过 canonical POSIX、no-symlink、resolved-root
  containment、regular-file 与 canonical/resolved/inode uniqueness 检查，再允许 hash/schema；
- 所有文件存在、SHA-256 匹配、`validate_episode().valid`，schema/action semantics/
  time alignment/control dt/action bounds 完全一致；
- train、validation、M3 IID、M4 locked 四集合逐项属于正确 namespace、内部唯一且两两
  无交集；
- validation/data/manual-review schemas 与 input hashes 对账，validation error 为 0，
  manual review 有 20 条 trial-level 事实；
- replay plan 是 manifest+`20260824` 可重算的 exact 20，trials 逐身份完整，summary
  hashes 相符且至少 18/20 success；
- normalization 尚未读取 validation、M3 IID 或 M4 locked 数据。

输出 canonical `m2_gate_receipt.json`，含每个显式输入的绝对/相对路径、file SHA-256、
content id、计数、集合检查和 `passed=true`。训练 CLI 必须要求 receipt 并重验全部 input
hashes；单独的 `--skip-m2-gate`、环境变量后门或 warning-only 模式都不允许。

允许在 M2 通过前编写 Dataset/model/checkpoint 单元测试以及用人工构造的微型 HDF5
做 shape 测试；任何创建 optimizer 并对真实或正式 M2 数据执行 step 的命令，包括
8-episode overfit，都由 receipt 阻止。

## 8. 两阶段训练与验收

### 8.1 8-episode overfit

对 BC-1 和 adoption receipt 选定的唯一 chunk backend 分别执行同一个 gate；不能只让
其中一个通过后就给另一个开始正式训练。official ACT 与 fallback 不能同时训练后再挑
test 表现更好的一个；backend 在任何真实 chunked overfit 前已由 smoke receipt 冻结。
8 个 episode 按 frozen train manifest 中 eligible seed 的数值升序取前 8 个，准确 seed
列表和选择规则写入 overfit manifest，避免手挑容易成功的轨迹。

固定诊断配置：

```text
train_seed=0
optimizer_steps=20_000          # 必须完整跑满；不是可早停上限
gate_eval_every=100 steps
rollout_max_control_steps=600
BC-1: H_pred=1, K_exec=1
selected chunk backend: H_pred=16, K_exec=4
```

在第 0 个 optimizer step 前，用固定顺序全量扫描 8 个 episode 得到
`initial_train_reconstruction_loss`。每次 gate eval 使用 `model.eval()`、同样顺序和
全部有效 action 元素：

- BC-1：normalized MSE；
- official ACT：zero-latent eval 下的 masked normalized L1（同时继续记录训练时
  total/L1/KL）；
- `act_like_minimal_v1`：masked normalized L1；没有 KL 字段，也不得伪造一个。

训练期间**不做任何闭环 rollout**。step 0 和每 100 steps 的全量重建 loss 都写入不可改写
记录，并在对应 step 原子保存 checkpoint。完整 20,000 steps 结束后才冻结唯一候选：在
step `100..20000` 的 finite 全量重建 loss 中取最小者；若数值完全相同，用较小 step
tie-break。不得用最后一个 minibatch、M2 validation、任何 rollout 或人工视频挑选。

按下式对这个 `selected_overfit_checkpoint` 计算：

```text
loss_drop_fraction = 1 - selected_checkpoint_reconstruction_loss
                         / initial_train_reconstruction_loss
```

loss 必须 finite，initial 必须 `>1e-8`。同一个 checkpoint 同时满足才通过：

```text
loss_drop_fraction >= 0.95
closed_loop_successes_on_exact_8_train_seeds >= 7/8
```

闭环每个 seed 重新构造/重置环境，策略只看 allowlisted observation，所有动作经
`PickPlace.step()`，最多 600 control steps；`env.success()` 的连续 1 秒条件是唯一成功
判定。只有 checkpoint id 与选择 receipt 已原子冻结后，才对 exact 8 train seeds 各运行
**一次**闭环。一个有效 `overfit_rollout_run_id` 必须得到恰好 8 条、每 seed 唯一的
`overfit-rollout-trial.v1` JSONL；每行固定 `ordinal=0..7`、`trial_id`、seed、checkpoint/config
hash、`outcome=success|policy_failure`，summary 必须逐 identity 对账且不接受 duplicate。
不提供 per-trial retry/attempt 机制。

若 process/simulator/GPU/I/O 等基础设施异常，当前 run 立即 invalid，保留其 run dir、已写
row 和 `failure.json`；不得在同一 run 补跑单个 seed，也不得把旧 row 与新 row 聚合。需要
恢复时只能创建全新 no-clobber `overfit_rollout_run_id`，引用旧 run id/失败原因，并从第一个
seed 开始重新执行 exact 8。只有机器可判定的 infrastructure error 可走整 run 重启；普通
timeout、控制步耗尽和环境未成功都属于 policy failure，必须进入 8 行分母。策略失败不能
重跑取最好结果，也不能回看其他 checkpoint。loss 下降但闭环 `<7/8` 必须报告为 overfit
FAIL，不得以离线 loss 替代，也不得试第二个 checkpoint。

### 8.2 正式 200/40 训练

只有对应 policy 的 overfit receipt 通过后才能启动。固定上限为 50,000 optimizer
steps，每 500 steps 做完整 validation loss、每 500 steps 原子保存 resume checkpoint；
完成训练/预注册早停规则后，以最低 validation reconstruction loss 选择唯一 checkpoint，
相同 loss 用较小 step tie-break。test/IID seed 不参与早停、checkpoint 选择或超参修改。

BC 与选定 chunk backend 都完成训练并各自冻结唯一 checkpoint 后，先发布同一个
`m3-iid-acceptance-plan.v1`：显式列出第 1.3 节 `10050..10149`、两个 checkpoint id、
normalization/config/M2 manifest/seed-protocol hashes 和每个 policy 的 100 个 trial id；
此后才允许首次打开这些在线 seed。每个 policy/checkpoint 对 exact 100 seeds 只运行一遍，
至少一个学习策略需要 `>=60/100`。这是未来必须真实运行的门禁，不是本文结果。
BC 与选定 chunk backend 的结果、训练时间、参数量和失败必须并列报告；不得预写
chunked policy 胜出。若采用 fallback，表头必须写 `ACT-like minimal`，不得写 `ACT`。
任何 M3 IID 结果都不能触发 checkpoint/超参修改后在同一 100 seeds 上复测；失败按第
1.3 节处理为 M3 FAIL/test exposure。

扩展的 3 个训练 seed 为 `{0,1,2}`，每个产生独立 run/checkpoint；只有资源允许时才做，
不能把同一权重重复 rollout 冒充 3 个训练 seed。

## 9. Checkpoint、hash 与兼容性

### 9.1 原子目录格式

每个 checkpoint 先写同目录临时目录，fsync、逐文件 hash 校验后 rename；目标目录存在
则拒绝覆盖：

```text
checkpoints/step-00000500/
  model.safetensors
  optimizer.pt                 # 仅本地产生的 trusted resume 使用
  trainer_state.json
  policy_config.json
  normalization.json
  provenance.json
  checkpoint_manifest.json
```

inference 只读取 safetensors/JSON，绝不 unpickle `optimizer.pt`。resume 只有显式
`--resume` 才读取本项目自己生成且 hash 匹配的 optimizer state。

### 9.2 canonical JSON 与非递归自哈希

所有带 content id 的 JSON 使用同一算法，不允许各模块自行解释“自身 hash”：

```python
canonical_bytes = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

canonical bytes 本身不含 LF；实际 JSON 文件写入完整文档的 canonical bytes 后追加一个
LF。每个 schema 只声明一个 identity 字段，例如 `normalization_id`、`manifest_id`、
`plan_id`、`receipt_id` 或 `checkpoint_id`。生成/验证算法固定为：浅拷贝完整文档，**只
删除该 schema 指定的 identity 字段**，计算
`"sha256:" + sha256(canonical_bytes(payload_without_identity)).hexdigest()`，再与完整
文档中的 identity 比较。不得排除 timestamp、path 或其他任意字段，也不得把完整文件
含 identity 的 bytes 反复自哈希。完整落盘文件另由父 manifest 记录普通 file SHA-256。

canonical JSONL 则逐行 canonicalize、每行恰好一个 LF，对最终全部 bytes 求普通
SHA-256；JSONL row 不放递归 file digest。所有 NaN/Infinity、重复 JSON key、非 UTF-8 或
CRLF 输入都 fail closed。第 1.3、3.1、6.4、7、9.1 节的 JSON identities 全部服从本节。

唯一实现必须是 stdlib-only `src/policy/canonical_json.py`；Python modules 直接 import，
shell launcher 只能用绝对 `/usr/bin/python3.12` 调它的 CLI。jq 可以构造/查询 JSON value，
但不得用 `-S` 输出冒充 canonical bytes，也不得在 shell 中自行 sha identity。测试冻结一个
同时含 `1.0`、Unicode、`false`、`null` 的 golden document，要求 manifest/config/receipt
三个调用点得到相同 canonical bytes/id；这专门防止 jq 把 `1.0` 编成 `1` 的数值差异。

```text
golden canonical bytes = {"false":false,"float":1.0,"null":null,"unicode":"机器人"}
golden SHA-256 = bde5262f0df421331496b60737cea47a92cd8fb49fa23694126b85921afcbc20
```

### 9.3 必存字段与 source provenance

`policy_config.json`：

- `checkpoint_schema_version="m3-checkpoint.v1"`；
- `policy_type=bc1|lerobot_act_0_6_1|act_like_minimal_v1` 与完整网络字段；
- exact observation allowlist、第 1.2 节两个 raw-source contract、唯一 adapter id、
  model-input dtype/shape、`action_dim=8`；
- `H_pred`；chunk backend id；official 时保存 LeRobot version/config，fallback 时保存
  失败的 smoke receipt hash 和第 5.3 节完整结构；
- preprocess/normalization mode 和 action semantics/bounds；
- optimizer、AMP、batch、accumulation、determinism 和 train seed。

`provenance.json`：

- Git full commit；tracked-only status clean flag/hash；
  `git status --porcelain=v1 -z --untracked-files=all` 的 stdout SHA-256；untracked file count
  与 NUL-separated path-list SHA-256。完整 status/path stdout 不落 artifact，避免泄露无关
  个人文件名；
- `pyproject.toml`、`uv.lock`、project training environment receipt 或
  `requirements.lerobot-act.lock.txt`、ResNet weight 文件 SHA-256；
- M2 ledger/manifest/gate、normalization、seed protocol、M3/M4 plans、train/val seed
  lists 的 content id 和 file SHA-256；
- HDF5 `SCHEMA_VERSION=1`、MuJoCo/LeRobot/Python/PyTorch/torchvision/h5py versions；
- host/device 信息、run seed、启动前的 `PYTHONHASHSEED` 和 launcher id。

formal admission 与当前 `expert._git_state` 语义一致，但不把 `worktree_clean` 当唯一门禁：

- 所有 tracked 文件必须 clean；tracked dirty 直接 fail；
- 枚举实际导入的本仓库 `.py`，以及 `src/`、`config/`/`configs/`、`scripts/`、
  `menagerie/`、根 launcher/pyproject/uv/requirements locks 等 runtime-input scope；每个
  实际输入必须 `git ls-files --error-unmatch`，且工作树 SHA-256 等于
  `git show HEAD:<path>`；
- relevant untracked/ignored source/config/launcher/lock/menagerie 必须安全地计算
  path/status/size/content SHA-256 并进入 `relevant_untracked_files`。formal 仍 fail closed，
  直到它们成为 tracked+HEAD-identical 的冻结输入；缺 hash 或扫描命令失败也 fail；
- 不在 runtime-input scope 的个人文档（例如用户保留的 report/logbook/notes）只影响
  `worktree_clean` 观测值和上述 count/hashes，不影响 `source_provenance_clean`，也不得要求
  删除、暂存、改名或加入 ignore。

因此 formal 门禁是 `tracked_worktree_clean && provenance_complete &&
source_provenance_clean`，不是“full status 必须空”。LeRobot wheel/source 由固定 upstream
commit、wheel/RECORD/lock hash 单独证明。

LeRobot adoption 另冻结 canonical `lerobot-source-input-manifest.v1`，并把它作为 terminal
receipt 的强制 lineage，而不是只记录一次 full status。manifest identity 按第 9.2 节只
排除 `manifest_id`；无 timestamp 字段，避免同一输入因时钟变化产生不同 identity。顶层至少
包含：

- `git_commit`、`head_tree`、`scope_config_sha256`、`path_set_sha256`、
  `index_entries_sha256`、`content_entries_sha256`、`tracked_status_sha256`、
  `source_provenance_clean`；
- 按 canonical repo-relative POSIX path 排序且唯一的 `entries`。每项含 `path`、Git mode、
  `index_blob_oid`、`head_blob_oid`、`size_bytes`、工作树 `content_sha256`；scope/path/hash
  command 失败即不完整。

full porcelain `-z` stdout hash、untracked count/path-list hash 写入独立 companion
`worktree-observation.json`，不保存 scope 外个人文件名。它的普通 file SHA 进入 receipt
evidence，但不进入 source manifest identity 或 admission；因此只新增 scope 外个人文档时，
观测 artifact 会改变，source manifest id 与 gate 结果不变。

版本化 `configs/m3/lerobot-source-inputs.json` 定义 closed scope，并且自身也必须入 scope。
最小闭包覆盖：`scripts/run_m3_lerobot.sh`、`src/policy/canonical_json.py`、
`src/policy/source_provenance.py`、全部被 official 路径导入的 `src/policy/**` adapter/wrapper、
其依赖的 `src/env/**` 与 `src/data/**`、`configs/m3/lerobot-act-smoke.json`、LeRobot source
scope config、`requirements.lerobot-act.in`/lock、`pyproject.toml`、`uv.lock` 与实际读取的
menagerie 文件。helper 从 Git/固定 scope 枚举，不接受 shell 当前目录 glob 或“最近文件”。
每个 scope path 必须 tracked，index blob 与 HEAD blob 相同，工作内容 SHA 与 HEAD 内容
相同；scope 内相关 untracked/ignored source/config/launcher/lock/menagerie 被 fingerprint
后仍使 `source_provenance_clean=false`。全局 tracked dirty 仍按 formal 规则失败；scope 外
用户个人 report/logbook/notes 只改变观测 hash/count，不改变该 manifest identity 或 admission。

adoption receipt 必须内嵌 `source_input_manifest_id`、`source_commit`、manifest 相对路径及
file SHA、path-set/index/content 摘要、三个 immutable verifier/canonical/scope snapshot
hash 和最后一次 recheck evidence path/hash。发布
`passed` 或 compatibility `failed` 前，当前重算值必须与启动时 manifest **逐字段完全相等**；
不完整、TOCTOU 漂移或 commit 变化一律把请求状态降为
`status="blocked",passed=null`。即使另一 clean commit 产生完全相同的
文件 bytes，`git_commit` 不同也禁止复用旧 receipt。train/rollout 同时核对 receipt 内引用
和外部 manifest；receipt path 存在不等于 lineage 通过。

diagnostic dirty run 可以继续，但必须 `non_acceptance=true`，保存 tracked binary diff
SHA-256，并生成 `source_input_manifest.json`：对**每个实际导入 source、config、launcher、
lock**记录 repo-relative/absolute path、Git 状态（`tracked_clean`、`tracked_modified`、
`untracked`、`ignored` 或 `external`）、size 和 content SHA-256。任何 untracked/ignored source/config 没有内容
条目即 fail closed；只保存 `git diff` 不足，因为它不覆盖 untracked。ignored data、
checkpoint、cache 等 runtime artifact 不要求入 Git，但必须由 M2/checkpoint/run manifest
逐文件 hash。不得把 token、credential 或完整敏感环境变量写入 provenance。

`checkpoint_manifest.json` 列出其余每个文件的 size/SHA-256，并对 canonical manifest
按第 9.2 节排除 `checkpoint_id` 后计算 id。正式 acceptance run 要求 tracked-clean 与
source-input provenance 两道检查都通过；无关 untracked 文档可以让
`worktree_clean=false`，但不会单独否决 acceptance。dirty diagnostic checkpoint 必须带
`non_acceptance=true`。

### 9.4 加载时 fail-closed 矩阵

| 检查 | inference | resume training |
|---|---|---|
| checkpoint schema 支持 | 必须 | 必须 |
| model/config/file hashes | 必须 | 必须 |
| raw-source/model-input keys/dtypes/shapes/action semantics | 必须 | 必须 |
| env action bounds/control frequency | 必须 | 必须 |
| normalization hash | 必须 | 必须 |
| chunk backend / smoke receipt | chunked policy 必须 | chunked policy 必须 |
| `K_exec <= H_pred` | 必须；K_exec 可变 | 必须 |
| M2 manifest/data hashes | 记录可定位 | 必须重验全部 |
| Git tracked/source-input manifest/lock exact match | acceptance 必须 | 必须 |
| optimizer config/state | 不读取 | 必须 |

允许 runtime 改变的只有设备、CPU/CUDA batch-free inference 实现和 `K_exec`；任何网络、
H_pred、normalization、allowlist 或 action contract 差异都报 incompatibility。不得提供
用于正式结果的 `--ignore-schema` / `--ignore-hash`。

## 10. CLI 与输出契约

### 10.1 环境与 preflight

未来实现后的规范调用形态：

```bash
set -euo pipefail
EMB="$(pwd -P)"
test -f "$EMB/README.md"
TRAIN_SEED=0
UV_BIN="$(command -v uv)"
PROJECT_PREFIX="$EMB/.venv"
PROJECT_PY="$PROJECT_PREFIX/bin/python"
test -x "$PROJECT_PY"
PROJECT_BIN="$(dirname -- "$PROJECT_PY")"
PROJECT_TMP="$EMB/cache/tmp-project"
M2_RUN_DIR="$EMB/runs/m2/m2-acceptance-001"  # frozen、no-clobber M2 run
SEED_PROTOCOL="$EMB/configs/m3/m3-m4-seed-protocol-v2.json"
M3_CONTRACT_DIR="$EMB/runs/m3/contracts-001" # 新 id；禁止复用已有目录
M2_GATE="$M3_CONTRACT_DIR/m2-gate-m2-acceptance-001.json"
PROJECT_TRAIN_RECEIPT="$M3_CONTRACT_DIR/project-train-env-receipt.json"
PROJECT_BOOTSTRAP_PY=/usr/bin/python3.12
PROJECT_CANONICAL_REL=src/policy/canonical_json.py
PROJECT_RECEIPT_TOOL_REL=src/policy/project_env_receipt.py
test -d "$M2_RUN_DIR"
test -x "$PROJECT_BOOTSTRAP_PY"

git check-ignore -q "$M2_RUN_DIR/"
git check-ignore -q "$EMB/runs/m3/"
mkdir -p "$EMB/runs/m3"
mkdir "$M3_CONTRACT_DIR"  # no-clobber；已存在即 nonzero

# pre-sync terminal receipt 不能依赖尚未完成 editable install 的 policy package。
# 只把已 tracked 且与 HEAD 内容相同的两个 stdlib-only publisher helper 固化到本 run；
# helper 本身不可信时 fail 2，不伪造可审计 receipt。
for bootstrap_rel in "$PROJECT_CANONICAL_REL" "$PROJECT_RECEIPT_TOOL_REL"; do
  git -C "$EMB" ls-files --error-unmatch "$bootstrap_rel" >/dev/null || exit 2
  git -C "$EMB" diff --quiet HEAD -- "$bootstrap_rel" || exit 2
done
PROJECT_CANONICAL_SNAPSHOT="$M3_CONTRACT_DIR/canonical_json.head.py"
PROJECT_RECEIPT_TOOL_SNAPSHOT="$M3_CONTRACT_DIR/project_env_receipt.head.py"
(set -o noclobber
 git -C "$EMB" show "HEAD:$PROJECT_CANONICAL_REL" >"$PROJECT_CANONICAL_SNAPSHOT"
 git -C "$EMB" show "HEAD:$PROJECT_RECEIPT_TOOL_REL" >"$PROJECT_RECEIPT_TOOL_SNAPSHOT") || exit 2

PROJECT_ENV_CMD=(
  env -u VIRTUAL_ENV -u PYTHONPATH
  "PATH=$PROJECT_BIN:/usr/local/bin:/usr/bin:/bin"
  "EMB=$EMB" "MUJOCO_GL=egl"
  "MENAGERIE=$EMB/menagerie"
  "UV_CACHE_DIR=$EMB/cache/uv"
  "PIP_CACHE_DIR=$EMB/cache/pip-project"
  "TMPDIR=$PROJECT_TMP" "XDG_CACHE_HOME=$EMB/cache/xdg-project"
  "HF_HOME=$EMB/hf" "TORCH_HOME=$EMB/cache/torch"
  "CUDA_CACHE_PATH=$EMB/cache/cuda-project"
  "CCACHE_DIR=$EMB/cache/ccache-project"
  "PYTHONPYCACHEPREFIX=$EMB/cache/pycache"
  "PYTHONDONTWRITEBYTECODE=1"
  "UV_PROJECT_ENVIRONMENT=$PROJECT_PREFIX"
  "UV_PYTHON_DOWNLOADS=never" "PYTHONHASHSEED=$TRAIN_SEED"
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
)
BOOTSTRAP_ENV_CMD=(
  env -u VIRTUAL_ENV -u PYTHONPATH
  "PATH=/usr/local/bin:/usr/bin:/bin" "EMB=$EMB" "MUJOCO_GL=egl"
  "UV_CACHE_DIR=$EMB/cache/uv" "PIP_CACHE_DIR=$EMB/cache/pip-project"
  "TMPDIR=$PROJECT_TMP" "XDG_CACHE_HOME=$EMB/cache/xdg-project"
  "HF_HOME=$EMB/hf" "TORCH_HOME=$EMB/cache/torch"
  "CUDA_CACHE_PATH=$EMB/cache/cuda-project"
  "CCACHE_DIR=$EMB/cache/ccache-project"
  "PYTHONPYCACHEPREFIX=$EMB/cache/pycache"
  "PYTHONDONTWRITEBYTECODE=1"
  "UV_PROJECT_ENVIRONMENT=$PROJECT_PREFIX"
  "UV_PYTHON_DOWNLOADS=never" "PYTHONHASHSEED=$TRAIN_SEED"
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
)

project_space_snapshot() {
  project_data_free_kib="$(df -Pk -- "$EMB" | awk 'NR==2 {print $4}')" || return 1
  project_root_free_kib="$(df -Pk -- / | awk 'NR==2 {print $4}')" || return 1
  [[ "$project_data_free_kib" =~ ^[0-9]+$ && "$project_root_free_kib" =~ ^[0-9]+$ ]]
}

project_blocked_exit() {
  local stage="$1" reason="$2" sync_exit_code=-1
  shift 2
  if (($# > 0)); then
    sync_exit_code="$1"
    shift
  fi
  "${BOOTSTRAP_ENV_CMD[@]}" "$PROJECT_BOOTSTRAP_PY" \
    "$PROJECT_RECEIPT_TOOL_SNAPSHOT" publish \
    --canonical-helper "$PROJECT_CANONICAL_SNAPSHOT" \
    --expected-prefix "$PROJECT_PREFIX" --pyproject "$EMB/pyproject.toml" \
    --lock "$EMB/uv.lock" --required-group train --force-status blocked \
    --failure-stage "$stage" --reason-code "$reason" \
    --sync-exit-code "$sync_exit_code" "$@" --output "$PROJECT_TRAIN_RECEIPT" || exit 70
  exit 3
}

project_resource_recheck() {
  local stage="$1" data_min_gib="$2" ignored_dir
  if ! project_space_snapshot; then
    project_blocked_exit "$stage" df_unreadable
  fi
  if ((project_data_free_kib < data_min_gib * 1024 * 1024 ||
       project_root_free_kib < 3 * 1024 * 1024)); then
    project_blocked_exit "$stage" insufficient_space
  fi
  for ignored_dir in \
    "$PROJECT_PREFIX" "$EMB/cache/uv" "$EMB/cache/pip-project" "$PROJECT_TMP" \
    "$EMB/cache/xdg-project" "$EMB/cache/torch" "$EMB/cache/cuda-project" \
    "$EMB/cache/ccache-project" "$EMB/cache/pycache" "$EMB/hf"; do
    git check-ignore -q "$ignored_dir/" || \
      project_blocked_exit "$stage" directory_not_ignored
  done
}

# 目录尚不存在时也以 trailing slash 检查 ignore；失败会留下 blocked receipt。
project_resource_recheck initial_project_setup 20
if ! mkdir -p \
  "$EMB/cache/uv" "$EMB/cache/pip-project" "$PROJECT_TMP" \
  "$EMB/cache/xdg-project" "$EMB/cache/torch" "$EMB/cache/cuda-project" \
  "$EMB/cache/ccache-project" "$EMB/cache/pycache" "$EMB/hf"; then
  project_blocked_exit project_cache_setup cache_directory_creation_failed
fi

# 这一步当前只能证明唯一 prefix 与基础包；不证明 PyTorch/CUDA ready。
if ! "${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -c \
  'import os,sys,h5py,mujoco,numpy; from pathlib import Path; p=Path(sys.argv[1]).resolve(); assert Path(sys.executable).resolve()==(p/"bin/python").resolve(); assert Path(sys.prefix).resolve()==p; assert "VIRTUAL_ENV" not in os.environ; assert "PYTHONPATH" not in os.environ' \
  "$PROJECT_PREFIX"; then
  project_blocked_exit base_interpreter prefix_or_base_package_mismatch
fi

# formal bootstrap 只接受已提交且与 HEAD 相同的 dependency inputs；实施改锁后须先审阅/提交。
for dependency_input in pyproject.toml uv.lock; do
  git ls-files --error-unmatch "$dependency_input" >/dev/null || \
    project_blocked_exit dependency_inputs untracked_dependency_input
  test "$(sha256sum "$dependency_input" | awk '{print $1}')" = \
       "$(git show "HEAD:$dependency_input" | sha256sum | awk '{print $1}')" || \
    project_blocked_exit dependency_inputs dirty_dependency_input
done
PROJECT_PYPROJECT_SHA256="$(sha256sum "$EMB/pyproject.toml" | awk '{print $1}')"
PROJECT_UV_LOCK_SHA256="$(sha256sum "$EMB/uv.lock" | awk '{print $1}')"

# 仅在非默认 train group 已写入 pyproject.toml 并进入 uv.lock 后执行；--locked 禁止改 lock。
# sync nonzero 必须原子写 blocked receipt，不能回退其他解释器。
project_resource_recheck initial_project_sync 20
PROJECT_SYNC_STDOUT="$M3_CONTRACT_DIR/project-sync.stdout.log"
PROJECT_SYNC_STDERR="$M3_CONTRACT_DIR/project-sync.stderr.log"
(set -o noclobber; : >"$PROJECT_SYNC_STDOUT"; : >"$PROJECT_SYNC_STDERR") || \
  project_blocked_exit project_sync log_capture_unavailable
if "${BOOTSTRAP_ENV_CMD[@]}" "$UV_BIN" sync --locked \
  --group test --group video --group train \
  --project "$EMB" --python "$PROJECT_PY" \
  >"$PROJECT_SYNC_STDOUT" 2>"$PROJECT_SYNC_STDERR"; then
  :
else
  project_sync_rc=$?
  if ((project_sync_rc >= 128)); then
    project_sync_reason=signal_or_interrupt
  elif /usr/bin/grep -aEqi \
    'ENOSPC|No space left on device|Disk quota exceeded|Input/output error|I/O error|timed out|timeout|Temporary failure in name resolution|Could not resolve host|connection reset' \
    "$PROJECT_SYNC_STDOUT" "$PROJECT_SYNC_STDERR"; then
    project_sync_reason=resource_network_or_io_error
  else
    project_sync_reason=locked_train_sync_not_ready
  fi
  project_blocked_exit project_sync "$project_sync_reason" "$project_sync_rc" \
    --stdout "$PROJECT_SYNC_STDOUT" --stderr "$PROJECT_SYNC_STDERR"
fi
project_resource_recheck post_project_sync 15
test "$PROJECT_PYPROJECT_SHA256" = \
     "$(sha256sum "$EMB/pyproject.toml" | awk '{print $1}')" || \
  project_blocked_exit post_project_sync pyproject_changed_during_sync
test "$PROJECT_UV_LOCK_SHA256" = \
     "$(sha256sum "$EMB/uv.lock" | awk '{print $1}')" || \
  project_blocked_exit post_project_sync uv_lock_changed_during_sync
PROJECT_CHECK_STDOUT="$M3_CONTRACT_DIR/project-sync-check.stdout.log"
PROJECT_CHECK_STDERR="$M3_CONTRACT_DIR/project-sync-check.stderr.log"
(set -o noclobber; : >"$PROJECT_CHECK_STDOUT"; : >"$PROJECT_CHECK_STDERR") || \
  project_blocked_exit project_sync_check log_capture_unavailable
if "${BOOTSTRAP_ENV_CMD[@]}" "$UV_BIN" sync --locked \
  --group test --group video --group train --check \
  --project "$EMB" --python "$PROJECT_PY" \
  >"$PROJECT_CHECK_STDOUT" 2>"$PROJECT_CHECK_STDERR"; then
  :
else
  project_check_rc=$?
  project_blocked_exit project_sync_check locked_environment_mismatch "$project_check_rc" \
    --stdout "$PROJECT_CHECK_STDOUT" --stderr "$PROJECT_CHECK_STDERR"
fi
if ! "${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -m policy.verify_project_training_environment \
  --expected-prefix "$PROJECT_PREFIX" --pyproject "$EMB/pyproject.toml" \
  --lock "$EMB/uv.lock" --required-group train --cuda-smoke \
  --expected-policy-root "$EMB/src/policy" --expected-editable-root "$EMB" \
  --require-pythonpath-unset \
  --expected-pyproject-sha256 "$PROJECT_PYPROJECT_SHA256" \
  --expected-lock-sha256 "$PROJECT_UV_LOCK_SHA256" \
  --sync-stdout "$PROJECT_SYNC_STDOUT" --sync-stderr "$PROJECT_SYNC_STDERR" \
  --sync-check-stdout "$PROJECT_CHECK_STDOUT" --sync-check-stderr "$PROJECT_CHECK_STDERR" \
  --data-min-free-gib 15 --root-min-free-gib 3 \
  --output "$PROJECT_TRAIN_RECEIPT"; then
  exit 3
fi

"${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -m policy.preflight \
  --data-root "$M2_RUN_DIR" \
  --manifest "$M2_RUN_DIR/collection-manifest.json" \
  --ledger "$M2_RUN_DIR/collection-ledger.jsonl" \
  --validation-report "$M2_RUN_DIR/validation-report.json" \
  --data-report "$M2_RUN_DIR/data-report.json" \
  --manual-review-trials "$M2_RUN_DIR/manual-review-trials.jsonl" \
  --replay-plan "$M2_RUN_DIR/replay/plan.json" \
  --replay-trials "$M2_RUN_DIR/replay/trials.jsonl" \
  --replay-summary "$M2_RUN_DIR/replay/summary.json" \
  --seed-protocol "$SEED_PROTOCOL" \
  --output "$M2_GATE"
```

pre-sync 的 `project_env_receipt.py` 是由 `/usr/bin/python3.12` 执行的 stdlib-only
publisher，且必须先从 tracked+HEAD-identical 内容固化 no-clobber snapshot；它只负责把
资源、sync 与失败日志原子写成 blocked terminal receipt，不导入项目 package。post-sync 的
`policy.verify_project_training_environment` 则必须从 locked editable root install 导入。
两者共同构成 fail-closed gate：所有 sync/resource/network/package/CUDA 分支在同一
no-clobber contract dir 原子发布一个
`project-train-env-receipt.v1` terminal receipt，并复用第 9.2 节唯一 canonical helper。
blocked receipt 保存 pre/post `pyproject.toml`/`uv.lock` hashes、sync/check logs hashes 和
reason；passed 只能在 post-sync space/ignore、absolute `.venv` prefix、未设置
`PYTHONPATH`、`policy` editable root、torch/torchvision/safetensors lock、sm_120 bf16
backward/save-load 全部复检后发布。当前仓库尚无该实现；
只读 probe 的结论仍是 training environment BLOCKED，而不是已有 receipt 或 ready。

official LeRobot setup/smoke 在另一个**未 source `env.sh`** 的 shell 中执行。launcher
不能依赖 bare `set -e`：从 no-clobber run dir 建立后起，每个 preflight 不足都必须先在
同目录原子发布 `status="blocked",passed=null` receipt 再退出 3。smoke 只写 evidence，
不能直接写 terminal receipt；passed receipt 只能由 launcher 在最后一次复检后发布：

```bash
set -uo pipefail
EMB="$(pwd -P)"
UV_BIN="$(command -v uv)"
JQ_BIN=/usr/bin/jq
LR_BOOTSTRAP_PY=/usr/bin/python3.12
CANONICAL_REL=src/policy/canonical_json.py
SOURCE_GATE_REL=src/policy/source_provenance.py
SOURCE_SCOPE_REL=configs/m3/lerobot-source-inputs.json
LR_LAUNCHER_REL=scripts/run_m3_lerobot.sh
LR_ENV="$EMB/.venv-lerobot"
LR_TMP="$EMB/cache/tmp-lerobot"
LR_RUN_ROOT="$EMB/runs/m3"
LR_RUN_DIR="$LR_RUN_ROOT/lerobot-adoption-001" # 每次尝试换新 id
LR_RECEIPT="$LR_RUN_DIR/lerobot_smoke_receipt.json"
LR_SMOKE_EVIDENCE="$LR_RUN_DIR/smoke-evidence.json"
SOURCE_MANIFEST="$LR_RUN_DIR/source-input-manifest.json"
WORKTREE_OBSERVATION="$LR_RUN_DIR/worktree-observation.json"
LOCK_FILE="$EMB/requirements.lerobot-act.lock.txt"

test -x "$JQ_BIN" || exit 2
test -x "$LR_BOOTSTRAP_PY" || exit 2
"$LR_BOOTSTRAP_PY" -c \
  'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || exit 2
LR_BOOTSTRAP_REALPATH="$(realpath -- "$LR_BOOTSTRAP_PY")"
LR_BOOTSTRAP_VERSION="$("$LR_BOOTSTRAP_PY" -c 'import platform; print(platform.python_version())')"
LR_BOOTSTRAP_SHA256="$(sha256sum "$LR_BOOTSTRAP_REALPATH" | awk '{print $1}')"

# 先建立最小 trust root；这里只要求 helper/launcher/scope 路径存在于 HEAD，随后一律从
# HEAD 固化 verifier。工作树内容是否相同由 source gate 判定并留下 blocked receipt。
for trusted_rel in \
  "$CANONICAL_REL" "$SOURCE_GATE_REL" "$SOURCE_SCOPE_REL" "$LR_LAUNCHER_REL"; do
  git -C "$EMB" ls-files --error-unmatch "$trusted_rel" >/dev/null || exit 2
  git -C "$EMB" cat-file -e "HEAD:$trusted_rel" || exit 2
done
mkdir -p "$LR_RUN_ROOT" || exit 2
mkdir "$LR_RUN_DIR" || exit 2  # no-clobber；此后所有终止路径必须写 receipt

# 后续 gate/publisher 只执行本次启动时的 immutable HEAD snapshots；工作树 helper 漂移不能
# 修改判定器。source scope snapshot 仍要求当前 repo 中同一路径与 HEAD 一致。
CANONICAL_TOOL="$LR_RUN_DIR/canonical_json.head.py"
SOURCE_GATE_TOOL="$LR_RUN_DIR/source_provenance.head.py"
SOURCE_SCOPE_SNAPSHOT="$LR_RUN_DIR/lerobot-source-inputs.head.json"
(set -o noclobber
 git -C "$EMB" show "HEAD:$CANONICAL_REL" >"$CANONICAL_TOOL"
 git -C "$EMB" show "HEAD:$SOURCE_GATE_REL" >"$SOURCE_GATE_TOOL"
 git -C "$EMB" show "HEAD:$SOURCE_SCOPE_REL" >"$SOURCE_SCOPE_SNAPSHOT") || exit 70
CANONICAL_TOOL_SHA256="$(sha256sum "$CANONICAL_TOOL" | awk '{print $1}')"
SOURCE_GATE_TOOL_SHA256="$(sha256sum "$SOURCE_GATE_TOOL" | awk '{print $1}')"
SOURCE_SCOPE_SNAPSHOT_SHA256="$(sha256sum "$SOURCE_SCOPE_SNAPSHOT" | awk '{print $1}')"
chmod 0444 "$CANONICAL_TOOL" "$SOURCE_GATE_TOOL" "$SOURCE_SCOPE_SNAPSHOT" || exit 70

SOURCE_GATE_ENV_CMD=(
  env -u VIRTUAL_ENV -u PYTHONPATH
  "PATH=/usr/local/bin:/usr/bin:/bin" "EMB=$EMB"
  "TMPDIR=$LR_RUN_DIR" "PYTHONDONTWRITEBYTECODE=1" "PYTHONHASHSEED=0"
)
SOURCE_BASELINE_VALID=0
SOURCE_RECHECK_SEQ=0
SOURCE_LAST_RECHECK=""
SOURCE_RECHECK_STDOUT=""
SOURCE_RECHECK_STDERR=""
SOURCE_RECHECK_LOGS_READY=0
SOURCE_BASELINE_MANIFEST_ID=""
SOURCE_BASELINE_COMMIT=""
SOURCE_BASELINE_PATH_SET_SHA256=""
SOURCE_BASELINE_INDEX_ENTRIES_SHA256=""
SOURCE_BASELINE_CONTENT_ENTRIES_SHA256=""
SOURCE_BASELINE_MANIFEST_SHA256=""
SOURCE_BASELINE_OBSERVATION_SHA256=""

trusted_snapshot_files_valid() {
  [[ "$(sha256sum "$CANONICAL_TOOL" | awk '{print $1}')" == "$CANONICAL_TOOL_SHA256" &&
     "$(sha256sum "$SOURCE_GATE_TOOL" | awk '{print $1}')" == "$SOURCE_GATE_TOOL_SHA256" &&
     "$(sha256sum "$SOURCE_SCOPE_SNAPSHOT" | awk '{print $1}')" == \
       "$SOURCE_SCOPE_SNAPSHOT_SHA256" ]]
}

source_manifest_valid() {
  trusted_snapshot_files_valid &&
    [[ -s "$SOURCE_MANIFEST" && -s "$WORKTREE_OBSERVATION" ]] &&
    "${SOURCE_GATE_ENV_CMD[@]}" "$LR_BOOTSTRAP_PY" "$SOURCE_GATE_TOOL" \
      validate-manifest --manifest "$SOURCE_MANIFEST" \
      --canonical-helper "$CANONICAL_TOOL" --require-schema lerobot-source-input-manifest.v1
}

cache_source_baseline() {
  SOURCE_BASELINE_MANIFEST_ID="$("$JQ_BIN" -er '.manifest_id' "$SOURCE_MANIFEST")" || return 1
  SOURCE_BASELINE_COMMIT="$("$JQ_BIN" -er '.git_commit' "$SOURCE_MANIFEST")" || return 1
  SOURCE_BASELINE_PATH_SET_SHA256="$("$JQ_BIN" -er '.path_set_sha256' "$SOURCE_MANIFEST")" || return 1
  SOURCE_BASELINE_INDEX_ENTRIES_SHA256="$(
    "$JQ_BIN" -er '.index_entries_sha256' "$SOURCE_MANIFEST"
  )" || return 1
  SOURCE_BASELINE_CONTENT_ENTRIES_SHA256="$(
    "$JQ_BIN" -er '.content_entries_sha256' "$SOURCE_MANIFEST"
  )" || return 1
  SOURCE_BASELINE_MANIFEST_SHA256="$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" || return 1
  SOURCE_BASELINE_OBSERVATION_SHA256="$(
    sha256sum "$WORKTREE_OBSERVATION" | awk '{print $1}'
  )" || return 1
}

source_baseline_artifacts_current() {
  ((SOURCE_BASELINE_VALID == 1)) && trusted_snapshot_files_valid &&
    [[ -s "$SOURCE_MANIFEST" && -s "$WORKTREE_OBSERVATION" ]] &&
    [[ "$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')" == \
       "$SOURCE_BASELINE_MANIFEST_SHA256" ]] &&
    [[ "$(sha256sum "$WORKTREE_OBSERVATION" | awk '{print $1}')" == \
       "$SOURCE_BASELINE_OBSERVATION_SHA256" ]] && source_manifest_valid
}

source_gate_recheck() {
  local stage="$1" rc
  SOURCE_RECHECK_SEQ=$((SOURCE_RECHECK_SEQ + 1))
  SOURCE_LAST_RECHECK="$LR_RUN_DIR/source-recheck-${SOURCE_RECHECK_SEQ}-${stage}.json"
  SOURCE_RECHECK_STDOUT="$LR_RUN_DIR/source-recheck-${SOURCE_RECHECK_SEQ}-${stage}.stdout.log"
  SOURCE_RECHECK_STDERR="$LR_RUN_DIR/source-recheck-${SOURCE_RECHECK_SEQ}-${stage}.stderr.log"
  SOURCE_RECHECK_LOGS_READY=0
  (set -o noclobber
   : >"$SOURCE_RECHECK_STDOUT"
   : >"$SOURCE_RECHECK_STDERR") || return 1
  SOURCE_RECHECK_LOGS_READY=1
  if ! trusted_snapshot_files_valid; then
    printf '%s\n' 'trusted source-gate snapshot hash mismatch' >>"$SOURCE_RECHECK_STDERR"
    return 3
  fi
  "${SOURCE_GATE_ENV_CMD[@]}" "$LR_BOOTSTRAP_PY" "$SOURCE_GATE_TOOL" verify-current \
    --repo "$EMB" --scope-config "$SOURCE_SCOPE_SNAPSHOT" \
    --scope-config-repo-path "$SOURCE_SCOPE_REL" \
    --baseline "$SOURCE_MANIFEST" --canonical-helper "$CANONICAL_TOOL" \
    --evidence-output "$SOURCE_LAST_RECHECK" \
    >"$SOURCE_RECHECK_STDOUT" 2>"$SOURCE_RECHECK_STDERR"
  rc=$?
  ((rc == 0)) || return "$rc"
  [[ -s "$SOURCE_LAST_RECHECK" ]] && "$JQ_BIN" -e \
    '.schema_version=="lerobot-source-recheck.v1" and .complete==true and
     .exact_match==true and .git_commit_match==true and .path_set_match==true and
     .index_entries_match==true and .content_entries_match==true' \
    "$SOURCE_LAST_RECHECK" >/dev/null
}

source_recheck_failure_json() {
  local evidence_sha="" stdout_sha="" stderr_sha="" logs_ready=false
  [[ -f "$SOURCE_LAST_RECHECK" ]] && \
    evidence_sha="$(sha256sum "$SOURCE_LAST_RECHECK" | awk '{print $1}')"
  ((SOURCE_RECHECK_LOGS_READY == 1)) && logs_ready=true
  [[ "$logs_ready" == true && -f "$SOURCE_RECHECK_STDOUT" ]] && \
    stdout_sha="$(sha256sum "$SOURCE_RECHECK_STDOUT" | awk '{print $1}')"
  [[ "$logs_ready" == true && -f "$SOURCE_RECHECK_STDERR" ]] && \
    stderr_sha="$(sha256sum "$SOURCE_RECHECK_STDERR" | awk '{print $1}')"
  "$JQ_BIN" -cn \
    --argjson logs_ready "$logs_ready" \
    --arg evidence_path "${SOURCE_LAST_RECHECK#"$EMB/"}" --arg evidence_sha256 "$evidence_sha" \
    --arg stdout_path "${SOURCE_RECHECK_STDOUT#"$EMB/"}" --arg stdout_sha256 "$stdout_sha" \
    --arg stderr_path "${SOURCE_RECHECK_STDERR#"$EMB/"}" --arg stderr_sha256 "$stderr_sha" \
    '{logs_ready:$logs_ready,evidence_path:$evidence_path,evidence_sha256:$evidence_sha256,
      stdout_path:$stdout_path,stdout_sha256:$stdout_sha256,
      stderr_path:$stderr_path,stderr_sha256:$stderr_sha256}'
}

source_manifest_failure_json() {
  local manifest_exists=false observation_exists=false trusted_snapshots_valid=false
  local actual_manifest_sha="" actual_observation_sha=""
  [[ -e "$SOURCE_MANIFEST" ]] && manifest_exists=true
  [[ -e "$WORKTREE_OBSERVATION" ]] && observation_exists=true
  if [[ "$manifest_exists" == true ]]; then
    actual_manifest_sha="$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')"
  fi
  if [[ "$observation_exists" == true ]]; then
    actual_observation_sha="$(sha256sum "$WORKTREE_OBSERVATION" | awk '{print $1}')"
  fi
  trusted_snapshot_files_valid && trusted_snapshots_valid=true
  "$JQ_BIN" -cn \
    --argjson manifest_exists "$manifest_exists" \
    --argjson observation_exists "$observation_exists" \
    --argjson trusted_snapshots_valid "$trusted_snapshots_valid" \
    --arg expected_manifest_sha256 "$SOURCE_BASELINE_MANIFEST_SHA256" \
    --arg actual_manifest_sha256 "$actual_manifest_sha" \
    --arg expected_observation_sha256 "$SOURCE_BASELINE_OBSERVATION_SHA256" \
    --arg actual_observation_sha256 "$actual_observation_sha" \
    '{manifest_exists:$manifest_exists,observation_exists:$observation_exists,
      trusted_snapshots_valid:$trusted_snapshots_valid,
      expected_manifest_sha256:$expected_manifest_sha256,
      actual_manifest_sha256:$actual_manifest_sha256,
      expected_observation_sha256:$expected_observation_sha256,
      actual_observation_sha256:$actual_observation_sha256}'
}

source_reference_json() {
  if ((SOURCE_BASELINE_VALID != 1)); then
    "$JQ_BIN" -cn '{source_input_manifest_id:null,source_commit:null,
      source_input_manifest:null,source_input_manifest_sha256:null,
      canonical_tool_snapshot_sha256:null,source_gate_tool_snapshot_sha256:null,
      source_scope_snapshot_sha256:null,
      worktree_observation:null,worktree_observation_sha256:null,
      last_recheck:null,last_recheck_sha256:null}'
    return
  fi
  local recheck_path="" recheck_sha256=""
  if [[ -n "$SOURCE_LAST_RECHECK" && -s "$SOURCE_LAST_RECHECK" ]]; then
    recheck_path="${SOURCE_LAST_RECHECK#"$EMB/"}"
    recheck_sha256="$(sha256sum "$SOURCE_LAST_RECHECK" | awk '{print $1}')"
  fi
  "$JQ_BIN" -cn \
    --arg source_input_manifest_id "$SOURCE_BASELINE_MANIFEST_ID" \
    --arg source_commit "$SOURCE_BASELINE_COMMIT" \
    --arg path_set_sha256 "$SOURCE_BASELINE_PATH_SET_SHA256" \
    --arg index_entries_sha256 "$SOURCE_BASELINE_INDEX_ENTRIES_SHA256" \
    --arg content_entries_sha256 "$SOURCE_BASELINE_CONTENT_ENTRIES_SHA256" \
    --arg manifest_path "${SOURCE_MANIFEST#"$EMB/"}" \
    --arg manifest_sha256 "$SOURCE_BASELINE_MANIFEST_SHA256" \
    --arg canonical_tool_snapshot_sha256 "$CANONICAL_TOOL_SHA256" \
    --arg source_gate_tool_snapshot_sha256 "$SOURCE_GATE_TOOL_SHA256" \
    --arg source_scope_snapshot_sha256 "$SOURCE_SCOPE_SNAPSHOT_SHA256" \
    --arg observation_path "${WORKTREE_OBSERVATION#"$EMB/"}" \
    --arg observation_sha256 "$SOURCE_BASELINE_OBSERVATION_SHA256" \
    --arg recheck_path "$recheck_path" --arg recheck_sha256 "$recheck_sha256" \
    '{source_input_manifest_id:$source_input_manifest_id,source_commit:$source_commit,
      path_set_sha256:$path_set_sha256,index_entries_sha256:$index_entries_sha256,
      content_entries_sha256:$content_entries_sha256,
      source_input_manifest:$manifest_path,source_input_manifest_sha256:$manifest_sha256,
      canonical_tool_snapshot_sha256:$canonical_tool_snapshot_sha256,
      source_gate_tool_snapshot_sha256:$source_gate_tool_snapshot_sha256,
      source_scope_snapshot_sha256:$source_scope_snapshot_sha256,
      worktree_observation:$observation_path,
      worktree_observation_sha256:$observation_sha256,
      last_recheck:(if $recheck_path=="" then null else $recheck_path end),
      last_recheck_sha256:(if $recheck_sha256=="" then null else $recheck_sha256 end)}'
}

publish_terminal_receipt() {
  local status="$1" passed_json="$2" stage="$3" reason="$4" evidence_json="$5"
  local payload_tmp document_tmp source_input_json source_failure_json

  # canonical snapshot 自身若被改写，已不存在可信 publisher；直接 exit70，不能制造 receipt。
  [[ "$(sha256sum "$CANONICAL_TOOL" | awk '{print $1}')" == \
      "$CANONICAL_TOOL_SHA256" ]] || return 1

  # terminal publisher 自身拥有最后一道 source gate。任何 requested failed/passed（以及普通
  # blocked）在 baseline 不 clean、commit/path/index/content 漂移或 evidence 不完整时都
  # 强制降为 source-lineage blocked；旧 receipt 因 git_commit 不同不能跨 clean commit 复用。
  if ! source_baseline_artifacts_current ||
     ! "$JQ_BIN" -e '.source_provenance_clean==true' "$SOURCE_MANIFEST" >/dev/null; then
    status=blocked; passed_json=null; stage=source_manifest
    reason=source_input_manifest_missing_malformed_or_snapshot_drift
    source_failure_json="$(source_manifest_failure_json)" || return 1
    evidence_json="$(
      "$JQ_BIN" -cn --argjson requested "$evidence_json" \
        --argjson source_manifest "$source_failure_json" \
        '{requested_terminal_evidence:$requested,source_manifest:$source_manifest}'
    )" || return 1
  elif ! source_gate_recheck "pre-receipt"; then
    status=blocked; passed_json=null; stage=source_recheck
    reason=source_input_drift_or_commit_change
    source_failure_json="$(source_recheck_failure_json)" || return 1
    evidence_json="$(
      "$JQ_BIN" -cn --argjson requested "$evidence_json" \
        --argjson source_recheck "$source_failure_json" \
        '{requested_terminal_evidence:$requested,source_recheck:$source_recheck}'
    )" || return 1
  fi
  source_input_json="$(source_reference_json)" || return 1
  payload_tmp="$(mktemp --tmpdir="$LR_RUN_DIR" .lerobot-payload.partial.XXXXXX)" || return 1
  document_tmp="$(mktemp --tmpdir="$LR_RUN_DIR" .lerobot-receipt.partial.XXXXXX)" || {
    rm -f -- "$payload_tmp"
    return 1
  }
  if ! "$JQ_BIN" -n \
      --arg schema_version "lerobot-smoke-receipt.v1" \
      --arg status "$status" --argjson passed "$passed_json" \
      --arg failure_stage "$stage" --arg reason_code "$reason" \
      --arg created_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      --argjson source_input "$source_input_json" --argjson evidence "$evidence_json" \
      '{schema_version:$schema_version,status:$status,passed:$passed,
        failure_stage:$failure_stage,reason_code:$reason_code,
        created_at_utc:$created_at_utc,source_input:$source_input,evidence:$evidence}' \
      >"$payload_tmp"; then
    rm -f -- "$payload_tmp" "$document_tmp"
    return 1
  fi
  # 唯一 Python helper 负责 parse/validate/canonicalize、排除 receipt_id 求 hash、
  # 注入 identity，并写 canonical document + 一个 LF；shell/jq 不实现第二套 canonical。
  if ! "$LR_BOOTSTRAP_PY" "$CANONICAL_TOOL" materialize-id \
    --input "$payload_tmp" --identity-field receipt_id --output "$document_tmp" || \
    ! sync -f "$document_tmp"; then
    rm -f -- "$payload_tmp" "$document_tmp"
    return 1
  fi
  if ! ln "$document_tmp" "$LR_RECEIPT"; then # same-filesystem atomic no-clobber publish
    rm -f -- "$payload_tmp" "$document_tmp"
    return 1
  fi
  rm -f -- "$payload_tmp" "$document_tmp"
  sync -f "$LR_RUN_DIR"
}

space_snapshot() {
  data_free_kib="$(df -Pk -- "$EMB" | awk 'NR==2 {print $4}')" || return 1
  root_free_kib="$(df -Pk -- / | awk 'NR==2 {print $4}')" || return 1
  [[ "$data_free_kib" =~ ^[0-9]+$ && "$root_free_kib" =~ ^[0-9]+$ ]]
}

space_evidence() {
  "$JQ_BIN" -cn --argjson data_free_kib "$data_free_kib" \
    --argjson root_free_kib "$root_free_kib" \
    '{data_free_kib:$data_free_kib,root_free_kib:$root_free_kib}'
}

check_ignored_directories() {
  local ignored_dir
  git check-ignore -q "$LR_ENV/" || return 1
  for ignored_dir in \
    "$LR_RUN_DIR" \
    "$EMB/cache/uv-lerobot" "$EMB/cache/pip-lerobot" "$LR_TMP" \
    "$EMB/cache/xdg-lerobot" "$EMB/cache/cuda-lerobot" \
    "$EMB/cache/ccache-lerobot" "$EMB/cache/pycache-lerobot" \
    "$EMB/cache/torch" "$EMB/hf"; do
    git check-ignore -q "$ignored_dir/" || return 1
  done
}

blocked_exit() {
  local stage="$1" reason="$2" evidence_json="$3"
  publish_terminal_receipt blocked null "$stage" "$reason" "$evidence_json" || exit 70
  exit 3
}

failed_exit() {
  local stage="$1" reason="$2" evidence_json="$3"
  publish_terminal_receipt failed false "$stage" "$reason" "$evidence_json" || exit 70
  exit 1
}

# launcher-start source gate：在 uv/venv/smoke 及任何 working-tree policy import 之前执行。
# expected source violations 也要原子留下 canonical manifest；tool crash/missing manifest 则
# terminal blocked receipt 中 source reference 为 null，且永远不能升级 failed/passed。
SOURCE_GATE_STDOUT="$LR_RUN_DIR/source-manifest.stdout.log"
SOURCE_GATE_STDERR="$LR_RUN_DIR/source-manifest.stderr.log"
(set -o noclobber; : >"$SOURCE_GATE_STDOUT"; : >"$SOURCE_GATE_STDERR") || exit 70
"${SOURCE_GATE_ENV_CMD[@]}" "$LR_BOOTSTRAP_PY" "$SOURCE_GATE_TOOL" snapshot \
  --repo "$EMB" --scope-config "$SOURCE_SCOPE_SNAPSHOT" \
  --scope-config-repo-path "$SOURCE_SCOPE_REL" \
  --canonical-helper "$CANONICAL_TOOL" --output "$SOURCE_MANIFEST" \
  --worktree-observation-output "$WORKTREE_OBSERVATION" \
  >"$SOURCE_GATE_STDOUT" 2>"$SOURCE_GATE_STDERR"
SOURCE_GATE_RC=$?
if source_manifest_valid && cache_source_baseline; then
  SOURCE_BASELINE_VALID=1
fi
source_gate_start_evidence="$(
  "$JQ_BIN" -cn --argjson exit_code "$SOURCE_GATE_RC" \
    --arg stdout_path "${SOURCE_GATE_STDOUT#"$EMB/"}" \
    --arg stderr_path "${SOURCE_GATE_STDERR#"$EMB/"}" \
    --arg stdout_sha256 "$(sha256sum "$SOURCE_GATE_STDOUT" | awk '{print $1}')" \
    --arg stderr_sha256 "$(sha256sum "$SOURCE_GATE_STDERR" | awk '{print $1}')" \
    '{exit_code:$exit_code,stdout_path:$stdout_path,stderr_path:$stderr_path,
      stdout_sha256:$stdout_sha256,stderr_sha256:$stderr_sha256}'
)"
if ((SOURCE_GATE_RC != 0 || SOURCE_BASELINE_VALID != 1)) ||
   ! "$JQ_BIN" -e '.source_provenance_clean==true' "$SOURCE_MANIFEST" >/dev/null; then
  blocked_exit launcher_start_source_gate source_input_not_clean_or_incomplete \
    "$source_gate_start_evidence"
fi
if ! source_gate_recheck launcher-start; then
  blocked_exit launcher_start_source_recheck source_input_drift_or_commit_change \
    "$source_gate_start_evidence"
fi

# Initial preflight：任何不足先留下 blocked receipt；尚未 mkdir cache 或安装。
if ! space_snapshot; then
  blocked_exit initial_space_preflight df_unreadable '{}'
fi
if ((data_free_kib < 20 * 1024 * 1024 || root_free_kib < 3 * 1024 * 1024)); then
  blocked_exit initial_space_preflight insufficient_space "$(space_evidence)"
fi
if ! check_ignored_directories; then
  blocked_exit initial_ignore_preflight directory_not_ignored "$(space_evidence)"
fi
if ! mkdir -p \
  "$EMB/cache/uv-lerobot" "$EMB/cache/pip-lerobot" "$LR_TMP" \
  "$EMB/cache/xdg-lerobot" "$EMB/cache/cuda-lerobot" \
  "$EMB/cache/ccache-lerobot" "$EMB/cache/pycache-lerobot" \
  "$EMB/cache/torch" "$EMB/hf"; then
  blocked_exit cache_setup cache_directory_creation_failed "$(space_evidence)"
fi

LR_ENV_CMD=(
  env -u VIRTUAL_ENV -u PYTHONPATH
  "PATH=$LR_ENV/bin:/usr/local/bin:/usr/bin:/bin"
  "EMB=$EMB" "PYTHONPATH=$EMB/src" "MUJOCO_GL=egl"
  "UV_CACHE_DIR=$EMB/cache/uv-lerobot"
  "PIP_CACHE_DIR=$EMB/cache/pip-lerobot"
  "TMPDIR=$LR_TMP" "XDG_CACHE_HOME=$EMB/cache/xdg-lerobot"
  "HF_HOME=$EMB/hf" "TORCH_HOME=$EMB/cache/torch"
  "CUDA_CACHE_PATH=$EMB/cache/cuda-lerobot"
  "CCACHE_DIR=$EMB/cache/ccache-lerobot"
  "PYTHONPYCACHEPREFIX=$EMB/cache/pycache-lerobot"
  "UV_PYTHON_DOWNLOADS=never" "PYTHONHASHSEED=0"
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
)

resource_recheck() {
  local stage="$1"
  if ! space_snapshot; then
    blocked_exit "$stage" df_unreadable '{}'
  fi
  if ((data_free_kib < 15 * 1024 * 1024 || root_free_kib < 3 * 1024 * 1024)); then
    blocked_exit "$stage" insufficient_space "$(space_evidence)"
  fi
  if ! check_ignored_directories; then
    blocked_exit "$stage" directory_not_ignored "$(space_evidence)"
  fi
}

run_logged_stage() {
  local stage="$1"
  shift
  STAGE_STDOUT="$LR_RUN_DIR/$stage.stdout.log"
  STAGE_STDERR="$LR_RUN_DIR/$stage.stderr.log"
  if ! (set -o noclobber; : >"$STAGE_STDOUT"; : >"$STAGE_STDERR"); then
    blocked_exit "$stage" log_capture_unavailable "$(space_evidence)"
  fi
  "$@" >"$STAGE_STDOUT" 2>"$STAGE_STDERR"
  STAGE_RC=$?
  return "$STAGE_RC"
}

stage_failure_evidence() {
  "$JQ_BIN" -cn --argjson exit_code "$STAGE_RC" \
    --arg stdout_path "${STAGE_STDOUT#"$EMB/"}" \
    --arg stderr_path "${STAGE_STDERR#"$EMB/"}" \
    --arg stdout_sha256 "$(sha256sum "$STAGE_STDOUT" | awk '{print $1}')" \
    --arg stderr_sha256 "$(sha256sum "$STAGE_STDERR" | awk '{print $1}')" \
    --argjson space "$(space_evidence)" \
    '{exit_code:$exit_code,stdout_path:$stdout_path,stderr_path:$stderr_path,
      stdout_sha256:$stdout_sha256,stderr_sha256:$stderr_sha256,space:$space}'
}

# 对 nonzero 先处理 resource/network/signal；return 0 仅表示可继续做确定性分类。
classify_common_indeterminate() {
  local stage="$1"
  if ! space_snapshot; then
    blocked_exit "$stage" df_unreadable '{}'
  fi
  if ((data_free_kib < 15 * 1024 * 1024 || root_free_kib < 3 * 1024 * 1024)); then
    blocked_exit "$stage" insufficient_space "$(stage_failure_evidence)"
  fi
  if ! check_ignored_directories; then
    blocked_exit "$stage" directory_not_ignored "$(stage_failure_evidence)"
  fi
  if ((STAGE_RC >= 128)); then
    blocked_exit "$stage" signal_or_interrupt "$(stage_failure_evidence)"
  fi
  if /usr/bin/grep -aEqi \
    'ENOSPC|No space left on device|Disk quota exceeded|Input/output error|I/O error|Read-only file system|timed out|timeout|Temporary failure in name resolution|Could not resolve host|connection reset' \
    "$STAGE_STDOUT" "$STAGE_STDERR"; then
    blocked_exit "$stage" resource_network_or_io_error "$(stage_failure_evidence)"
  fi
}

structured_install_failure_or_block() {
  local stage="$1" evidence_file="$2" failure_class evidence
  classify_common_indeterminate "$stage"
  if [[ ! -s "$STAGE_STDOUT" && ! -s "$STAGE_STDERR" ]]; then
    blocked_exit "$stage" empty_failure_logs "$(stage_failure_evidence)"
  fi
  if [[ ! -f "$evidence_file" ]] || ! "$JQ_BIN" -e \
    '.schema_version=="lerobot-environment-evidence.v1" and
     .complete==true and .deterministic_incompatibility==true and
     (.failure_class as $c |
       (["package_version_mismatch","wheel_source_mismatch","lock_mismatch",
         "platform_wheel_incompatible"] | index($c)) != null)' \
    "$evidence_file" >/dev/null; then
    blocked_exit "$stage" missing_or_unclassified_install_evidence \
      "$(stage_failure_evidence)"
  fi
  failure_class="$("$JQ_BIN" -r '.failure_class' "$evidence_file")"
  evidence="$(
    "$JQ_BIN" -cn --argjson process "$(stage_failure_evidence)" \
      --arg evidence_path "${evidence_file#"$EMB/"}" \
      --arg evidence_sha256 "$(sha256sum "$evidence_file" | awk '{print $1}')" \
      --arg failure_class "$failure_class" \
      '{process:$process,evidence_path:$evidence_path,
        evidence_sha256:$evidence_sha256,failure_class:$failure_class}'
  )"
  failed_exit "$stage" "$failure_class" "$evidence"
}

if run_logged_stage create_venv \
  "${LR_ENV_CMD[@]}" "$UV_BIN" venv "$LR_ENV" --python "$LR_BOOTSTRAP_PY"; then
  :
else
  classify_common_indeterminate create_venv
  blocked_exit create_venv indeterminate_venv_failure "$(stage_failure_evidence)"
fi
resource_recheck post_venv_preflight

if run_logged_stage locked_sync \
  "${LR_ENV_CMD[@]}" "$UV_BIN" pip sync --python "$LR_ENV/bin/python" "$LOCK_FILE"; then
  :
else
  classify_common_indeterminate locked_sync
  # 只有版本化白名单中的确定性 lock/resolution/platform-wheel 诊断才是 compatibility failed。
  if /usr/bin/grep -aEqi \
    'No solution found when resolving dependencies|No compatible distribution found|not compatible with the current platform|no wheel with a matching Python' \
    "$STAGE_STDOUT" "$STAGE_STDERR"; then
    failed_exit locked_sync dependency_resolution_incompatible "$(stage_failure_evidence)"
  fi
  blocked_exit locked_sync indeterminate_sync_failure "$(stage_failure_evidence)"
fi
resource_recheck post_sync_preflight

if run_logged_stage post_sync_interpreter \
  "${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -c \
  'import os,sys; from pathlib import Path; expected=Path(sys.argv[1]).resolve(); assert Path(sys.executable).resolve()==(expected/"bin/python").resolve(); assert Path(sys.prefix).resolve()==expected; assert "VIRTUAL_ENV" not in os.environ' \
  "$LR_ENV"; then
  :
else
  classify_common_indeterminate post_sync_interpreter
  blocked_exit post_sync_interpreter indeterminate_interpreter_failure \
    "$(stage_failure_evidence)"
fi
if run_logged_stage post_sync_pip_check \
  "${LR_ENV_CMD[@]}" "$UV_BIN" pip check --python "$LR_ENV/bin/python"; then
  :
else
  classify_common_indeterminate post_sync_pip_check
  blocked_exit post_sync_pip_check unclassified_pip_check_failure \
    "$(stage_failure_evidence)"
fi
POST_SYNC_PACKAGE_EVIDENCE="$LR_RUN_DIR/post-sync-package-evidence.json"
if run_logged_stage post_sync_packages \
  "${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -m policy.verify_lerobot_environment \
    --lock "$LOCK_FILE" --expected-prefix "$LR_ENV" \
    --source-input-manifest "$SOURCE_MANIFEST" \
    --evidence-output "$POST_SYNC_PACKAGE_EVIDENCE"; then
  :
else
  structured_install_failure_or_block post_sync_packages "$POST_SYNC_PACKAGE_EVIDENCE"
fi
if [[ ! -f "$POST_SYNC_PACKAGE_EVIDENCE" ]] || ! "$JQ_BIN" -e \
  '.schema_version=="lerobot-environment-evidence.v1" and
   .complete==true and .all_required_checks_passed==true' \
  "$POST_SYNC_PACKAGE_EVIDENCE" >/dev/null; then
  blocked_exit post_sync_package_evidence incomplete_or_invalid_package_evidence \
    "$(space_evidence)"
fi

# Smoke 只能发布 evidence；terminal status 仍不存在。launcher 独立捕获完整 logs。
if ! source_gate_recheck pre-smoke; then
  blocked_exit pre_smoke_source_recheck source_input_drift_or_commit_change '{}'
fi
if run_logged_stage compatibility_smoke \
  "${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -m policy.act_compat_smoke \
    --config "$EMB/configs/m3/lerobot-act-smoke.json" \
    --source-input-manifest "$SOURCE_MANIFEST" \
    --microbatch-ladder 4 2 1 --evidence-output "$LR_SMOKE_EVIDENCE"; then
  if ! source_gate_recheck post-smoke; then
    blocked_exit post_smoke_source_recheck source_input_drift_or_commit_change '{}'
  fi
else
  smoke_process_rc="$STAGE_RC"
  if ! source_gate_recheck post-smoke; then
    blocked_exit post_smoke_source_recheck source_input_drift_or_commit_change '{}'
  fi
  STAGE_RC="$smoke_process_rc"
  classify_common_indeterminate compatibility_smoke
  if [[ ! -s "$STAGE_STDOUT" && ! -s "$STAGE_STDERR" ]]; then
    blocked_exit compatibility_smoke empty_failure_logs "$(stage_failure_evidence)"
  fi
  if [[ ! -f "$LR_SMOKE_EVIDENCE" ]] || ! "$JQ_BIN" -e \
    '.schema_version=="lerobot-adoption-evidence.v1" and
     .complete==true and .all_required_checks_passed==false and
     .attempt_reconciliation_complete==true and (.attempts|type=="array") and
     (.failure_class as $c |
       (["batch1_cuda_oom","kernel_incompatible","nonfinite_loss",
         "save_load_mismatch","adapter_incompatible","cpu_path_incompatible"] |
        index($c)) != null)' \
    "$LR_SMOKE_EVIDENCE" >/dev/null; then
    blocked_exit compatibility_smoke missing_or_partial_failure_evidence \
      "$(stage_failure_evidence)"
  fi
  smoke_failure_class="$("$JQ_BIN" -r '.failure_class' "$LR_SMOKE_EVIDENCE")"
  smoke_failure_evidence="$(
    "$JQ_BIN" -cn --argjson process "$(stage_failure_evidence)" \
      --arg evidence_path "${LR_SMOKE_EVIDENCE#"$EMB/"}" \
      --arg evidence_sha256 "$(sha256sum "$LR_SMOKE_EVIDENCE" | awk '{print $1}')" \
      --arg failure_class "$smoke_failure_class" \
      '{process:$process,evidence_path:$evidence_path,
        evidence_sha256:$evidence_sha256,failure_class:$failure_class}'
  )"
  failed_exit compatibility_smoke "$smoke_failure_class" "$smoke_failure_evidence"
fi
SMOKE_STDOUT="$STAGE_STDOUT"
SMOKE_STDERR="$STAGE_STDERR"
resource_recheck post_smoke_preflight

# Final preflight：passed 只能在下面全部重验成功后，作为最后一步原子发布。
if run_logged_stage final_interpreter \
  "${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -c \
  'import os,sys; from pathlib import Path; expected=Path(sys.argv[1]).resolve(); assert Path(sys.executable).resolve()==(expected/"bin/python").resolve(); assert Path(sys.prefix).resolve()==expected; assert "VIRTUAL_ENV" not in os.environ' \
  "$LR_ENV"; then
  :
else
  classify_common_indeterminate final_interpreter
  blocked_exit final_interpreter indeterminate_interpreter_failure \
    "$(stage_failure_evidence)"
fi
if run_logged_stage final_pip_check \
  "${LR_ENV_CMD[@]}" "$UV_BIN" pip check --python "$LR_ENV/bin/python"; then
  :
else
  classify_common_indeterminate final_pip_check
  blocked_exit final_pip_check unclassified_pip_check_failure \
    "$(stage_failure_evidence)"
fi
FINAL_PACKAGE_EVIDENCE="$LR_RUN_DIR/final-package-evidence.json"
if run_logged_stage final_packages \
  "${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -m policy.verify_lerobot_environment \
    --lock "$LOCK_FILE" --expected-prefix "$LR_ENV" \
    --source-input-manifest "$SOURCE_MANIFEST" \
    --evidence-output "$FINAL_PACKAGE_EVIDENCE"; then
  :
else
  structured_install_failure_or_block final_packages "$FINAL_PACKAGE_EVIDENCE"
fi
resource_recheck final_resource_preflight
if [[ ! -f "$FINAL_PACKAGE_EVIDENCE" ]] || ! "$JQ_BIN" -e \
  '.schema_version=="lerobot-environment-evidence.v1" and
   .complete==true and .all_required_checks_passed==true' \
  "$FINAL_PACKAGE_EVIDENCE" >/dev/null; then
  blocked_exit final_package_evidence incomplete_or_invalid_package_evidence \
    "$(space_evidence)"
fi
if ! "$JQ_BIN" -e \
  '.schema_version=="lerobot-adoption-evidence.v1" and
   .complete==true and .all_required_checks_passed==true and
   .attempt_reconciliation_complete==true and
   (.selected_microbatch as $b | ([4,2,1] | index($b)) != null)' \
  "$LR_SMOKE_EVIDENCE" >/dev/null; then
  blocked_exit final_smoke_evidence incomplete_or_invalid_smoke_evidence "$(space_evidence)"
fi
smoke_sha256="$(sha256sum "$LR_SMOKE_EVIDENCE" | awk '{print $1}')"
final_evidence="$(
  "$JQ_BIN" -cn --argjson space "$(space_evidence)" \
    --arg smoke_evidence "${LR_SMOKE_EVIDENCE#"$EMB/"}" \
    --arg smoke_sha256 "$smoke_sha256" \
    --arg smoke_stdout "${SMOKE_STDOUT#"$EMB/"}" \
    --arg smoke_stdout_sha256 "$(sha256sum "$SMOKE_STDOUT" | awk '{print $1}')" \
    --arg smoke_stderr "${SMOKE_STDERR#"$EMB/"}" \
    --arg smoke_stderr_sha256 "$(sha256sum "$SMOKE_STDERR" | awk '{print $1}')" \
    --arg bootstrap_python "$LR_BOOTSTRAP_REALPATH" \
    --arg bootstrap_python_version "$LR_BOOTSTRAP_VERSION" \
    --arg bootstrap_python_sha256 "$LR_BOOTSTRAP_SHA256" \
    --arg package_evidence "${FINAL_PACKAGE_EVIDENCE#"$EMB/"}" \
    --arg package_evidence_sha256 "$(sha256sum "$FINAL_PACKAGE_EVIDENCE" | awk '{print $1}')" \
    --arg lock_sha256 "$(sha256sum "$LOCK_FILE" | awk '{print $1}')" \
    '{space:$space,smoke_evidence:$smoke_evidence,
      smoke_sha256:$smoke_sha256,smoke_stdout:$smoke_stdout,
      smoke_stdout_sha256:$smoke_stdout_sha256,smoke_stderr:$smoke_stderr,
      smoke_stderr_sha256:$smoke_stderr_sha256,bootstrap_python:$bootstrap_python,
      bootstrap_python_version:$bootstrap_python_version,
      bootstrap_python_sha256:$bootstrap_python_sha256,
      package_evidence:$package_evidence,
      package_evidence_sha256:$package_evidence_sha256,lock_sha256:$lock_sha256}'
)"
publish_terminal_receipt passed true complete all_checks_passed "$final_evidence" || exit 70
```

`publish_terminal_receipt` 使用同目录临时文件、fsync 和 hard-link no-clobber 发布，并只
调用启动时已验证并固化的 HEAD `canonical_json.py` snapshot 来计算 identity；jq 只构造
JSON value，不参与 canonical bytes/hash。publisher 在任何 terminal 状态前亲自运行
source recheck，并把 source manifest id/commit/path/index/content 摘要及最后 recheck hash
放进 receipt 顶层 lineage；漂移强制降为 blocked。blocked/failed/passed 三种 terminal
receipt 互斥；同一 adoption run 绝不覆盖。实现测试以含 `1.0`、Unicode、`false`、`null`
的 golden vector 同时覆盖 manifest/config/receipt，并断言没有第二套 shell/jq hash 实现。
launcher-start 验证成功时须立即缓存 manifest/observation 的 identity 与 hashes，后续 publisher
不得为构造 lineage 重新解析可能已被删除或损坏的文件。所有 recheck path/log-ready 变量在
`set -u` 下先初始化；每次 recheck 在任何 snapshot/source 检查早退前先分配 no-clobber
stdout/stderr，分配失败则写明确 `logs_ready=false` evidence。manifest 删除/非 JSON/hash
变化、source-gate/scope snapshot 漂移都由仍可信的 canonical snapshot 发布
`status="blocked",passed=null`，receipt 同时保存 cached expected 与 current existence/hash。
只有 canonical snapshot 自身 hash 漂移时已无可信 publisher，明确 exit 70，不伪造 receipt。

状态机测试必须注入 initial、post-venv、post-sync、post-smoke/final space/ignore 失败，
以及 venv/sync/smoke 的 ENOSPC、network timeout、signal、missing/partial smoke evidence
和 valid compatibility evidence。前几类必须留下 blocked/indeterminate terminal receipt，
只有最后一类可 failed；每条路径都断言 logs/evidence paths+hashes 完整、terminal receipt
唯一，且任何 final preflight 之前都不存在 `passed=true`。source harness 还必须强制注入：
启动后改单个 adapter byte、改变 index entry/path set、把 HEAD 移到另一 clean commit、传旧
receipt，以及只新增 scope 外个人文档。前四类都必须在首次 import/receipt 发布前 exit 3
并留下 blocked evidence；最后一类只改变 full-status 观测 hash/count，manifest identity 与
gate 结果保持不变。另在 `set -u` forced Bash harness 中覆盖 recheck path 初始空值、manifest
删除、malformed manifest 与 scope snapshot 漂移；上述四种在 canonical snapshot 仍可信时
均必须原子留下 blocked receipt/显式 evidence，不能以 rc 127/70 代替。

`m2-acceptance-001` 是规范命名示例；实际执行前必须把 `M2_RUN_DIR` 明确设为一个已存在
且不可覆盖的 M2 artifact 目录。CLI 不能自动选择“最新”目录，因为那会改变 data
lineage。M3 receipt 写入独立 `M3_CONTRACT_DIR`，不得回写或修改冻结的 M2 run。
LeRobot adoption 则使用独立 no-clobber `LR_RUN_DIR`；两者都必须位于 `$EMB/runs/m3/`。

### 10.2 normalization

```bash
"${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -m policy.compute_normalization \
  --manifest "$M2_RUN_DIR/collection-manifest.json" \
  --m2-gate "$M2_GATE" \
  --project-env-receipt "$PROJECT_TRAIN_RECEIPT" \
  --split train \
  --output "$M3_CONTRACT_DIR/normalization.json"
```

命令拒绝非 `train` split 和现有 output。
`--project-env-receipt` 不是路径装饰：normalization、BC/fallback train 和 project rollout
每次启动都重算 receipt identity，核对 `pyproject.toml`/`uv.lock` hashes、当前 absolute
interpreter/prefix 与精确 packages；任一漂移即 nonzero。共享 `PROJECT_ENV_CMD` 不能替代
这条 lineage gate。

### 10.3 overfit、正式训练、闭环

official 示例假设仍在第 10.1 节构造 `LR_ENV_CMD` 的同一个干净 shell；不得为方便另行
`source env.sh`。BC、fallback、M2 preflight、normalization 及其 rollout 则必须复用同一
`PROJECT_ENV_CMD`；更换 train seed 时在新 shell 重建该数组，不能进程内补设。

每个 official train/rollout 启动前，launcher 还必须从 adoption run 使用 immutable
`source_provenance.head.py`，而不是当前工作树 verifier，执行下面的 gate。它重算当前
commit/path set/index/content manifest，验证 adoption receipt 的 source 引用和外部 manifest
完全一致；调用前还要把 verifier/canonical/scope snapshot 当前 file hash 与 receipt 逐项
对账。任一不同返回 3/BLOCKED，因而旧 receipt 不能在另一 clean commit 复用。
随后 CLI 的 minimal `__main__` 在导入 adapter/train/rollout 实现前重复相同检查；
`policy/__init__.py` 必须 side-effect-free。scope 外个人文档不参与 identity。

```bash
LR_ADOPTION_DIR="$EMB/runs/m3/lerobot-adoption-001"
LR_BACKEND_RECEIPT="$LR_ADOPTION_DIR/lerobot_smoke_receipt.json"
LR_SOURCE_MANIFEST="$LR_ADOPTION_DIR/source-input-manifest.json"
LR_SOURCE_GATE_TOOL="$LR_ADOPTION_DIR/source_provenance.head.py"
LR_CANONICAL_TOOL="$LR_ADOPTION_DIR/canonical_json.head.py"
LR_SOURCE_SCOPE_SNAPSHOT="$LR_ADOPTION_DIR/lerobot-source-inputs.head.json"
LR_POLICY_GATE_CMD=(
  env -u VIRTUAL_ENV -u PYTHONPATH
  "PATH=/usr/local/bin:/usr/bin:/bin" "EMB=$EMB"
  "TMPDIR=$EMB/cache/tmp-lerobot" "PYTHONDONTWRITEBYTECODE=1" "PYTHONHASHSEED=0"
)
lerobot_source_gate_or_block() {
  [[ "$(sha256sum "$LR_SOURCE_GATE_TOOL" | awk '{print $1}')" == \
     "$(/usr/bin/jq -er '.source_input.source_gate_tool_snapshot_sha256' \
       "$LR_BACKEND_RECEIPT")" ]] || return 3
  [[ "$(sha256sum "$LR_CANONICAL_TOOL" | awk '{print $1}')" == \
     "$(/usr/bin/jq -er '.source_input.canonical_tool_snapshot_sha256' \
       "$LR_BACKEND_RECEIPT")" ]] || return 3
  [[ "$(sha256sum "$LR_SOURCE_SCOPE_SNAPSHOT" | awk '{print $1}')" == \
     "$(/usr/bin/jq -er '.source_input.source_scope_snapshot_sha256' \
       "$LR_BACKEND_RECEIPT")" ]] || return 3
  "${LR_POLICY_GATE_CMD[@]}" /usr/bin/python3.12 "$LR_SOURCE_GATE_TOOL" \
    verify-receipt-current --repo "$EMB" \
    --scope-config "$LR_SOURCE_SCOPE_SNAPSHOT" \
    --scope-config-repo-path configs/m3/lerobot-source-inputs.json \
    --canonical-helper "$LR_CANONICAL_TOOL" \
    --source-input-manifest "$LR_SOURCE_MANIFEST" \
    --backend-receipt "$LR_BACKEND_RECEIPT" || return 3
}

"${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -m policy.train \
  --config "$EMB/configs/m3/bc1-overfit.json" \
  --project-env-receipt "$PROJECT_TRAIN_RECEIPT" \
  --m2-gate "$M2_GATE" \
  --run-dir "$EMB/runs/m3/bc1-overfit-seed0"

lerobot_source_gate_or_block || exit 3
"${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -m policy.train \
  --config "$EMB/configs/m3/lerobot-act-overfit.json" \
  --backend-receipt "$LR_BACKEND_RECEIPT" \
  --source-input-manifest "$LR_SOURCE_MANIFEST" \
  --m2-gate "$M2_GATE" \
  --run-dir "$EMB/runs/m3/lerobot-act-overfit-seed0"

"${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -m policy.train \
  --config "$EMB/configs/m3/act-like-minimal-overfit.json" \
  --project-env-receipt "$PROJECT_TRAIN_RECEIPT" \
  --backend-receipt "$LR_BACKEND_RECEIPT" \
  --source-input-manifest "$LR_SOURCE_MANIFEST" \
  --m2-gate "$M2_GATE" \
  --run-dir "$EMB/runs/m3/act-like-minimal-overfit-seed0"
```

上面两个 chunk train 命令互斥：receipt `status="passed",passed=true` 时 CLI 只允许
`lerobot-act-overfit.json`，`status="failed",passed=false` 时只允许
`act-like-minimal-overfit.json`；`status="blocked"` 时两个命令都必须退出 3。
rollout 必须使用与 checkpoint backend 相同的 interpreter。以 official backend 为例：

```bash
lerobot_source_gate_or_block || exit 3
"${LR_ENV_CMD[@]}" "$LR_ENV/bin/python" -m policy.rollout \
  --checkpoint "$EMB/runs/m3/lerobot-act-overfit-seed0/selected-checkpoint" \
  --backend-receipt "$LR_BACKEND_RECEIPT" \
  --source-input-manifest "$LR_SOURCE_MANIFEST" \
  --seed-plan "$EMB/runs/m3/lerobot-act-overfit-seed0/overfit-seeds.json" \
  --k-exec 4 \
  --max-control-steps 600 \
  --output "$EMB/runs/m3/lerobot-act-overfit-seed0/rollout.jsonl"
```

BC/fallback rollout 的规范形态与训练使用同一 project launcher，例如：

```bash
"${PROJECT_ENV_CMD[@]}" "$PROJECT_PY" -m policy.rollout \
  --checkpoint "$EMB/runs/m3/bc1-overfit-seed0/selected-checkpoint" \
  --project-env-receipt "$PROJECT_TRAIN_RECEIPT" \
  --seed-plan "$EMB/runs/m3/bc1-overfit-seed0/overfit-seeds.json" \
  --k-exec 1 --max-control-steps 600 \
  --output "$EMB/runs/m3/bc1-overfit-seed0/rollout.jsonl"
```

正式 config 与 overfit config 是不同的 immutable JSON；CLI 展开后的完整 effective
config 总会复制到 run dir。run dir、checkpoint、日志和 summary 均 no-clobber。

退出码：`0` 成功完成命令，`2` CLI/config 错误，`3` M2/overfit gate 未通过，`4`
checkpoint/schema/hash 不兼容，`1` 其他运行失败。gate 未达标属于明确 nonzero，不是
“命令成功但指标较低”。

## 11. 日志与诊断

### 11.1 run 目录

```text
RUN_DIR/
  run.json                 # immutable effective config + provenance
  events.jsonl             # append-only machine-readable events
  checkpoints/
  validation.jsonl
  rollout.jsonl
  summary.json             # 仅在完整结束后原子发布
  failure.json             # 异常结束时发布，和 summary 二选一
```

每个 train event 至少记录：`schema_version`、run/policy id、wall time、global step、
samples seen、split、total/reconstruction/KL loss（适用时）、LR、grad norm、throughput、
loader wait、CUDA allocated/reserved/peak、AMP/determinism、data/normalization/config hashes。
stdout 只打印同一事件的短摘要，JSONL 是事实来源。

rollout 每个 episode 记录 seed、checkpoint id/hash、K_exec、control steps、success、
timeout、推理 p50/p95 原始样本链接、clip count 和异常；不得记录或消费 privileged 真值
作为动作输入。

### 11.2 故障码和第一诊断动作

| code | 现象 | fail-closed 诊断 |
|---|---|---|
| `M2_GATE_INCOMPLETE` | 计数/回放不足 | 停止；打开 receipt 的缺项，不训练 |
| `DATA_HASH_MISMATCH` | manifest 与 HDF5 不一致 | 停止；定位具体 path/expected/actual SHA |
| `M2_PATH_ESCAPE` | absolute/alias/`..`/symlink/hard-link duplicate 或 resolved root 外路径 | 打开 HDF5 前停止；报告 path class，不读取 root 外内容 |
| `SCHEMA_INCOMPATIBLE` | v1/shape/dtype/alignment 不符 | 停止；输出 validator code/location |
| `RAW_OBSERVATION_INCOMPATIBLE` | env/HDF5 raw dtype、shape 或 finite 不符 | 停止；不得在 adapter 外静默 cast |
| `POLICY_KEY_VIOLATION` | 缺键或额外 observation | 停止；打印 missing/extra，不忽略 |
| `SPLIT_LEAKAGE` | namespace 错、seed 重复或 stats 读到 val/M3/M4 | 停止；打印四集合交集和访问路径 |
| `CHECKPOINT_INCOMPATIBLE` | config/hash/env contract 不同 | 停止；逐字段 diff |
| `PROJECT_TRAIN_ENV_BLOCKED` | `.venv` 缺 locked train group/package/CUDA smoke，或 sync resource/network 中断 | 保存 blocked receipt；BC/fallback/normalization/rollout 均拒绝，不换解释器 |
| `LEROBOT_PREFLIGHT_BLOCKED` | space/ignore/resource/network/signal/证据不足，兼容性尚未确定 | 保存 logs/hash 与 blocked receipt；不启用 fallback |
| `LEROBOT_COMPAT_FAILED` | 完整证据命中 resolver/platform 或 smoke compatibility allowlist | 保存 evidence/log hashes；才可选择已定义 fallback |
| `BACKEND_RECEIPT_MISMATCH` | config 与 smoke 决策不符 | 停止；不得同时训练 official/fallback 后挑 test |
| `LEROBOT_SOURCE_LINEAGE_MISMATCH` | receipt/manifest 与当前 commit/path/index/content 任一不同 | 在任何 policy import 前 exit 3/BLOCKED；旧 receipt 不得跨 clean commit 复用 |
| `CUDA_OOM` | 显存不足 | official adoption 按 4→2→1 新进程阶梯；保留每档 config/峰值，batch1 才 fallback |
| `SOURCE_PROVENANCE_INCOMPLETE` | tracked dirty 或 relevant untracked/ignored runtime input 未 tracked+HEAD-identical/无 hash | formal 停止；无关个人文档只记 status hash/count，不要求处理 |
| `HDF5_WORKER_ERROR` | fork/handle/I/O 问题 | workers 改 0 做定位；检查 worker 是否 lazy-open |
| `NONFINITE_LOSS` | NaN/Inf | 停止；保存最后安全 checkpoint，检查 stats/AMP/输入 |
| `LOSS_PLATEAU` | 8 条轨迹不下降 | 先查 pre-action 对齐、mask、stats、LR、grads；不扩模型 |
| `OFFLINE_ONLY_FAILURE` | loss gate 过但闭环 `<7/8` | 维持失败；检查分布偏移/执行语义，不改成功定义 |
| `OVERFIT_ROLLOUT_INFRA_INVALID` | exact-8 run 中出现机器可判定基础设施异常 | 保留旧 run；新 run id 从第一个 frozen seed 起完整重跑，不聚合/单 trial 补跑 |
| `ACTION_CLIP_EXCESS` | 大量预测越界 | 报 clip 率/幅度；查 inverse normalization/action semantics |
| `DETERMINISM_UNSUPPORTED` | deterministic op 报错 | 停止 acceptance；记录具体 op，不静默关闭 |

诊断 run 必须写新的 run dir。不得覆盖失败证据、删掉低成功 rollout 或把一次 process
启动写成 gate 通过。

## 12. 未来实施文件边界与顺序

建议的最小文件结构；最终名称如需调整，接口和测试语义仍须保持：

```text
src/policy/__init__.py           # side-effect-free；允许 entrypoint 先做 lineage gate
src/policy/contracts.py          # raw-source/model-input contracts、config dataclasses
src/policy/preprocessing.py      # 唯一 env/HDF5 state dtype 适配边界
src/policy/canonical_json.py     # 第 9.2 节唯一 JSON identity/hash 实现
src/policy/source_provenance.py  # stdlib-only source scope manifest/recheck CLI
src/policy/project_env_receipt.py # pre-sync stdlib-only blocked receipt publisher
src/policy/hdf5_dataset.py       # streaming sample/chunk/pad mask
src/policy/normalization.py      # train-only Welford stats + transforms
src/policy/bc1.py                # 唯一 BC-1 网络
src/policy/act_adapter.py        # 唯一 LeRobot 边界 + adoption smoke
src/policy/act_like_minimal.py   # 仅 smoke 失败可启用的精确 fallback
src/policy/checkpoint.py         # atomic save/load/hash/compatibility
src/policy/backend_receipt.py    # blocked/failed/passed terminal receipt 原子发布
src/policy/preflight.py          # M2 gate CLI
src/policy/train.py              # overfit/formal runner CLI
src/policy/rollout.py            # H_pred/K_exec + PickPlace.step runner
scripts/run_m3_project.sh        # 唯一 project preflight/normalization/BC/fallback launcher
scripts/run_m3_lerobot.sh        # 不 source env.sh 的唯一 official launcher
tests/test_policy_contract.py
tests/test_hdf5_policy_dataset.py
tests/test_normalization_lineage.py
tests/test_bc1.py
tests/test_act_adapter.py
tests/test_policy_checkpoint.py
tests/test_policy_rollout.py
tests/test_backend_receipt.py
tests/test_source_provenance.py
requirements.lerobot-act.in
requirements.lerobot-act.lock.txt
configs/m3/m3-m4-seed-protocol-v2.json
configs/m3/lerobot-source-inputs.json
configs/m3/*.json
```

- [ ] **Task 1 — M2 receipt、seed 与唯一 adapter：** 先写第 7 节各 schema/hash/随机
  exact-20/trial 身份、manifest canonical relative path/root containment/no-symlink/unique inode、
  四 namespace、extra-key 和真实 env `float64[8] -> float32[8]` 的
  失败/一致性测试，再实现 contracts/preprocessing/preflight；无完整 receipt 时 train
  CLI 必须退出 3。测试所有 project CLI 都经同一 `PROJECT_ENV_CMD`，绝对 interpreter、
  cache/EGL/PYTHONHASHSEED/CUBLAS 任一偏差均 fail。
- [ ] **Task 2 — streaming Dataset 与 stats：** 用跨 episode 边界和 pad mask 回归测试
  驱动实现；用“val 极端值不改变 stats”证明 train-only lineage。
- [ ] **Task 3 — project train group 与 BC-1：** 在 `pyproject.toml`/`uv.lock` 唯一锁定
  torch 2.13.0、torchvision 0.28.0、safetensors 0.8.0 与 CUDA 13 wheel source；先测
  package discovery 包含 `policy*`、locked sync 保留 editable root install，且所有 project
  launcher 都 unset `PYTHONPATH`；post-sync 从 distribution metadata 与 `policy.__file__`
  证明导入来自 `$EMB/src/policy`。再测 project receipt/cache/space/prefix/sm_120 bf16
  backward/save-load、精确 shapes、共享 parameter identity 和一批梯度并实现网络。常规 CI
  不默认安装 train group。
- [ ] **Task 4 — checkpoint/provenance：** 先测 canonical self-id、单字节篡改、
  `1.0`/Unicode/false/null golden vector、tracked dirty、relevant untracked/ignored runtime
  input、schema/action bounds/H_pred mismatch 都 fail；另测无关 untracked 个人文档只改变
  status hash/count、不阻塞 source gate。再实现唯一 canonical helper、原子保存和
  inference/resume 两条加载路径。
- [ ] **Task 5 — backend adoption：** 先补并验证 `.venv-lerobot`/cache/temp ignore，使用
  不激活 `env.sh` 的 launcher 完成第 6.4 节 smoke；测试 4 OOM/2 pass 和 4+2 OOM/1 pass
  都选择 official，只有 batch1 OOM/真实兼容失败才 fallback，blocked 不得 fallback；
  `--no-deps` 环境不能通过。另注入 initial/post-venv/post-sync/post-smoke/final space、
  ignore、ENOSPC、network/signal、interpreter、package、missing/partial/valid-failure smoke
  evidence，证明不确定路径 blocked、只有完整白名单 compatibility evidence 才 failed，
  且 final gate 前不可能出现 passed。先 TDD 实现 versioned source scope 与 canonical
  manifest；launcher-start、pre/post-smoke、pre-receipt 都重算。强制 adapter 单 byte/index/
  path/commit 漂移和另一 clean commit 复用旧 receipt 全部 blocked；scope 外个人文档不阻塞。
- [ ] **Task 6 — runner：** 用 fake policy 验证 `H_pred=16,K_exec=4` 只执行前 4 步并
  重新观察；真实 runner 只能调用 `PickPlace.observe/step/success`。每个 official
  train/rollout 先由 immutable adoption verifier 重算 source manifest，再由 minimal CLI
  entry 重验 receipt lineage；commit/path/index/content mismatch 在 import 训练栈前退出 3。
- [ ] **Task 7 — M2 后的 overfit：** 读取通过的 M2/backend receipts 后分别执行 BC
  与选定 chunk backend；完整跑满 20k 后按最低全量 loss/最小 step 冻结一个 checkpoint，
  每个 run id 才且仅运行一次 exact-8 rollout。基础设施异常使整 run invalid，只能保留旧
  run 并用新 id 从头跑 8 个 seed，禁止 per-trial retry/跨 run 聚合。未同时达到 `>=95%`
  和 `>=7/8` 就停止。
- [ ] **Task 8 — 正式训练与锁定 IID：** 仅对已过 overfit 的 policy 启动 200/40；先
  冻结两策略 checkpoint 和 exact `10050..10149` plan，再运行一次 acceptance；M4
  `10000..10049` 保持未暴露，原样报告通过或失败。

每个 task 先跑目标测试，再跑最小相关 integration suite，最后审阅 scoped diff。本文
不授权 commit/push；是否提交由当次实施任务的用户授权决定。

## 13. 完成证据清单

M3 未来只有同时具备以下原始 artifacts 才能标记完成：

- 所有 M2/M3 runtime artifacts 位于被完整 ignore 的 `$EMB/runs/m2|m3/` no-clobber
  目录；full status hash/count 仍记录，但 admission 使用 tracked/source-input gate；
- M2 ledger/manifest/validation/data/manual-review/replay plan+trials+summary 及 gate receipt
  的逐项 hashes/identities；
- LeRobot adoption smoke receipt；通过则证明 official backend，失败则固定并解释
  `act_like_minimal_v1`，4→2→1 每次 attempt 都保留；blocked 不得伪装 fallback；passed
  receipt 另有 post-venv/post-sync/post-smoke/final preflight、完整 logs 与 smoke evidence
  hashes，以及绑定 commit/path/index/content 的 `lerobot-source-input-manifest.v1` 与末次
  recheck hash，且无提前发布或跨 clean commit 复用途径；
- `PROJECT_ENV_CMD` unset `PYTHONPATH`、绝对 `.venv` interpreter、数据盘 cache、EGL 和
  determinism 验证，`policy*` package discovery/locked editable root import 证据，以及
  train group/lock/CUDA smoke 的 passed project environment receipt；
- 真实 env float64 state 在唯一边界先验证再 cast float32、且与 HDF5 训练输入一致的测试；
- train-only normalization file/hash 和无泄漏测试；
- 两种 policy（BC + receipt 选定 chunk backend）的 effective config、锁文件、
  Git status hash/count、tracked/source-input/data/checkpoint provenance；
- BC 与选定 chunk backend 各自 8-episode 初始/selected-checkpoint 全量 loss、选择 receipt
  和单一有效 run id 的精确 8-seed/8-row trial-level rollout；
- 每个进入正式训练的 policy 均已满足 loss drop `>=95%`、闭环 `>=7/8`；
- `m3-m4-seed-protocol.v2`、未暴露 M4 plan，以及 frozen checkpoint 对 exact
  `10050..10149` 的 100-IID trial-level JSONL；
- 至少一个学习策略真实达到 `>=60/100`，否则 M3 明确 FAIL/PENDING；
- 离线 loss、闭环成功和 process/queue 状态分开表述。

截至 2026-08-24 当前候选，项目 train group/lock、editable `policy` 基础包、source
provenance helper、project verifier、development CUDA smoke 与 fixed-HEAD formal project
environment receipt 均已实现并验证。LeRobot isolate 因不可满足的 dependency-security gate
保持 BLOCKED，且未创建环境。M2 正式 200+40、训练数据人工审核、BC/ACT-like 网络、
checkpoint、overfit、100-IID 与 M4 runner/正式实验均尚未运行，因此 M3 总里程碑仍为
PENDING，不能由环境准入 receipt 代替。
