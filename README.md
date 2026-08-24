# Panda Reactive-IL：视觉模仿学习抓放与扰动恢复

> **这是本仓库唯一的项目与唯一执行入口。**
> 旧版 `DAY1.md`、`PLAN.md` 和七天计划已经停止执行，只保留作历史参考。
> 今后不按“每天学多少”推进，只按下面的工程门禁推进：上一关没有客观通过，就不进入下一关。

公开仓库：<https://github.com/rikfish163-rgb/embodied>

## 当前阶段与快速验证

当前已通过 **M0 工程底座**。M1 脚本专家的控制链和历史运行已经完成，但旧的
seed 0–99、99/100 与 100 个 MP4 绑定的是后来被修正的“中心点近似入盒”谓词，
不能当作严格旋转角点判定下的当前正式验收证据；新的 no-clobber 全视频 canonical
运行仍待生成。M2 的单 episode HDF5 契约、writer、reader 与 validator 已实现，
但 200+40 正式采集和完整门禁尚未完成，因此仓库不会声称已经完成模仿学习或 ACT。

干净环境使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/rikfish163-rgb/embodied.git
cd embodied
env -u PYTHONPATH uv sync --locked --group test --group video
env -u PYTHONPATH MUJOCO_GL=egl uv run --locked python scripts/check_foundation.py
env -u PYTHONPATH MUJOCO_GL=egl uv run --locked pytest tests -q
env -u PYTHONPATH MUJOCO_GL=egl uv run --locked python -m expert.evaluate \
  --output-dir runs/m1/acceptance-001 --record failures
```

本机已有由 `uv sync --locked` 创建的 `.venv/` 时，也可以：

```bash
source ./env.sh
python scripts/check_foundation.py
pytest tests -q
```

### MuJoCo 原生 3D 窗口

这不是浏览器界面，而是 MuJoCo 自带的 GLFW `Simulate` GUI：

```bash
source ./env.sh
MUJOCO_GL=glfw python -m env.native_viewer --seed 0
```

窗口包含原生左右控制栏、运行/暂停、关节与执行器面板，并支持鼠标旋转、
平移和缩放视角。需要显示 TCP 与法兰调试 site 时追加 `--debug-sites`：

```bash
MUJOCO_GL=glfw python -m env.native_viewer --seed 0 --debug-sites
```

该命令必须在有 `DISPLAY` 或 `WAYLAND_DISPLAY` 的桌面会话中运行；CI 和无窗口
回归测试继续使用 `MUJOCO_GL=egl`。原生 Viewer 用于人工检查场景与物理交互，
正式 episode runner 仍必须通过 `PickPlace.step()` 推进并累计成功保持时间。

默认 `pytest` 运行当前工程的活动回归套件，包含工程底座、M1、M2 数据契约、M4
评测工具与集成卫生测试。`src/robotics/tests/test_so3.py` 是停止执行的旧版手写练习，
五个函数仍为 `TODO(you)`；它不属于活动门禁，也没有被伪装成已经通过。

`uv.lock` 是公开仓库的跨机器安装契约；默认依赖仍保持最小，MP4 编码器位于
独立的 `video` 依赖组，CUDA/PyTorch 位于非默认 `train` 组。常规 CI 不安装 train 组，
避免为了验证 MuJoCo 场景下载整套训练栈。`requirements.lock.txt` 仅保留旧本机环境快照，
不参与解析、安装或正式 provenance。

当前交付形态是 **GitHub source checkout + locked editable install**，不是 PyPI/wheel
发行包：环境默认从 checkout 内的 `menagerie/franka_emika_panda` 读取已审计资产，独立
wheel 不包含这棵 repo-root 资产树。发布 GitHub release 不得把可构建 wheel 误写成已经
通过独立安装运行验收；若未来发布 wheel，必须另做资产打包、许可证和隔离安装 smoke。

## 许可证状态

仓库根目前没有项目级 `LICENSE`。公开可见不等于授予复制、修改或再分发许可；
`menagerie/franka_emika_panda/LICENSE` 只覆盖对应第三方资产，不能当作本项目源码许可。
在仓库所有者明确选择 Apache-2.0、MIT 或其他许可证之前，本项目只发布可审阅源码候选，
不创建“可复用正式发行版”tag，也不把第三方 Apache-2.0 外推到项目代码。

## 一句话目标

在 MuJoCo 中让 Franka Panda 仅根据 **前视 RGB + 腕部 RGB + 关节状态**，学习把随机位置的立方体抓起并放入随机位置的盒子；比较单步行为克隆与 action-chunk 策略，并定量评测新位置泛化、3 cm 中途扰动恢复和推理延迟。

```text
脚本专家
  → 同步的视觉/状态/动作数据集
  → 单步 BC 基线
  → ACT/action chunk 策略
  → IID / OOD / 扰动恢复 / 延迟评测
  → 演示视频 + 结果表 + 失败分析
```

这不是五个小项目，而是一个从数据到部署评测的完整项目。

## 为什么选它投实习

截至 2026-08-23，岗位与官方技术栈的共同要求已经很明确：

- [宇树当前招聘](https://www.unitree.com/cn/position/)直接要求机器人多模态数据管线、模型训练与集成、低延迟“感知—决策—执行”、异常恢复和完整评测体系；
- [宇树官方开源页](https://www.unitree.com/cn/mobile/opensource/)与 [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) 已把 LeRobot、ACT/DP、数据转换、训练、仿真/真机评测连成完整流程；
- [大疆 2027 校招热招方向](https://careers.dji.com/zh-CN/campus/hot-jobs?source=campus_hotjobs)强调视觉、多模态、三维物理世界、数据闭环与算法落地；
- [千寻智能当前校招实习入口](https://nwd4iy9rd2s.jobs.feishu.cn/campusofSpiritAI)覆盖数据算法、VLA 操作算法、后训练和模型评测等方向。

因此本项目优先对准：**机器人学习 / VLA 操作 / 具身数据 / 模型评测实习**。它不假装是“大模型预训练项目”，而是证明你能独立闭环数据、策略、控制和评测。

## 项目边界

### 必须做

- 单臂 Panda top-down pick-and-place；
- 训练时随机化 cube 和 box 位置，训练/验证/测试 seed 严格分开；
- 视觉策略只能看两路 RGB 与 Panda 本体状态；
- 对比单步 BC 与 action chunk；
- 固定扰动时机，把 cube 横移 3 cm，测恢复率；
- 报告成功率、Wilson 95% 区间、p95 推理延迟和失败类型。

### 暂时不做

- 不把完整 SO(3)/SE(3) 手写实现当作开工前置；
- 不上 ROS 2、Isaac Lab、强化学习或真机；
- 不训练 SmolVLA、π0/π0.5 等超出 8 GB 显存的模型；
- 不给单一固定任务硬塞一条恒定文本并称为 VLA。

如果主项目全部通过，再加“三种颜色 + 两种任务 + 自然语言指令”。在此之前，恒定语言输入没有信息量，只是简历包装。

## 固定任务协议

### 策略可见输入

| 键 | 内容 |
|---|---|
| `observation.images.front` | 128×128 RGB 前视图 |
| `observation.images.wrist` | 128×128 RGB 腕部视图 |
| `observation.state` | 7 维臂关节角 + 1 维夹爪状态 |

### 策略输出

`action` 是 8 维关节目标：7 维臂目标 + 1 维归一化夹爪目标。

- BC-1 每次预测下一步动作；
- ACT 预测未来 `H` 步动作块；
- `H_pred` 表示预测长度，`K_exec` 表示重新观察前实际执行多少步；两者不能混为一谈。

### 严格防泄漏

cube/box 真值位置、TCP 真值和随机化参数可以存到 `privileged/*`，供脚本专家与评测使用，但绝不能进入策略输入。训练数据与测试数据按 **episode seed** 切分，不能把同一条轨迹的相邻帧随机拆到两边。

### 成功定义

cube 必须完整进入盒子、落到底部附近，并在物理仿真中连续保持 1 秒，才计成功。
碰到盒沿、短暂经过盒内、悬在盒壁高度或直接瞬移物体都不算。
所有 episode runner 必须通过 `PickPlace.step()` 推进物理；直接绕过环境调用
`mujoco.mj_step()` 不会更新连续保持计数，不能用于正式判分。

## 五个工程门禁

### M1：脚本专家闭环

做什么：用 MuJoCo 的 site Jacobian + 阻尼最小二乘控制器完成“上方接近 → 下降 → 合爪 → 抬升 → 移动 → 松爪”。这里使用引擎 Jacobian，不先手写整套运动学。

**状态：历史运行已通过，当前严格谓词的正式证据待重跑。** commit `a97977a` 在
固定 seed 0–99 上记录过 99/100、2 次物理重抓恢复和 100 个可解码 MP4，但这些
数字只描述旧的中心点近似成功谓词，不能沿用为当前严格角点谓词的结果。控制器、
修订边界和新证据要求见 [M1 专家验收报告](docs/M1_EXPERT_REPORT.md)。

控制链不是“调用一个黑盒 IK”：

```text
TCP 位置/姿态误差
  → 统一到世界坐标系的 6D 任务增量 Δx
  → MuJoCo mj_jacSite 得到 J=[J_position; J_rotation]
  → Δq = Jᵀ(JJᵀ + λ²I)⁻¹Δx + 零空间回中项
  → 限速、关节范围和 command lead 裁剪
  → 8D action=[7 维关节目标, 1 维夹爪目标]
  → PickPlace.step() 推进 20 Hz 控制与 500 Hz 物理
```

专家只在控制阶段读取 cube/box/TCP 真值；reset 之后不写 cube 位姿，不提高摩擦，
也不绕过环境成功计数。批量评测会为每个 seed 写 success、failure stage、attempts
和最终位姿；追加 `--record all` 可生成逐 seed 的前视+腕部 MP4。

通过条件：

- 100 个固定但未手调的随机 seed 中成功不少于 90 次；
- 每次成功均由物理接触产生，不修改 cube 位姿作弊；
- 输出逐 episode 的 seed、成功/失败、失败阶段和轨迹视频；
- 你能画出 `末端误差 → Jacobian → 关节增量 → actuator target` 控制链。

### M2：可审计数据集

做什么：先采集 200 条成功训练轨迹和 40 条验证轨迹，20 Hz、双相机、HDF5 流式写盘；随后提供到 LeRobotDataset 键名的转换器。测试 episode 不提前写入训练数据，而是在评测时用独立 seed 生成。

通过条件：

- schema、dtype、shape、时间戳、动作范围、episode 边界检查为 0 错误；
- 随机回放 20 条专家动作，至少 18 条仍成功；
- 任取一个时刻，都能回答“这一帧观测对应哪一步动作”；
- 数据报告包含成功率、长度分布、动作分布和人工查看的 20 条失败/异常记录。

### M3：先做小基线，再做 ACT

做什么：

1. `BC-1`：共享 ResNet-18 编码两路图像，拼接 proprioception，MLP 输出下一步动作；
2. `ACT`：相同输入和数据，输出动作块；优先对接官方 LeRobot ACT，不为了“从零”重写成熟框架；
3. 正式训练前，先让模型过拟合 8 条轨迹，证明数据和 loss 链路正确。

通过条件：

- 8 条轨迹的训练 loss 相对初始值下降至少 95%，闭环回放至少成功 7/8；
- 从未见过的 100 个 IID seed 上，至少一个学习策略成功率达到 60%；
- checkpoint、配置、Git commit、数据版本和随机 seed 全部可追溯；
- 不能拿离线 action loss 代替闭环成功率。

### M4：2026 风格的反应性评测

固定同一 ACT checkpoint，只改变 `K_exec ∈ {1, 4, 8, 16}`。在抓取前的固定阶段把 cube 横移 3 cm，观察执行块越长时策略何时失去反应能力。主实验不得同时修改网络、数据或 checkpoint。

每个设置至少跑 50 个 trial，输出：

- IID 成功率；
- OOD 位置成功率；
- 扰动后的恢复率；
- Wilson 95% 置信区间；
- p50/p95 推理延迟与动作平滑度；
- `未对准 / 抓空 / 滑落 / 碰盒 / 超时` 失败分类。

项目最低通过线：M1、M2 全通过，学习策略 IID ≥ 60%，所有评测格子都有可复现的原始记录。优秀作品目标是 IID ≥ 80%、OOD ≥ 60%、扰动恢复 ≥ 50%，并用 3 个训练 seed 报告均值与波动。ACT 是否胜过 BC 必须由实验决定，不能预写结论。

### M5：求职交付

- 90–120 秒视频：正常成功、3 cm 扰动恢复、一个典型失败；
- 一张方法图：观测 → 编码器 → action chunk → 闭环执行；
- 一张主结果表和一张 `K_exec—恢复率` 曲线；
- 一页失败分析：最常见的三类失败、证据和下一步；
- 简历一句话必须包含任务、数据量、基线、评测次数和真实数字。

## 只学这五组材料

学习不是按视频数量验收。每看一组，必须立刻产出右栏结果，否则停止继续看。

| 顺序 | 精华材料 | 只学什么 | 学完的判分方式 |
|---|---|---|---|
| 1 | Northwestern Modern Robotics：[齐次变换](https://www.youtube.com/watch?v=vlb3P7arbkU)、[Space Jacobian](https://www.youtube.com/watch?v=KbI8HN3imtQ)、[数值 IK 1](https://www.youtube.com/watch?v=VhUA0jf7tI8)、[数值 IK 2](https://www.youtube.com/watch?v=24cXvgQl-nk) | 坐标变换、Jacobian 把末端速度映射到关节速度、迭代 IK；总计约 22 分钟 | 不看资料画出 M1 控制链，并解释为什么奇异位形要加阻尼 |
| 2 | Sergey Levine 的 [Berkeley CS185/285 2026](https://rail.eecs.berkeley.edu/deeprlcourse/)：只看 Lecture 2、3；视频用 [2023 官方播放列表](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps) 对应课次 | Behavior cloning、分布偏移、恢复数据、为什么离线 loss 不等于闭环成功 | 用自己的项目举例说明一个动作误差如何把后续观测带出训练分布 |
| 3 | PyTorch 官方 [Training with PyTorch](https://www.youtube.com/watch?v=jF43_wj_DCQ) | `Dataset/DataLoader`、train/eval、loss、optimizer、checkpoint | 独立读懂并讲清 8-episode overfit 的每一行日志 |
| 4 | Tony Zhao / Chelsea Finn 等人的 [ALOHA + ACT 官方项目页](https://tonyzhaozh.github.io/aloha/) 与 Hugging Face [LeRobot ACT 官方教程](https://www.youtube.com/watch?v=ft73x0LfGpM) | ACT 输入输出、action chunk、CVAE style、预测长度与执行长度 | 画出本项目 ACT 图，并回答 `H_pred=16, K_exec=4` 实际意味着什么 |
| 5 | Hugging Face [LeRobotDataset v3 文档](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3) 与 [宇树 unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) | 多相机视频、state/action/timestamp、episode metadata、转换/回放/训练接口 | 把本项目每个 HDF5 键一一映射到 LeRobot 键，并解释时间对齐 |

加分材料只在 M4 之后看：[LeRobot Real-Time Chunking](https://huggingface.co/docs/lerobot/main/en/rtc)。它主要针对高延迟 flow-matching VLA，不直接套到 ACT；我们只借它理解“平滑、推理延迟与反应性”的权衡。

## 你和 Codex 的分工

Codex 可以写大量代码。判断标准不是“每一行是否手敲”，而是关键决策是否由你做出、你是否审过 diff、跑过验证并能解释。

| 必须由你负责 | 可以交给 Codex |
|---|---|
| 定义成功、失败与作弊边界 | MuJoCo 环境封装、CLI 和重复样板 |
| 决定 observation/action/schema 与时间对齐 | HDF5 writer、LeRobot adapter、可视化工具 |
| 划分 seed，人工看数据并删除坏轨迹 | 单元测试、数据校验器、批量运行脚本 |
| 决定唯一变化的实验变量 | PyTorch 训练脚手架、checkpoint 管理 |
| 亲自看 rollout、分类失败、解释数字 | 画图、汇总原始结果、润色 README |
| 面试时从白板讲完整链路 | 扮演面试官追问、检查你的解释漏洞 |

每次让 Codex 改代码前，你先说清三件事：输入输出、通过条件、哪些信息禁止使用。每次改完，你必须看 `git diff`、跑最小测试，并用自己的话复述为什么通过。

## 本机适配

2026-08-24 实测：RTX 5070 Laptop 8151 MiB、30 GiB RAM。项目 `.venv` 已按
`uv.lock` 的非默认 `train` group 精确同步 torch 2.13.0+cu130、torchvision
0.28.0+cu130 与 safetensors 0.8.0；bf16 backward、双视角 ResNet18 和跨进程
safetensors development smoke 均通过。第一次同步因系统盘瞬时跌破 3 GiB 被安全终止，
第二次受控同步才完成；当前数据盘约余 31 GiB、系统盘约余 4.2 GiB。

候选 source/lock 进入提交 `0063226` 后，fixed-HEAD formal verifier 已通过：receipt 为
`sha256:e31baea7dc02a965dcf165c534fac275110af1bdd63337d746efcd7deb5cc373`，文件位于
`runs/m3/project-train-env-formal-20260824-003/formal-project-train-env-receipt.json`。
这只解除项目训练环境门禁，不代表 BC/ACT-like 网络、数据、checkpoint 或闭环训练已经
完成。`requirements.lock.txt` 只是旧本机快照，不是 resolver source；正式合同是
`pyproject.toml` + `uv.lock`。LeRobot isolate 因上游依赖安全修复版本与 v0.6.1 约束
不可满足而保持 `dependency_security` BLOCKED，`.venv-lerobot` 未创建。

- 图像固定 128×128，双相机；
- ACT 从 `batch_size=4`、bf16 AMP、`num_workers=2` 开始；
- HDF5 流式读取，禁止把整套图像载入 RAM；
- 所有缓存和数据继续放 `$EMB/cache`、`$EMB/hf`、`$EMB/runs`；
- PyTorch/torchvision 已进入独立 `train` group 与 `uv.lock`，并在 `.venv` 通过
  development CUDA smoke 与 fixed-HEAD formal environment receipt；实际训练仍须等待
  M2 正式数据及 M3 policy/training artifacts；
- LeRobot 继续使用隔离路线，但当前依赖安全门禁阻断，禁止创建或启用该环境；
- 不在系统盘创建新 Conda 环境，不安装 Isaac Lab 或大 VLA。

LeRobot 当前硬件指南给 ACT 类轻量 BC 的参考峰值是 batch 8 时约 2–6 GB，因此本机 8 GB 显存适合 ACT；SmolVLA 的参考是 10–16 GB，大型 VLA 是 24–40 GB，不适合作为本项目主线。

## 现在只做什么

当前只进入 **M2 可审计数据集**。`src/robotics/so3.py` 继续冻结；M1 的控制动作与
阶段标签将直接复用，但观测必须在动作执行前按 20 Hz 对齐写入，策略输入不能包含
cube/box/TCP 真值。

```bash
source ./env.sh
MUJOCO_GL=egl python -m expert.evaluate \
  --output-dir runs/m1/acceptance-001 --record failures
```

M2 单 episode HDF5 schema、时间对齐、train/val seed 契约，以及集合级
manifest/checksum/split 校验、action-only replay、PENDING 人工审核报告和 LeRobot 0.6.1
key adapter 都已实现。当前只完成 1-episode EGL smoke；下一步是在 clean fixed HEAD 下正式
采集 200 条成功训练轨迹与 40 条成功验证轨迹，做 cross-manifest 零错误、冻结 20 条 replay
至少成功 18 条，并逐条完成人工 verdict。在这些证据完成前，不创建训练模型。

## 第三方模型边界

`menagerie/` 是 Google DeepMind MuJoCo Menagerie 的 Panda 快照，由外层 `embodied` 仓库直接管理，没有第二个 `.git`。来源和固定 revision 见 `menagerie/VENDORED_REVISION`；不直接修改上游 XML/mesh，自定义 TCP、相机与任务场景都在 `src/env/` 中程序化注入。
