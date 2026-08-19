# embodied Panda 全链路项目完整计划

> **项目定位**：一周承重墙 + 可扩展底座。
> 目标不是漂亮 demo，是四个你能在面试里讲透四十分钟的东西：
> 自写运动学、自定数据 schema、从零 action-chunk 策略、严谨评测协议。
>
> **目标岗位**：千寻智能 / 智元 / 星海图 一档的「VLA操作算法 / 数据算法 / 模型评测」
> **约束**：纯仿真（无真机）· RTX 5070 Laptop **8GB** · 一周 · 每天 13h+
> **开工**：2026-08-18

---

## 0. 环境事实（Day 0 实测，不是估计）

| 项 | 值 | 影响 |
|---|---|---|
| 项目根 | `/media/hetaisheng/044A81D94A81C83E/embodied` | 数据盘 NVMe，ntfs3；2026-08-19 实测系统盘仅余约 1.2G |
| 弃用磁盘 | `sda1` (USB, 29G) | ext4 块位图损坏，只写过 4MB 就坏 → **设备本身有问题，别再用** |
| GPU | RTX 5070 Laptop, **8151 MiB**, sm_120 (Blackwell) | 驱动 580.173；torch 必须 CUDA 12.8+ 构建 |
| torch | 2.13.0 + CUDA 13 | PyPI 默认构建已含 sm_120 |
| MuJoCo | 3.11.0 | 有 `MjSpec`（程序化改模型），有 `mj_jacSite`（Day2 判卷器） |
| 渲染 | EGL，**1501 fps/相机** | 400 ep × 80 步 × 双相机 ≈ **43 秒** |
| RAM | 30G（2026-08-19 可用约 8.9G），swap 15G（已用约 11G） | ⚠️ 训练前必须关 Chrome/微信/VSCode 等重进程，先运行 `free -h` |
| 网络 | `raw.githubusercontent` 被墙；`pypi.tuna` 5MB/s；GitHub API 通 | 装包走清华源，拉 GitHub 文件走 `gh api` |

### Day 0 测出的四个数（Day1-3 会直接用）

1. **`hand` 局部 `+z` 指向世界 `-z`** —— 夹爪朝下时局部 z 轴就是抓取方向。
   `hand` 带 `quat="0 0 0 1"`（绕 z 转 180°），**这是最容易搞错的地方**。
   腕部相机因此需 `quat=(0,1,0,0)` 翻转 180° 才能看向抓取区。
2. **qpos 布局**：臂 `[0:7]` · 夹爪 `[7:9]` · cube freejoint `[9:16]` · 干扰物 `[16:23]`/`[23:30]`
   （代码里用 `body_jntadr` 反查 + 断言，不硬编码）
3. **夹爪**：滑动关节沿 hand 局部 y 轴，单指行程 0–0.04m（开口 8cm），
   执行器 `actuator8` ctrl 范围 **0–255**（不是 0–1）
4. **TCP**：flange + 0.1034m（Franka 官方值），实测 flange z=0.926 / TCP z=0.8226，差值精确吻合

### 每次开工

```bash
source /media/hetaisheng/044A81D94A81C83E/embodied/env.sh
```

---

## 1. 五个锁定交付物

按**不可砍 → 可砍**排序。东西一定会出问题，从底部开始砍。

| # | 交付物 | 为什么值钱 |
|---|---|---|
| **1** | **action chunk 消融 + 扰动恢复实验 + eval 协议** | 最不可能是 AI 代做的；对口千寻「模型评测工程师」实岗 |
| **2** | **自写 FK/Jacobian + 与 `mj_jacSite` 的 1e-9 级验证** | 防守性最强；区分"懂机器人"和"会调模型" |
| **3** | 数据 schema + 脚本专家 | 对口「数据算法工程师」；schema 是你定的，你就知道每个字段为什么在 |
| **4** | Mini-ACT Transformer 版 | 可降级为 CNN+MLP |
| **5** | 零空间投影、三色语言条件、演示视频 | 加分项 |

---

## 2. 逐日计划

### Day 0 ✅ 已完成（今晚 3h）

环境 · Panda 模型 67/67 · TCP site · 双相机 · 任务场景（台面/立方体/容器/干扰物）· 防泄漏验证 · `so3.py` 骨架 + 24 个测试

---

### Day 1（13h）：`so3` → `se3` → FK

全程纯 NumPy，**不依赖 torch**。

| 任务 | 时长 | 验收 |
|---|---|---|
| `so3.py` 5 个函数 | 5h | **24 个测试全绿** |
| `se3.py` | 3h | `T @ inv(T) == I`；`Ad`/`ad` 与数值结果一致 |
| `kinematics.py` FK | 4h | **你的 FK vs `data.site_xpos` 误差 < 1e-9** |
| 缓冲 | 1h | |

**`so3.py` 三个必须自己处理的分支**（否则 `test_small_angle_no_nan`、`test_near_pi_no_nan`、`test_log_tolerates_noisy_rotation` 必 fail）：

| 陷阱 | 症状 | 对策 |
|---|---|---|
| θ→0 | `sinθ/θ` 和 `(1-cosθ)/θ²` 变 0/0 | 泰勒展开：`1-θ²/6`、`1/2-θ²/24`；阈值 ~1e-8（想清楚为什么不能取 1e-16） |
| θ→π | `θ/(2sinθ)` 爆炸，`R-Rᵀ→0` 退化 | 独立分支：`(R+I)/2 = ω̂ω̂ᵀ`，取模长最大的列归一化 |
| 浮点噪声 | `(tr(R)-1)/2` 越界 → `arccos` 返回 nan | 先 `clip` |

**推导任务（纸上先做，不看参考）**：用 `hat(ω̂)³ = -hat(ω̂)` 把 `exp` 的泰勒级数折叠成 Rodrigues 公式。

**FK 的做法**：从 MuJoCo 模型读 `jnt_axis` / `body_pos` / `body_quat`，**自己重建正运动学链**，再和 `mj_forward` 的结果对。
这是全项目最干脆的一次验证——你在重新推导 MuJoCo 内部做的事，用它自己的模型数据。

---

### Day 2（13h）：Jacobian → DLS IK → 零空间

| 任务 | 时长 | 验收 |
|---|---|---|
| 数值 Jacobian（FK 有限差分） | 2h | 与解析版一致 |
| 解析几何 Jacobian | 3h | **vs `mujoco.mj_jacSite` 误差 < 1e-9** |
| DLS IK | 4h | 100 个随机可达目标收敛率 > 95% |
| λ 扫描 | 1h | 记录 λ ∈ {0, 1e-4, 1e-2, 1e-1} 在奇异位姿附近的行为（λ=0 会炸，要亲眼看到） |
| 零空间投影 | 2h | 同一末端轨迹，加零空间项前后的关节限位越界次数对比 |
| 缓冲 | 1h | |

**核心公式**（自己推，别抄）：

- 几何 Jacobian，转动关节 i：`J_v[:,i] = z_i × (p_e − p_i)`，`J_w[:,i] = z_i`
- 阻尼最小二乘：`q̇ = Jᵀ(JJᵀ + λ²I)⁻¹ v`
- 零空间投影：`q̇ = J⁺v + (I − J⁺J)q̇₀`

**为什么选 Panda**：7 自由度 → 有 1 维零空间。6 轴臂给不了你这个话题，而"零空间你怎么用"是高频面试题。

> ⚠️ **Day 2 22:00 检查点**：若 DLS IK 仍不收敛 → 立刻降级：
> 保留手写 FK + Jacobian 及其验证（核心资产），IK 改用 `scipy.optimize.least_squares`，**砍掉零空间**。
> 不要在这里超支，Day 3 依赖 IK 能用。

---

### Day 3（13h）：脚本专家 + 数据集

**纯仿真路线的最大红利**：没有遥操硬件，但你可以用刚写的 IK 造专家。这不是妥协，是把 Day1-2 的成果立刻变现。

**专家状态机**（笛卡尔路点 + 你的 DLS IK）：

```
接近位(物体上方 8cm) → 下降 → 闭夹爪 → 抬升 → 移到盒上方 → 松开
```

**随机化范围 = 训练分布定义**（eval 的 L1 外推档必须落在它之外，所以要显式记录）：

| 参数 | 范围 |
|---|---|
| cube x | 0.42 – 0.60 |
| cube y | −0.16 – 0.16 |
| cube yaw | −π/4 – π/4 |
| box x | 0.44 – 0.56 |
| box y | 0.26 – 0.36 |
| 光照 / 桌面纹理 / 相机抖动 | 待定，Day3 敲定并记进 `PROTOCOL.md` |

**数据 schema（★ 你自己定，这是防守要点）**：

| 字段 | 内容 |
|---|---|
| `obs/wrist_rgb` | 128×128×3 uint8 |
| `obs/front_rgb` | 128×128×3 uint8 |
| `obs/qpos` | 7 关节角 |
| `obs/qvel` | 7 关节速度 |
| `obs/gripper` | 夹爪开合（归一化到 [0,1]，注意 ctrl 是 0–255） |
| `obs/ee_pose` | 末端 SE(3) —— **用你的 FK 算，不读 MuJoCo** |
| `action` | `[arm_target(7), gripper_cmd(1)]`，共 8 维绝对目标，H 步一组；夹爪命令归一化到 `[0,1]`，在环境边界再映射到 `ctrl[7]∈[0,255]` |
| `lang` | 指令字符串 |
| `meta` | 随机化参数、成功标志、episode 长度、seed |

**存储**：单个 HDF5，uint8，mmap 读。400 episodes ≈ 3.1GB（RAM 紧张，别一次全载）。

**验收**：400 episodes · 专家成功率 > 90% · 数据体检报告出图（动作幅值分布、episode 长度直方图、时间轴对齐检查、退化 episode 检出数）

**可选升级（+1.5h，强烈建议）**：三色立方体 + 指令 `"pick the {red/blue/green} cube"`。
成本很小，但让**语言维度不再是假的**——否则 Day 6 的 L5 指令改写档只能诚实地标注"退化，不测"。

> ⚠️ **Day 3 检查点**：专家成功率 < 80% → 简化任务：取消朝向约束，只做 top-down 抓取。

---

### Day 4（13h）：Mini-ACT Stage A — CNN + MLP

先把"chunk 是什么、怎么执行、时间轴怎么对齐"跑通。**这一步就能闭环。**

```
ResNet18(两路相机共享权重) + proprio(qpos+gripper) → concat → MLP → 输出 H×8 动作块
loss: L1
```

**必须想清楚的三件事**（面试会问）：

1. 为什么不是只预测 `a_t`？（长时任务的误差累积）
2. chunk 太长为什么变 open-loop？
3. 推理延迟怎么造成动作不连续？→ 这决定你要不要做 temporal ensemble

**验收**：闭环跑通，L0 成功率 > 40%

**显存预算（8GB）**：`bs=16` · AMP bf16 · 128² 双相机 · ResNet18 → 预估占用 < 4GB。
参数量约 17M，对 8GB 是很轻的负载。

---

### Day 5（13h）：Mini-ACT Stage B — Transformer + 挂夜跑

```
图像 token + proprio token → Transformer encoder(d=256, 4层, nhead=8)
                            → H 个 learned action query → decoder → H×8
```

**明确跳过 CVAE**（原 ACT 的 style variable）。理由写进报告：
CVAE 是为建模人类演示的多模态性；**脚本专家是单模态的，CVAE 没有可学的东西**。
—— 这句话本身就是强面试答案，证明你读懂了论文而不是照抄。

**验收**：成功率超过 Stage A

**先把两个量分开**：

- `H_pred`：策略一次预测多少步；
- `K_exec`：闭环重新观测前实际执行多少步，且 `K_exec ≤ H_pred`。

**Day 5 收尾必做**：先训练一份 `H_pred=64` 的冻结 checkpoint，供 Day 6 只改变
`K_exec ∈ {8,16,32,64}`。若资源与时间允许，再挂
`H_pred ∈ {8,16,32,64} × 3 seeds`，且评测时固定 `K_exec=8`。
若 `H_pred=64` 不收敛，就使用已收敛的最大 `H_pred`，并相应截断 `K_exec` 网格，
不能为了保住预设图表而换用失败 checkpoint。

单次训练耗时必须以 Day 4/5 的实测为准，不能把预估写成结果。

> ⚠️ **Day 5 22:00 检查点**：Transformer 不收敛 → 用 Stage A 跑消融。
> **消融不可砍，网络结构可降级。**

---

### Day 6（13h）：这一天决定项目值多少钱

#### 6.1 预测长度与执行长度的受控消融（带误差棒）

控制频率 20Hz 时，真正的 **open-loop 时长 = `K_exec / 20` 秒**，不是自动等于
`H_pred / 20`。只有整块执行（`K_exec=H_pred`）时两者才相等。

主实验固定同一个 `H_pred=64` checkpoint，只改变 `K_exec`：

| K_exec | open-loop 时长 | 预期现象 |
|---|---|---|
| 8 | 0.4s | 推理调用频繁，latency 占比高 |
| 16 | 0.8s | 通常最优区 |
| 32 | 1.6s | 平滑但反应变钝 |
| 64 | 3.2s | 扰动后基本救不回来 |

这样恢复率变化可以归因于重规划频率，而不是同时换了模型输出头。若夜间训练完成，
再做次实验：固定 `K_exec=8`，比较 `H_pred∈{8,16,32,64}`，隔离预测长度本身的影响。
所有组各 50 trials，报告成功率 + Wilson 95% 区间；运行时间以实测为准。

#### 6.2 ★ 杀手实验：执行中途扰动物体，测恢复率 vs K_exec

在固定 seed、固定任务阶段且立方体尚未被夹持时，把立方体横向挪 3cm，记录恢复率；
不要在夹爪已接触/夹持时瞬移物体，以免把不物理的穿透冲量混进实验。
**预期：恢复率随 `K_exec` 增大而下降，且在「剩余 open-loop 时长 > 扰动后可用反应时间」时崩塌。**

极少有学生做这个实验，但它是"为什么 chunk 不能太长"的**直接实验证据**。
面试官问“为什么不整块执行 64 步”，你的答案不是“论文这么写”，而是同一 checkpoint、
只改变 `K_exec` 的恢复率曲线。若被问 `H_pred`，再展示固定 `K_exec` 的次实验，避免混淆。

4 个 `K_exec` × 50 trials；运行时间、触发时刻与有效 trial 数全部按实测记录。

#### 6.3 eval 协议（最优 H × 6 档 × 50 trials）

```
L0 基准       训练分布内
L1 位姿外推   cube 位置落在训练范围外 20% 区域
L2 干扰物     +2 个非目标立方体
L3 光照/纹理  侧光、照度 −50%、桌面换纹理
L4 相机扰动   相机位姿抖动超出训练随机化范围
L5 指令改写   "red" → "crimson"（★ 仅在做了三色升级时才测）

成功判据：立方体完全落入容器内且未掉落，300 步超时算失败
统计：成功率 + Wilson 95% 置信区间
失败分类：{抓空, 滑落, 碰撞, 未对准, 放置失败, 超时}
```

**关键**：若没做三色升级，就**明确写出"语言维度在单任务上退化，故不测"**。
诚实地拒绝一个假指标，比硬凑一个数字更有说服力 —— 2026 年这个行业刚因榜单可信度出过事
（RoboArena 六月的刷榜争议、榜单被拆成 Official 与全量两版），你报告里有这一句，分量很重。

**Day 6 eval 总预算**：约 1.5h 计算 + 大量分析时间。

---

### Day 7（13h）：交付

- `README.md`：一句话结论 + 3 张核心图 + 复现命令
- `REPORT.md` 8–12 页：方法 · 验证残差 · 消融 · 失败分析 · **方法学局限**
- 演示视频：成功案例 **+ 失败案例**（放失败案例比只放成功更专业）
- `LOGBOOK.md` 整理：每个决策的"为什么"
- 简历条目草稿
- **扩展接口标注**（你会继续推进，必做）：
  `policies/` 留 VLA 插入点 · `deploy/` 留 HIL 接管点 · `control/` 留阻抗控制接口

---

## 3. 仓库结构

```
embodied/
├── env.sh                     # source 我
├── PLAN.md  README.md  REPORT.md  LOGBOOK.md
├── PROTOCOL.md                # ★ 采集协议
├── EVAL_PROTOCOL.md           # ★ 评测协议
├── menagerie/                 # Panda 模型 (67 mesh, 已就位)
├── src/
│   ├── robotics/              # ★★ Day1-2 你的手写代码
│   │   ├── so3.py             #    ✅ 骨架就位, 24 测试待绿
│   │   ├── se3.py
│   │   ├── kinematics.py      #    FK + 解析 Jacobian
│   │   ├── ik.py              #    DLS IK + 零空间
│   │   └── tests/             # ★★ 与 mj_jacSite / site_xpos 交叉验证
│   ├── env/                   # ✅ 我的脚手架, 已完成
│   │   ├── scene.py           #    MjSpec 注入 TCP + 双相机
│   │   └── pick_place.py      #    任务场景 + 随机化 + 成功判据
│   ├── data/
│   │   ├── expert.py          # ★ 脚本专家 (调用你的 IK)
│   │   ├── schema.py          # ★ 你定义的数据格式
│   │   ├── collect.py
│   │   └── inspect.py         #   数据体检
│   ├── policies/
│   │   ├── mini_act_a.py      #   Stage A: CNN+MLP
│   │   ├── mini_act_b.py      # ★ Stage B: Transformer
│   │   └── train.py
│   ├── eval/
│   │   ├── protocol.py        # ★ 扰动网格 L0-L5
│   │   ├── runner.py          #   N×trials → JSONL
│   │   ├── disturb.py         # ★ 中途扰动恢复实验
│   │   └── analyze.py         #   Wilson CI + 失败矩阵 + H 曲线
│   └── deploy/                #   [扩展位] HIL / 真机 / 安全层
├── data/                      # HDF5 数据集
└── runs/                      # checkpoint + 日志
```

`★` = 你的核心资产，必须手写。其余我可以代写。

---

## 4. 消融实验表模板（Day 6 填）

**单变量原则**：一次只动一列。

| exp_id | policy | n_demo | H_pred | K_exec | seed | 视角 | 成功率(L0) | 95%CI | 主要失败模式 | 推理延迟(ms) | 恢复率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B-h64-k8-s0 | Trans | 400 | 64 | 8 | 0 | w+f | | | | | |
| B-h64-k16-s0 | Trans | 400 | 64 | 16 | 0 | w+f | | | | | |
| B-h64-k32-s0 | Trans | 400 | 64 | 32 | 0 | w+f | | | | | |
| B-h64-k64-s0 | Trans | 400 | 64 | 64 | 0 | w+f | | | | | |
| B-h16-k8-s0（次实验） | Trans | 400 | 16 | 8 | 0 | w+f | | | | | |
| A-h16-k8-s0 | CNN+MLP | 400 | 16 | 8 | 0 | w+f | | | | | |

**若时间允许再加的维度**（按价值排序）：数据量 `n_demo ∈ {100,200,400}` > 视角组合（仅腕部 / 仅第三人称 / 双路）> 失败样本混入比。

---

## 5. 降级预案

| 触发条件 | 动作 |
|---|---|
| Day 2 22:00 IK 不收敛 | 保留手写 FK+Jacobian，IK 换 `scipy.optimize`，砍零空间 |
| Day 3 专家成功率 < 80% | 取消朝向约束，只做 top-down 抓取 |
| Day 5 22:00 Transformer 不收敛 | 用 Stage A 固定 checkpoint 跑 `K_exec` 消融（**消融不可砍**） |
| 任何一天落后 > 4h | 从交付物列表底部砍：视频 → 三色语言 → 零空间 → L4 档 |
| 训练被 oomd 杀 | 关 Chrome/微信/VSCode；`num_workers` 降到 1；数据集减到 200 ep |
| 显存 OOM | `bs` 16→8；图像 128→96；ResNet18→自定义小 CNN |

---

## 6. AI 使用红线

**必须你自己写**：
`robotics/` 全部 · 数据 schema · action chunk 的时间轴对齐逻辑 · eval 协议设计 · 失败分类判据

**AI 可以大量帮**：
MuJoCo API 排查 · CUDA/依赖报错 · 可视化绘图 · 训练脚本样板 · 报告排版 · 消融批跑脚本

**每写完一个模块，问自己三句**：
1. 这个数为什么是这个数？
2. 改成 2 倍会怎样？
3. 出错时先看哪个 log？

答不上来就是没懂。

---

## 7. 面试问答映射

做完这个项目，下面这些题你是**用自己的实验数据**回答的，不是背的。
（前四题来自牛客真实面经）

| 面试题 | 你的答案来源 |
|---|---|
| VLA 的 action head 有哪些常见设计？ | Day4/5 的回归 vs Transformer chunk 对照 |
| VLA 微调的数据量一般是多少？ | Day 6 的 `n_demo` 消融曲线（若做了） |
| 遥操作数据采集原理？ | Day 3 的脚本专家 + 为什么它替代了遥操、代价是什么 |
| π0 / π0.5 / π\*0.6 的区别？ | 报告里的路线综述 + 你跳过 CVAE 的理由 |
| `H_pred` 与 `K_exec` 有什么区别，为什么不整块执行 64 步？ | **Day 6.1/6.2 的受控消融与扰动恢复率曲线** |
| Panda 的零空间你怎么用？ | Day 2 的关节限位越界次数对比 |
| 手眼标定残差多少？ | 纯仿真无标定 → **诚实说明，改讲 FK 与 `mj_jacSite` 的 1e-9 验证** |
| sim 成功率高真机低，你怎么归因？ | 报告的"方法学局限"章节（本项目未做真机，要明确边界） |
| normalization stats 用全量还是训练集？ | Day 4 的实现细节 + 混了会怎样 |

---

## 8. 一周之后（扩展路线，不在本周范围）

按性价比排序：

1. **HIL / DAgger 闭环** —— 策略出错时人工接管，收 recovery 轨迹重训。
   `Demo → Policy₀ → Failure → Human Correction → Dataset₁ → Policy₁`
   这是 2026 味道最浓的一层，LeRobot 官方有现成 workflow。
2. **SmolVLA-450M** —— 8GB 需 bf16 + 冻结 VLM 前若干层 + grad accum，bs 1-2。
   π0(3B) / GR00T 本地跑不了（LoRA 也要 24G+），要租云卡。
3. **真机** —— SO-ARM101 约 ¥1200-1950。注意：**位置控制舵机做不了真正力控**。
4. **力控 / 接触装配** —— 需要力矩控制或 F/T 传感，SO-101 不满足，要 Franka/UR/Aubo。
5. **移动操作** —— 复用你已有的 SLAM/导航栈（FAST-LIO + TEB + NBV），
   做 base placement 与可达性求解。**这是你相对其他候选人最大的结构性优势**：
   绝大多数人只有"臂"或只有"车"。

---

## 9. 当前状态

- ✅ Day 0 完成
- ✅ torch 2.13.0+cu130 已可用；RTX 5070 Laptop、sm_120、BF16 已实测通过
- ✅ MuJoCo Panda 任务场景可构建并完成双相机渲染
- 🔄 `so3.py` 仍为 5 个 TODO；24 个测试按预期全部失败于 `NotImplementedError`
- ⏭️ **下一步：`robotics/so3.py` 五个 TODO → `pytest robotics/tests/test_so3.py` 24 绿**
