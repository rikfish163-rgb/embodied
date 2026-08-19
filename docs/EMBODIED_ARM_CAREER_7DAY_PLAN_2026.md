# 2026 具身智能机械臂岗位分析与七天项目计划

> 调研日期：2026-08-19（Asia/Shanghai）  
> 项目根：`/media/hetaisheng/044A81D94A81C83E/embodied`  
> 定制假设：按此前背景“985 AI 本科、低年级、项目积累较少”设计；毕业年份仍需在实际投递前逐岗核对。  
> 证据边界：招聘官网和当前职位页用于判断岗位方向；第三方招聘页只作为技能信号，不等于岗位仍开放。

## 文档关系与执行入口

这不是一套独立于仓库的第二方案，而是当前项目的“岗位依据 + 总路线”层：

- 本文回答为什么选择 Panda、运动学、Mini-ACT 和鲁棒评测，以及它们对应哪些岗位；
- 根目录 [`PLAN.md`](../PLAN.md) 是统一技术规格，负责七天工程门禁和系统边界；
- 根目录 [`DAY1.md`](../DAY1.md) 是 Day 1 唯一执行清单，分钟级命令和验收以它为准；
- 根目录 [`README.md`](../README.md) 是仓库入口，并记录 vendored Menagerie 的来源、锁定 revision 和升级方法。

`docs/jobs/` 当前为空，只是未来招聘证据快照的预留位置；空目录和本文中的第三方链接都不能单独证明岗位仍开放。实际投递当天必须重开官方页面，核对毕业年份、地点、实习时长和岗位状态。

---

## 0. 先给结论

这一周最值得做的不是下载一个大 VLA、改配置跑 Demo，而是完成一个能从底层数学讲到策略评测的项目：

**Panda Manipulation Systems Lab：手写运动学与 IK → 脚本专家采集数据 → Mini-ACT action chunk 策略 → 扰动恢复与 OOD 评测。**

它同时对准三类当前岗位：

1. **操作 / 运控算法**：SO(3)、SE(3)、FK、Jacobian、DLS IK、零空间、轨迹与控制接口；
2. **VLA / 模仿学习**：多视角观测、动作空间、数据时序、ACT、action chunk、闭环推理；
3. **具身数据 / 模型评测**：数据 schema、质量检查、训练/测试隔离、扰动恢复、失败分类、置信区间。

“高级”不等于模型参数多。对本科生作品集，更高级的信号是：

- 每个坐标系和动作定义都能解释；
- 模型前后都有可靠的系统模块；
- 有严格的实验协议和反例；
- 能说明纯仿真、无力传感器、无真机各自限制；
- 结果可复现，不挑最好 seed，不把失败藏起来。

---

## 1. 本机实测约束与技术选择

| 项目 | 2026-08-19 实测 | 对计划的影响 |
|---|---:|---|
| CPU | Ryzen 9 7845HX，12 核 24 线程 | 适合 MuJoCo 并行采样、离线数据检查 |
| 内存 | 30 GiB；可用量是动态值，开工前重新执行 `free -h` | HDF5 流式读取；`num_workers=1~2`，不把图像全集载入 RAM |
| Swap | 15 GiB；2026-08-19 23:45 已用约 13 GiB | 当前内存压力偏高；训练前关闭不需要的重进程并重新检查 |
| GPU | RTX 5070 Laptop，8151 MiB，sm_120 | Mini-ACT / ResNet18 / 小 Transformer 可行；不本地全参训练大 VLA |
| PyTorch | 2.13.0+cu130；CUDA 可用；BF16 支持 | 使用 BF16 autocast；每次先跑显存 smoke test |
| 数据盘 | 626 GiB，总空闲约 42 GiB | 数据、缓存、checkpoint 全放 `$EMB`；单项目预算不超过 25 GiB |
| 系统盘 | 98 GiB；2026-08-19 23:45 仅约 979 MiB 空闲 | 禁止把 Hugging Face、Torch、uv 缓存写到 `~/.cache` |
| 当前工具 | Docker/Conda/Python 可用；shell 找不到 ROS 2、Gazebo、`nvcc` | 本周不临时安装大型 ROS/Isaac 栈；先用现成 MuJoCo 环境闭环 |

### 为什么本周不装 Isaac Lab

NVIDIA 当前 Isaac Lab 安装文档建议至少 **32 GB RAM、16 GB VRAM**；Isaac Sim 还给出约 **50 GB SSD** 的最低存储要求。当前机器只有 8 GB VRAM且数据盘仅剩约 42 GB，因此本周本地安装 Isaac Lab 是高风险、低产出的路线。Isaac Lab 仍然是第二阶段要学的能力，但用官方文档、云 GPU 或实验室机器完成。

官方依据：

- [Isaac Lab 安装与系统要求](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html)
- [Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)
- [Isaac Lab SkillGen（含 VRAM 实测说明）](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)

### 为什么本周选 MuJoCo + Mini-ACT

- 当前场景、Panda 模型、EGL 渲染和 PyTorch 已就位；没有重新安装成本。
- LeRobot 官方把 ACT 定位为低计算成本、单卡数小时可训练的入门首选，建议从 batch size 8 起步。
- ACT 的 action chunk 同时连接“模型结构、实时性、扰动恢复、评测方法”，比只复现单步 BC 更有面试深度。

官方依据：

- [LeRobot ACT 文档](https://huggingface.co/docs/lerobot/act)
- [LeRobot 端到端真机/数据/训练教程](https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot)
- [ACT 原论文](https://tonyzhaozh.github.io/aloha/aloha.pdf)

---

## 2. 2025–2026 招聘市场：企业真正购买什么能力

### 2.1 当前岗位证据

| 公司 / 岗位族 | 当前公开信号 | 项目里必须拿出的证据 |
|---|---|---|
| **宇树科技**：具身智能软件、数据管线、具身数据评估、AI 大模型、强化学习运控 | 官方岗位强调端侧部署、低延迟“感知-决策-执行”、数据采集/清洗/标注、难例挖掘、模型评估、异常恢复与真机长测 | 推理延迟；数据 schema；难例/OOD；失败分类；安全停机接口；闭环而非只报 loss |
| **千寻智能 Spirit AI**：VLA、数据算法、机器学习系统、系统集成、运控/力控 | 官方职位覆盖 VLA 训练推理、数据质量闭环、采集-仿真-部署全生命周期；运控岗直接写机械臂路径、轨迹、运动学、动力学、建模辨识；力控岗写阻抗/导纳、碰撞检测、力位混合 | 手写 FK/Jacobian/IK；动作/状态数据对齐；训练与评测；明确无真机/无力传感器边界 |
| **千寻 2027 校招实习** | 2026-06 发布的信息列出数据算法、VLA 操作、模型评测、机器人软件、AI 平台等；毕业窗口为 2026-09 至 2027-12 | 若毕业时间匹配可投；若不匹配，把 JD 当能力清单，不错误投递 |
| **DJI 大疆**：控制、视觉、机器人系统、RoboMaster 相关研发 | 当前官网确认 2027 校招已于 2026-06-25 开启，实习全年开放；公开门户未稳定暴露“机械臂算法”单岗详情 | 项目应突出控制/仿真/鲁棒工程，而不是把 DJI 误写成当前明确在招的 VLA 机械臂岗 |
| **智元机器人**：世界模型、WAM/VLA、具身 Agent、全身控制 | 2027 优才方向把世界动作模型、物理交互预测、Verifier/Value Model、VLA/WAM、长时规划列为前沿方向 | 本周只读懂“慢系统规划 + 快系统动作”与 verifier；不假装一周训练世界模型 |
| **星海图**：模型评测、数据与 VLA 平台 | 当前可查岗位重视仿真/真机自动化评测、开环/闭环回灌、数据覆盖/偏差、延迟/吞吐/资源、Python/C++/ROS2/CI | Day 6 的协议、Wilson 区间、延迟、恢复率、失败矩阵正对口 |
| **海康机器人**：机械臂运动控制、智能控制 | 2026 当前职位要求运动学/动力学、位置/力控制、RL/IL、C++/Python、仿真到实物部署；另有岗位要求 LQR/MPC/ADRC 与控制指标 | Day 1–2 是门票；后续需补动力学、阻抗控制、C++ 与真实硬件 |
| **美的具身团队**：仿真、场景、策略、Sim2Real | 2026 实习信息把 Isaac/MuJoCo/Gazebo、ROS2/MoveIt、ACT/Diffusion Policy/OpenVLA、数据全链路列成可演示能力 | 本项目先把 MuJoCo+ACT 跑深；第二阶段接 ROS2/MoveIt，不要浅尝十个框架 |
| **传统机械臂厂 / 集成商**：JAKA、Dobot、Mech-Mind 等 | 运动学/动力学、轨迹规划、碰撞、抓取、力控、Linux C++、精度/稳定性/节拍长期高频 | 增加碰撞约束、轨迹平滑、控制频率、失败诊断；VLA 不能替代机器人学基本功 |

关键来源：

- [宇树科技当前官方职位](https://www.unitree.com/cn/position/)
- [千寻智能当前官方职位](https://www.spirit-ai.com/career)
- [千寻智能 2027 届校招实习信息（南开就业平台）](https://career.nankai.edu.cn/correcruit/content/id/116118.html)
- [DJI 2027 校招与全年实习](https://careers.dji.com/zh-CN/campus?source=RM-Banner)
- [智元机器人官方招聘入口](https://www.agibot.com.cn/join_us)
- [智元 2027 优才方向（转引智元招聘公众号）](https://www.pd-italent.com/Article/202606/202606250014.shtml)
- [星海图模型评测岗位存档](https://www.shushuqiuzhi.com/position/159280)
- [海康机器人智能控制职位](https://job.hikrobotics.com/society/position?postId=05B62984F3CBD9D7EBCACFD3FD6C9CF4)
- [机械臂运动控制职位](https://talent.hikvision.com/society/position?postId=0FEFBE9947ABD182F6A32351AAD5223C)
- [2026 VLA/机械臂/ROS2/MoveIt 岗位样本](https://career.nankai.edu.cn/correcruit/content/id/116183.html)
- [美的具身智能算法实习样本](https://www.shixiseng.com/intern/inn_1ktmoxpe4ilq)

### 2.2 高频能力矩阵

| 优先级 | 能力 | 岗位覆盖 | 本周如何证明 |
|---|---|---|---|
| S | Python、PyTorch、Linux、Git、可测试代码 | 几乎所有算法/数据/评测岗 | 单测、固定 seed、配置与复现命令 |
| S | 坐标系、SO(3)/SE(3)、FK/IK/Jacobian | 运控、规划、力控、系统集成 | 与 SciPy/MuJoCo 独立实现交叉验证 |
| S | 数据采集、同步、schema、质量与版本 | VLA、数据算法、平台、评测 | HDF5/LeRobot 兼容映射、数据体检报告 |
| S | 闭环评测、鲁棒性、失败归因 | 模型评测、产品化、系统集成 | L0–L4、扰动恢复、95% CI、失败矩阵 |
| A | 模仿学习、ACT/Diffusion Policy、action chunk | VLA 操作、智能控制 | BC 基线与 Mini-ACT；`H_pred`/`K_exec` 受控消融 |
| A | C++、ROS2、MoveIt、实时控制接口 | 工业机械臂、系统集成、部署 | 本周做接口设计；第二阶段容器化接入 |
| A | 真机调试、标定、时延、异常处理 | 所有产品化岗位 | 本周只能做仿真替代与局限说明，不能宣称 Sim2Real |
| B | 力控、动力学、系统辨识、MPC/WBC | 高级运控/力控 | 第二项目；当前位置控制 Panda 场景只能打基础 |
| B | 大规模预训练、分布式训练、推理内核 | 预训练/AI Infra | 当前硬件与阶段不匹配，不作为七天主项目 |

### 2.3 对你最现实的岗位顺序

1. **模型评测 / 具身数据算法 / 机器人学习工程实习**：最容易用纯仿真项目给出完整证据。
2. **VLA 操作算法实习**：项目完成后具备可信入场券，但仍需论文阅读、更多策略和最好一段真机/实验室经历。
3. **机器人软件 / 算法系统集成**：再补 ROS2、MoveIt、C++、部署与安全层。
4. **运动控制算法**：再补动力学、轨迹优化、阻抗/导纳、系统辨识和真机。
5. **具身基础模型预训练 / AI Infra**：通常依赖硕博研究、论文或大规模训练系统经历，不应把它当一周后的主要投递方向。

---

## 3. 2025–2026 前沿：本周实现什么，阅读什么

| 前沿方向 | 行业意义 | 本周策略 |
|---|---|---|
| Action chunk + 实时重规划 | ACT、π 系列、SmolVLA 都围绕连续动作块；预测长度 `H_pred` 与实际执行长度 `K_exec` 共同影响平滑、延迟和恢复 | **分离 `H_pred`/`K_exec` 做受控消融** |
| 异步推理 / Real-Time Chunking | 让感知推理与动作执行解耦，减少停顿；是 2025–2026 的部署热点 | 本周记录同步推理延迟并预留接口；第二周实现 async/RTC |
| 数据质量与 HIL/DAgger | 官方 LeRobot 已把人工接管 recovery 数据纳入工作流；企业 JD 高频出现难例挖掘与数据闭环 | 本周做失败样本 schema；第二周做仿真接管/纠错轨迹 |
| 轻量 VLA 与 PEFT | SmolVLA 450M、LoRA/PEFT 降低后训练门槛 | 本周只把数据格式做兼容；完成 ACT 后再用云卡或小 batch 实测，不承诺 8 GB 成功 |
| 世界动作模型 / Verifier | 智元、银河、千寻等开始强调预测动作后果、失败风险、慢思考/快执行 | 阅读架构并在报告写“扩展设计”；本周不训练 WAM |
| 合成数据 / SkillGen | 少量演示扩增大量轨迹，是数据规模化的重要路线 | 当前 8 GB 不跑 Isaac SkillGen；脚本专家是轻量替代，但必须声明差异 |
| 可信评测 | 开环指标无法代表闭环任务；OOD、扰动、失败分类和统计不确定性越来越重要 | **Day 6 是项目价值最高的一天，不得删** |

前沿阅读：

- [SmolVLA 官方介绍与异步推理](https://huggingface.co/blog/smolvla)
- [SmolVLA 官方文档](https://huggingface.co/docs/lerobot/smolvla)
- [LeRobot PEFT/LoRA 文档](https://huggingface.co/docs/lerobot/peft_training)
- [LeRobot Human-in-the-Loop 数据闭环](https://huggingface.co/docs/lerobot/hil_data_collection)
- [OpenVLA 官方仓库（含 OFT 与 FAST 更新）](https://github.com/openvla/openvla)
- [π0.5 论文](https://www.physicalintelligence.company/download/pi05.pdf)
- [NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)

---

## 4. 推荐项目组合

### P1（本周主项目）：Panda 全链路操作系统 + Mini-ACT 鲁棒评测

**链路**：模型与坐标约定 → FK/Jacobian/IK → 专家轨迹 → 数据 schema → BC → Mini-ACT → 闭环评测 → 报告。  
**岗位**：VLA 操作、具身数据、模型评测、机器人学习、运控基础。  
**硬件**：完全适配当前机器。  
**独特性**：能回答“为什么”，且有手写机器人学、策略、统计实验三层证据。

### P2（第 2 周）：Embodied Data Flywheel

**链路**：LeRobotDataset 兼容适配 → 时间同步/异常检测 → 难例挖掘 → 仿真人工接管 → DAgger/HIL 重训 → 数据价值评估。  
**岗位**：宇树/千寻的数据算法、数据管线、评测、平台。  
**高级点**：比较“随机新增 50 条演示”与“针对失败新增 50 条 recovery”对成功率的边际收益。

### P3（第 3–4 周）：ROS2/MoveIt 混合规划与安全层

**链路**：URDF/TF → PlanningScene → OMPL/MTC → MoveIt Servo → 策略目标 → 碰撞/限速/看门狗 → ROS bag 回放评测。  
**岗位**：机器人软件、系统集成、规划控制、工业机械臂。  
**硬件**：用 Docker + RViz/仿真；先释放磁盘，再安装。  
**核心实验**：学习策略直接输出 vs 安全层过滤后的碰撞率、任务成功率、延迟。

### P4（第 2 个月）：接触丰富操作数字孪生

**链路**：动力学 → 重力/摩擦辨识 → 阻抗/导纳 → 力位混合 → 接触插入任务 → 残差学习。  
**岗位**：千寻高级力控、海康/协作臂控制、机械臂核心算法。  
**边界**：纯仿真只能证明算法与实验方法；没有扭矩控制真机和 F/T 传感器，不能宣称真实力控落地。

### P5（P1/P2 完成后）：SmolVLA PEFT + 异步推理基准

**链路**：LeRobot 数据 → LoRA/PEFT → 本地或云卡训练 → 同步/异步/RTC → 延迟、吞吐、扰动恢复。  
**岗位**：VLA 后训练、推理部署、模型系统。  
**硬件策略**：先用 8 GB 做小 batch smoke test；若 OOM 或速度不可接受，租 L4/A10/A100。官方文档给出的 20k 步参考是单 A100 约 4 小时，因此不能把“consumer hardware”宣传语直接当作本机训练承诺。

**只能选一个七天主项目：P1。** 同时开 P2–P5 会得到五个浅 Demo，反而削弱简历。

---

## 5. P1 的完整系统边界

```text
MuJoCo Panda + 双相机 + 任务随机化
            │
            ├── 你手写：SO(3)/SE(3)/FK/Jacobian/DLS IK
            │                         │
            │                         └── 脚本专家
            │                                  │
            └── RGB / qpos / qvel / gripper / action / meta
                                               │
                                        数据体检与切分
                                               │
                             ┌─────────────────┴─────────────────┐
                             │                                   │
                          BC 基线                         Mini-ACT chunk
                             │                                   │
                             └─────────────────┬─────────────────┘
                                               │
                              L0–L4 + 中途扰动 + 延迟 + 失败矩阵
                                               │
                                  README / REPORT / 视频 / 简历
```

### 五个不可妥协的交付物

1. 机器人学核心的独立交叉验证；
2. 数据契约与泄漏检查；
3. 至少一个可闭环运行的学习策略；
4. action chunk 长度与恢复率实验；
5. 可复现命令、失败案例和局限说明。

---

## 6. AI 使用合同

### 必须由你亲手完成

- 纸上推导 SO(3)/SE(3)、FK、Jacobian、DLS、零空间；
- `robotics/` 核心循环与关键数值分支；
- 明确 frame、twist、qpos、action、时间戳和单位；
- 决定数据 schema、训练/验证/测试切分和随机化边界；
- action chunk 标签生成、执行/重规划逻辑；
- 在看结果前写评测假设和成功判据；
- 亲看失败轨迹并分类；
- 用自己的语言写简历第一稿、README 结论和 10 分钟口述；
- 逐岗判断自己是否满足毕业年份、学历、时长和地点。

### AI 可以大量协助

- 查 MuJoCo/LeRobot/MoveIt API 与文档；
- 生成非核心 I/O、CLI、配置、绘图和报告排版脚手架；
- 写独立测试、边界 case、profile 脚本和批量运行脚本；
- 解释错误日志、定位 shape/device/依赖问题；
- 对你的推导和代码做 code review，但不替你填核心答案；
- 对你已经写出的简历和口述稿做表达优化。

### 禁止交给 AI

- 直接生成 `so3.py`、`se3.py`、FK/Jacobian/IK 的最终实现；
- 先看全部结果再挑一个有利协议；
- 只展示最好 seed，删除失败 run；
- 虚构真机、Sim2Real、力控、招聘投递或性能数字；
- 自动代投、自动写“本人贡献”或在你没看懂时替你答面试题。

每天收尾都必须不看代码回答：

1. 这个模块的输入、输出、坐标系、单位是什么？
2. 最危险的两个失败模式是什么？
3. 我用了哪个独立判官验证它？
4. 一个超参数翻倍会发生什么？
5. 当前结论只在哪个分布、哪台机器、哪种控制接口上成立？

---

## 7. 七天逐时计划

每天按 **10–11 小时专注 + 1 小时缓冲**设计。若只能做 8 小时，优先保留标记“核心”的任务。夜间训练必须先有 50-step smoke test、checkpoint 和资源上限。

### Day 1：SO(3) → SE(3) → Panda FK

**当天结论目标**：你能从关节角独立得到 TCP 位姿，并解释每个坐标变换。

| 时间 | 任务 | 你必须完成 | AI 可协助 | 验收 |
|---|---|---|---|---|
| 00:00–00:20 | 环境门禁 | 执行下方命令并解释为什么当前应有 24 个 `TODO` 失败 | 解释环境输出 | 导入路径指向 `embodied/src`；失败原因只有 `TODO` |
| 00:20–01:40 | SO(3) 推导 | 纸上推 Rodrigues、log 普通分支、0/π 数值问题 | 审阅推导、出反例 | 笔记可脱稿讲 8 分钟 |
| 01:40–04:30 | `so3.py` | 亲写 5 个函数和数值分支 | 给单测提示，不给最终实现 | 24 个测试全绿；与 SciPy 误差达标 |
| 04:30–05:00 | 复盘 | 写错过的 3 个点 | 整理格式 | `notes/day1_so3.md` |
| 05:00–07:20 | SE(3) | 亲写齐次变换、逆、Adjoint、`ad` | 设计随机 property tests | 100 组随机测试全绿 |
| 07:20–10:20 | Panda FK | 从模型 body/joint 链重建，不读末端真值作输入 | 查 MuJoCo 字段 | 100 个随机姿态，位置/姿态与 MuJoCo 交叉验证 |
| 10:20–11:00 | 口述与提交前检查 | 录 10 分钟屏幕讲解；写 Day 1 报告 | 润色报告 | 能回答 frame、左乘/右乘、θ≈π |

开工命令：

```bash
source /media/hetaisheng/044A81D94A81C83E/embodied/env.sh
export PYTHONPYCACHEPREFIX="$EMB/cache/pycache"
python -c 'import robotics.so3 as s; print(s.__file__)'
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q
```

`PYTHONPYCACHEPREFIX` 用来避开迁移前缓存里保留的旧 traceback 路径，不需要删除用户文件。

学习资源：

- [Modern Robotics 视频：Chapter 3–5](https://hades.mech.northwestern.edu/index.php/Modern_Robotics_Videos)
- [Modern Robotics 免费教材](https://hades.mech.northwestern.edu/images/b/b2/MR-2up.pdf)
- [MIT 2025 Robotic Manipulation 课程笔记](https://manipulation.mit.edu/index.html)
- [MuJoCo Python API](https://mujoco.readthedocs.io/en/latest/python.html)

**降级**：时间不足时保留 SO(3) 全绿、SE(3) 基本变换、FK 位置与姿态验证；Adjoint 的扩展测试可移到 Day 2 开头。

### Day 2：Jacobian → DLS IK → 零空间

**当天结论目标**：你能让 Panda 稳定跟踪末端目标，并用实验说明阻尼和冗余自由度的意义。

| 时间 | 任务 | 验收 |
|---|---|---|
| 00:00–00:40 | 复述 Day 1 + 修复遗留测试 | 不看代码画出变换链 |
| 00:40–02:00 | FK 有限差分 Jacobian | 步长扫描；误差随步长先降后升 |
| 02:00–04:00 | 解析几何 Jacobian | 与 `mj_jacSite` 独立比对；记录最大残差 |
| 04:00–06:30 | DLS IK | 100 个固定 seed 可达目标，收敛率 >95%；记录迭代数 |
| 06:30–07:30 | λ 消融 | λ∈{0,1e-4,1e-2,1e-1}；必须展示奇异位姿失败 |
| 07:30–09:20 | 零空间/关节限位 | 对比越界次数、末端误差和最小 joint margin |
| 09:20–10:20 | API 与日志 | 明确 `solve(target, q0, limits, tol)` 的失败返回 |
| 10:20–11:00 | 口述 | 推导 `Jᵀ(JJᵀ+λ²I)⁻¹v`；解释为什么 Panda 有零空间 |

你必须亲写：Jacobian 核心、DLS 更新、停止条件、零空间投影、失败判据。  
AI 可写：随机目标生成、画图、批量测试、profile。

**22:00 门禁**：若 IK 仍不稳，保留 FK/Jacobian 核心证据，专家暂用 `scipy.optimize.least_squares`，零空间延后。不能让 Day 2 吞掉整周。

### Day 3：脚本专家、数据契约与质量闭环

**当天结论目标**：从任务分布定义到可训练数据全部可追溯。

| 时间 | 任务 | 验收 |
|---|---|---|
| 00:00–01:00 | 先写 `PROTOCOL.md` | 在采数前冻结训练分布、seed、成功判据和异常规则 |
| 01:00–03:30 | 笛卡尔路点专家 | approach→descend→close→lift→place→open；每段有超时/失败码 |
| 03:30–04:30 | 50 episode pilot | 成功率、控制频率、episode 长度、磁盘占用实测 |
| 04:30–06:00 | 数据 schema | 图像、qpos/qvel、gripper、EE pose、`action=[7 维臂目标,1 维夹爪命令]`、lang、timestamp、meta |
| 06:00–07:30 | 时间对齐/归一化 | observation 与 action 的索引关系写成图；统计量只用 train split |
| 07:30–08:40 | 数据体检 | NaN、常量通道、跳变、重复帧、失败轨迹、相机泄漏、分布覆盖 |
| 08:40–10:00 | 扩到 200–400 ep | 达到 >90% 专家成功率或触发降级 |
| 10:00–11:00 | 数据卡 | 数据量、许可、生成器版本、随机化、已知偏差、失败比例 |

你必须决定：动作是绝对关节目标还是增量、控制频率、chunk 标签、时间戳语义、train/val/test 的 seed 隔离。  
AI 可写：HDF5 I/O、压缩、并行采集、直方图和 HTML/Markdown 报告。

额外要求：写一个到 LeRobotDataset 字段的**映射文档**，本周不必安装完整 LeRobot。

资源：

- [LeRobotDataset 与工具总入口](https://huggingface.co/docs/lerobot/en/index)
- [LeRobot 真机数据记录→训练→评测教程](https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot)
- [MuJoCo Python 数据视图注意事项](https://mujoco.readthedocs.io/en/latest/python.html)

**降级**：专家成功率 <80% 时取消姿态约束，只做 top-down grasp；先保数据正确，再加任务难度。

### Day 4：闭环 BC 基线 + action chunk 时间轴

**当天结论目标**：先建立一个可学习、可闭环、可诊断的下限，再上 Transformer。

| 时间 | 任务 | 验收 |
|---|---|---|
| 00:00–01:00 | 数据 loader 单测 | episode 边界不串；最后 H 步 padding/mask 正确 |
| 01:00–02:00 | 过拟合 8 条样本 | loss 显著下降；失败则先查数据/归一化，不调大模型 |
| 02:00–04:20 | Stage A：共享 ResNet18 + MLP | 双相机+proprio→H×8；shape/参数量/显存报告 |
| 04:20–05:20 | 50-step GPU smoke | BF16、batch 8 起；峰值显存和 step time 写入日志 |
| 05:20–07:30 | 正式训练 | 固定 seed、checkpoint、train/val 曲线 |
| 07:30–09:00 | 闭环 rollout | 50 trials；成功率、延迟、动作抖动、失败视频 |
| 09:00–10:00 | `H_pred=16` 下 `K_exec=1` vs 16 | 同一 checkpoint 只改变实际执行步数，观察累计误差与平滑性 |
| 10:00–11:00 | 解释 | 画出 `obs_t → a[t:t+H_pred] → execute K_exec → replan` 时间轴 |

你必须亲写/审核：chunk label 生成、padding mask、归一化、防止跨 episode、闭环执行语义。  
AI 可写：ResNet 调用样板、训练 CLI、TensorBoard/W&B 替代日志、显存 profile。

验收门槛不是某个虚构成功率。最低门槛是：过拟合小样本成功、闭环确实控制仿真、失败可复现。目标 L0 >40%，但结果低于目标也必须如实保留。

### Day 5：Mini-ACT + 延迟感知执行

**当天结论目标**：理解并实现 action query 如何生成未来动作块，跑出可比较的 Transformer 策略。

| 时间 | 任务 | 验收 |
|---|---|---|
| 00:00–01:00 | 精读 ACT 图 4 与 LeRobot ACT 文档 | 手画 token、encoder、query、decoder、输出 shape |
| 01:00–03:30 | Mini-ACT forward | d=256、4 层、8 heads 起；逐张量注释 shape |
| 03:30–04:30 | 8 条样本过拟合 | 证明模型和 loader 能学，不用大训练掩盖 bug |
| 04:30–05:30 | 资源门禁 | batch 8→4；记录 BF16 峰值显存；保留 >1 GiB 安全余量 |
| 05:30–08:00 | 正式训练 | 与 Stage A 使用同一数据、split、训练预算 |
| 08:00–09:20 | 闭环 50 trials | 成功率 + latency p50/p95 + jitter |
| 09:20–10:20 | temporal ensemble/重规划设计 | 明确 `H_pred` 与 `K_exec`；执行步数不能被含糊地都叫 H |
| 10:20–11:00 | 挂夜间消融 | 先得到 `H_pred=64` 冻结 checkpoint；若不收敛则用最大已收敛 horizon；有余力再提交 `H_pred×seed` 队列 |

资源：

- [LeRobot ACT 教程](https://huggingface.co/docs/lerobot/act)
- [ACT 原论文](https://tonyzhaozh.github.io/aloha/aloha.pdf)
- [CS285 Imitation Learning 视频](https://www.youtube.com/watch?v=tbLaFtYpWWU)
- [Diffusion Policy 论文/项目](https://diffusion-policy.cs.columbia.edu/diffusion_policy_2023.pdf)

本周不照搬完整 CVAE：脚本专家的行为近似单模态，先验证 deterministic action chunk。报告必须写明这是与原 ACT 的差异，而不是偷偷省略。

**22:00 门禁**：Mini-ACT 不收敛就用 Stage A 的冻结 checkpoint 做 `K_exec` 消融。网络可降级，评测协议不可删。

### Day 6：预注册评测、扰动恢复与失败科学

**当天结论目标**：回答“模型为什么成功/失败”，而不仅是“成功率多少”。

| 时间 | 任务 | 验收 |
|---|---|---|
| 00:00–00:50 | 冻结 `EVAL_PROTOCOL.md` | 看正式结果前固定指标、trial 数、seed、失败类 |
| 00:50–02:30 | 执行长度主消融 | 固定同一 `H_pred=64` checkpoint，只改 `K_exec={8,16,32,64}`；隔离重规划频率 |
| 02:30–04:00 | 扰动恢复 | 在固定阶段且物体未被夹持时横向移动 3 cm；测恢复率与反应延迟；有余力再固定 `K_exec=8` 比较不同 `H_pred` |
| 04:00–05:30 | L0/L1 | IID 与位置外推；报告 Wilson 95% CI |
| 05:30–07:00 | L2/L3/L4 | 干扰物、光照纹理、相机扰动；一次只改一因子 |
| 07:00–08:20 | 延迟与资源 | p50/p95/p99、GPU 峰值、吞吐、控制停顿 |
| 08:20–09:40 | 人工看失败 | 每档至少看 20 个失败，分类：抓空/滑落/碰撞/未对准/放置/超时 |
| 09:40–10:30 | 反事实检查 | chunk 长导致恢复差，还是模型容量/训练量/执行步数混入？ |
| 10:30–11:30 | 核心图 | `K_exec`-成功率、`K_exec`-恢复率、OOD 矩阵、失败堆叠图；有次实验再画 `H_pred` 曲线 |

统计纪律：

- 不用训练集归一化以外的测试信息；
- 不根据测试结果回去挑最有利的 seed 或阈值；
- 调参用 validation，最终表只跑一次冻结 test；
- 每个 trial 保存配置、seed、checkpoint hash、结果和失败码；
- 只有一组 seed 时明确写“诊断结果”，不包装成稳定结论。

### Day 7：工程交付 + 求职证据 + 口述防守

**当天结论目标**：陌生人能复现，你能讲透，招聘方能在 60 秒内看到价值。

| 时间 | 任务 | 你必须完成 | AI 可协助 |
|---|---|---|---|
| 00:00–01:00 | 从干净 shell 复现 | 按 README 命令重跑 smoke/eval | 检查命令遗漏 |
| 01:00–03:00 | README | 你写一句话结论、3 张核心图、边界、复现 | 排版与语言精简 |
| 03:00–05:00 | REPORT 8–12 页 | 你写方法选择、实验解释、失败和局限 | 图表、引用、语法 |
| 05:00–06:00 | 视频 | 成功 + 失败 + 扰动恢复；口述，不配“炫技空镜” | 剪辑字幕 |
| 06:00–07:00 | 仓库卫生 | 小文件入 Git；数据/checkpoint 不误提交；许可证/来源 | `.gitignore` review |
| 07:00–08:30 | 简历 | 先写 3 条真实 bullet，每条含动作、方法、证据、边界 | 压缩表达，不造数字 |
| 08:30–09:30 | 岗位映射 | 为评测/数据/VLA/运控各写一版项目摘要 | JD 关键词对齐 |
| 09:30–10:30 | 10 分钟答辩 | 录屏，一镜讲完；不能看 AI 稿 | 根据录音给反馈 |
| 10:30–11:30 | 30 题压力面 | 坐标系、IK、chunk、数据泄漏、失败、无真机边界 | AI 当面试官追问 |

简历 bullet 模板（数字必须由真实结果填）：

```text
从零实现 Panda 的 SO(3)/SE(3)、FK、解析 Jacobian 与 DLS IK，
以 SciPy/MuJoCo 独立实现交叉验证，在 N 个随机姿态上达到 [真实误差]。

构建含双视角 RGB、关节状态、末端位姿和动作时间戳的模仿学习数据链路，
采集 [真实数量] 条轨迹并通过 [真实检查项] 发现/过滤 [真实数量] 个异常 episode。

实现 Mini-ACT action-chunk 策略并预注册闭环评测，分离比较 H_pred={...} 与 K_exec={...}，
在 [真实 trial 数] 次实验中报告成功率、Wilson 95% CI、p95 延迟与中途扰动恢复率；
明确结果仅为 MuJoCo 诊断证据，未宣称真机/Sim2Real。
```

---

## 8. 完成定义与降级顺序

### 一周“完成”必须同时满足

- `robotics/` 核心测试通过且有独立交叉验证；
- 固定 seed 能生成并读取一份数据集；
- 至少一个策略从 checkpoint 闭环运行；
- 至少 2 个 `K_exec` 在同一冻结 checkpoint 上完成公平对比，且有扰动恢复实验；
- README 有完整复现命令；
- 报告包含失败案例、局限和未完成项；
- 你能在不看代码/AI 的情况下讲 10 分钟并回答追问。

### 砍项顺序

从上到下依次砍，不能反过来：

1. 三色语言泛化；
2. temporal ensemble；
3. Transformer Stage B（保留 Stage A）；
4. 零空间优化（保留 FK/Jacobian/DLS）；
5. L4 相机扰动；
6. 演示视频美化。

**绝不砍**：测试、数据时序、闭环策略、扰动恢复、失败分析、复现命令。

---

## 9. 七天以后四周路线

| 周次 | 目标 | 产出 |
|---|---|---|
| Week 2 | LeRobotDataset 适配 + 仿真 HIL/DAgger | recovery 数据价值曲线；数据版本卡 |
| Week 3 | Docker 中 ROS2/MoveIt 2 + PlanningScene/MTC/Servo | 学习策略与安全规划混合闭环；碰撞率/延迟对照 |
| Week 4 | SmolVLA PEFT smoke + 云卡正式训练 | ACT vs SmolVLA；同步 vs async/RTC；成本与延迟表 |
| Month 2 | 实验室真机或合适机械臂 | 标定、时延、控制接口、安全、Sim2Real 误差；再谈真实落地 |

MoveIt 学习入口：

- [MoveIt 2 官方教程](https://moveit.picknik.ai/main/doc/tutorials/tutorials.html)
- [MoveIt Task Constructor Pick-and-Place](https://github.com/moveit/moveit2_tutorials/blob/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.rst)
- [MoveIt Servo 实时控制教程](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html)

---

## 10. 当前仓库状态（2026-08-19）

- 本次整合审计的 Git 起点是 `main` 提交 `e19d84c`（`day1`）；本轮文档整合尚未自动提交。
- `env.sh` 已把 uv/HF/Torch 缓存路由到数据盘。
- `menagerie/` 是由主仓库直接管理的 Google DeepMind Panda 第三方快照；内层 Git 已移除，来源 revision 锁定为 `da76818e269b82289eba39808e2fb91d679d6994`。
- Panda 的上游 XML/mesh 不直接修改；TCP 与双相机由 `src/env/scene.py` 使用 `MjSpec` 在内存中注入。
- MuJoCo Panda 场景实测可构建：`nq=30`、`nu=8`，双相机渲染成功。
- PyTorch 能识别 RTX 5070 Laptop，计算能力 `(12,0)`，BF16 可用。
- `so3.py` 仍是 5 个 `TODO`；24 个测试当前全部按预期失败于 `NotImplementedError`。
- 迁移前 `__pycache__` 会让 traceback 显示旧 `panda-week` 路径；设置 `PYTHONPYCACHEPREFIX=$EMB/cache/pycache` 后，回溯已确认指向当前 `embodied` 源码。
- 下一步不是继续搜资料，而是按 Day 1 从 `hat/vee` 开始亲手实现。
