# Embodied Panda Manipulation Systems Lab

这个仓库的目标是完成一条可解释、可测试、可复现的机械臂学习链路：

```text
SO(3)/SE(3) → FK/Jacobian/IK → 脚本专家 → 数据契约
→ BC/Mini-ACT → 闭环扰动评测 → 求职交付物
```

## 从哪里开始

| 文档 | 作用 | 发生冲突时的优先级 |
|---|---|---|
| [`docs/EMBODIED_ARM_CAREER_7DAY_PLAN_2026.md`](docs/EMBODIED_ARM_CAREER_7DAY_PLAN_2026.md) | 招聘证据、项目选择、七天总路线、后续四周路线 | 只负责回答“为什么做、对准什么岗位” |
| [`PLAN.md`](PLAN.md) | 技术架构、七天工程门禁、数据与评测协议 | 负责回答“整个项目怎样闭环” |
| [`DAY1.md`](DAY1.md) | Day 1 分钟级学习、实践、AI Coding、命令和验收 | Day 1 执行细节以此文件为准 |

三份文件不是三套项目。它们描述的是同一个 **Panda Manipulation Systems Lab**，只是粒度不同。不要同时来回执行三份时间表：今天直接执行 `DAY1.md`，完成后再按 `PLAN.md` 进入 Day 2。

`docs/jobs/` 预留给招聘证据快照，目前为空。空目录不构成岗位仍开放的证据；以后每份快照至少记录原始 URL、抓取日期、岗位状态和是否为官方来源，禁止只保存脱离日期的 JD 文本。

## 当前真实状态

- Day 0 环境和 MuJoCo Panda 场景已经就位；
- `src/env/scene.py` 能在上游 Panda 模型上注入 TCP、腕部相机和第三人称相机；
- `src/robotics/so3.py` 仍保留 5 个 `TODO(you)`，24 个测试当前按预期失败；
- `se3.py`、`kinematics.py` 和后续策略模块仍待你按计划完成；
- 当前成果是“环境与执行计划已验证”，不是“七天项目已经完成”。

## 每次开工

```bash
source /media/hetaisheng/044A81D94A81C83E/embodied/env.sh
export PYTHONPYCACHEPREFIX="$EMB/cache/pycache"
test -f "$MENAGERIE/franka_emika_panda/scene.xml"
python -m env.scene
```

`env.sh` 会切换到 `$EMB/src`，并把 uv、Hugging Face、Torch 缓存留在数据盘；上面的 `PYTHONPYCACHEPREFIX` 另外迁移 Python 字节码缓存。系统盘当前接近满载，不要在本周临时安装 Isaac Lab、ROS 2 或其他大型仿真栈。

## 第三方模型：MuJoCo Menagerie

`menagerie/` 不是本项目原创代码，而是从 [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 固定 revision 提取的 Panda 第三方快照。它现在由**外层主仓库直接管理**，目录内不再保留第二个 `.git`：

| 项 | 当前锁定值 |
|---|---|
| 上游 | `https://github.com/google-deepmind/mujoco_menagerie` |
| revision | `da76818e269b82289eba39808e2fb91d679d6994` |
| vendored path | `franka_emika_panda` |
| Panda 模型许可 | Apache-2.0，见 `menagerie/franka_emika_panda/LICENSE` |
| 版本锁定文件 | `menagerie/VENDORED_REVISION` |
| Git 管理 | 随主仓库一起 clone、审查和提交 |

本项目不直接修改上游 XML 或 mesh。自定义 TCP 和相机由 `src/env/scene.py` 通过 `MjSpec` 在内存中注入，从而保持“上游模型”和“本项目场景逻辑”的边界。

验证来源文件、许可证和“仓库内只有一个 `.git`”：

```bash
test -f menagerie/franka_emika_panda/scene.xml
test -f menagerie/franka_emika_panda/LICENSE
grep '^revision=' menagerie/VENDORED_REVISION
find . -type d -name .git -print -prune
```

最后一条预期只输出仓库根目录的 `./.git`。新 clone 会直接包含模型，不需要第二次 clone。若以后升级 Menagerie，必须在临时目录检出新 revision，审查 Panda XML/mesh 差异，更新 `VENDORED_REVISION`，再重跑 FK、碰撞和控制基线；不能在七天实验中静默替换模型。

## AI 使用边界

你必须亲手完成机器人学公式、坐标约定、FK/Jacobian/IK 核心循环、数据时序、评测协议与失败解释。AI 可以帮助查 API、生成非核心脚手架、设计反例、解释单个报错和整理报告。完整边界及可复制提示词见 `DAY1.md`。
