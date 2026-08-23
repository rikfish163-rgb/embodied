# M1 脚本专家验收报告

## 结论

M1 已通过：固定、未逐个手调的 seed `0..99` 中成功 `99/100`，超过门槛
`90/100`。唯一失败是 seed 34 在第一次运输滑落后，第二次抬升仍未形成稳定夹持；
它被记录为 `failure_stage="lift"`，没有被删除或改成成功。

| 指标 | 结果 |
|---|---:|
| 固定 episode 数 | 100 |
| 成功 | 99 |
| 成功率 | 99% |
| 门槛 | ≥90% |
| 物理重抓后恢复成功 | 2 |
| 最终失败 | 1（seed 34） |
| 平均仿真时长 | 11.94 s |
| 控制频率 / 物理频率 | 20 Hz / 500 Hz |

## 控制器与状态机

状态机是：

```text
pregrasp → descend → close → lift → transport → lower → open → retreat → settle
                                      └─ 滑落 → recover → 最多重抓一次
```

每个控制周期先计算 TCP 的位置与姿态误差。`mju_subQuat` 给出的姿态修正在 TCP
局部坐标系，而 `mj_jacSite` 的旋转 Jacobian 位于世界坐标系，因此必须先用当前
TCP 旋转矩阵把误差变到世界系。随后使用阻尼最小二乘：

```text
Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ Δx + (I - J⁺J) k(q_home - q)
```

这里 `λ=0.025`；TCP 单周期位移限制为 8 mm，旋转限制为 0.04 rad，关节命令增量
限制为 0.05 rad。最终输出仍是环境定义的 8D action，而不是直接写仿真状态。

## 防作弊边界

- cube、box、TCP 真值只允许脚本专家用于生成控制目标；它们不会进入后续学习策略；
- `run_episode()` 在 reset 后不修改 cube 的 qpos；所有运动来自指爪接触和动力学；
- 所有仿真推进均调用 `PickPlace.step()`，连续 1 秒成功保持计数没有被绕过；
- 没有为专家提高 cube 摩擦、夹爪力或改变成功容差；
- seed 34 的失败被保留在逐 episode 记录中。

## 可复现命令

```bash
source ./env.sh
MUJOCO_GL=egl python -m expert.evaluate \
  --num-seeds 100 \
  --required-successes 90 \
  --output-dir runs/m1/acceptance-001 \
  --record all
```

输出目录包含：

- `episodes.jsonl`：每个 seed 的成功、失败阶段、attempts、仿真时长与最终位姿；
- `summary.json`：汇总指标、完整控制器配置、MuJoCo 版本和 Git 状态；
- `videos/seed_XXXX.mp4`：128×128 前视和腕部画面横向拼接的 H.264 轨迹视频。

生成器拒绝覆盖已有输出目录；需要复跑时必须选择新目录，防止原始结果被静默改写。

## 尚未完成

M1 通过不代表项目完成。目前没有 HDF5 数据集、回放验收、BC、ACT、OOD 或扰动
评测。下一门禁是 M2：先冻结数据 schema 和观测—动作时间对齐，再采集 200 条
train 与 40 条 validation 成功轨迹。
