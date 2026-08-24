# M1 脚本专家验收报告

## 结论

历史 canonical M1 run 已通过：固定、未逐个手调的 seed `0..99` 中成功
`99/100`，超过门槛 `90/100`。唯一失败是 seed 34 的第二次抬升未形成稳定夹持；
它被记录为 `failure_stage="lift"`，没有被删除或改成成功。现有 JSONL 没有单独记录
第一次尝试的失败阶段，因此这里不从最终字段反推未落盘的逐尝试结论。

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

## 物理正确性修订后的证据边界

上表和下文的 `2` 次恢复、平均时长、JSONL 与 100 个视频都只描述 commit
`a97977ad5267562f97b5c1a43623e0a2ce23d79c` 对应的历史 canonical run。该版本的
成功谓词用 cube 中心和固定半边长近似“完整入盒”；它没有检查旋转后的实际 geom
角点。因此，这批历史产物不能被回填或重新解释为已经通过后来的严格角点谓词。

2026-08-24 的物理正确性修订改为用实际 `cube_geom` 世界姿态的 OBB support（与逐个
检查 8 个角点数学等价），相对实际盒壁内面、底面顶面和墙顶面计算 clearance；连续
1 秒和 `success_z_tolerance` 均未改变。修订后的无视频 EGL 重跑仍为 `99/100`，
唯一失败仍是 seed 34，因此 `>=90/100` 的代码级门槛没有回归。但严格真值会改变部分
边缘 episode 是否进入第二次物理尝试，所以历史 `recovered_successes=2` 不能沿用为
修订后计数。该无视频重跑也不替代历史 100 个 MP4；若要生成新的 formal canonical
证据，必须用新输出目录重跑 JSONL、summary 和视频，并绑定当时的代码/资产 provenance。

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

## 未来 formal run 的资产字段边界

上面的 canonical M1 产物生成于资产 provenance 字段加入之前；不会回填、改写或把
历史 `summary.json` 重新解释成已经记录了 Menagerie 内容身份。未来由当前
`expert.evaluate` 生成的 formal summary 才会在既有顶层 `git` 字段旁增加：

```json
{
  "asset_provenance": {
    "canonical_root": "menagerie/franka_emika_panda",
    "revision": {
      "repository": "https://github.com/google-deepmind/mujoco_menagerie",
      "revision": "da76818e269b82289eba39808e2fb91d679d6994",
      "vendored_path": "franka_emika_panda",
      "license": "Apache-2.0"
    },
    "file_count": 81,
    "aggregate_manifest_sha256": "a90bfc375cb4ef3286dc104ca3e4b8045eb1e96ef54547f7178816c835bbc37a"
  }
}
```

`file_count` 包含 tracked `menagerie/VENDORED_REVISION` 和 Panda 子树内的 tracked
regular files。aggregate 对按项目相对路径排序的 `path`、`size_bytes`、`sha256`
清单使用项目 canonical JSON 规则后再做 SHA-256；summary 不记录 XML、mesh 或其他
资产文件内容。正式 evaluate 会在创建输出目录和启动环境前拒绝外部
`MENAGERIE`、symlink、缺失或非 regular 资产。MuJoCo native viewer 的人工查看边界
不因此改变。

## 尚未完成

M1 的历史运行不代表项目完成。M2A 已实现单 episode HDF5 schema、20 Hz `pre_action`
时间对齐、writer、reader 与 validator；目前仍没有正式 200+40 集合 manifest、20 条
回放验收、BC、ACT、OOD 或扰动评测。下一门禁是 M2B：完成集合级 provenance/split/
checksum 审计，采集 200 条 train 与 40 条 validation 成功轨迹，再随机回放 20 条。
