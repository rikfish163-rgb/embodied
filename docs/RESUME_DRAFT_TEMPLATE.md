# 具身智能岗位简历草稿模板

> 当前可用范围：**CODE-PUBLIC / CURRENT-STRICT-EVIDENCE-PENDING / RELEASE-PENDING**。
> M1 代码提交 [`a97977ad5267562f97b5c1a43623e0a2ce23d79c`](https://github.com/rikfish163-rgb/embodied/commit/a97977ad5267562f97b5c1a43623e0a2ce23d79c)
> 已发布到公开 `main`，对应 CI
> [run 32658620777](https://github.com/rikfish163-rgb/embodied/actions/runs/32658620777)
> 为 `success`；旧中心点成功谓词下的 M1 运行仅是 `HISTORICAL-LOCAL`，不能作为
> 当前严格旋转角点谓词的结果。新的 canonical M1 原始证据和 M5 交付包仍未生成/release。
> M2 数据、M3 BC/ACT、M4 扰动与反应性评测、M5 release 仍是 `PENDING`。
> 本文中的 `[待填]` 必须由真实原始记录替换；未替换前不得投递。

## 1. 一条合格 bullet 的结构

推荐结构：

```text
[动作/责任] + [关键方法与工程约束] + [可复核结果] + [适用边界]
```

自检：

- **动作**：用了“实现、构建、冻结、评测、定位”等可说明本人贡献的动词。
- **方法**：点出真正的难点，不堆工具名。
- **证据**：有分子/分母、原始来源、失败样本或可复现命令。
- **边界**：明确 MuJoCo、脚本专家/学习策略、evidence-local/released、IID/OOD 等范围。

状态标记只用于写作和审核，投递版可删去方括号，但其对应证据和人读边界不能删。
当前严格谓词结果尚不存在，旧 99/100、2 次恢复和 11.9375 s 只能作为带 commit 和
old-predicate 限定的历史审计说明，不能移入当前投递 bullet。代码事实若进入简历，必须
保留可见文字：**“严格谓词证据重跑中，release 待完成”**。

- `[VERIFIED-LOCAL]`：本地有原始证据但尚未随 release 发布；不表示代码未 push。
- `[HISTORICAL-LOCAL]`：旧定义下的本地历史事实，只能用于审计，不得充当当前结果。
- `[PENDING]`：只有计划或占位，不得作为完成经历。
- `[RELEASED]`：release/tag 与证据已从远端复核。

## 2. 当前安全版本与历史审计参考

### 项目标题

```text
Panda Reactive-IL：MuJoCo 视觉模仿学习抓放与扰动恢复（进行中）
```

如果版面很紧，当前阶段应写成：

```text
MuJoCo Panda 抓放：脚本专家与可审计评测（严格 M1 证据重跑中）
```

### 一句话摘要

```text
[CODE-PUBLIC] 面向机器人学习/VLA 操作评测，搭建 Panda 随机化抓放任务与
基于 6D DLS IK 的物理脚本专家；已将完整入盒判定修订为旋转角点几何门禁，当前
100-seed 逐 episode 视频与严格 provenance 证据正在 no-clobber 重跑，结果和 release
均待完成。
```

> 历史审计参考：`runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/summary.json`
> 与同目录 `episodes.jsonl` 绑定公开 `a97977a`，但使用旧中心点成功谓词。其 99/100、
> recovery、失败 seed 和平均时长必须标为 `[HISTORICAL-LOCAL]`，即使未来上传历史附件
> 也不能重解释为当前严格谓词结果。

### 可组合 bullet

**控制与系统实现**

```text
[CODE-PUBLIC] 基于 MuJoCo site Jacobian 实现世界坐标系 6D 阻尼最小二乘控制、
零空间回中、关节限幅与抓放状态机，通过环境 action 接口和物理接触完成 Panda
抓放，未直接写物体位姿或绕过连续成功判定；严格谓词批量证据与 release 待完成。
```

**批量评测与异常恢复**

```text
[HISTORICAL-LOCAL — 不得作为当前结果投递] commit `a97977a` 的旧中心点谓词运行记录
固定 100 个 seed 中成功 99/100、2 个 episode 第二次物理尝试成功、唯一失败 seed 34
（lift）及平均仿真时长 11.9375 s/episode；严格旋转角点谓词下须用新证据重新计算。
```

**工程可复核性**

```text
[HISTORICAL-LOCAL] 旧 `a97977a` 运行将每个 episode 的 seed、成功/失败、阶段、重试和最终位姿写入
JSONL，并输出汇总配置与双视角 MP4，使总体指标、恢复样本和失败案例可回溯；
该历史证据不证明当前严格谓词；新 canonical 与 release 待完成，不宣称 BC/ACT、真机
或 Sim2Real 结果。
```

## 3. 完整项目 bullet 模板（M2–M5 待真实结果）

以下段落是写作槽位，不是项目现状。只有对应行在
[`M5_DELIVERY_CHECKLIST.md`](M5_DELIVERY_CHECKLIST.md) 中变为 `RELEASED` 后，
才能移入投递版。

### M2：可审计数据链路

```text
[PENDING] 构建双视角 RGB + 本体状态 + action 的同步模仿学习数据链路，采集
[成功轨迹数] 条 train / [成功轨迹数] 条 validation 轨迹；通过 [schema/time alignment/
range/episode boundary 检查] 发现并处理 [真实异常数] 个 episode，并在 [真实回放数]
次动作回放中成功 [真实成功数] 次。
```

必须附：

- `[dataset manifest 路径/URL]`
- `[validator report 路径/URL]`
- `[replay trial-level records 路径/URL]`
- `[data version + split seed contract + commit]`

### M3：BC-1 与 ACT

```text
[PENDING] 在相同观测、数据切分和闭环协议下实现/接入 BC-1 与 ACT action-chunk
策略；使用 [真实数据量]、[真实训练 seed] 和冻结配置训练，在 [真实 trial 数] 次
[IID/OOD] 闭环评测中分别达到 [真实结果]，而非用离线 action loss 代替任务成功率。
```

必须附：

- `[checkpoint + config + data version + Git commit]`
- `[train logs and overfit-smoke evidence]`
- `[trial-level rollout records]`
- `[失败样本和策略标签明确的视频]`

不得预写：

- “ACT 优于 BC”——必须由同协议实验决定。
- “VLA”——当前固定任务没有有信息量的语言条件，不应只为包装而使用该词。
- “实现了 ACT”——若实际使用成熟库，应准确写“接入/适配”，并说明自己负责的部分。

### M4：反应性与扰动评测

```text
[PENDING] 冻结同一 ACT checkpoint，控制变量比较不同 K_exec 的闭环反应性；在
[真实 trial 数] 次 IID、OOD 与固定干预实验中报告成功率、Wilson 95% 区间、恢复率、
p95 推理延迟与失败分类。固定干预只允许在抓取前 `pregrasp` 阶段、cube 未夹持时
横移 3 cm（`[0,0.03,0] m`），并用 [真实证据] 解释 chunk 执行长度的权衡。
```

必须附：

- `[预注册/冻结的 eval protocol]`
- `[同一 checkpoint 的身份与 checksum]`
- `[逐 trial 干预记录和原始计时]`
- `[协议校验：pregrasp + cube 未夹持 + 实际横移 3 cm + 仅触发一次]`
- `[曲线底层 CSV/JSON + 绘图命令]`

M4 对错幅度或错时机 fail closed：协议违规 trial 不得计入结果、视频或 release。
不得把 M1 的第二次物理尝试恢复写成 M4 的固定干预恢复；两者干预机制和结论对象
不同。M1 的 JSON 只能直接证明 `attempts == 2` 后成功；若简历描述“运输滑落”，
必须引用已人工配对的视频或更细记录并明确标注“视觉复核”，不能把它写成机器字段结论。

### M5：交付与失败分析

```text
[PENDING] 将任务、数据、BC/ACT 基线、反应性评测和失败分类打包为可复现 GitHub
release，提供 [真实 release URL]、原始结果、方法图和 90–120 秒演示；从全新 clone
复核 [真实命令/结果]，并公开 [真实主要失败类型] 与适用边界。
```

## 4. 按岗位方向选择 bullet

### 机器人学习 / 操作策略

保留：数据契约、BC/ACT 公平对比、闭环成功、扰动反应性、失败分布。

```text
[PENDING] 面向随机化 Panda 抓放构建从脚本专家、同步视觉数据到 BC-1/ACT 闭环
评测的完整链路；在 [数据量] 与 [trial 数] 下取得 [真实结果]，并通过固定扰动和
失败分类定位 [真实结论]（MuJoCo-only）。
```

### 具身数据 / 模型评测

保留：schema、时间对齐、seed lineage、回放验收、置信区间、原始记录。

```text
[PENDING] 设计可审计的具身数据与评测协议，按 episode seed 隔离 train/val/test，
验证 [真实检查项] 并保存逐 trial 记录；在 [真实 trial 数] 下报告 [真实指标与区间]，
保留 [真实失败数量/类型] 供复核。
```

### 机器人控制 / 仿真工程

直接选用第 2 节的 `[CODE-PUBLIC]` 控制 bullet；旧批量数字只能保留为明确标注
`[HISTORICAL-LOCAL] / old center predicate` 的审计参考，不能进入当前结果 bullet。如果岗位
更重控制，应减少 ACT 术语，突出坐标系、Jacobian、DLS、限幅和接触动力学。

## 5. 最终压缩模板

### 中文两条版

```text
• [任务与本人责任]：实现 [方法]，在 [数据/seed/trial 范围] 达到 [真实结果与区间]；
  原始记录与 [commit/release] 可复核，结论限定为 [仿真/分布/策略]。
• [最有区分度的工程或实验]：冻结 [控制变量] 比较 [基线]，用 [指标/失败证据]
  定位 [真实结论]；公开 [主要失败与未完成边界]。
```

### English two-bullet version

```text
• Built [task and owned subsystem] using [method]; achieved [verified result with
  numerator/denominator or interval] over [data/seed/trial scope], with raw records
  tied to [commit/release] and conclusions limited to [simulation/distribution/policy].
• Evaluated [baselines/controlled variable] under a frozen [protocol/checkpoint],
  reporting [verified metrics] and [failure evidence]; documented [known limitations]
  instead of claiming untested real-robot or Sim2Real performance.
```

### 项目页一句话（最终版槽位）

```text
在 MuJoCo Panda 随机化抓放中，基于 [真实数据量] 比较 [真实基线]，完成 [真实评测
次数/分布] 的闭环评测并取得 [真实核心数字]；所有结果绑定 [release/tag]，不外推到
未测试的真机或 Sim2Real。
```

## 6. 数字审核表

投递前为每条出现的数字填一行；没有来源就删掉该数字或整条 bullet。

| 简历文本片段 | claim_id | 原始记录 | 计算方式 | commit/release | 审核人 | 状态 |
|---|---|---|---|---|---|---|
| `旧谓词历史 99/100` | `M1-HISTORICAL-SUCCESS` | `runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/summary.json` | `.successes / .num_episodes`；不得表述为当前结果 | `a97977a` public main / artifact local | `[待签]` | `HISTORICAL-LOCAL` |
| `旧谓词历史 2 个 episode 第二次尝试成功` | `M1-HISTORICAL-RECOVERY` | 同目录 `summary.json` + `episodes.jsonl` | `.recovered_successes == 2`，且逐行筛选 `success == true and recovered == true and attempts == 2` | `a97977a` public main / artifact local | `[待签]` | `HISTORICAL-LOCAL` |
| `旧谓词历史唯一失败 seed 34` | `M1-HISTORICAL-FAILURE` | 同目录 `episodes.jsonl` | 唯一 `success == false` 行 | `a97977a` public main / artifact local | `[待签]` | `HISTORICAL-LOCAL` |
| `旧谓词历史 11.9375 s/episode` | `M1-HISTORICAL-MEAN-TIME` | 同目录 `summary.json` | 历史固定 seed episode 仿真时长均值 | `a97977a` public main / artifact local | `[待签]` | `HISTORICAL-LOCAL` |
| `[M2 数字]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待签]` | `PENDING` |
| `[M3 数字]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待签]` | `PENDING` |
| `[M4 数字]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待签]` | `PENDING` |

## 7. 禁用表述与安全替换

| 当前不可用表述 | 原因 | 当前安全替换 |
|---|---|---|
| “完成视觉模仿学习/ACT” | M2/M3 未验证 | “正在构建；当前仅 M1 脚本专家本地通过” |
| “ACT 优于 BC” | 尚无公平闭环对比 | 删除，等待同协议结果 |
| “扰动恢复率达到……” | M4 未运行 | “已定义固定扰动协议，结果待验证” |
| “运输滑落后重抓 2 次” | 当前 episode JSON 只编码第二次尝试后成功，不编码第一次尝试的视觉事件 | “2 个 episode 通过第二次物理尝试恢复成功”；若需“运输滑落”，另引人工配对视频/细记录并标“视觉复核” |
| “seed 34 是典型策略失败” | 它只被证明是已知且唯一的 M1 失败，M3/M4 失败分布尚无结果 | “已知且唯一的 M1 失败为 seed 34”；等待 M3/M4 分布后再判断典型性 |
| “100% 可复现” | 尚未从公开 release/tag 做全新 clone 复核 | “公开 main 的提交有本地原始记录，摘要字段 `.git.tracked_worktree_clean == true`；远端 release 复核待完成” |
| “已发布 M1 完整证据包” | M1 代码已公开，但 `runs/` 原始证据与 M5 文档仍是本地素材 | “M1 代码已发布到公开 main；证据包尚未 release” |
| “真机可部署 / Sim2Real” | 只有 MuJoCo 证据 | “MuJoCo 仿真验证，未测试真机/Sim2Real” |
| “大模型/VLA 训练” | 当前没有有效语言条件或大模型训练证据 | 准确写“视觉模仿学习计划”或当前 M1 控制工作 |

## 8. 面试可复核问题

每条最终 bullet 都应能不看代码回答：

- 策略可见输入与脚本专家 privileged 信息如何隔离？
- 为什么姿态误差与 Jacobian 必须处在一致坐标系？
- 为什么 `99/100` 不能被一条成功视频替代？
- 两个 episode 的第二次物理尝试恢复说明什么，为什么单靠 JSON 不能称为“运输滑落”？
- seed 34 为什么是已知且唯一的 M1 失败，又为什么还不能称为学习策略的典型失败？
- 为什么离线 action loss 不能替代闭环成功率？
- `H_pred` 与 `K_exec` 分别控制什么，比较时冻结了哪些因素？
- 当前结论为什么只能限定在 MuJoCo，距离真机还缺哪些证据？
