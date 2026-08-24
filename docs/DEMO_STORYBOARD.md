# 90–120 秒最终 Demo 分镜与方法图文字规范

> 当前状态：**PENDING / BLOCKED / RELEASE-PENDING**。
> M1 代码提交 [`a97977ad5267562f97b5c1a43623e0a2ce23d79c`](https://github.com/rikfish163-rgb/embodied/commit/a97977ad5267562f97b5c1a43623e0a2ce23d79c)
> 已在公开 `main`，对应 CI
> [run 32658620777](https://github.com/rikfish163-rgb/embodied/actions/runs/32658620777)
> 为 `success`；分镜文档、原始证据附件和 release 仍未发布。
> 下表采用 100 秒最终交付参考分镜；时间码是编辑计划，不是实验指标。
> 当前只有 M1 脚本专家代码事实可以填实；旧 `a97977a` 数字只允许带
> `HISTORICAL-LOCAL / old center predicate` 水印作内部审计参考，不得作为当前成片结果。
> M2–M4 的策略、固定扰动和结果镜头保持
> `PENDING`，没有原始记录和配对视频前不得用动画或 M1 第二次尝试恢复片段冒充。
> M4 固定干预已锁定为进入抓取前 `pregrasp` 固定阶段、cube 尚未夹持时横移 3 cm
>（`[0,0.03,0] m`），每个 trial 只触发一次；错幅度或错时机的 trial/镜头 fail closed，
> 不得进入成片、指标或 release。

## 1. 成片目标

招聘方看完应能回答：

1. 任务是什么，策略真正能看到什么？
2. 当前已完成的是脚本专家，还是已经训练的 BC/ACT？
3. action chunk 如何进入闭环，为什么 `K_exec` 影响反应性？
4. 成功、固定扰动恢复和失败分别有什么原始证据？
5. 哪些结论只在 MuJoCo 成立，哪些阶段仍未完成？

剪辑原则：

- 先给任务行为，再给方法，不用长片头和无信息空镜。
- 每个行为镜头常驻策略/控制器标签，禁止把 `scripted expert` 标成 `ACT`。
- 双视角素材保持前视/腕部并排；裁切时保留视角标签。
- 结果角标必须带 trial 集合或 `claim_id`，不能只写百分比。
- 失败镜头保留完整因果链，不只截最后一帧。
- 旁白只解释画面和证据，不朗读 README。

## 2. 100 秒最终分镜

> 最终 M5 版本必须含正常成功、合规的固定干预恢复和有证据的失败。seed 34 只能称
> “已知且唯一的 M1 失败”；除非 M3/M4 失败分布支持，不得在最终学习策略 demo 中
> 预称它或任一单例为“典型失败”。当前可先做 M1-only 粗剪，但第 7、8 镜头只有在
> M4 真实完成并具备本地可复核证据后才能转为 `VERIFIED-LOCAL`。

| 时间码 | 画面 | 屏幕文字 | 旁白要点 | 证据/素材 | 状态 |
|---|---|---|---|---|---|
| `00:00–00:06` | 最清楚的抓取到放置瞬间，前视/腕部并排 | `MuJoCo Panda · Reactive Imitation Learning Benchmark` | “目标不是单个好看 demo，而是从专家、数据、策略到反应性评测的可审计闭环。” | 最终应使用已配对的成功 trial | `M1 候选可用；策略版 PENDING` |
| `00:06–00:14` | 任务随机化静帧；cube、box 与两相机用简洁标注 | `Input: front RGB + wrist RGB + proprioception` | 说明策略输入；privileged cube/box/TCP 真值只给脚本专家与评测，不进入学习策略。 | 场景静帧 + 协议文字 | `协议已定义；策略实现 PENDING` |
| `00:14–00:25` | 方法图由左到右逐层高亮 | `observe → encode → action chunk → execute → re-observe` | 说明 `H_pred` 是预测长度，`K_exec` 是重新观察前的执行长度；二者不能混同。 | 第 4 节方法图 | `图待绘制；M3 路径 PENDING` |
| `00:25–00:38` | M1 控制链：TCP 误差、Jacobian、DLS、关节目标；旧 rollout 若叠加须带历史水印 | `M1 scripted expert · privileged controller` | “专家通过世界系 6D DLS IK 和状态机生成物理 action，不写 cube 位姿，也不绕过成功判定。” | `src/expert/scripted.py` 为 `CODE-PUBLIC`；旧配对 MP4 为 `HISTORICAL-LOCAL` | `代码 CODE-PUBLIC；当前严格行为证据 PENDING` |
| `00:38–00:51` | 当前严格谓词的新 canonical 成功全流程；尚未生成前只能内部展示旧片并常驻历史水印 | `[PENDING] Current strict M1 result` | 新运行完成前不得念 99/100 或 11.9375 秒；旧片旁白只能说“这是 `a97977a` 旧中心点谓词的历史运行”。 | 当前严格 `M1-CURRENT-STRICT-CANONICAL` 待生成；旧 `M1-HISTORICAL-*` 仅审计参考 | `CURRENT PENDING；旧片 HISTORICAL-LOCAL` |
| `00:51–01:03` | 当前严格谓词的新 recovery 素材；尚未生成前不得沿用旧 recovery 数字 | `[PENDING] Current strict recovery evidence` | 旧记录中的 2 个第二次尝试成功仅属于 `a97977a` 历史谓词；机器字段不单独证明第一次是运输滑落，也不是策略固定干预实验。 | 当前严格 recovery 待新 JSONL/视频；旧 `M1-HISTORICAL-RECOVERY` 仅审计参考 | `CURRENT PENDING；旧片 HISTORICAL-LOCAL` |
| `01:03–01:17` | 进入抓取前 `pregrasp` 固定阶段且 cube 未夹持时，cube 按 `[0,0.03,0] m` 横移 3 cm；画面显示实际触发阶段/位移、`K_exec` 与重新观察时刻 | `Frozen checkpoint · pregrasp · cube +3 cm lateral` | 只在 M4 完成后说真实恢复结果；同一 checkpoint、只改变声明变量。错幅度或错时机的 trial/镜头 fail closed。 | M4 trial JSONL + 干预日志 + MP4 + 协议校验 | `PENDING / 当前不得填结果` |
| `01:17–01:30` | `K_exec`—恢复率曲线与主结果表，逐项出现置信区间/延迟/失败类型 | `IID · OOD · recovery · latency` | 用实际数据说最重要的一条结论；没有完整格子就不展示“胜出”。 | 底层 CSV/JSON + 绘图命令 | `PENDING` |
| `01:30–01:40` | 当前严格谓词失败链；若暂用 seed 34 旧片，必须全程标 `a97977a historical old predicate` | `[PENDING] Current strict failure`；`MuJoCo only · release URL: PENDING` | 旧 seed 34 只是在旧谓词运行中唯一失败，不代表当前 M1 或学习策略的典型失败；未测试真机或 Sim2Real。 | 当前严格失败待新证据；旧 `M1-HISTORICAL-FAILURE` 仅审计参考 | `CURRENT PENDING；旧片 HISTORICAL-LOCAL` |

参考总时长：100 秒，最终成片必须保持在 README 约定的 90–120 秒。调整时优先改变
片头和方法图动画长度，不删除失败镜头、证据来源或边界声明。

## 3. 当前可制作的 M1-only 粗剪

在 M2–M4 未完成时，可以制作内部审阅版，但它不是当前结果成片，且标题必须明确为：

```text
M1 Scripted Expert — Historical a97977a Old-Predicate Cut (Internal Only)
```

内部粗剪只允许使用第 5 节的历史 M1 字幕和第 7 节前三类已有素材，并始终保留
`M1 scripted expert`、`HISTORICAL-LOCAL`、`old center predicate` 和“非当前结果/非最终策略
demo”标签。它不得直接导出为公开招聘成片。

必须删除/留空：

- BC-1 或 ACT rollout；
- 固定中途扰动后的“恢复率”；
- `K_exec`—恢复率曲线；
- IID/OOD、Wilson 区间、推理延迟等尚未生成的结果；
- GitHub release 已发布的表述。

## 4. 方法图文字规范

### 4.1 信息架构

采用横向闭环，主路径从左到右，环境回箭头从右侧回到观测：

```text
MuJoCo Panda task
    → front RGB / wrist RGB / proprioception
    → visual encoder + state projection
    → BC-1 next action  OR  ACT action chunk
    → chunk executor (K_exec)
    → environment action interface
    → physics step
    ↺ next observation
```

在主路径下方放一条独立的 M1 专家支路：

```text
privileged cube / box / TCP state
    → scripted state machine + world-frame 6D DLS IK
    → expert action
    → dataset collection (M2 pending)
```

评测支路从 `physics step` 向下：

```text
fixed evaluation protocol
    → IID / OOD / pregrasp cube +3 cm intervention / latency
    → trial-level records
    → success + interval / recovery / failure taxonomy
```

### 4.2 必须表达的边界

- 用竖直“policy boundary”分隔策略可见输入和 privileged 信息。
- privileged cube/box/TCP 真值不得连到 encoder 或 learned policy。
- `H_pred` 标在 ACT 输出上；`K_exec` 标在执行器上，不写成同一个 chunk length。
- M1 专家代码支路用实线并标 `CODE-PUBLIC`；旧运行镜头只能标
  `HISTORICAL-LOCAL / old center predicate`，当前严格结果节点仍为 `PENDING`。
- M2、M3、M4 尚未验证的节点用灰色虚线并标 `PENDING`。
- 固定干预从环境侧注入，不从模型输入侧偷偷添加真值信号；唯一有效 R1 事件是
  `pregrasp` 进入时、cube 未夹持、按 `[0,0.03,0] m` 横移 3 cm 且仅触发一次。
- 图注写清“系统目标图不代表所有节点均已实现”。

### 4.3 视觉语言

| 元素 | 规范 |
|---|---|
| 已公开 M1 代码 | 深蓝实线、白底；角标 `CODE-PUBLIC`；当前严格结果另用 `PENDING` |
| 旧 M1 运行 | 蓝灰实线、浅灰底；常驻角标 `HISTORICAL-LOCAL / old center predicate` |
| 待验证 M2–M4 | 中灰虚线、浅灰底；角标 `PENDING` |
| 数据/记录 | 青绿色；使用文档或数据库形状，不画成模型 |
| 固定干预 | 橙色箭头，从评测协议指向环境 |
| 失败路径 | 红色细线，只连接到失败分类，不覆盖主路径 |
| 禁止信息流 | 红色断线并加叉，例如 privileged state → policy |

版式要求：

- 白底或近白底，避免黑底霓虹和装饰性 3D 图标。
- 最多两层主信息：主闭环 + 专家/评测支路；细节放图注。
- 节点采用短名词，动词放箭头；不在框内塞整段说明。
- 导出 SVG 与高分辨率 PNG；在 README 宽度和视频画面中都能读清。
- 图题建议：`Panda Reactive-IL: auditable data-to-policy closed loop`。

### 4.4 当前与最终两版图

**当前 M1 图**：只高亮 `privileged state → DLS expert → action → physics → records`；
其余节点全部灰色 `PENDING`。

**最终 M5 图**：只有当 M2–M4 逐项有原始证据后，才把相应节点转为实线。不要仅因
代码文件存在就把节点标成完成；以闭环验收为准。

## 5. 字幕与旁白模板

### 固定角标

```text
Controller: [scripted expert / BC-1 / ACT]
Split: [fixed seeds / IID / OOD / intervention]
Evidence: [claim_id or experiment_id]
Status: [VERIFIED-LOCAL / RELEASED / PENDING]
```

### M1 历史字幕（仅内部审计参考，不是当前结果）

```text
HISTORICAL-LOCAL — a97977a — old center predicate
Historical fixed seeds: 99/100 success
Historical second-attempt physical successes: 2 episodes
Historical run's only failure: seed 34 (lift)
Historical mean simulation time: 11.9375 s/episode
NOT THE CURRENT STRICT M1 RESULT
```

来源统一指向：

```text
runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/{summary.json,episodes.jsonl}
```

### M4 结果字幕（未验证占位）

```text
[PENDING — DO NOT EXPORT]
Checkpoint: [id]
Protocol: [id]
Intervention: pregrasp, cube ungrasped, lateral +3 cm ([0,0.03,0] m), once
K_exec: [value]
Recovery: [successes/trials, interval]
Latency: [definition and verified value]
```

## 6. 素材采集与配对

### 现有视频入口

canonical M1 目录已有 100 个逐 seed MP4；2026-08-24 使用 `ffmpeg` 全帧解码复核
100/100 通过。该检查只证明本地文件可解码，不表示素材已经随 release 发布。
仓库当前的正式录制入口是：

```bash
source ./env.sh
MUJOCO_GL=egl python -m expert.evaluate \
  --seed-start [seed] \
  --num-seeds [count] \
  --required-successes [threshold] \
  --output-dir runs/m1/[new-run-id] \
  --record all
```

它会录制前视/腕部并排 H.264 MP4；输出目录拒绝覆盖。`--record failures` 可在批量
评测后只重放失败 episode，但必须检查重放结果与原始判定一致。

### 现有静帧入口

```bash
source ./env.sh
MUJOCO_GL=egl python -m env.pick_place
```

当前实现只向 `/tmp/task_front.png` 与 `/tmp/task_wrist.png` 写静帧，不会生成带实验
manifest 的发布素材。用于最终方法图或封面时，应把静帧复制到版本化素材目录并在
manifest 中记录生成 commit、seed、命令和 SHA-256；本任务不执行该素材写入。

### 每个镜头的 manifest 行

```text
asset_id: [stable id]
experiment_id: [run id]
controller: [scripted expert / BC-1 / ACT]
seed_or_trial_id: [id]
source_video: [path]
source_record: [JSONL path + row key]
commit: [sha]
config: [path/checksum]
trim: [in/out]
transform: [crop/speed/overlay only; no hidden cuts]
sha256: [digest]
status: [HISTORICAL-LOCAL / CURRENT-PENDING / RELEASED]
```

配对门禁：

- [ ] 视频对应的 episode 在 JSONL 中存在且 controller/seed/trial 一致。
- [ ] 成功、恢复、失败标签来自评测记录，不由剪辑者凭画面猜测。
- [ ] `M1-HISTORICAL-RECOVERY` 机器字幕必须先写明 `a97977a / old center predicate`，
  再写“历史记录中 2 个 episode 通过第二次物理尝试恢复成功”；
  “运输滑落/重抓”只能来自已配对视频或更细记录，并在 manifest 标 `visual-review`。
- [ ] 慢放、裁切或加速写入 `transform`；不能剪掉会改变结论的中间过程。
- [ ] M1 第二次尝试恢复与 M4 固定干预恢复使用不同标签和不同素材 ID。
- [ ] M4 镜头的日志证明 `pregrasp`、cube 未夹持、实际横移 3 cm、仅触发一次；
  错幅度或错时机即 fail closed，不进入成片、结果或 release。
- [ ] 所有对外片段都能从 release 附件或稳定链接取得原文件。

## 7. 当前素材缺口与替换单

| 分镜 | 当前候选 | 缺口 | 替换条件 |
|---|---|---|---|
| 正常成功 | canonical M1 目录含 100 个配对双视角 MP4，`M1-VIDEO-DECODE` 为 100/100 | 尚未裁切、加角标并写素材 manifest | 从配对 trial 选片并锁定 checksum |
| M1 第二次尝试恢复 | canonical M1 记录含 2 个 `attempts == 2` 后成功的 episode 和对应 MP4 | 尚未裁切；机器字段不证明“运输滑落” | 依据 episode record 选片并锁定 checksum；若写“运输滑落/重抓”，追加人工配对视频/细记录和 `visual-review` 标记 |
| seed 34 失败 | canonical M1 目录含配对失败 MP4 | 尚未裁切、解释并写入发布 manifest；不能预称学习策略“典型失败” | 保留完整因果链并锁定 checksum；仅标“已知且唯一的 M1 失败”，典型性等待 M3/M4 分布 |
| 固定干预恢复 | 无 | M4 尚未完成 | 真实日志证明抓取前 `pregrasp`、cube 未夹持、横移 3 cm、仅一次；冻结 checkpoint、trial record、视频齐全；错幅度/错时机 fail closed |
| 主结果表/曲线 | 无 | M3/M4 原始结果不存在 | 从锁定原始记录生成，不手工填图 |
| 方法图 | 本文仅有文字规范 | 缺可编辑源和导出图 | 按第 4 节绘制并过信息流审查 |
| 结尾 release 链接 | M1 代码已在公开 `main`；无 release URL | M5 文档/附件尚未 tag/release | 远端发布并从未登录环境复核 |

## 8. 导出前终检

- [ ] 最终成片总时长处于 90–120 秒，音轨、字幕和画面同步。
- [ ] 每个策略画面都有 controller 标签；M1 不伪装成 BC/ACT。
- [ ] 所有结果数字存在于 [`M5_DELIVERY_CHECKLIST.md`](M5_DELIVERY_CHECKLIST.md) 的证据表。
- [ ] 正常成功、固定扰动恢复和失败均能回到 trial-level 记录。
- [ ] 固定干预镜头只来自抓取前 `pregrasp` 阶段 cube 横移 3 cm 的合规 trial；错幅度
  或错时机的镜头 fail closed，不导出、不计数、不发布。
- [ ] seed 34 只标为已知且唯一的 M1 失败，没有被美化成恢复成功，也没有在缺少
  M3/M4 失败分布时预称为学习策略的“典型失败”。
- [ ] 图表不是截图手填，底层数据随 release 提供。
- [ ] 结尾明确 `MuJoCo only`，没有真机/Sim2Real 暗示。
- [ ] GitHub release URL、tag、附件和 SHA-256 已从远端下载复核；否则保留 evidence-local/release-pending 标识。
