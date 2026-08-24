# M5 求职交付检查表

> 用途：把项目从“本地实验”收敛为招聘方可快速理解、可下载、可复核的交付包。
>
> 当前状态：**CODE-PUBLIC / EVIDENCE-LOCAL / RELEASE-PENDING**。截至 2026-08-24，
> M1 代码提交 [`a97977ad5267562f97b5c1a43623e0a2ce23d79c`](https://github.com/rikfish163-rgb/embodied/commit/a97977ad5267562f97b5c1a43623e0a2ce23d79c)
> 已精确发布到公开 GitHub `main`，对应 CI
> [run 32658620777](https://github.com/rikfish163-rgb/embodied/actions/runs/32658620777)
> 已完成且结论为 `success`。canonical M1 运行产物、本文和同批 M5 文档仍仅在
> 本地，尚未组装或发布为 GitHub release；公开仓库当前也没有 tag/release。
> M2–M5 均为 **PENDING / 待验证**，不得把占位字段改写成完成态。

## 1. 状态标签与填写规则

所有交付物、简历 bullet、图表和视频镜头都必须带下列状态之一：

| 标签 | 含义 | 允许对外使用 |
|---|---|---|
| `VERIFIED-LOCAL` | 本地原始证据可复核，但尚未随 release 远端发布；该标签不表示代码未 push | 可用于内部审阅；对外时必须同时写明 evidence-local/release-pending |
| `RELEASED` | 原始记录、代码提交、文档和发布附件已绑定到同一 release/tag，并从远端复核 | 可以 |
| `PENDING` | 只有协议、模板或计划，没有完成实验 | 不得写成结果 |
| `BLOCKED` | 所需代码、素材或证据不存在/不完整 | 不得用替代素材假装完成 |
| `N/A` | 经说明后不适用 | 可以，但必须保留原因 |

填写规则：

- 结果数字只能来自锁定的机器可读产物，不从终端截图、口头回忆或最佳单例抄写。
- 每个结果必须能回到：实验 ID → 原始记录 → 配置/seed → Git commit → 汇总脚本。
- 百分比必须同时给出分子/分母；均值必须给出所汇总的样本集合。
- 视频只证明该 episode 的可视行为，不单独证明总体成功率。
- 旧成功谓词下的 M1 数字只允许写成带 commit 的历史事实；在严格旋转角点谓词下
  重新生成 no-clobber JSONL、summary 和逐 episode 视频前，不得写成当前 M1 结果。
- `README` 中的最低线、优秀目标和未来 trial 数是协议，不是已取得结果。
- M2 数据、M3 BC/ACT、M4 IID/OOD/扰动/延迟、M5 发布状态当前全部保持 `PENDING`。
- M4 的唯一有效固定干预锁定为：进入抓取前 `pregrasp` 固定阶段、确认 cube 尚未被
  夹持时，将 cube 横移 3 cm（协议向量 `[0, 0.03, 0] m`），且每个 trial 只触发一次。
  实际幅度不是 3 cm 或触发时机不是该固定阶段时，该 trial、视频和汇总一律判为
  协议违规并 fail closed，不得进入 checklist 勾选、release 或恢复率结果。

## 2. 当前允许引用的事实

下表是本版交付材料唯一可预填的结果数字。显示值与来源路径必须一起保留。

| claim_id | 可引用表述 | 机器可读来源 | 复核字段/方法 | 当前状态 |
|---|---|---|---|---|
| `M1-HISTORICAL-FIXED-SEEDS` | commit `a97977a` 的旧谓词历史运行包含固定 seed 0–99 共 100 个 episode | `runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/summary.json` | `.seed_start == 0` 且 `.num_episodes == 100`；必须同时标注旧谓词 | `VERIFIED-LOCAL` |
| `M1-HISTORICAL-SUCCESS` | 旧谓词历史运行记录成功 99/100 | 同上 | `.successes == 99`；`.success_rate == 0.99`；不得表述为当前严格谓词结果 | `VERIFIED-LOCAL` |
| `M1-HISTORICAL-RECOVERY` | 旧谓词历史运行记录 2 个 episode 通过第二次物理尝试恢复成功 | 同上 + 同目录 `episodes.jsonl` | `.recovered_successes == 2`；历史事实，不沿用到修订后 | `VERIFIED-LOCAL` |
| `M1-HISTORICAL-FAILURE` | 旧谓词历史运行唯一失败为 seed 34 / `lift` | 同目录 `episodes.jsonl` | 历史运行筛选仅一行 `.seed == 34`、`.failure_stage == "lift"` | `VERIFIED-LOCAL` |
| `M1-HISTORICAL-MEAN-TIME` | 旧谓词历史运行平均仿真时长 11.9375 s/episode | 同目录 `summary.json` | `.mean_sim_time_s == 11.937500000000604`；对外必须带历史限定 | `VERIFIED-LOCAL` |
| `M1-HISTORICAL-VIDEO-DECODE` | 旧谓词历史运行的 100 个逐 seed MP4 全部可逐帧解码 | 同目录 `videos/seed_0000.mp4` 至 `seed_0099.mp4` | 只证明历史文件可解码，不证明当前严格谓词 | `VERIFIED-LOCAL` |
| `M1-CURRENT-STRICT-CANONICAL` | 严格旋转角点谓词下的 100-seed JSONL、summary 与逐 episode 视频 | `[待新 no-clobber run]` | `[成功率 / recovery / failure / 100 MP4 decode / provenance]` | `PENDING` |

人工复核命令：

```bash
jq '{seed_start,num_episodes,successes,success_rate,recovered_successes,failure_counts,mean_sim_time_s,git}' \
  runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/summary.json

jq -c 'select(.recovered == true or .success == false) |
  {seed,success,recovered,failure_stage,attempts,sim_time_s}' \
  runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/episodes.jsonl
```

### M1 provenance 状态

历史 canonical M1 证据目录是
`runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/`：

- `summary.json` 记录 commit `a97977ad5267562f97b5c1a43623e0a2ce23d79c`；
- `.git.tracked_worktree_clean == true`、`.record == "all"`；前者表示该运行绑定提交时
  已跟踪文件干净，不把未跟踪或被忽略的本地产物误写成“整个工作区干净”；
- 同目录有完整 `summary.json`、`episodes.jsonl` 和逐 seed MP4；
- `videos/` 中 100 个逐 seed MP4 已于 2026-08-24 用 `ffmpeg` 全帧解码复核，
  100/100 通过；这只证明本地文件可解码，不代表附件已发布；
- 当前 `summary.json` 的 SHA-256 是
  `caa3d21ba56e9a8fc31400db93970d2752779eb5bd9e5bac8e9e84f007ac2635`；
- 当前 `episodes.jsonl` 的 SHA-256 是
  `41380a1b4e4eeaa52462843aa73c8b4f47bdb995e07b046d3a82865ff21fc9c8`。

这些产物位于被 `.gitignore` 排除的 `runs/`，目前仍是本地证据，不等于 GitHub
release 已可下载。发布时应以 canonical 目录为源生成 manifest/附件并重新核对
checksum。较早的 `codepath_100seed_20260824T0200Z` 虽得到相同汇总数字，但其
Git 元数据是旧提交加 dirty worktree，不应再作为对外 canonical 证据。

当前 M1 报告 [`M1_EXPERT_REPORT.md`](M1_EXPERT_REPORT.md) 是人读摘要；发生差异时，
以锁定的 `summary.json`、`episodes.jsonl` 和对应代码提交为准。

## 3. 证据与指标来源表（后续填充）

禁止删除 `PENDING` 行；完成一项实验时，把占位符替换为真实路径，并附生成命令。

| 阶段 | 指标/结论 | 展示值 | 原始来源 | 配置与 commit | 汇总/绘图入口 | 状态 |
|---|---|---|---|---|---|---|
| M1 历史 | 旧中心近似谓词的固定 seed 成功率 | `99/100` | `runs/m1/final_a97977a_100seed_all_egl_20260823T183117Z/{summary.json,episodes.jsonl}` | commit `a97977a`；不得表述为当前严格谓词 | `python -m expert.evaluate` | `VERIFIED-LOCAL` |
| M1 历史 | 旧谓词第二次物理尝试恢复 | `2` 个 episode 成功 | 同上 | 同上；历史限定 | `jq` 筛选历史记录 | `VERIFIED-LOCAL` |
| M1 历史 | 旧谓词唯一失败 | `seed 34 / lift` | 同上 `episodes.jsonl` | 同上；历史限定 | 失败行筛选 | `VERIFIED-LOCAL` |
| M1 历史 | 旧谓词平均仿真时长 | `11.9375 s/episode` | 同上 `summary.json` | 同上；历史限定 | JSON 汇总字段 | `VERIFIED-LOCAL` |
| M1 历史 | 旧谓词视频可解码性 | `100/100` | 同目录 `videos/seed_*.mp4` | 同上；只证明历史文件可解码 | 逐文件 `ffmpeg` 全帧解码 | `VERIFIED-LOCAL` |
| M1 当前 | 严格旋转角点谓词正式验收 | `[待填]` | `[新 canonical JSONL / summary / videos]` | `[当前 commit / asset provenance]` | `[100-seed all-video no-clobber run]` | `PENDING` |
| M2 | 数据集规模、切分、schema 检查、回放验收 | `[待填]` | `[dataset manifest / validation report]` | `[config / data version / commit]` | `[validator command]` | `PENDING` |
| M3 | BC-1 训练与闭环结果 | `[待填]` | `[train log / checkpoint / rollout JSONL]` | `[config / seeds / commit]` | `[evaluation command]` | `PENDING` |
| M3 | ACT 训练与闭环结果 | `[待填]` | `[train log / checkpoint / rollout JSONL]` | `[config / seeds / commit]` | `[evaluation command]` | `PENDING` |
| M4 | IID / OOD 成功率与 Wilson 区间 | `[待填]` | `[trial-level records]` | `[frozen checkpoint / protocol / commit]` | `[analysis command]` | `PENDING` |
| M4 | 抓取前固定阶段 cube 横移 3 cm 的恢复率 | `[待填]` | `[trial-level records + intervention logs + videos]` | `[same checkpoint / pregrasp / ungrasped / [0,0.03,0] m]` | `[protocol validator + analysis command]` | `PENDING` |
| M4 | `K_exec`—恢复率曲线 | `[待填]` | `[plot data CSV/JSON]` | `[same checkpoint / varied K_exec only]` | `[plot command]` | `PENDING` |
| M4 | 推理延迟与动作平滑度 | `[待填]` | `[raw timing/action trace]` | `[hardware / warm-up / sync protocol]` | `[analysis command]` | `PENDING` |
| M4 | 失败分类 | `[待填]` | `[annotated failure manifest]` | `[taxonomy version / reviewer]` | `[report generator]` | `PENDING` |
| M5 | 演示视频、方法图、结果表、失败分析 | `[待填 release URLs]` | `[release attachments + checksums]` | `[release tag / commit]` | `[build/export command]` | `PENDING` |

## 4. 交付物清单

### 4.1 仓库内小文件

- [ ] README 的当前阶段、复现命令、结果和边界与 release 一致。
- [ ] 方法图同时提供可编辑源文件和导出的 SVG/PNG。
- [ ] 主结果表同时提供渲染图和底层 CSV/JSON。
- [ ] `K_exec`—恢复率曲线同时提供渲染图、底层数据和绘图命令。
- [ ] 失败分析包含分类定义、样本索引、证据片段和下一步，而不只放“失败截图”。
- [ ] 许可证、MuJoCo Menagerie 来源和第三方模型/权重归属完整。
- [ ] 所有文档内链在全新 clone 中可打开。

### 4.2 Release 附件

- [ ] `demo.mp4`：遵循 README 的 90–120 秒最终版本；固定扰动镜头只能使用抓取前
  `pregrasp` 阶段、cube 横移 3 cm 的有效 M4 trial。错幅度或错时机的素材 fail closed，
  不得剪入 release。片中 seed 34 只称“已知且唯一的 M1 失败”；最终学习策略的
  “典型失败”必须等 M3/M4 失败分布支持后再选。
- [ ] `poster.png`：不依赖视频即可看懂任务、方法、关键结果和未完成边界。
- [ ] `method.svg` / `method.png`：按 [`DEMO_STORYBOARD.md`](DEMO_STORYBOARD.md) 的文字规范绘制。
- [ ] `main_results.csv` / `main_results.png`：每个展示值可回到 trial-level 记录。
- [ ] `reactivity_curve.csv` / `reactivity_curve.png`：同一冻结 checkpoint，只改变已声明变量。
- [ ] `failure_analysis.pdf` 或 Markdown 导出：至少包含样本索引、现象、原因假设和可证伪的下一步。
- [ ] `evidence_manifest.json`：列出每个附件的来源实验、commit、配置和 SHA-256。
- [ ] `SHA256SUMS`：覆盖所有 release 附件。

### 4.3 简历和投递材料

- [ ] 从 [`RESUME_DRAFT_TEMPLATE.md`](RESUME_DRAFT_TEMPLATE.md) 选择与岗位匹配的 bullet。
- [ ] 每条 bullet 都含“动作 + 方法 + 可复核证据 + 边界”。
- [ ] M2–M4 未验证前，只使用 M1-safe 版本；不得出现“训练了 BC/ACT”或“完成扰动恢复评测”。
- [ ] GitHub 链接打开后默认落到与简历数字一致的 release/tag，而不是漂移中的分支页面。
- [ ] 中文/英文简历中的数字、单位、分母和限定词逐项一致。
- [ ] 准备一段不看稿的项目口述，能解释输入输出、泄漏边界、失败和为什么离线 loss 不等于闭环成功。

## 5. GitHub release 门禁

### A. 冻结实验前

- [ ] 写明 release 候选 commit；确认没有未授权或无关改动。
- [ ] `git status --short --branch` 干净，或明确记录每个残留文件的归属。
- [ ] 记录 Python、MuJoCo、CUDA/设备信息和依赖锁文件版本。
- [ ] 固定数据版本、训练配置、评测 seed、checkpoint 和失败分类规则。
- [ ] 在看正式结果前冻结评测协议；禁止看结果后挑有利 seed 或指标。

### B. 生成证据

- [ ] 从候选 commit 的干净环境运行最小 smoke 和完整目标验收。
- [ ] 原始输出写入新的、不可覆盖目录。
- [ ] 保存逐 trial 记录，不只保存 summary。
- [ ] 对 summary、原始记录、checkpoint、图表数据和视频生成 SHA-256。
- [ ] 抽查汇总值能由原始记录重新计算；抽查视频对应的 seed/策略/配置无误。
- [ ] 记录失败样本，不删除已知且唯一的 M1 失败或难看片段。

### C. 组装发布包

- [ ] 文档中的每个数字在证据来源表中有唯一 `claim_id`。
- [ ] 图题写明环境、分布、trial 集合、策略/checkpoint 和误差条定义。
- [ ] 视频角标区分 `scripted expert`、`BC-1`、`ACT`，避免观众误认。
- [ ] M4 干预日志同时证明 `stage == pregrasp`、cube 未夹持、实际横移为 3 cm 且仅
  触发一次；任一条件不满足就阻断 checklist、视频与 release，不以重命名素材放行。
- [ ] 仿真结论明确限定为 MuJoCo；没有真机/Sim2Real 证据就不使用相应表述。
- [ ] 大数据、checkpoint、原始视频不误提交进 Git；通过 release 附件或外部版本化存储分发。
- [ ] release notes 同时列“已验证”“已知失败”“尚未完成”。

### D. 发布与远端复核

- [ ] 仅在获得本批 M5 文档/附件的提交、push、tag 和 release 授权后执行外部写操作。
- [ ] tag 精确指向候选 commit；release 页面显示同一 tag。
- [ ] 从全新临时目录 clone/tag，按 README 复现 smoke 和至少一个可审计评测入口。
- [ ] 以未登录浏览器检查 README、图片、视频、附件和许可证链接。
- [ ] 下载 release 附件并用 `SHA256SUMS` 复核。
- [ ] 将最终远端 URL 回填到简历和证据表；此时才能把相应状态改为 `RELEASED`。

### Release notes 模板

```markdown
## What is verified

- [只写有 claim_id、原始记录和 commit 的结果]

## Reproduce

- Environment: [lockfile / hardware note]
- Commands: [exact commands]
- Evidence: [manifest and raw-record links]

## Known failures

- [保留的失败类型、样本索引和影响]

## Not claimed

- [未完成的 M2–M5 项、真机/Sim2Real 边界]
```

## 6. 当前真实素材盘点与缺口

### 已存在但仍是本地素材

- M1 canonical 目录含 100-episode `summary.json`、`episodes.jsonl` 和逐 seed MP4，
  可支持第 2 节六项机器事实与正常/第二次尝试恢复/失败 episode 回溯。当前 JSON
  只能证明 2 个 episode 在 `attempts == 2` 后成功，不能单独证明第一次尝试发生了
  “运输滑落”；该视觉描述必须另引已人工配对的视频或更细记录，并标明“视觉复核”。
- `python -m expert.evaluate --record all|failures` 能生成前视/腕部并排的 H.264 MP4；默认从控制循环按 stride 采帧。
- canonical 批量录制与 `a97977a` 配对，且摘要记录 `.git.tracked_worktree_clean == true`；
  但目录被 `.gitignore` 排除，
  尚未组装成带 manifest/checksum 的公开附件。
- `python -m env.pick_place` 当前能向 `/tmp` 写前视和腕部 PNG；仓库尚无专用、版本化的截图导出 CLI。

### 发布前仍缺少的真实素材

| 缺口 | 当前边界 | 完成条件 |
|---|---|---|
| M1 证据包与成片素材 | canonical 原始记录和 100 个可解码 MP4 仅在本地；恢复机器字段不直接编码“运输滑落” | 生成 manifest/全附件 checksum，完成正常、第二次尝试恢复、失败片段的裁切与标注；若称“运输滑落”，附人工配对视频/细粒度记录和视觉复核标记；提供稳定下载链接 |
| M2 数据链路 | `PENDING` | 数据 manifest、验证报告、回放记录和可公开样例齐全 |
| M3 学习策略 | `PENDING` | 真实训练日志、checkpoint、BC-1/ACT 闭环 trial 与失败视频齐全 |
| M4 反应性评测 | `PENDING`；当前不可展示扰动恢复率 | 抓取前 `pregrasp` 固定阶段、cube 未夹持、横移 3 cm 的干预记录通过协议校验；主结果表、`K_exec` 曲线底层数据、区间和延迟原始记录齐全；错幅度/错时机 fail closed |
| 图与最终视频 | 仅有文字规范 | 方法图可编辑源/导出图和 README 约定的 90–120 秒成片、字幕、旁白、封面及许可确认齐全 |
| 远端发布 | M1 代码已公开；M5 文档/附件尚未发布 | tag/release、附件下载校验和远端复现记录齐全后，才能写“已发布” |

## 7. 最终签字

发布者在 release 前逐项回答：

- [ ] 我能从每条简历数字定位到原始记录，而不是只定位到 README。
- [ ] 我没有把脚本专家结果写成学习策略结果。
- [ ] 我只把机器记录表述为“2 个 episode 通过第二次物理尝试恢复成功”；若写
  “运输滑落”，我同时提供人工配对视频/更细记录并标明视觉复核。
- [ ] 我没有把 M1 的第二次尝试恢复写成 M4 固定干预恢复；M4 只接受抓取前
  `pregrasp` 阶段 cube 横移 3 cm，错幅度或错时机均 fail closed。
- [ ] 我没有把协议门槛或目标值写成实际结果。
- [ ] 我保留 seed 34，并只称其为“已知且唯一的 M1 失败”；没有 M3/M4 失败分布时，
  不把它预称为最终学习策略 demo 的“典型失败”。
- [ ] 我明确区分 code-public、evidence-local、release-pending、MuJoCo-only 和 no-Sim2Real 边界。
- [ ] 我在远端重新下载并复核过最终交付，而不只确认命令已启动。
