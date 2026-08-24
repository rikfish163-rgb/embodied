# M2 HDF5 数据契约（schema v1）

本文冻结 M2 单 episode 文件的可审计存储契约。它描述已经实现的 writer、reader
和 validator，不代表已经采集 200 条训练成功轨迹、40 条验证成功轨迹，也不代表
M2 门禁已经通过。

## 文件与发布边界

- 一个 HDF5 文件只包含一个 episode；`T = num_steps >= 1`。
- writer 每次只追加一个 transition，图像不会整集缓存在内存中。图像 dataset
  以单帧为 chunk 并使用 LZF；状态、动作、时间戳和阶段按可扩展的一维时间轴写入。
  writer 的 state/action/timestamp/stage 时间轴 chunk 分别固定为 256 行，stage 的每行
  是 64-byte fixed UTF-8 item。
- 写入期间使用目标目录中的隐藏临时文件
  `.<target-name>.partial-<uuid>`。只有完成元数据写入、flush、fsync 和完整校验均成功
  后才用同文件系统 hard link 原子发布目标名。
- 目标文件在构造 writer 时已经存在，或在发布前被其他进程抢先创建，都会报
  `FileExistsError`；实现绝不覆盖现有 episode。发布前异常会删除未发布的 partial。
- schema v1 是 closed-world：下表之外的根 dataset 或 group 都会被 validator
  报为错误。
- 六个 schema 根键都必须是当前 episode 文件内的普通 hard link，目标必须是六个彼此
  不同的具体 `Dataset`；不能用第二个 hard link 让双相机或任意其他 schema 键共享同一
  HDF5 object。`SoftLink`、`ExternalLink`（包括断链）、virtual dataset (`is_virtual`)
  和 external raw storage (`Dataset.external`) 一律拒绝；不能让另一个文件在本 episode
  的 SHA-256 不变时改变训练 payload。
- 根 attributes 必须精确等于本文列出的 11 个冻结字段；任何额外 scalar、text、array、
  object reference 或 region reference 都是错误。六个 schema dataset 自身的 attributes
  必须为空。validator 只枚举 attribute 名称，不读取未知值，也不沿 reference 遍历匿名
  object graph。
- 完全不可达、且没有被根键、根 attribute 或 schema dataset attribute 引用的 HDF5
  内部空闲块不属于 policy 数据，也不要求 validator 扫描；一旦对象可从这些 schema
  入口到达，对应的额外 root key/attribute 就会使文件 fail closed。

## Transition 时间语义

时间轴固定为 **20 Hz**，所以根属性 `control_dt_s` 必须是 `0.05` 秒。对所有
`t > 0`：

```text
timestamp[t] - timestamp[t - 1] = 0.05 s
```

每一行采用 pre-action 对齐：

```text
环境状态 s_t
  -> observe() 得到双相机图像与 8D state
  -> 专家根据 s_t 生成 action[t]
  -> 写入 observation[*][t], action[t], timestamp[t], stage[t]
  -> PickPlace.step(action[t]) 推进到 s_(t+1)
```

因此 `observation[*][t]` 正是产生 `action[t]` 时可见的观测；不存在把
`action[t]` 与执行后的 `observation[t+1]` 错配的情况。`timestamp[t]` 是动作执行前
的 MuJoCo 仿真秒数，必须有限、非负并严格递增。首个时间戳通常是 `0.0`，但契约只
要求它非负。最后一条动作之后不额外存一行 terminal observation。

`stage[t]` 是产生 `action[t]` 的专家状态机阶段标签，不是策略输入。

## 根 datasets

所有 shape 都以同一个时间长度 `T` 开头，且长度必须等于 `@num_steps`。
图像轴顺序是 `T, H, W, C`，颜色顺序是 RGB。

| HDF5 键 | dtype | shape | 单步语义 |
|---|---|---|---|
| `/observation.images.front` | `uint8` | `[T, 128, 128, 3]` | 动作执行前的固定前视 RGB |
| `/observation.images.wrist` | `uint8` | `[T, 128, 128, 3]` | 动作执行前的腕部 RGB |
| `/observation.state` | `float32` | `[T, 8]` | 7 个臂关节位置（rad）加 1 个归一化夹爪开度 `[0, 1]` |
| `/action` | `float32` | `[T, 8]` | 7 个绝对臂关节位置目标（rad）加 1 个归一化夹爪命令 `[0, 1]` |
| `/timestamp` | `float64` | `[T]` | pre-action 仿真时间（s） |
| `/stage` | fixed UTF-8（64-byte item） | `[T]` | 非空专家阶段标签；UTF-8 编码后最多 64 bytes |

以上六个 dataset 都不得携带 dataset-level attributes；schema v1 没有这类 metadata
扩展点。`/stage` 的 chunk layout 还必须精确为 `[256]`；variable-length string、ASCII
fixed string、其他 fixed itemsize 或其他 stage chunk geometry 都不属于 schema v1。
writer 在 resize/写盘前按 UTF-8 bytes 检查 `stage`，不会依赖 HDF5 的静默截断。

所有 chunked schema dataset 的单个 decoded chunk 固定上限为 16 MiB：
`product(chunks) * dtype.itemsize <= 16 * 1024 * 1024`。这是 validator 和 Reader 共同继承
的消费边界；即使一次逻辑 slice 只有 256 行，也不能用超大物理 chunk 迫使 HDF5 解码
巨量内存。writer 当前最大 chunk 是单帧 RGB 的 49,152 bytes，远低于该上限。

动作的规范字符串是：

```text
absolute_joint_position_targets_rad[7]+normalized_gripper_open[1]
```

`/action` 的每一维都必须落在根属性 `@action_min` 和 `@action_max` 的闭区间内。
前 7 维边界来自 Panda 臂 actuator `ctrlrange`，第 8 维固定为 `[0, 1]`。validator
使用 `1e-6` 容差处理写入 `float32` 后的边界舍入。

## 根 attributes

| 属性 | HDF5 值 | 约束 |
|---|---|---|
| `@schema_version` | integer | 固定为 `1` |
| `@seed` | integer | 非负 episode reset seed；也是 split 审计主键 |
| `@success` | boolean | 环境连续成功判定的 episode 结果 |
| `@failure_stage` | UTF-8 | 成功时为空字符串；失败时非空且必须出现在 `/stage` |
| `@num_steps` | integer | 正整数，且等于所有 dataset 的 `T` |
| `@control_dt_s` | float | 固定 `0.05`（20 Hz） |
| `@time_alignment` | UTF-8 | 固定 `pre_action` |
| `@action_semantics` | UTF-8 | 固定为上面的动作规范字符串 |
| `@action_min` | float64 array | finite `[8]`，逐维严格小于 `@action_max` |
| `@action_max` | float64 array | finite `[8]`，逐维严格大于 `@action_min` |
| `@complete` | boolean | 只有校验前的完整临时文件为 `true`；发布文件必须为 `true` |

根 attributes 的集合必须与表中 11 个名称完全相等，而不只是“至少包含”这些名称。
未知 attribute 即使值可被 HDF5 正常解码，也不能进入 schema v1。

成功 episode 用 HDF5 空字符串表示 `@failure_stage`；reader 将其恢复为 Python
`None`。`@success`、`@failure_stage`、`/stage` 与 `@seed` 共同提供 episode 级和
transition 级审计链。writer 对非空 `failure_stage` 使用与 `/stage` 相同的 64 UTF-8
bytes 上限，避免产生不可能由 canonical stage 列表示的失败标签。

## 策略 allowlist 与 privileged 边界

策略观测的唯一 allowlist 是：

```text
observation.images.front
observation.images.wrist
observation.state
```

reader 的 `Transition.observation` 只返回这三个键。`action` 是监督标签，
`timestamp` 和 `stage` 是对齐/审计字段，都不是策略输入。

脚本专家可以在生成动作时读取 cube、box、TCP 真值和随机化参数，但 schema v1
故意不把这些信息写进策略 episode；validator 也会拒绝额外的 `privileged` 根
group。需要保留这类评测证据时，应放进与策略数据分离的审计 sidecar。若未来
schema 升级后加入 `privileged/*`，仍必须位于独立 namespace，且不得被加入上述
allowlist、归一化统计或模型 batch。

allowlist 也约束图像内容，不只是 Python 键名。writer 的 `capture()` 在每一帧
`observe()` 前要求 `env.cfg.debug_viz is False`，模型中必须恰好各有一个 `tcp` 和
`flange` site，且两者 RGBA alpha 都必须有限并精确为 `0`。缺失/重复 site、NaN alpha、
cfg/model 不一致或运行中重新显示 marker 都会在图像落盘前拒绝。M2 collection
manifest 后续还必须冻结环境配置与模型 hash；单 episode schema 不能替代该集合级证据。

## Train / validation / test 的 seed 隔离

split 以 **episode seed** 决定，绝不按帧随机拆分。单 episode schema v1 不重复
保存一个可被篡改的 `split` 字符串；collection manifest 必须记录 split，并按下面
冻结的、互不相交的 candidate seed namespace 验证它：

| split | seed namespace | M2 使用方式 |
|---|---:|---|
| train | `0 <= seed < 1_000` | 从 0 递增尝试，达到 200 条成功轨迹即停止；不得跨入 validation namespace |
| validation | `1_000 <= seed < 10_000` | 从 1000 递增尝试，达到 40 条成功轨迹即停止 |
| formal evaluation | `10_000 <= seed < 10_050` | 明确保留，collection manifest 必须拒绝；冻结策略评测时在线生成 |
| unassigned | `seed >= 10_050` | 未经协议版本升级不得用于 train、validation 或正式结果 |

失败尝试仍须在采集 ledger 中保留 `seed/success/failure_stage`，不能静默换 seed
后只报告成功样本。正式 manifest 至少执行以下集合检查：

```text
train_seeds ∩ validation_seeds = ∅
train_seeds ∩ test_seeds = ∅
validation_seeds ∩ test_seeds = ∅
```

同一 seed 的所有帧只能属于一个 episode 和一个 split。归一化统计、数据增强参数
拟合与模型训练只能读取 train；validation 只用于模型选择；test/evaluation 在策略
和超参数冻结后才运行。当前仓库尚未生成正式 manifest 或 240 条成功轨迹，因而这些
数量和回放门禁仍是待验证项。

## 校验与读取边界

`validate_episode(path)` 返回带错误 code 和 HDF5 location 的报告，并检查必需键、
未知根项、shape、dtype、统一长度、有限值、动作范围、20 Hz 时间差、元数据一致性
和 failure-stage lineage，也会把间接/外部 storage 与断链报告为结构化错误，而不是
跟随它们读取或把 `KeyError`/`OSError` 泄漏给调用者。`HDF5EpisodeReader` 只打开一次
目标文件，并在同一个只读 handle 上完成完整验证、metadata 提取和逐 transition 读取；
因此不存在“按路径验证后重新打开另一个 inode”的窗口，也不会一次把双相机 episode
载入 RAM。打开或验证失败时 reader 会关闭已经创建的 handle。

validator 先完成六个 dataset 的 link/storage/object identity、ndim、尾轴、精确 dtype、
`T`、合法 `@num_steps` 和 chunk-byte/layout 检查，再决定是否读取 value payload。任意
schema dataset 的 shape/dtype/length/layout 不合规时，整个文件在第一次 dataset payload
读取前 fail closed；因此 malformed sparse 尾轴、伪造长度或超大 HDF5 chunk 都只产生
结构化 issue，不会继续扫描其余“看似正常”的列。

合法 state、action、timestamp 和 fixed UTF-8 stage 都以最多 256 行的逻辑块扫描；
timestamp 的单调性/20 Hz 检查显式跨块衔接，stage 只保留 empty/
failure-stage-observed 聚合状态，不构造 `O(T)` Python 字符串列表。固定 64-byte item
同时把每个 stage 元素和一次 stage scan 的解码分配约束在明确上限内。variable-length
stage 在 dtype metadata 阶段直接拒绝，validator 不会先 materialize 一个超长 vlen
元素再检查长度。图像 payload 不由 schema validator 整集读取。

writer 在校验 partial 前后记录并比较完整 filesystem snapshot：device、inode、size、
mtime_ns 和 ctime_ns；validation 稳定后计算 validated SHA-256，hash 前后的完整 snapshot
也必须相同。publish 前、hard-link 后、post-link schema validation 后、partial unlink
前后以及目录 fsync 后都会重新验证稳定的完整 snapshot 和同一个 validated digest。
hard-link/unlink 本身会合法改变 ctime，所以每次 link-count 变化后建立新的完整稳定
snapshot，但 device/inode/size/mtime 和 SHA-256 必须继续绑定 validated 文件。path 替换、
同 inode overwrite、恢复 mtime 或 hash 期间变化都不能把旧校验结果变成正常成功。
只有删除 partial 和目录 fsync 也成功、最后一次 digest 绑定复验通过，`finalize()` 才返回。

该保证的边界是 `finalize()` 正常返回时 target 对应刚复验的 validated digest；实现不会
无限追逐返回之后发生的外部替换。collection manifest 在登记 SHA-256 时必须重新读取
target，并在训练/replay 消费时再次核验冻结 digest。

如果 `os.link` 已成功，而 target 复验、partial unlink 或目录 fsync 失败，writer 不会
把结果伪装成普通失败，也不会删除可能已经公开的 target。它会从 `data.hdf5` 抛出
`EpisodePublicationError`，其中 `published=True`、
`state="publication_indeterminate"`，并携带 `target_path`、`partial_path`、
`target_matches_source`、`target_valid` 和现场 `target_sha256`。collection ledger 必须先
记录这条 indeterminate outcome，再用 target path、schema 和 SHA-256 对账；只有证明
target 是本次有效 episode 后才能登记为已采集，并在持久化 ledger 后清理同 inode
partial。若 target 无效或 identity 不符，应隔离并人工恢复。不得盲目重试同目标、删除
不明 target 或绕过 no-clobber；collection 启动时也必须恢复/对账所有 indeterminate
记录。

单文件 validator 不能替代集合级审计：重复 seed、200/40 成功数、split 交集、长度/
动作分布、20 条人工异常记录，以及随机回放 20 条至少成功 18 条，都必须由后续
manifest/报告和真实回放证据验证。

## M2B collection manifest（集合 schema v1）

`data.collection` 把一个 split 写入全新的 run root。正式 train 固定从 seed `0`
向上单次尝试，直到精确获得 `200` 个 success；正式 validation 固定从 `1000`
向上单次尝试，直到精确获得 `40` 个 success。train 最多尝试到 `999`，validation
最多尝试到 `9999`；namespace 用尽时有界失败，不跨 split，也不接触任何
`seed >= 10000`。只有显式 `--smoke --target-successes N` 才允许较小目标，且产物
永久 `formal=false`。`--diagnostic-allow-dirty` 同样永久降级为非正式。

run root 的冻结布局是：

```text
<run-root>/
  attempts.jsonl
  episodes/seed_<六位十进制 seed>.h5
  manifest.json                 # 只有达到目标并通过自校验才发布
```

`attempts.jsonl` 在每次 attempt 后立即 append、flush、fsync。正常记录的字段集合
精确为 `attempt_index, seed, status, success, failure_stage, path, num_steps, sha256,
error`；最终 manifest 的 `attempts` 必须与 JSONL 逐条、逐字段相等，不能缺少、增加、
重复或改写记录。专家正常失败也保留完整 HDF5；只有 `success=true` 的条目按原顺序进入
`eligible_successes`。runner/writer 异常会写 `status=exception` 后终止，且不发布 manifest。

若 episode 的 hard link 已公开，但 partial unlink 或目录 fsync 后的 durability 无法确认，
ledger 写 `status=publication_indeterminate`，记录 target/partial 相对路径、现场 SHA-256、
`target_matches_source` 与 `target_valid`。此状态不得盲目重试或删除 target。
`reconcile_indeterminate_run()` 只接受 ledger 最后一条且唯一的 indeterminate 记录；它在
同一个 anchored target 上重新校验 schema、metadata、digest，并且仅在 partial 与 target
是同 inode、同 digest 时删除 partial，最后另写 no-clobber reconciliation receipt。它不会
继续采集，也不会发布 manifest。

`manifest.json` 顶层字段集合精确为：

```text
schema_version, split_protocol, split, formal, target_successes,
attempt_count, success_count, attempts, eligible_successes, controller,
git, assets, environment, generated_at, cli_config, ledger_path, episodes_root
```

- `schema_version = "m2-collection-manifest.v1"`；
- `split_protocol.name = "m3-m4-seed-protocol.v2"`，并冻结该 split 的 namespace、
  scan start 与 `reserved_seed_min=10000`；
- `git` 保存 full HEAD、tracked/source cleanliness、status/untracked hashes 和相关
  untracked runtime source fingerprints；正式运行在创建 output 前要求完整、干净、固定
  full HEAD，并在发布 manifest 前再次要求 Git/assets/environment 与起点完全一致；
- `assets` 保存 canonical Menagerie summary、逐文件 content manifest 与聚合 SHA-256；
- `environment` 保存 MuJoCo 版本、完整 `TaskConfig`、`debug_viz=false` 和 compiled MJB
  fingerprint；compiled fingerprint 还冻结维数及恰好各一个、alpha 为 0 的 `flange/tcp`
  site；
- `controller` 是完整 `ExpertConfig`。validator 从它计算 `run_episode()` 可达的精确最大
  callback 数；默认配置上限为 `1902`，且所有配置还受 `10000` transition 的硬上限。

所有被消费的路径都必须是 canonical POSIX relative path：拒绝空串、绝对路径、NUL、
反斜杠、空 component、`.`、`..`、任意 symlink component 与非普通文件 leaf。消费者先
逐 component `lstat` 并证明 strict resolve 留在 run root 内，再通过 directory fd 和
`O_NOFOLLOW` 打开 leaf。SHA、HDF5 schema、metadata 与 payload 始终绑定同一个 anchored
file handle；消费前后重算同一 fd 的 SHA/fstat，并重查 path 和完整 directory-chain
snapshot。因此 path swap/restore 的 ABA、同 inode 改写、symlink 和 hard-link alias 都
fail closed。manifest validator 还要求每个 episode path、object identity/inode、seed
唯一，并提供 train/validation pair 的 disjoint 验证。

JSON summary/receipt 通过同目录 partial + hard link 做 no-clobber 发布。若 link 后的
partial unlink 或目录 fsync 失败，抛出 `AtomicPublicationError`；它明确携带
`published=true`、`state=publication_indeterminate`、target/partial、target digest、
validity 与是否匹配 source。此异常不能当成普通的“未写出”而直接重试。

## 确定性 replay 与报告

`data.replay` 只从 `eligible_successes` 选择 episode。正式选择固定为 NumPy PCG64
无放回 permutation、`selection_seed=20260824`、精确 `20` 条；更小 count 或不同 seed
必须显式 `--smoke`，summary 永久非正式。每条 trial 用原 seed reset，只按 HDF5 顺序
逐行读取 `/action` 并调用 `PickPlace.step(action)`；不读取 `stage` 或 object/TCP 真值来
控制动作。执行完所有 action 后才调用环境完整 hold 判定 `success()`。正式 gate 固定为
`>=18/20`。

`trials.jsonl` 每条立即 fsync；`summary.json` 记录 trials SHA-256、确定性选择、选中 seed、
算术计数、gate、CLI config，以及 replay 当时的完整 Git/assets/environment provenance。
正式 replay 在创建 output 前要求当前 clean full HEAD 与 source collection commit 一致，
且 assets、MuJoCo/TaskConfig/compiled model fingerprint 完全相等。replay validator 会重新
验证 source manifest 与每个 HDF5、稳定 hash summary/ledger、逐条核对 trial index/seed/
path/action count/outcome、递归校验 replay Git/assets/environment provenance、重建选中集合
并重算 gate；任何伪造、extra/missing trial 或选择篡改都不能作为 report linkage。

`data.reporting` 从 anchored HDF5 分块读取 action（最多 256 行），计算 episode length
分布及 8 个 action 维度的 min/max/mean/std 和 p01/p05/p25/p50/p75/p95/p99。percentile
spool 是 output/run 同一数据盘上的临时 memmap，不使用系统默认 `/tmp`；总 spool 冻结为
最多 2 GiB，创建前另要求 64 MiB 空闲余量。人工复核候选精确 20 条：失败优先，再按
确定性 length/action outlier 排序补足。正式候选 seed 同样固定为 `20260824`；不同 seed
或较小数量只允许 smoke。每个候选只读取 front/wrist 的首、中、末索引帧生成 contact
sheet。`manual_review.jsonl` 初始 verdict 只能是 `PENDING`，报告始终
`manual_review_complete=false`，没有真人判断时绝不合成通过结论。

## LeRobot optional boundary

`data.lerobot_adapter` 只实现经审计的纯 mapping，不安装、不导入、不上传 LeRobot。
版本边界冻结为 LeRobot `0.6.1` / dataset codebase `v3.0` / `20 fps`。传给官方
`add_frame` 的用户键只允许双相机、`observation.state`、`action` 和非空 `task`；
`timestamp/frame_index/index/episode_index/task_index` 由官方 writer 管理。`stage`、seed、
outcome、failure stage 从不进入 policy observation 或 frame feature。

## CLI 与完成边界

诊断命令示例（路径必须全新）：

```bash
env -u PYTHONPATH MUJOCO_GL=egl uv run --locked python -m data.manifest \
  --manifest runs/m2/train-smoke/manifest.json \
  --report runs/m2/train-smoke-validation.json

env -u PYTHONPATH MUJOCO_GL=egl uv run --locked python -m data.collection \
  --split train --output-dir runs/m2/train-smoke \
  --target-successes 1 --smoke --diagnostic-allow-dirty

env -u PYTHONPATH MUJOCO_GL=egl uv run --locked python -m data.replay \
  --manifest runs/m2/train-smoke/manifest.json \
  --output-dir runs/m2/replay-smoke --count 1 --smoke

env -u PYTHONPATH MUJOCO_GL=egl uv run --locked python -m data.reporting \
  --manifest runs/m2/train-smoke/manifest.json \
  --output-dir runs/m2/report-smoke --manual-review-count 1 --smoke
```

smoke 只证明流水线与真实环境可运行，不能替代 README M2 门禁。正式完成仍要求独立新
目录内的 `200 train success + 40 validation success`、两个 manifest 零错误且 seed
disjoint、冻结选择的 replay `>=18/20`、20 条真人 review verdict。当前实现不会自动
采集这 240 条，也不会把 diagnostic evidence 表述为正式完成。
