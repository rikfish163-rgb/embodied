# M4 评估协议（`m4-reactivity-robustness-v1`）

## 状态与证据边界

本文件冻结的是评估方法，不是模型成绩。当前已经用合成 episode records 验证：

- 成功率与 Wilson 双侧 95% 置信区间；
- 控制/推理延迟的 p50、p95、p99；
- 分阶段失败类型交叉表；
- action smoothness 的确定公式；
- 固定 seed、扰动参数和 trial 展开；
- 冻结 protocol identity、policy/checkpoint identity 与 8D action contract；
- 从 JSONL 生成严格 JSON 报告，并按 trial 身份指出缺失、意外和重复记录；
- CLI 默认对 coverage 不完整或正式证据未验证的报告返回非零，同时仍以独占创建方式
  留下可审计报告。

当前**没有** BC/ACT checkpoint、BC/ACT rollout 或 L0–L5 实测结果；协议模块也不会修改
MuJoCo 环境。任何成功率、恢复率或模型优劣数字都必须来自后续保存的原始 episode
records，不能由本工具生成或补齐。M1 专家记录可以进入同一汇总 API，但它没有推理延迟
时，报告会给 `count=0` 和 `null`，不会用控制周期或仿真时间冒充实测延迟。

## 冻结项

正式运行前必须冻结并写入 plan/manifest：

1. 顶层 `policy_id`、`checkpoint_id`（非空、无首尾空白的规范字符串）、训练配置和数据
   manifest 哈希；这两个 identity 必须原样出现在每一个 planned/observed trial；
2. Git commit、环境版本、设备、精度和 batch size（闭环默认 batch size 1）；
3. 本协议版本、完整 seed 列表、condition、`K_exec` 和每一项扰动参数；
4. 每个 trial 的原始 JSONL 记录，以及扰动实际触发的 step/仿真时间；
5. action 表示和固定 `action_scale`。v1 固定为 8D：前 7 维 joint-position targets
   （弧度），第 8 维 normalized gripper-opening target；逐维 scale 固定为
   `[1,1,1,1,1,1,1,1]`。不同表示或 scale 的 smoothness 不可横比。
6. reset admission 固定为 `pick_place_collision_free_rejection_v1`：target、box 与所有
   distractor 必须满足环境的联合无初始碰撞约束；每个 distractor 和 box 各最多尝试
   256 次。预算耗尽记为 `invalid_reset`，保留审计记录但不得进入成功率分母。

正式 test 只运行冻结版本一次。选 checkpoint、`K_exec` 或阈值只能使用 validation；不得看
test 结果后换 seed、删失败或改变触发时机。主 `K_exec` 消融固定同一 checkpoint、数据、
`H_pred` 和环境，只改变 `K_exec`。

## Seed 与配对设计

`src/evaluation/protocol.py` 保留 test seeds `10000..10049`（共 50 个），同一个 seed 在
各 condition 和各 `K_exec` 中成对复用，以减少场景差异带来的方差。`K_exec` 主实验固定为
`{1, 4, 8, 16}`；协议 v1 的鲁棒性 `selected_k_exec` 固定为 `8`，L5 固定禁用。

只有 seeds、完整 R1 K grid、`selected_k_exec=8`、L0–L4/L5/R1 condition 定义和 action
contract **逐项等于**上述冻结值时，`build_m4_plan()` 才会写
`protocol_id=m4-reactivity-robustness-v1` 与 `protocol_mode=frozen`。任何 seed/K/condition
覆盖都会得到基于完整自定义配置哈希的 `custom-m4-reactivity-robustness-*` identity、
`protocol_mode=custom_non_formal` 和 `customization_reasons`；它可用于诊断 coverage，绝不
冒充正式 v1。report 会重新生成并逐字段核对 frozen/custom M4 shape，手改 protocol ID、
删 R1 或重写 trial list 都会使 `protocol_validation.valid=false`。

这 50 个 seed 尚未与最终 M2 collection manifest 做过交集验证，因此目前只是“保留”而非
“已证明未见”。`build_m4_plan()` 因而固定写入
`seed_disjointness={status: unchecked, collection_manifest_id: null}`，绝不把保留 seed
冒充已检查。采集 split 冻结后，正式运行前必须执行：

```python
from evaluation.protocol import M4_EVAL_SEEDS, assert_disjoint_seeds

assert_disjoint_seeds(M4_EVAL_SEEDS, train_seeds | validation_seeds)
```

有任何重叠就阻断正式评估，先重新冻结数据 split 或新协议版本；不能静默跳过重叠 seed。
当前 report 工具不会读取、哈希或解析 M2 collection manifest，因此即使输入 plan 自报
`status=checked` 并附一个 `collection_manifest_id`，也只把它当作未验证声明：completion
中的有效状态仍是 `unchecked`，正式 readiness 仍为 `false`。只有未来加入对实际 manifest
内容的可执行核验后，工具才可以输出有效的 `checked`；本轮不预留“字符串自证”通道。

默认完整计划包括：

- R1 反应性：4 个 `K_exec` × 50 seeds = 200 trials；
- L0–L4 鲁棒性：冻结的 `selected_k_exec=8` × 5 conditions × 50 seeds = 250 trials；
- L5 在 v1 中禁用。显式启用会增加 50 trials，同时生成 custom/non-formal identity；若要
  成为正式协议，必须另发新版本，不能沿用 v1 ID。

v1 的 `8` 是协议身份的一部分，不表示它已经被证明最优。若 validation 选择其他值，必须
生成 custom plan 或冻结新协议版本，不能在 v1 下替换。

## Condition 与扰动参数

下面的数值全部序列化进 plan。它们是 runner 必须实现和核验的请求，不代表当前环境已经
支持这些修改。

| ID | 单一变化 | 冻结参数与触发 |
|---|---|---|
| L0 | IID 基准 | `kind=none`；使用训练分布 reset 范围 |
| L1 | 位置外推 | 训练内框 `x=[0.42,0.60] m, y=[-0.16,0.16] m`；每侧外扩一个原区间宽度的 20%，外框 `x=[0.384,0.636], y=[-0.224,0.224]`；先采样“外框减内框”的边界壳，再通过冻结的联合无碰撞 admission |
| L2 | 干扰物 | reset 时 `n_distractors=2` |
| L3 | 光照/纹理 | reset 时 key light 强度乘 `0.5`、方向 `[1,-1,-1]`，桌面纹理 `eval_checker_v1` |
| L4 | 相机外参 | front、wrist 同时平移 `[0.02,0,0] m`、RPY 旋转 `[0,5,0]°` |
| L5 | 指令替换 | episode start 时 `red → crimson`；单色任务默认关闭 |
| R1 | 中途扰动 | 进入 `pregrasp` 时 cube 平移 `[0,0.03,0] m`；前置条件是 cube 尚未被夹持 |

R1 只允许触发一次。runner 必须记录 plan 中的 spec 及 `applied=true/false`、实际 step、
仿真时间和“cube 未夹持”检查结果。若前置条件不成立或实际位移不等于 3 cm，该 trial 是
协议违规：保留唯一原记录、令本次 formal run 失败，不得在同一 run 内重跑该 `trial_id`，
也不能把它悄悄当作有效恢复 trial或在夹持后瞬移 cube。排除基础设施原因后只能用新的
`run_id` 从完整冻结计划起点重新执行；不同 run 不得拼接。L1–L4 若 runner/环境尚不支持
对应参数，应明确阻断该 condition，不能把未施加扰动的 L0 rollout 改标签充数。

L1 外框并不保证任意独立的 cube/box 样本都物理有效。例如 cube 接近
`y=0.224 m`、box 接近 `y=0.26 m` 且 x 投影重叠时可能在 reset 就穿入盒壁。runner 必须
调用版本化的联合 rejection sampler。冻结身份为
`pick_place_collision_free_rejection_v1`，receipt schema 为
`pick_place_reset_receipt_v1`，候选序列按
`sha256-canonical-json-array-v1` 编码：每项严格为
`{candidate_index,collision_free,xy}`，0-based index，JSON 使用
`sort_keys=true, separators=(",",":"), ensure_ascii=true, allow_nan=false`。
trial seed 必须进入 `numpy.random.default_rng`/`PCG64`；NumPy 版本由环境 provenance 绑定。
draw schedule 固定为 target XY、target yaw、每个 distractor 的 XY rejection（仅接受后抽 yaw）、
最后 box XY rejection，禁止为了得到容易样本额外耗用 RNG。

L1 的 proposal 不是无条件矩形均匀分布，而是把“外框减训练内框”分成互不重叠的
left/right/bottom/top 四个矩形，按面积选区后区内均匀采样，再以同一联合无碰撞规则做
conditioning。正式描述只能写“在冻结 collision-free joint admission 下的条件分布”。
所有区间使用 NumPy uniform 的低端包含、高端排除（受浮点舍入边界影响）语义；receipt
逐条记录 condition、proposal id、effective cube/box ranges、RNG API/bit generator、target
accepted XY/yaw/L1 partition，以及各 distractor/box 的 accepted XY、yaw（如适用）、尝试次数、
拒绝次数、accepted index、最终 clearance、完整有界 candidate ledger 与其 SHA-256。每个 ledger
最多 256 项，因此审计内容有硬上限；只给一个无法重算的 hash 不合格。

distractor 按 index 顺序放置，与 target 和先前已接受 distractor 的 XY 中心 L2 距离必须
严格 `> 0.08 m`。box admission 使用实际旋转 geom 的 XY AABB 与四墙真实
`0.063 m` 外包络计算，最小
separating clearance 必须严格 `> 1e-9 m`。每个 distractor 与 box 各最多 256 次；成功时
必须满足 `attempts=rejections+1`、`accepted_candidate_index=rejections`、lower-hex 64 位
hash 和 clearance 阈值。达到 256 次预算仍无有效候选时，异常 receipt 必须是
`attempts=rejections=256`、accepted index 为 null、`collision_free=false`。该计划 trial 的
唯一执行记录标为 `invalid_reset`，本次 formal run 失败：不能改记策略失败、不能在同一
run 内重试，也不能从计划中删除后只汇总容易的 seed。若冻结 ranges 本身不可行，应发布
新协议身份；若是已修复的基础设施故障，只能以新 `run_id` 从完整计划重新开始。原始 attempt
ledger 必须保留 `run_id + trial_id + attempt_id=1`，而指标 JSONL 对每个 run/trial 仍严格
一对一；report 因而会继续把重复 identity 全部排除，而不是猜测哪一行“最权威”。
report 会对 `reset_receipt` 做 closed-world 字段、版本、trial seed/condition、proposal/ranges、
RNG、attempt/rejection/index 核验，并从 bounded ledger 重算 canonical SHA-256、distractor
pairwise distance、旋转 cube AABB 对 box 外包络 clearance、target L1 shell partition 及所有
candidate 的 half-open range。已提供但不合法的 receipt 从指标分母排除并给出
`reset_admission_invalid`；缺失 receipt 同样不进入指标分母，并给出
`reset_admission_unverified`。即使 receipt 内容重算
通过，当前 report 仍把 `runner_provenance_verified=false`、`complete=false` 并保留
`reset_admission_unverified`：完整 outer-shell runner、attempt ledger 与当前 source/environment
provenance 尚未落地前，正式 M4 仍为 PENDING，不能凭一份自报 JSON 解锁。

## Episode record 输入契约

`evaluation.report.build_report()` 接受 `Mapping`，也接受带同名属性的对象。这样当前
`expert.scripted.EpisodeResult` 与未来 BC/ACT runner 可以共享指标代码。正式 BC/ACT JSONL
推荐每行一个对象：

```json
{
  "schema_version": "episode-record.v1",
  "episode_id": "ACT-L0-k008-s10000",
  "trial_id": "L0-k008-s10000",
  "policy_id": "ACT",
  "checkpoint_id": "sha256:<content-hash>",
  "condition_id": "L0",
  "seed": 10000,
  "k_exec": 8,
  "success": false,
  "failure_stage": "lift",
  "failure_type": "object_slip",
  "actions": [[0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785, 1.0]],
  "inference_latency_ms": [4.2],
  "control_latency_ms": [5.1],
  "perturbation": {
    "spec": {"kind": "none", "parameters": {}, "trigger": {"event": "reset"}},
    "applied": true,
    "applied_step": 0,
    "applied_sim_time_s": 0.0
  }
}
```

只有 `success` 是不带 trial 清单的通用汇总最低必需字段；缺少 action 或 latency 时相关
计数为 0。当 protocol 含 `reactivity_trials` 或 `robustness_trials` 时，protocol 顶层和
每个 planned/observed trial 都必须显式包含完全相同的 canonical `policy_id` 与
`checkpoint_id`，并用 `trial_id + seed + condition_id + k_exec` 逐条对账。任一 identity
显式为 `null`、缺失、空白、有首尾空白或上下层冲突，都会令
`protocol_validation.valid=false` 或产生 `missing + unexpected`，因此
`completion.complete=false`。原始行数和不合格记录的 index/trial ID/error 仍写入审计
字段，但正式 `metrics/groups` **只汇总**协议有效、身份 expected、全局唯一且记录 schema
有效的行；unexpected、duplicate 的全部副本以及 outcome/action/latency 非法行不会污染
成功率、失败率或 smoothness，也不会被删除、补造或悄悄改成有效。

正式 BC/ACT 记录必须包含示例中的全部审计字段。`success=true` 时 direct
`failure_stage/failure_type` 与 nested `failure.stage/type` 必须缺失、`null` 或规范空字符串
`""`；任何非空/空白-only label 都是矛盾。`success=false` 必须从 direct 或 nested **恰好
一种**来源同时提供非空、ASCII 小写 snake_case 的 stage 和冻结 taxonomy 中除
`unclassified` 外的
canonical type。两种来源重复（即使值相同）、冲突，或 stage/type 跨来源拼接都无效。
stage 仍是开放词表；stage/type 的映射证据来自同一完整 failure 对，而不是根据最终 stage
猜测 type。

actions、episode `action_scale` 和两类 latency 必须使用真正的 JSON number；boolean、数字
字符串（含空白字符串）、NaN/Infinity、非正 scale、负 latency、空/非二维 action matrix
均结构化写入 `record_schema_evidence` 并排除出指标，而不是在生成审计报告前抛异常。
episode 还应在独立 manifest 中记录 Git、设备、环境和 checkpoint 证据，避免每个 step
重复写大字段。

M2 HDF5 的 `action` 数据集可以导出为这里的 `actions`；`stage` 可以用于人工复核和失败
阶段标注。不要从图像时间戳或 `control_dt_s` 推算墙钟延迟。

## 指标定义

### 成功率与 Wilson 95% 区间

对 `n>0` 个通过上述资格筛选的**实际观测** trial，成功数为 `x`，`p̂=x/n`，
`z=1.9599639845`：

```text
center = (p̂ + z²/(2n)) / (1 + z²/n)
radius = z * sqrt(p̂(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
CI = [max(0, center-radius), min(1, center+radius)]
```

底层 `wilson_interval()` 拒绝空分母；report 若没有合格记录则明确输出
`trials=0, rate=null, wilson=null`。报告中的 `completion.complete=false` 表示计划 trial
尚未逐条收齐或证据/schema 不合格；即使观测数量等于 `planned_trials`，只要有身份错配、
重复、失败分类缺失或记录 schema 错误，也仍是不完整。不能因当前已观测子集看起来较好
就发布正式结果。

R1 恢复率就是**协议有效且确实施加 R1** 的 trial 成功率，同样使用 Wilson 区间。正式
runner 必须先做上述触发有效性筛选并把无效 trial 单独列出。

### 延迟

- `inference_latency_ms`：单次 policy forward 的墙钟时间，从输入已就绪到 action/chunk
  已可读；GPU 必须在计时边界同步；不含首次 warm-up 和模型加载；
- `control_latency_ms`：同一次重规划从观测交付给 runner，到 action 完成校验并可交给
  `env.step()` 的端到端墙钟时间；包含预处理、推理和后处理；
- 使用单调高精度时钟，原始样本逐次保存；报告全部有效样本的线性插值
  p50/p95/p99 和样本数。不得用固定 20 Hz 控制周期或 MuJoCo `sim_time` 代替延迟。

### Action smoothness

对一条 episode 的 action 序列 `a_t ∈ R^D` 和预先冻结的正 scale 向量 `s`：

```text
S = (1 / (T-1)) * Σ[t=1..T-1] || (a_t - a_(t-1)) / s ||₂
```

数值越低越平滑。无 action contract 的通用汇总默认 `s=[1,...,1]`；M4 v1 plan 会把上述
8D 表示、语义和逐维全 1 scale 完整序列化，report 必须使用 plan scale。episode 若声明
`action_scale`，它必须与 plan 相等；无 plan 时缺省 scale 等价于逐维 1。混合 action
维度、混合 scale、scale 长度错误、没有 actions 却单独声明 scale，或与 plan 冲突都会
进入记录级无效证据并排除，不能把不可比数值加权到一起。
少于两步没有 transition，值为 `null` 而非 0。总报告按 transition 数加权平均 episode
的 `S`。它衡量 action 一阶变化，不等同于物理 jerk，也不能单独证明控制稳定。

### 分阶段失败 taxonomy

阶段是开放词表，可直接保留专家的 `pregrasp/descend/lift/transport/lower/retreat/settle`
以及未来 runner 的阶段。类型使用固定代码：

| 代码 | 中文含义 |
|---|---|
| `misalignment` | 未对准 |
| `empty_grasp` | 抓空 |
| `object_slip` | 滑落 |
| `box_collision` | 碰盒/碰撞 |
| `placement_failure` | 放置失败 |
| `timeout` | 超时 |
| `invalid_action` | action 非法 |
| `inference_error` | 推理失败 |
| `environment_error` | 环境/runner 失败 |
| `unclassified` | 尚未人工归类 |

底层诊断函数仍会把未知/缺失类型计入 `unclassified` 并把原始 label 列在
`raw_unclassified_types`，以便离线清洗；report 的正式指标资格门禁更严格：未知、缺失或
显式 `unclassified` 的失败行只出现在 `outcome_evidence` 审计中，不进入 `by_stage`、
`by_type` 或 `stage_by_type`。失败类型应由轨迹证据和固定判据标注，不应只从最终 stage
猜测。例如 `failure_stage=lift` 不自动等于 `object_slip`。

## JSON 报告 schema

`evaluation-report.v1` 顶层字段为：

```text
schema_version              固定为 evaluation-report.v1
protocol                    完整 plan/protocol JSON
observed_episode_count      实际读取的 JSONL 行数
qualified_episode_count     通过 protocol/identity/outcome/record schema 门禁的行数
completion                  observed, planned, trial_coverage_complete, complete；
                            有 trial 清单时另含 matching=trial_identity、missing、
                            unexpected、duplicate；始终含 protocol_validation、
                            outcome_evidence、reset_admission_evidence、
                            record_schema_evidence、record_qualification、
                            seed_disjointness、formal_ready 与 formal_readiness_blockers
metrics.success             successes, trials, rate, Wilson interval
metrics.failures            stage/type 计数与交叉表
metrics.latency_ms          inference/control 的 count,p50,p95,p99
metrics.action_smoothness   公式、聚合、episode/transition 数和值
groups[]                    按 policy/checkpoint/condition/K_exec 的同构指标
```

报告只含标准 JSON 值；protocol 中的 NaN/Infinity 会被替换为可审计 marker 并令协议无效，
record 中的非有限 action/scale/latency/failure label 会产生结构化无效证据，因此输出本身
绝不含 NaN/Infinity。有 trial 清单时按身份判断 coverage；只有
`planned_trials` 时退化为数量判断；两者都没有时 `trial_coverage_complete=null`。
`completion.complete` 还要求 `protocol_validation.valid=true`，且任何要求 reset admission
证据的 M4/custom-M4 协议都必须有 runner provenance 验证通过的 receipt；当前该验证尚未
实现，所以它们固定保持 false。即使自报 trial list
全部匹配，伪造 frozen ID、删 grid、篡改 action contract 或 identity freeze 违规也不能
complete；outcome 与 record schema evidence 也必须完整。`protocol_validation.formal=true`
只表示输入逐字段等于 frozen v1 shape；custom 与 generic 始终为 false。当前
`formal_ready` 仍固定为 `false`，因为工具尚不能验证 collection manifest、真实 reset
admission，也不能验证 R1 扰动实际施加；blocker 还会明确列出 protocol、coverage/身份
清单、outcome/record schema 问题及任一 episode 缺失的 action、inference latency、
control latency。汇总器不会自行补齐缺失 trial，也不会运行 expert、BC 或 ACT。

## 命令

先加载仓库环境。以下命令只生成计划，不运行模型：

```bash
source ./env.sh
python - <<'PY'
import json
from pathlib import Path

from evaluation.protocol import build_m4_plan

plan = build_m4_plan(
    policy_id="ACT",
    checkpoint_id="sha256:REPLACE_WITH_REAL_CONTENT_HASH",
    selected_k_exec=8,
)
path = Path("runs/m4/plan.json")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("x", encoding="utf-8") as handle:
    json.dump(plan.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
print(f"planned {plan.to_dict()['planned_trials']} trials -> {path}")
PY
```

后续 runner 写出真实 `episodes.jsonl` 后再汇总。输出文件使用独占创建，不覆盖旧报告：

```bash
source ./env.sh
python -m evaluation.report \
  --episodes runs/m4/episodes.jsonl \
  --protocol runs/m4/plan.json \
  --output runs/m4/report.json
```

CLI 总是先写出并打印报告。只有未来获得可执行证据并由工具给出
`completion.formal_ready=true` 时才默认返回 0；当前 coverage 即使完整，也会因 seed/R1
等证据未验证而返回 2，但报告仍保留用于审计。确需继续诊断流水线时可显式加
`--allow-incomplete`，它只把退出码改为 0，不会改写 `completion.complete`，也不允许把
部分结果当作正式 M4 证据。若目标文件已存在，独占创建会报错，不能静默覆盖旧报告。

验证工具本身：

```bash
source ./env.sh
pytest -q tests/test_evaluation_metrics.py
ruff check src/evaluation tests/test_evaluation_metrics.py
```

这些测试通过只证明公式、适配、计划展开和序列化实现符合本协议，不证明任何未来模型达到
IID、OOD 或恢复率门槛。
