# Day 1 执行计划：SO(3) → SE(3) → Panda FK

> 适用方式：从上到下顺序执行，不跳步骤。标准版约 12–13 小时；文末有 8 小时压缩版。
> 今天全部是 CPU 数值计算和 MuJoCo 判卷，不训练模型、不安装 ROS、不占用 RTX 5070 Laptop 的 8 GB 显存。

本机开工快照（2026-08-19 实测）：Ryzen 9 7845HX（12 核 24 线程）、30 GiB 内存、RTX 5070 Laptop 8151 MiB；根分区只余约 979 MiB，项目所在数据盘余约 42 GiB。因而 Day 1 只用 NumPy/SciPy/MuJoCo 的现有环境，缓存指向 `$EMB/cache`，不下载大模型、不装新仿真栈。GPU 留给后续策略训练；今天即使 `nvidia-smi` 利用率为 0 也完全正常。

## 今天的唯一目标

今天不碰 VLA、数据集、训练、强化学习和 IK。只完成一件事：

> 用自己的数学和代码解释一个 Panda 末端位姿怎样从关节角得到。

今天结束时必须完成：

1. 自己实现并验证 SO(3) 的基本运算；
2. 自己实现 SE(3) 齐次变换、逆、Ad；
3. 自己沿 MuJoCo Panda 的 body/joint 链重建 TCP FK；
4. 对所有结果给出测试数字，并能口述公式和坐标约定。

核心代码必须由你亲手写：

- src/robotics/so3.py
- src/robotics/se3.py
- src/robotics/kinematics.py 的 FK 核心循环

AI 可以帮助查询 MuJoCo 字段、解释具体报错、设计测试和整理报告；不能直接生成以上三个文件的完整答案。

## 今天真正要带走的求职能力

今天不是“抄一个 FK demo”。完成后，你应当能在机械臂算法面试里完整讲清这条链：

~~~text
关节角 q
  → 每个关节的局部旋转 exp([axis * q]×)
  → 父子坐标变换逐层相乘
  → hand 位姿
  → TCP 固定偏置
  → 世界系中的 TCP 位姿 T_world_tcp
  → 用独立物理引擎 MuJoCo 做数值判卷
~~~

这条链直接支撑后续 Jacobian、DLS IK、轨迹跟踪、操作数据采集和策略学习。Day 1 不追求“功能多”，追求公式、代码、测试、证据四者闭环。

### 完成态文件清单

当天结束前应存在：

~~~text
src/robotics/so3.py
src/robotics/se3.py
src/robotics/kinematics.py
src/robotics/tests/test_so3.py          # 已提供，不修改
src/robotics/tests/test_se3.py          # 你创建
src/robotics/tests/test_kinematics.py   # 你创建
notes/day1_so3.md
notes/day1_se3.md
notes/day1_fk.md
notes/day1_ai_log.md
DAY1_REPORT.md
~~~

报告、测试文件和机械性脚手架可以借助 AI；三个机器人学核心实现及其坐标约定必须由你亲手完成。

---

## 0. 开始前的 20 分钟

~~~bash
source /media/hetaisheng/044A81D94A81C83E/embodied/env.sh
export PYTHONPYCACHEPREFIX="$EMB/cache/pycache"
cd "$EMB/src"
python -V
python -c 'import numpy, scipy, mujoco; print(numpy.__version__, scipy.__version__, mujoco.__version__)'
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q
~~~

预期基线是 **24 failed**；当前 so3.py 是 TODO 骨架，失败原因应当都是或主要都是 `NotImplementedError("TODO(you)")`。这是基线成功，不是环境失败。

如果出现 `ModuleNotFoundError`、MuJoCo XML/asset 错误或 Python 解释器不在 `$EMB/venv`，先修环境，不进入编码。确认：

~~~bash
which python
python -c 'import os, robotics; print(os.environ["EMB"]); print(robotics.__file__)'
git -C "$EMB" status --short
df -h / "$EMB"
~~~

本机根分区空间紧张，所以把 Python 字节码缓存放到数据盘，并关闭 pytest 缓存。今天不要升级 CUDA、PyTorch、MuJoCo，也不要新建 Conda 环境。

建立记录文件：

~~~bash
mkdir -p "$EMB/notes" "$EMB/results"
touch "$EMB/notes/day1_so3.md" \
      "$EMB/notes/day1_se3.md" \
      "$EMB/notes/day1_fk.md" \
      "$EMB/notes/day1_ai_log.md" \
      "$EMB/LOGBOOK.md" \
      "$EMB/DAY1_REPORT.md"
~~~

在 `$EMB/notes/day1_so3.md` 写下：

~~~text
旋转约定：列向量；R_ab 把 b 坐标表示转换到 a
twist 顺序：xi = [v_x, v_y, v_z, w_x, w_y, w_z]
模型：MuJoCo Franka Panda；TCP site 名称：tcp
~~~

---

## 1. 时间安排

| 时间块 | 学习与实践 | AI Coding 环节 | 离开该阶段的硬条件 |
|---|---|---|---|
| 0:00–0:20 | 环境、基线、记录文件 | AI 不介入 | 导入成功；看到预期 24 failed |
| 0:20–0:50 | 看旋转矩阵、指数坐标视频 | 让 AI 只出 5 道理解题 | 能说出 SO(3)、so(3)、hat、exp 的关系 |
| 0:50–1:40 | 白纸推 Rodrigues 与 log | AI 只审推导，不给代码 | 笔记有完整推导和 3 个数值陷阱 |
| 1:40–2:20 | 手写 hat、vee、is_rotation | AI 审形状和边界 | 对应测试通过 |
| 2:20–3:20 | 手写 exp 与小角度分支 | 单失败诊断模式 | exp 独立对照误差 < 1e-12 |
| 3:20–4:30 | 手写 log 三个分支 | 单失败诊断模式 | 普通角、零角、近 pi、噪声测试通过 |
| 4:30–5:20 | 全量测试、SciPy 交叉验证、笔记 | AI 可审测试输出 | SO(3) 24 passed |
| 5:20–5:40 | 休息并写 LOGBOOK | 禁止看代码 | 记录一个你亲手定位的 bug |
| 5:40–6:10 | 看 SE(3)、twist 视频 | AI 出口述题 | 固定 `[v,w]` 与变换方向 |
| 6:10–7:15 | 手写 se3.py | AI 只审单函数 | 9 个 API 完成 |
| 7:15–8:20 | 写 invariant tests 并判卷 | AI 可补重复测试脚手架 | 五类代数恒等式全绿 |
| 8:20–8:40 | 休息 | — | 离屏、走动、喝水 |
| 8:40–9:20 | 看 FK 视频并打印真实 Panda 链 | AI 解释 MuJoCo 字段 | 手绘 link0→hand→tcp 链 |
| 9:20–10:40 | 手写 body_chain、fk_body、fk_site | AI 只定位首个分歧 body | 零位和 home_q 对齐 |
| 10:40–11:40 | 100 个随机姿态交叉验证 | AI 可整理失败样本 | 位置、姿态最大误差均 < 1e-9 |
| 11:40–12:20 | 报告、diff、自查 | AI 只润色非核心文字 | 报告含命令、数字、失败记录 |
| 12:20–13:00 | 关闭 AI，10 分钟口述 + 追问 | AI 最后扮演面试官 | 不看代码讲完整核心链路 |

如果今天只有 8 小时，保留顺序：so3 全绿 → se3 基本变换 → FK 位置验证 → FK 姿态验证 → 报告；`ad` 的扩展随机测试可明确记录后顺延。

### 每个编码小节都使用同一闭环

1. 先在纸上写输入、输出、矩阵形状、坐标方向和不变量；
2. 亲手只写一个函数；
3. 只跑与它有关的最小测试；
4. 失败时先自己写出三个可能原因；
5. 仍卡住再把“单个失败 + 你的三个猜测”交给 AI；
6. 修复后不看 AI，自己解释根因，并记入 `notes/day1_ai_log.md`。

### AI Coding 总提示词：每次对话先贴这一段

~~~text
你是我的机器人学代码审查员，不是代写者。
项目约定：列向量；R_ab 把 b 系表达转到 a 系；twist=[v,w]。
禁止给出 so3.py、se3.py、kinematics.py 的完整实现，禁止直接替我填 TODO。
你可以：检查我的推导、指出第一个逻辑错误、解释一个报错、建议测试、核对数组形状。
每次只给：1) 最可能根因；2) 两个验证实验；3) 一个最小修改方向。
除非我贴出自己已写的函数，否则不要给实现代码。
~~~

四种可直接复制的后续提示词：

**推导审查**

~~~text
下面是我亲手写的推导。不要重写标准答案，只指出第一处不成立的等式，
说明它违反了什么性质，再问我一个问题让我自己修正：<粘贴推导>
~~~

**单失败诊断**

~~~text
这是唯一一个失败和我的三个猜测。不要给完整函数。
失败：<粘贴 pytest 的一个失败>
猜测：1)... 2)... 3)...
请把猜测按概率排序，并给每个猜测一个不超过 3 行的验证实验。
~~~

**测试脚手架**

~~~text
只帮我设计 pytest 的测试名称、输入分布和数学 invariant，不写被测实现。
固定随机种子；说明每个容差的量纲和理由；必须包含正常值、边界值和反例。
模块：<SE(3) 或 FK>
~~~

**模拟面试**

~~~text
你扮演 2025–2026 机械臂算法面试官，围绕我今天的 SO(3)、SE(3)、Panda FK
连续问 10 个递进问题。一次只问一个，不先给答案；我的回答有坐标系歧义时追问。
~~~

`notes/day1_ai_log.md` 每次只记四项：你问了什么、AI 建议什么、你采用/拒绝什么、为什么。这个文件能证明你是在使用 AI 放大效率，而不是把理解外包给 AI。

---

## 2. 模块 A：SO(3) / so(3)

### A0. 学习材料，30 分钟，按顺序看

只看与今天编码直接相关的内容，不打开推荐算法继续刷视频：

1. [Rotation Matrices Part 1（2:54）](https://www.youtube.com/watch?v=OZucG1DY_sY)
2. [Rotation Matrices Part 2（4:14）](https://www.youtube.com/watch?v=6KIPusOv5fA)
3. [Exponential Coordinates of Rotation Part 1（2:04）](https://www.youtube.com/watch?v=v_KBHaG0mas)
4. [Exponential Coordinates of Rotation Part 2（3:43）](https://www.youtube.com/watch?v=WHn9xJl43nY)
5. 对照 [Modern Robotics 免费教材](https://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf) 第 3.2.1、3.2.3 节；只查视频没讲清的公式。

官方的 [Modern Robotics 视频目录](https://hades.mech.northwestern.edu/index.php/Modern_Robotics_Videos) 明确建议“先看短视频，再读教材并做题”；今天按这个顺序执行。看完立刻合上资料，在笔记中默写 `R.T @ R`、`det(R)`、hat、Rodrigues 和 log 普通分支。默写不出再回看，禁止边看边抄代码。

### A1. 纸上推导，40 分钟

对 w=[w1,w2,w3] 写出：

~~~text
hat(w) = [[  0, -w3,  w2],
          [ w3,   0, -w1],
          [-w2,  w1,   0]]
~~~

证明 hat(w) @ v = w × v。

令 theta=||w||，u=w/theta。由矩阵指数展开，并利用：

~~~text
hat(u)^3 = -hat(u)
~~~

把奇数项、偶数项分别合并，得到：

~~~text
R = I + (sin(theta)/theta) hat(w)
      + ((1-cos(theta))/theta^2) hat(w)^2
~~~

在笔记中回答：为什么公式里用 hat(w)，不是 hat(u)？

同时写出 log 的普通分支：

~~~text
theta = acos((trace(R)-1)/2)
hat(w) = theta/(2 sin(theta)) (R-R.T)
~~~

并说明 theta 接近 0、接近 pi 时为什么失效。

### A2. 实现顺序，2 小时 50 分钟

文件：/media/hetaisheng/044A81D94A81C83E/embodied/src/robotics/so3.py

按顺序写，每完成一个函数就运行相关测试：

1. hat(w)
2. vee(W)
3. is_rotation(R)
4. exp(w) 普通角度
5. exp(w) 小角度
6. log(R) 单位阵和普通角度
7. log(R) 小角度
8. log(R) 接近 pi 和轻微噪声

每个里程碑对应的最小命令：

~~~bash
# hat / vee
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q \
  -k 'hat or vee'

# is_rotation；此时 exp 还没完成，所以只选反例测试
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q \
  -k 'is_rotation_rejects_bad_input'

# exp；其中 output_is_rotation 会同时复核 is_rotation
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q \
  -k 'exp or small_angle_first_order'

# log 全部分支
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q \
  -k 'log or near_pi or noisy_rotation'
~~~

每次测试失败只阅读第一个 traceback。先把当前输入的 `theta`、trace、det、正交误差打印出来，再考虑修改公式。不要一次改多个分支，否则你无法知道哪项改动解决了问题。

必须处理三个数值问题：

- theta 小于约 1e-8：使用 sin(theta)/theta 和 (1-cos(theta))/theta^2 的 Taylor 展开；
- theta 接近 pi：从 (R+I) 的最大范数列提取旋转轴；
- acos 前先 clip 到 [-1,1]，避免浮点噪声产生 NaN。

不要把输入矩阵偷偷投影到最近旋转矩阵来掩盖问题。今天要验证的是公式在轻微噪声下仍然有限。

### A3. SO(3) 验收，1 小时

~~~bash
python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q
~~~

必须 24 个测试全绿。然后做独立交叉验证：

~~~bash
python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
from robotics import so3

rng = np.random.default_rng(1)
max_exp = 0.0
max_roundtrip = 0.0
for _ in range(1000):
    w = rng.normal(size=3)
    w = w / np.linalg.norm(w) * rng.uniform(0, np.pi - 1e-7)
    R = so3.exp(w)
    max_exp = max(max_exp, np.max(np.abs(
        R - Rotation.from_rotvec(w).as_matrix())))
    max_roundtrip = max(max_roundtrip, np.linalg.norm(so3.log(R) - w))
print("max exp-vs-scipy:", max_exp)
print("max log-exp roundtrip:", max_roundtrip)
PY
~~~

目标：普通角度下 exp 对照误差小于 1e-12，log-exp 误差小于 1e-9。

把完整命令、`24 passed`、两个最大误差和至少一个失败样例写进 `$EMB/notes/day1_so3.md`。AI 可以帮你把测试输出压缩成表格，但“为什么近 pi 不能用普通公式”这一段必须自己写。

### A4. 口述门槛

不看代码回答：

1. SO(3) 和 so(3) 的关系；
2. hat 为什么等价于叉积；
3. Rodrigues 的两个系数分别来自哪两组 Taylor 项；
4. theta 接近 0 时为什么会有消去误差；
5. theta 等于 pi 时为什么 R-R.T 不能提取轴；
6. 为什么 log(exp(w)) 不是全局一一对应。

---

## 3. 模块 B：SE(3)

### B0. 学习材料，30 分钟

按顺序看：

1. [Homogeneous Transformation Matrices（6:22）](https://www.youtube.com/watch?v=vlb3P7arbkU)
2. [Twists Part 1（5:00）](https://www.youtube.com/watch?v=mvGZtO_ruj0)
3. [Twists Part 2（2:39）](https://www.youtube.com/watch?v=VTv0qmLNvjg)
4. 阅读 [Modern Robotics 免费教材](https://hades.mech.northwestern.edu/images/2/25/MR-v2.pdf) 第 3.3.1–3.3.2 节中齐次变换、twist 和 Adjoint 的定义。

看完后不看资料完成一个具体纸上例子：坐标系 B 相对 A 绕 z 轴 90°、沿 A 的 x 轴平移 1 m。写出 `T_AB`，选一个 `p_B`，手算 `p_A = T_AB @ [p_B,1]`，再手算逆变换还原。只要方向含糊，就先不要写 `se3.py`。

### B1. 固定约定

本项目使用：

~~~text
xi = [v, w] = [vx, vy, vz, wx, wy, wz]

hat6(xi) = [[hat(w), v],
            [   0,   0]]
~~~

不要混用 [w,v]。如果以后接 ROS 或 Pinocchio，在接口边界显式转换。

### B2. 新建 se3.py

文件：/media/hetaisheng/044A81D94A81C83E/embodied/src/robotics/se3.py

实现：

~~~python
make(R, t) -> T
rotation(T) -> R
translation(T) -> t
inv(T) -> T_inv
hat(xi) -> Xi
vee(Xi) -> xi
adjoint(T) -> Ad_T
ad(xi) -> ad_xi
is_transform(T) -> bool
~~~

齐次变换和逆：

~~~text
T = [[R, t],
     [0, 1]]

T^-1 = [[R.T, -R.T @ t],
        [  0,      1   ]]
~~~

Adjoint 和 Lie bracket：

~~~text
Ad_T = [[R, hat(t) @ R],
        [0,     R     ]]

ad([v,w]) = [[hat(w), hat(v)],
             [  0,     hat(w)]]
~~~

### B3. SE(3) 测试

新建 robotics/tests/test_se3.py，使用至少 100 组随机 R、t、xi、eta，验证：

~~~text
T @ inv(T) == I
inv(T1 @ T2) == inv(T2) @ inv(T1)
vee(hat(xi)) == xi
Ad_T @ xi == vee(T @ hat(xi) @ inv(T))
ad(xi) @ eta == vee(hat(xi) @ hat(eta) - hat(eta) @ hat(xi))
~~~

为保证下面的分阶段命令可直接使用，测试命名至少包含：

~~~text
test_make_rotation_translation
test_inverse_identity
test_hat_vee_roundtrip
test_adjoint_conjugation
test_ad_matches_lie_bracket
test_is_transform_rejects_invalid
~~~

目标：

~~~text
齐次变换代数误差 < 1e-12
Adjoint 误差       < 1e-10
ad 误差            < 1e-10
~~~

推荐执行顺序：

~~~bash
# 先做基础块，不让后续复杂恒等式掩盖简单错误
python -m pytest -p no:cacheprovider robotics/tests/test_se3.py -q \
  -k 'make or rotation or translation or inv or hat or vee'

# 再做 Adjoint 和 Lie bracket
python -m pytest -p no:cacheprovider robotics/tests/test_se3.py -q \
  -k 'adjoint or ad'

# 最后全量
python -m pytest -p no:cacheprovider robotics/tests/test_se3.py -q
~~~

你必须亲手决定五条 invariant 和容差。AI 可以生成固定随机种子、参数化装饰器、重复的随机样本循环，但你要逐条审核矩阵形状和等式方向，不能复制一个自己解释不了的测试。

在 `$EMB/notes/day1_se3.md` 记录：`[v,w]` 约定、每个 6×6 块的物理含义、五项最大误差，以及一次因为乘法顺序导致的反例。如果一次就全通过，也要主动构造 `T1 @ T2 != T2 @ T1` 的例子。

口述回答：

1. 为什么逆平移是 -R.T @ t，而不是 -t；
2. Ad_T 转换的是什么；
3. 为什么选择 [v,w]；
4. T1 @ T2 的顺序如何对应坐标链。

---

## 4. 模块 C：Panda 自写 FK

### C0. 学习材料，40 分钟

1. [Product of Exponentials in the Space Frame（6:31）](https://www.youtube.com/watch?v=hE_Duih_7JE)
2. [Forward Kinematics Example（3:28）](https://www.youtube.com/watch?v=cKHsil0V6Qk)
3. 阅读 [MIT Robotic Manipulation：Basic Pick and Place](https://manipulation.mit.edu/pick.html) 的坐标系记法、kinematic tree 和 forward kinematics 部分。
4. 需要查 Python 字段时只看 [MuJoCo 官方 Python 文档](https://mujoco.readthedocs.io/en/latest/python.html) 的 Basic usage、Structs、Named access；不要让 AI 猜字段。

视频讲的是通用 FK/PoE 思想；今天的代码实现要读取当前 MJCF 编译后的真实 body/joint 链。不要把教材里的 UR5 参数或网上另一版 Panda DH 表抄进项目。

### C1. 先读真实模型

~~~bash
python - <<'PY'
import mujoco
from env.scene import build_model

m, d = build_model()
for b in range(m.nbody):
    print("body", b, mujoco.mj_id2name(
        m, mujoco.mjtObj.mjOBJ_BODY, b),
        "parent", int(m.body_parentid[b]),
        "pos", m.body_pos[b], "quat", m.body_quat[b])
for j in range(m.njnt):
    print("joint", j, mujoco.mj_id2name(
        m, mujoco.mjtObj.mjOBJ_JOINT, j),
        "body", int(m.jnt_bodyid[j]),
        "qadr", int(m.jnt_qposadr[j]),
        "axis", m.jnt_axis[j])
PY
~~~

把输出整理为一张手绘表，至少包含：`body_name`、`parent`、`joint_name`、`qpos address`、`axis`、`body_pos`、`body_quat`。AI 可以解释字段，但表中的每个值必须来自你刚运行的本机输出。

已实测但仍要自己读出来的事实：

- link0 → link1 → ... → link7 → hand 是末端链；
- 臂关节在 qpos[0:7]；
- 夹爪在 qpos[7:9]，不参与 TCP FK；
- 本项目 `env/pick_place.py` 的 `home_q` 是 `[0.0, 0.35, 0.0, -2.2, 0.0, 2.55, 0.785]`；
- tcp site 挂在 hand，局部位置为 [0,0,0.1034]；
- 当前 Panda hinge 的 jnt_pos 都为零；
- 四元数是 MuJoCo 的 (w,x,y,z)，不是 SciPy 的 (x,y,z,w)。

### C2. FK 的正确变换顺序

对链上的每个 body，构造：

~~~text
T_parent_to_body(q)
  = Trans(body_pos)
    @ Rot(body_quat)
    @ Rot(axis_in_body_frame * q_joint)
~~~

然后：

~~~text
T_world_body = T_world_parent @ T_parent_to_body(q)
~~~

最后：

~~~text
T_world_tcp = T_world_hand
              @ Trans(site_pos)
              @ Rot(site_quat)
~~~

注意：对本模型这套 body/joint 组合，固定 body 旋转后再乘局部关节旋转，不能擅自换成 joint_then_body。你要用零位和随机位姿与 MuJoCo 验证，而不是凭感觉决定顺序。

### C3. 新建 kinematics.py

文件：/media/hetaisheng/044A81D94A81C83E/embodied/src/robotics/kinematics.py

建议 API：

~~~python
body_chain(model, body_name="hand") -> list[int]
fk_body(model, q_arm, body_name="hand") -> np.ndarray
fk_site(model, q_arm, site_name="tcp") -> np.ndarray
~~~

硬要求：

- 不能用 data.site_xpos 或 data.site_xmat 作为实现结果；
- 不能用 mj_forward 生成你的 FK 结果；
- 可以读取静态模型字段；
- qpos 地址用 jnt_qposadr 查询，不要到处硬编码；
- MuJoCo 只负责独立判卷。

### C4. FK 验收

在 test_kinematics.py 中验证零位、home_q 和 100 个关节限位内随机姿态：

~~~text
test_fk_zero
test_fk_home
test_fk_random_100
~~~

~~~python
m, d = build_model()
q = rng.uniform(m.jnt_range[:7, 0], m.jnt_range[:7, 1])
d.qpos[:7] = q
mujoco.mj_forward(m, d)

T_my = fk_site(m, q, "tcp")
np.testing.assert_allclose(T_my[:3, 3], d.site_xpos[sid_tcp], atol=1e-9)
np.testing.assert_allclose(T_my[:3, :3],
                           d.site_xmat[sid_tcp].reshape(3, 3),
                           atol=1e-9)
~~~

先不要直接上 100 个随机样本，按下面的诊断顺序跑：

~~~bash
# 1. 零位：最容易看固定偏置和四元数顺序
python -m pytest -p no:cacheprovider robotics/tests/test_kinematics.py -q \
  -k 'zero'

# 2. home_q：让多个关节同时非零
python -m pytest -p no:cacheprovider robotics/tests/test_kinematics.py -q \
  -k 'home'

# 3. 随机位姿：覆盖关节限位内的组合
python -m pytest -p no:cacheprovider robotics/tests/test_kinematics.py -q
~~~

报告：

- 最大位置误差，目标小于 1e-9 m；
- 最大旋转矩阵元素误差，目标小于 1e-9；
- 齐次矩阵底行必须是 [0,0,0,1]。

误差大时排查顺序：

1. body 变换乘法顺序；
2. 四元数顺序；
3. 是否漏掉 hand 或 tcp site；
4. 是否混用了局部坐标和世界坐标；
5. qpos 地址是否用了错误的关节。

如果零位通过而随机位姿失败，优先检查 joint rotation 插入位置和 axis 所在坐标系；如果位置通过而姿态失败，优先检查四元数顺序和 `site_xmat.reshape(3,3)`；如果从某个 link 开始位置、姿态同时失败，打印逐 link 误差并只修第一个分歧点。

把最大误差、最坏样本的 `q`、第一个分歧 body、修复前后数字写入 `$EMB/notes/day1_fk.md`。不要只贴一张仿真截图：截图证明“看起来像”，随机数值交叉验证才证明 FK 链正确。

---

## 5. 最终验收命令

~~~bash
source /media/hetaisheng/044A81D94A81C83E/embodied/env.sh
export PYTHONPYCACHEPREFIX="$EMB/cache/pycache"
cd "$EMB/src"

python -m pytest -p no:cacheprovider robotics/tests/test_so3.py -q
python -m pytest -p no:cacheprovider robotics/tests/test_se3.py -q
python -m pytest -p no:cacheprovider robotics/tests/test_kinematics.py -q

# 审查自己实际改了什么；不要在 Day 1 自动提交
git -C "$EMB" diff -- \
  src/robotics/so3.py \
  src/robotics/se3.py \
  src/robotics/kinematics.py \
  src/robotics/tests/test_se3.py \
  src/robotics/tests/test_kinematics.py
~~~

Day 1 完成的硬标准：

- so3 测试全绿；
- se3 测试全绿；
- FK 通过零位、home_q、100 个随机姿态；
- 报告有所有最大误差数字；
- 能不看代码回答 10 分钟口述题。

最后再做一次防止“测试写错了所以假绿”的人工审查：

1. 临时在脑中假设 `inv(T)` 返回 `T`，指出哪个测试一定失败；
2. 临时假设 FK 漏掉 TCP 的 0.1034 m 偏置，指出误差应出现在哪里；
3. 临时假设四元数按 `(x,y,z,w)` 读取，指出零位和随机位姿可能出现什么现象；
4. 确认 `test_kinematics.py` 的真值来自 `mj_forward`，而被测实现没有读取 `data.site_xpos/site_xmat`；
5. 确认随机数种子固定、关节样本在限位内、报告中样本数确实是 100。

如果只完成 so3 和 se3，报告必须写“FK 未完成”，不能把部分完成说成全链路完成。

---

## 6. 降级规则

### so3 超过 3 小时仍未全绿

保留 hat/vee/exp，停止扩展；记录失败样例和原因，不让 AI 直接补完整 log。

### se3 超过 90 分钟仍卡住

先完成 make、rotation、translation、inv、hat、vee；Ad/ad 放到 Day 2 早段。

### FK 误差大于 1e-6

不要放宽阈值。打印每一个 body 的中间 T_world_body，与 data.xpos/xmat 对比，找到第一个开始出现误差的 body。

### 代码能跑但讲不清

该模块不算完成。先关闭 AI，重新在白纸上写变量含义、矩阵形状和坐标方向。

### 8 小时压缩版

压缩的是范围，不是正确性：

| 时间 | 必做内容 | 可顺延到 Day 2 |
|---|---|---|
| 0:00–0:20 | 环境和基线 | 无 |
| 0:20–1:10 | 只看 SO(3) 4 个短视频并完成推导 | 教材扩展题 |
| 1:10–3:30 | so3.py + 24 tests + SciPy 对照 | 无；这是硬门槛 |
| 3:30–4:50 | SE(3) 的 make/inv/hat/vee/adjoint + 核心 invariant | `ad` 的扩展随机测试 |
| 4:50–5:10 | 休息 | 无 |
| 5:10–5:40 | 打印真实 Panda 链、手绘坐标链 | PoE 扩展阅读 |
| 5:40–7:20 | FK 零位、home_q、20 个随机姿态 | 把随机样本扩展到 100 |
| 7:20–8:00 | 报告、口述、记录未完成项 | 报告排版 |

8 小时版不能把 20 个随机姿态写成“完成 100 个验证”；状态应标为“核心链已贯通，完整随机回归待补”。

---

## 7. 今天结束时的反馈格式

~~~text
Day 1 状态：完成 / 部分完成

环境：Python / NumPy / SciPy / MuJoCo 版本

1. SO(3)
   - pytest：passed / failed
   - exp vs SciPy 最大误差：
   - log(exp(w)) 最大误差：
   - 我能解释的三个数值分支：

2. SE(3)
   - pytest：passed / failed
   - T @ inv(T) 最大误差：
   - Adjoint 最大误差：
   - ad / bracket 最大误差：

3. Panda FK
   - 测试姿态：zero + home_q + N random
   - 最大位置误差（m）：
   - 最大旋转矩阵元素误差：
   - 最坏 q：

4. 我亲手定位并修复的 bug：
   - 症状：
   - 根因：
   - 最小复现实验：
   - 为什么修复有效：

5. AI 使用审计
   - AI 帮助了：
   - 我拒绝了：
   - 仍由我独立完成的核心：

6. 仍然解释不清或未完成：
7. Day 2 入口：Jacobian → 数值微分判卷 → DLS IK
~~~

报告中不得只写“误差很小”“全部正确”；必须填真实数字和实际命令输出。把这份反馈发回来后，Day 2 才根据你的真实失败点调整，不按假定进度继续。
