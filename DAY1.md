# Day 1：SO(3) → SE(3) → Panda FK

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

---

## 0. 开始前的 20 分钟

~~~bash
source /media/hetaisheng/044A81D94A81C83E/panda-week/env.sh
cd /media/hetaisheng/044A81D94A81C83E/panda-week/src
export PYTHONPATH="$PWD"
python -V
python -c 'import numpy, scipy, mujoco; print(numpy.__version__, scipy.__version__, mujoco.__version__)'
pytest robotics/tests/test_so3.py -q
~~~

基线失败是正常的；当前 so3.py 是 TODO 骨架。失败原因应主要是 NotImplementedError，而不是导入错误。

建立记录文件：

~~~bash
mkdir -p notes results
touch notes/day1_so3.md DAY1_REPORT.md
~~~

在 notes/day1_so3.md 写下：

~~~text
旋转约定：列向量；R_ab 把 b 坐标表示转换到 a
twist 顺序：xi = [v_x, v_y, v_z, w_x, w_y, w_z]
模型：MuJoCo Franka Panda；TCP site 名称：tcp
~~~

---

## 1. 时间安排

| 时间块 | 任务 | 验收 |
|---|---|---|
| 0:00–0:20 | 环境、基线、日志 | 能导入并看到预期基线失败 |
| 0:20–1:40 | SO(3) 推导 | 笔记中有完整 Rodrigues 推导 |
| 1:40–5:20 | so3.py | 24 个测试全绿 |
| 5:20–5:40 | 休息和记录 | LOGBOOK 有一条决策记录 |
| 5:40–8:20 | se3.py | SE(3) 代数测试全绿 |
| 8:20–8:40 | 休息 | — |
| 8:40–12:00 | Panda FK | 100 个随机姿态验证通过 |
| 12:00–13:00 | 口述考试、报告、收尾 | DAY1_REPORT.md 完成 |

如果今天只有 8 小时，保留顺序：so3 全绿 → se3 基本变换 → FK 位置验证 → FK 姿态验证 → Ad/ad 和报告。

---

## 2. 模块 A：SO(3) / so(3)

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

### A2. 实现顺序，2 小时 20 分钟

文件：/media/hetaisheng/044A81D94A81C83E/panda-week/src/robotics/so3.py

按顺序写，每完成一个函数就运行相关测试：

1. hat(w)
2. vee(W)
3. exp(w) 普通角度
4. exp(w) 小角度
5. log(R) 单位阵和普通角度
6. log(R) 小角度
7. log(R) 接近 pi
8. is_rotation(R)

必须处理三个数值问题：

- theta 小于约 1e-8：使用 sin(theta)/theta 和 (1-cos(theta))/theta^2 的 Taylor 展开；
- theta 接近 pi：从 (R+I) 的最大范数列提取旋转轴；
- acos 前先 clip 到 [-1,1]，避免浮点噪声产生 NaN。

不要把输入矩阵偷偷投影到最近旋转矩阵来掩盖问题。今天要验证的是公式在轻微噪声下仍然有限。

### A3. SO(3) 验收，1 小时

~~~bash
pytest robotics/tests/test_so3.py -q
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

### B1. 固定约定

本项目使用：

~~~text
xi = [v, w] = [vx, vy, vz, wx, wy, wz]

hat6(xi) = [[hat(w), v],
            [   0,   0]]
~~~

不要混用 [w,v]。如果以后接 ROS 或 Pinocchio，在接口边界显式转换。

### B2. 新建 se3.py

文件：/media/hetaisheng/044A81D94A81C83E/panda-week/src/robotics/se3.py

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

目标：

~~~text
齐次变换代数误差 < 1e-12
Adjoint 误差       < 1e-10
ad 误差            < 1e-10
~~~

口述回答：

1. 为什么逆平移是 -R.T @ t，而不是 -t；
2. Ad_T 转换的是什么；
3. 为什么选择 [v,w]；
4. T1 @ T2 的顺序如何对应坐标链。

---

## 4. 模块 C：Panda 自写 FK

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

已实测但仍要自己读出来的事实：

- link0 → link1 → ... → link7 → hand 是末端链；
- 臂关节在 qpos[0:7]；
- 夹爪在 qpos[7:9]，不参与 TCP FK；
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

文件：/media/hetaisheng/044A81D94A81C83E/panda-week/src/robotics/kinematics.py

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

---

## 5. 最终验收命令

~~~bash
source /media/hetaisheng/044A81D94A81C83E/panda-week/env.sh
cd /media/hetaisheng/044A81D94A81C83E/panda-week/src
export PYTHONPATH="$PWD"

pytest robotics/tests/test_so3.py -q
pytest robotics/tests/test_se3.py -q
pytest robotics/tests/test_kinematics.py -q
~~~

Day 1 完成的硬标准：

- so3 测试全绿；
- se3 测试全绿；
- FK 通过零位、home_q、100 个随机姿态；
- 报告有所有最大误差数字；
- 能不看代码回答 10 分钟口述题。

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

---

## 7. 今天结束时的反馈格式

~~~text
Day 1 状态：完成 / 部分完成

1. so3：测试通过数；exp 对照最大误差；roundtrip 最大误差
2. se3：测试通过数；Ad 最大误差；ad 最大误差
3. FK：随机样本数；最大位置误差；最大姿态误差
4. 我自己修复的 bug：
5. 仍然解释不清的问题：
6. 明天准备继续：Jacobian / 数值微分 / DLS IK
~~~

