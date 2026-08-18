"""so(3) / SO(3) —— ★ 这个文件由你手写，不要让 AI 代写 ★

背景：三维旋转有 3 个自由度，但旋转矩阵有 9 个数（正交性 + det=1 占掉 6 个约束）。
so(3) 是它的最小表示：轴角向量 ω ∈ R³，方向 = 转轴，模长 = 转角。

你要实现 5 个函数。跑 `pytest robotics/tests/test_so3.py` 验收。


────────────────────────────────────────────────────────
推导任务 1：Rodrigues 公式（先在纸上推，再写代码）
────────────────────────────────────────────────────────
反对称矩阵算子 hat(ω) 满足 hat(ω) @ v == cross(ω, v)。

设 θ = ||ω||，ω̂ = ω/θ（单位轴）。矩阵指数按定义展开：

    exp(hat(ω)) = I + hat(ω) + hat(ω)²/2! + hat(ω)³/3! + ...

关键性质（自己验证一下）：       hat(ω̂)³ = -hat(ω̂)

用它把所有 3 次以上的项折叠回 hat 和 hat²：
  - 奇数次项归并 → 系数凑成 sin θ
  - 偶数次项归并 → 系数凑成 (1 - cos θ)

最终得到：

    R = I + (sinθ/θ)·hat(ω) + ((1-cosθ)/θ²)·hat(ω)²

★ 自己推一遍再写。背公式和推出来，在面试里是两种人。


────────────────────────────────────────────────────────
推导任务 2：log（R → ω）
────────────────────────────────────────────────────────
对 Rodrigues 两边取迹（用 tr(hat)=0, tr(hat²)=-2θ²·... 自己算）可得：

    θ = arccos((tr(R) - 1) / 2)

再看 R - Rᵀ 只保留反对称部分：

    hat(ω) = θ/(2 sinθ) · (R - Rᵀ)


────────────────────────────────────────────────────────
★ 三个数值陷阱 —— 这是"能跑"和"工程可用"的分界线
────────────────────────────────────────────────────────
(1) θ → 0：sinθ/θ 和 (1-cosθ)/θ² 都是 0/0。
    用泰勒展开代替：
        sinθ/θ        ≈ 1 - θ²/6   + θ⁴/120
        (1-cosθ)/θ²   ≈ 1/2 - θ²/24 + θ⁴/720
    切换阈值取 θ < 1e-8 左右（想一下为什么不能取 1e-16）。

(2) θ → π：sinθ → 0，log 的 θ/(2sinθ) 爆炸，且 R - Rᵀ → 0（退化）。
    必须走独立分支：此时 R 近似对称，用 R + I 的列向量提取转轴。
    提示：Rodrigues 在 θ=π 时化简为 R = I + 2·hat(ω̂)²，
          于是 (R+I)/2 = I + hat(ω̂)² = ω̂ ω̂ᵀ（外积！）
          取 (R+I) 中模长最大的一列归一化即得 ±ω̂。

(3) 浮点误差让 (tr(R)-1)/2 略微超出 [-1, 1] → arccos 返回 nan。
    先 clip。

★ 关于 θ=π 的符号歧义：此时 ω 和 -ω 表示同一旋转，log 的返回值有符号歧义。
  这不是 bug，是 so(3) → SO(3) 不是全局一一对应的体现。
  所以测试在 θ≈π 处比较的是 exp(log(R)) 与 R，而不是 ω 本身。
"""
from __future__ import annotations

import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
    """(3,) -> (3,3) 反对称矩阵，满足 hat(w) @ v == np.cross(w, v)。"""
    raise NotImplementedError("TODO(you)")


def vee(W: np.ndarray) -> np.ndarray:
    """(3,3) -> (3,)  hat 的逆运算。"""
    raise NotImplementedError("TODO(you)")


def exp(w: np.ndarray) -> np.ndarray:
    """(3,) -> (3,3)  Rodrigues 公式。必须处理 θ→0。"""
    raise NotImplementedError("TODO(you)")


def log(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (3,)  必须处理 θ→0 和 θ→π 两个分支。"""
    raise NotImplementedError("TODO(you)")


def is_rotation(R: np.ndarray, tol: float = 1e-8) -> bool:
    """检查 RᵀR == I 且 det(R) == +1。"""
    raise NotImplementedError("TODO(you)")
