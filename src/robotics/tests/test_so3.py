"""so3 判卷脚本 —— 我写的，你不用改，只管跑到全绿。

    cd $PW/src && pytest robotics/tests/test_so3.py -v

第二判官用 scipy.spatial.transform.Rotation：它是独立实现，
所以"你的实现 == scipy"是真正的交叉验证，不是自己验证自己。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotics import so3

rng = np.random.default_rng(0)


def rand_axis(n: int = 1):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ─────────────────────── hat / vee ───────────────────────
def test_hat_is_cross_product():
    """hat(w) @ v 必须等于 cross(w, v) —— 这是 hat 的定义。"""
    for _ in range(300):
        w, v = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(so3.hat(w) @ v, np.cross(w, v), atol=1e-14)


def test_hat_is_antisymmetric():
    for _ in range(100):
        W = so3.hat(rng.normal(size=3))
        np.testing.assert_allclose(W, -W.T, atol=1e-14)


def test_vee_inverts_hat():
    for _ in range(300):
        w = rng.normal(size=3)
        np.testing.assert_allclose(so3.vee(so3.hat(w)), w, atol=1e-14)


def test_hat_cubed_identity():
    """hat(ŵ)³ == -hat(ŵ) —— 推导 Rodrigues 的关键性质，顺手验一下。"""
    for u in rand_axis(50):
        H = so3.hat(u)
        np.testing.assert_allclose(H @ H @ H, -H, atol=1e-13)


# ─────────────────────── exp ───────────────────────
def test_exp_matches_scipy():
    """交叉验证：与 scipy 的独立实现逐个比对。"""
    for _ in range(500):
        w = rand_axis()[0] * rng.uniform(0.0, np.pi)
        np.testing.assert_allclose(
            so3.exp(w), Rotation.from_rotvec(w).as_matrix(), atol=1e-12
        )


def test_exp_output_is_rotation():
    for _ in range(300):
        w = rand_axis()[0] * rng.uniform(0.0, 3.0)
        assert so3.is_rotation(so3.exp(w))


def test_exp_zero_is_identity():
    np.testing.assert_allclose(so3.exp(np.zeros(3)), np.eye(3), atol=1e-14)


# ─────────────────────── log ───────────────────────
def test_log_roundtrip():
    """log ∘ exp == id （避开 θ≈0 和 θ≈π 的歧义区）。"""
    for _ in range(500):
        w = rand_axis()[0] * rng.uniform(1e-6, np.pi - 1e-4)
        np.testing.assert_allclose(so3.log(so3.exp(w)), w, atol=1e-9)


def test_log_matches_scipy():
    for _ in range(300):
        w = rand_axis()[0] * rng.uniform(1e-6, np.pi - 1e-4)
        R = so3.exp(w)
        np.testing.assert_allclose(
            so3.log(R), Rotation.from_matrix(R).as_rotvec(), atol=1e-9
        )


def test_log_identity_is_zero():
    np.testing.assert_allclose(so3.log(np.eye(3)), np.zeros(3), atol=1e-12)


# ─────────────────────── ★ 陷阱 1: θ → 0 ───────────────────────
@pytest.mark.parametrize("theta", [0.0, 1e-16, 1e-12, 1e-9, 1e-7, 1e-5])
def test_small_angle_no_nan(theta):
    """★ 只照抄 Rodrigues 会在这里出 nan。必须用泰勒展开分支。"""
    for axis in rand_axis(5):
        w = axis * theta
        R = so3.exp(w)
        assert np.all(np.isfinite(R)), f"exp 出现 nan/inf @ theta={theta}"
        assert so3.is_rotation(R), f"exp 输出不是旋转矩阵 @ theta={theta}"
        w2 = so3.log(R)
        assert np.all(np.isfinite(w2)), f"log 出现 nan/inf @ theta={theta}"
        np.testing.assert_allclose(w2, w, atol=1e-9)


def test_small_angle_first_order():
    """小角度下 exp(w) ≈ I + hat(w)，一阶精度要对。"""
    w = np.array([1e-6, -2e-6, 3e-7])
    np.testing.assert_allclose(so3.exp(w), np.eye(3) + so3.hat(w), atol=1e-11)


# ─────────────────────── ★ 陷阱 2: θ → π ───────────────────────
@pytest.mark.parametrize("theta", [np.pi - 1e-5, np.pi - 1e-8, np.pi - 1e-12, np.pi])
def test_near_pi_no_nan(theta):
    """★ log 的 θ/(2sinθ) 在这里爆炸，必须走 (R+I) 提取轴的分支。

    比较 exp(log(R)) 与 R 而不是 ω，因为 θ=π 时 ω 与 -ω 等价（符号歧义）。
    """
    axes = list(np.eye(3)) + list(rand_axis(5))
    for axis in axes:
        w = axis * theta
        R = so3.exp(w)
        w2 = so3.log(R)
        assert np.all(np.isfinite(w2)), f"log 出现 nan/inf @ theta={theta}"
        np.testing.assert_allclose(np.linalg.norm(w2), theta, atol=1e-5)
        np.testing.assert_allclose(so3.exp(w2), R, atol=1e-6)


# ─────────────────────── 陷阱 3: 数值噪声 ───────────────────────
def test_log_tolerates_noisy_rotation():
    """浮点噪声让 (tr(R)-1)/2 略微越界时 arccos 会返回 nan。必须 clip。"""
    for _ in range(200):
        R = so3.exp(rand_axis()[0] * rng.uniform(0, np.pi))
        Rn = R + rng.normal(scale=1e-12, size=(3, 3))  # 轻微破坏正交性
        assert np.all(np.isfinite(so3.log(Rn)))


def test_log_handles_exact_pi_trace():
    """θ=π 时 tr(R) = -1，(tr-1)/2 = -1 正好在 arccos 边界上。"""
    for axis in np.eye(3):
        R = so3.exp(axis * np.pi)
        np.testing.assert_allclose(np.trace(R), -1.0, atol=1e-12)
        assert np.all(np.isfinite(so3.log(R)))


# ─────────────────────── is_rotation ───────────────────────
def test_is_rotation_rejects_bad_input():
    assert not so3.is_rotation(np.eye(3) * 2.0)  # 非正交（有缩放）
    assert not so3.is_rotation(np.diag([1.0, 1.0, -1.0]))  # det = -1（反射）
    assert not so3.is_rotation(np.ones((3, 3)))  # 完全不是旋转
    assert so3.is_rotation(np.eye(3))
