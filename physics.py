import numpy as np


def pbc_diff(p1, p2, L):
    d = p1 - p2
    return d - np.round(d / L) * L


def solve_three_spheres(p1, r1, p2, r2, p3, r3, r_new):
    dp21 = p2 - p1
    d12  = np.sqrt(np.sum(dp21 ** 2))
    s1, s2, s3 = r1 + r_new, r2 + r_new, r3 + r_new

    if d12 > s1 + s2 or d12 < abs(s1 - s2):
        return False, np.zeros(3), np.zeros(3)

    ex   = dp21 / d12
    dp31 = p3 - p1
    i    = np.dot(ex, dp31)
    ey_v = dp31 - i * ex
    d_ey = np.sqrt(np.sum(ey_v ** 2))

    if d_ey < 1e-7:
        return False, np.zeros(3), np.zeros(3)
    ey = ey_v / d_ey
    ez = np.cross(ex, ey)

    x    = (s1 ** 2 - s2 ** 2 + d12 ** 2) / (2 * d12)
    y    = (s1 ** 2 - s3 ** 2 + i ** 2 + d_ey ** 2) / (2 * d_ey) - (i * x) / d_ey
    z_sq = s1 ** 2 - x ** 2 - y ** 2

    if z_sq < 0:
        return False, np.zeros(3), np.zeros(3)
    z = np.sqrt(z_sq)

    sol1 = p1 + x * ex + y * ey + z * ez
    sol2 = p1 + x * ex + y * ey - z * ez
    return True, sol1, sol2


def check_collision(sol, r_new, all_pos, all_rad, L, collision_tol):
    """检查候选点 sol 与所有已有粒子是否碰撞（向量化）。返回 (collision, coordination)。"""
    diffs = all_pos - sol                        # (N, 3)
    diffs -= np.round(diffs / L) * L             # PBC wrap
    dists = np.linalg.norm(diffs, axis=1)        # (N,)
    r_sums = all_rad + r_new                     # (N,)
    gaps   = dists - r_sums                      # (N,)
    if np.any(gaps < -collision_tol * r_sums):
        return True, 0
    coordination = int(np.sum(gaps < collision_tol * r_sums))
    return False, coordination


def check_single_collision(sol, r_new, new_pos, new_rad, L, collision_tol):
    """检查候选点 sol 与单个粒子是否碰撞。返回 (collision, touching)。"""
    d     = new_pos - sol
    d    -= np.round(d / L) * L
    dist  = np.linalg.norm(d)
    r_sum = new_rad + r_new
    gap   = dist - r_sum
    return gap < -collision_tol * r_sum, gap < collision_tol * r_sum


def get_pbc_center_of_mass(pos_array, L):
    theta   = 2 * np.pi * pos_array / L
    cos_sum = np.mean(np.cos(theta), axis=0)
    sin_sum = np.mean(np.sin(theta), axis=0)
    phi     = np.arctan2(sin_sum, cos_sum)
    phi     = np.where(phi < 0, phi + 2 * np.pi, phi)
    return (phi / (2 * np.pi)) * L
