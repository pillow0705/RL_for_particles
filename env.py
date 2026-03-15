import numpy as np
from itertools import combinations

from config import Config
from physics import (pbc_diff_njit, solve_three_spheres_njit,
                     check_collision_njit, check_single_collision_njit,
                     get_pbc_center_of_mass)


class ConstructEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        avg_vol   = (4.0 / 3.0) * np.pi * ((cfg.diameters.mean()/2) ** 3)
        total_vol = cfg.target_N * avg_vol / cfg.target_phi
        self.L    = total_vol ** (1.0 / 3.0)
        self.reset()

    def reset(self):
        cfg = self.cfg
        r0, r1, r2, r3 = np.random.choice(cfg.diameters)/2,np.random.choice(cfg.diameters)/2,np.random.choice(cfg.diameters)/2,np.random.choice(cfg.diameters)/2
        p0 = np.array([self.L / 2, self.L / 2, self.L / 2])
        p1 = p0 + np.array([r0 + r1, 0, 0])

        d01 = r0 + r1
        d02 = r0 + r2
        d12 = r1 + r2
        x2  = (d02 ** 2 - d12 ** 2 + d01 ** 2) / (2 * d01)
        y2  = np.sqrt(max(0.0, d02 ** 2 - x2 ** 2))
        p2  = p0 + np.array([x2, y2, 0])

        valid, p3, _ = solve_three_spheres_njit(p0, r0, p1, r1, p2, r2, r3)
        if not valid:
            p3 = p0 + np.array([0, 0, r0 + r3])

        self.pos = [p % self.L for p in [p0, p1, p2, p3]]
        self.rad = [r0, r1, r2, r3]
        self.n   = 4
        self.current_candidates = []
        self._last_cand_stats   = {}

        # 候选点集合 & 三元组集合
        self._candidate_set    = {}   # cand_id -> feat[5]
        self._triplet_set      = {}   # (i,j,k) -> set of cand_id
        self._cand_to_triplets = {}   # cand_id -> set of (i,j,k)
        self._cand_counter     = 0

        self._init_sets()
        return self._get_obs()

    # ------------------------------------------------------------------
    # 候选点集合增量维护
    # ------------------------------------------------------------------
    def _new_cand_id(self):
        cid = self._cand_counter
        self._cand_counter += 1
        return cid

    def _process_triplet(self, i, j, k):
        cfg     = self.cfg
        all_pos = np.array(self.pos, dtype=np.float64)
        all_rad = np.array(self.rad, dtype=np.float64)
        p_i, r_i = all_pos[i], all_rad[i]
        p_j, r_j = all_pos[j], all_rad[j]
        p_k, r_k = all_pos[k], all_rad[k]

        p_j = p_i + pbc_diff_njit(p_j, p_i, self.L)
        p_k = p_i + pbc_diff_njit(p_k, p_i, self.L)

        triplet_cands = set()
        for r_new in cfg.diameters / 2.0:
            valid, sol1, sol2 = solve_three_spheres_njit(p_i, r_i, p_j, r_j, p_k, r_k, r_new)
            if not valid:
                continue
            for sol in (sol1, sol2):
                collision, coord = check_collision_njit(
                    sol, r_new, all_pos, all_rad, self.L, cfg.collision_tol)
                if not collision:
                    cid  = self._new_cand_id()
                    feat = np.array([*(sol % self.L), r_new, coord], dtype=np.float32)
                    self._candidate_set[cid]    = feat
                    self._cand_to_triplets[cid] = {(i, j, k)}
                    triplet_cands.add(cid)

        if triplet_cands:
            self._triplet_set[(i, j, k)] = triplet_cands

    def _init_sets(self):
        for i, j, k in combinations(range(self.n), 3):
            self._process_triplet(i, j, k)

    def _add_new_triplets(self, new_idx):
        for j, k in combinations(range(new_idx), 2):
            self._process_triplet(new_idx, j, k)

    def _filter_candidates(self, new_pos, new_rad):
        new_pos_np = np.array(new_pos, dtype=np.float64)
        to_delete  = []
        for cid, feat in self._candidate_set.items():
            sol   = feat[:3].astype(np.float64)
            r_new = float(feat[3])
            collision, touching = check_single_collision_njit(
                sol, r_new, new_pos_np, new_rad, self.L, self.cfg.collision_tol)
            if collision:
                to_delete.append(cid)
            elif touching:
                feat[4] += 1
        for cid in to_delete:
            self._remove_candidate(cid)

    def _remove_candidate(self, cid):
        if cid not in self._candidate_set:
            return
        del self._candidate_set[cid]
        for triplet in self._cand_to_triplets.get(cid, set()):
            if triplet in self._triplet_set:
                self._triplet_set[triplet].discard(cid)
                if not self._triplet_set[triplet]:
                    del self._triplet_set[triplet]
        del self._cand_to_triplets[cid]

    # ------------------------------------------------------------------
    # 环境接口
    # ------------------------------------------------------------------
    def _get_candidates(self):
        return list(self._candidate_set.values())

    def get_graph_data(self):
        """返回当前已放置粒子的坐标和半径（供 Transformer 编码）。"""
        return (np.array(self.pos, dtype=np.float64),
                np.array(self.rad, dtype=np.float64))

    def _get_obs(self):
        cfg   = self.cfg
        cands = self._get_candidates()

        if len(cands) > 0:
            cands_np = np.array(cands)
            idx      = np.lexsort((cands_np[:, 3], cands_np[:, 4]))[::-1]
            cands_np = cands_np[idx]
            self.current_candidates = cands_np.tolist()
        else:
            self.current_candidates = []

        obs  = np.zeros((cfg.max_candidates, 5), dtype=np.float32)
        mask = np.zeros(cfg.max_candidates, dtype=np.float32)

        n_valid = min(len(self.current_candidates), cfg.max_candidates)
        if n_valid > 0:
            c        = np.array(self.current_candidates[:n_valid])
            center   = get_pbc_center_of_mass(np.array(self.pos), self.L)
            c[:, :3] -= center
            c[:, :3]  = c[:, :3] - np.round(c[:, :3] / self.L) * self.L
            c[:, :3] *= 0.5
            obs[:n_valid]  = c
            mask[:n_valid] = 1.0

        return obs, mask

    def step(self, action_idx):
        cfg = self.cfg

        if action_idx >= len(self.current_candidates):
            obs, mask = self._get_obs()
            return (obs, mask), 0.0, True

        chosen  = self.current_candidates[action_idx]
        pos_new = np.array(chosen[:3])
        r_new   = chosen[3]

        chosen_feat = np.array(chosen, dtype=np.float32)
        for cid, feat in list(self._candidate_set.items()):
            if np.allclose(feat, chosen_feat, atol=1e-5):
                self._remove_candidate(cid)
                break

        self.pos.append(pos_new)
        self.rad.append(r_new)
        self.n += 1

        n_before_filter = len(self._candidate_set)
        self._filter_candidates(pos_new, r_new)
        n_filtered = n_before_filter - len(self._candidate_set)

        n_before_add = len(self._candidate_set)
        self._add_new_triplets(self.n - 1)
        n_added = len(self._candidate_set) - n_before_add

        self._last_cand_stats = {
            'n_before'  : n_before_filter,
            'n_filtered': n_filtered,
            'n_added'   : n_added,
            'n_after'   : len(self._candidate_set),
        }

        done = self.n >= cfg.max_particles
        obs, mask = self._get_obs()

        if np.sum(mask) == 0 and not done:
            done = True

        reward = self.get_phi() if done else 0.0
        return (obs, mask), reward, done

    def get_phi(self):
        box_vol   = self.L ** 3
        total_vol = sum((4.0 / 3.0) * np.pi * (r ** 3) for r in self.rad)
        return total_vol / box_vol
