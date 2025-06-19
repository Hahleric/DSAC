# envs.py  ──────────────────────────────────────────────────────────────
import gym, numpy as np
from gym import spaces
from Vehicle import Crossroad
from torch_geometric.data import Data
import torch as t

from utils.utils import bloom_encode


# --------------------------------------------------------------------- #
# 通用：RSU + Vehicles 组图                                             #
# --------------------------------------------------------------------- #
def _build_rsu_vehicle_graph(vehicles, num_files, max_nodes):
    feat_dim = num_files + 1                     # 最后一维: RSU flag
    x    = np.zeros((max_nodes, feat_dim), np.float32)
    mask = np.zeros(max_nodes, np.int32)

    # RSU 节点 (id=0)
    x[0, -1] = 1.0
    mask[0]  = 1

    # 车辆节点
    for i, v in enumerate(vehicles[: max_nodes-1], start=1):
        x[i, : num_files] = v.interest_vector
        mask[i] = 1

    # 边：RSU ↔ 每辆车
    src, dst = [], []
    for i in range(1, mask.sum()):
        src += [0, i];  dst += [i, 0]
    edge_index = np.array([src, dst], np.int64)
    if edge_index.size == 0:
        edge_index = np.zeros((2, 0), np.int64)

    return Data(x=t.tensor(x),
                edge_index=t.tensor(edge_index),
                mask=t.tensor(mask)), mask

def _topk_interest(v, K, num_files):
        """return (2K,) float32:  [idx_norm_0, prob_0, idx_norm_1, prob_1, ...]"""
        probs = v.interest_vector
        if K >= len(probs):  # K>=F 退化为全量
            idx = np.arange(len(probs))
            top_idx = idx[np.argsort(-probs)]  # 从大到小
        else:
            top_idx = np.argpartition(-probs, K)[:K]
            top_idx = top_idx[np.argsort(-probs[top_idx])]  # 再排序
        vals = probs[top_idx]
        idx_norm = top_idx.astype(np.float32) / num_files  # ∈[0,1]
        return np.vstack([idx_norm, vals]).T.reshape(-1)  # (2K,)


# ===================== 1. DynamicRecCacheEnv ========================= #
class DynamicRecCacheEnv(gym.Env):
    """
    obs = {"preferences":[max_V,F], "cache_status":[F], "mask":[max_V]}
    act = {"cache":[F], "recommend":[max_V,F]}
    """
    def __init__(self, cfg, *, user_interest_dict=None):
        self.cfg        = cfg
        self.num_files  = cfg.num_files
        self.max_V      = cfg.gnn_max_vehicles

        self.cross = Crossroad(width=200, height=200,
                               spawn_rate=cfg.spawn_rate,
                               user_interest_dict=user_interest_dict)

        self.cache_cap  = cfg.cache_capacity_rsu
        self.lambda_r   = cfg.lambda_rate

        self.K_pref = getattr(self.cfg, "top_k_pref", 30)
        self.eta        = cfg.preference_update_rate
        self.steps_per_episode = cfg.steps_per_episode
        self.B_cache = getattr(cfg, "bloom_B", 512)   # <<< 新超参
        self.H_hash  = getattr(cfg, "bloom_H", 4)
        self.C = self.cache_cap  # 要缓存多少部
        self.K = self.cfg.max_recommend_per_vehicle
        self.current_step = 0
        self.cache_status = np.zeros(self.num_files, np.int32)

        self.observation_space = spaces.Dict({
            "preferences": spaces.Box(0., 1., (self.max_V, self.K_pref*2), np.float32),
            "cache_status": spaces.Box(0, 1, (self.B_cache,), np.float32),  # <<< 长 B
            "mask": spaces.Box(0, 1, (self.max_V,), np.int32)
        })
        self.action_space = spaces.Dict({
            "cache_idx": spaces.MultiDiscrete([self.num_files] * self.C),
            "recommend_idx": spaces.MultiDiscrete([self.num_files] * (self.max_V * self.K))
        })
        self.state = self._pack_state()

    def _idxs_to_mask(self, idx_arr, length):
        m = np.zeros(length, np.int32)
        m[idx_arr] = 1
        return m
    def _pack_state(self):
        prefs = np.zeros((self.max_V, self.K_pref * 2), np.float32)
        mask = np.zeros(self.max_V, np.int32)
        cache_enc = bloom_encode(self.cache_status,
                                 B=self.B_cache, H=self.H_hash)
        for i, v in enumerate(self.cross.vehicles[: self.max_V]):
            prefs[i] = _topk_interest(v, self.K_pref, self.num_files)
            mask[i] = 1
        return {"preferences": prefs,
                "cache_status": cache_enc,
                "mask": mask}



    # -------- 违约处理 -------- #
    def _violated(self, msg):
        self.current_step += 1
        self.cross.simulate_step()
        self.state = self._pack_state()
        return self.state, -self.max_V, False, {"error": msg,
                                                "hit_ratio": 0.0,
                                                "total_utility": 0.0}

    def step(self, action):
        cache_idx = action["cache_idx"].astype(np.int32)  # (C,)
        rec_idx = action["recommend_idx"].astype(np.int32)  # (V,K)

        mask = self.state["mask"]  # (V,)
        cache = np.zeros(self.num_files, np.int32)
        cache[cache_idx] = 1  # 把选中的 C 部影片位置设 1
        # ---------- rec:  (V, F) one-hot ----------
        rec = np.zeros((self.max_V, self.num_files), np.int32)
        rows = np.repeat(np.arange(self.max_V), self.K)
        rec[rows, rec_idx.reshape(-1)] = 1
        # ---------- 1) 处理 cache ---------- #
        if len(cache_idx) > self.cache_cap:
            # 随机丢弃超出的
            cache_idx = np.random.choice(cache_idx, self.cache_cap, replace=False)
        cache_set = set(int(x) for x in cache_idx)  # O(1) membership

        # ---------- 2) 处理 recommend ---------- #
        # rec_dict:  {v_id: list[file_id]}
        k_max = self.cfg.max_recommend_per_vehicle
        for v in range(self.max_V):
            if mask[v] == 0:
                continue
            # ① 拿到这一行所有 1 的列号
            cols = np.flatnonzero(rec[v])
            if cols.size > k_max:
                cols = np.random.choice(cols, k_max, replace=False)
            # ② 只留在 cache 中的
            cols = [c for c in cols if c in cache_set]
            # ③ 重新写回该行
            rec[v] *= 0
            rec[v, cols] = 1
        req_vec = np.zeros(self.num_files, np.int32)
        for idx, veh in enumerate(self.cross.vehicles[: self.max_V]):
            base = veh.interest_vector
            rec_files = rec[idx]  # 0/1 向量
            k = rec_files.sum()
            bonus = (1.0 / k) if k else 0.0

            eff = (1 - self.lambda_r) * base
            if k:
                eff[rec_files == 1] += self.lambda_r * bonus
            eff /= eff.sum()

            requested = np.random.choice(self.num_files, p=eff)
            req_vec[requested] += 1
            # ---- 偏好演化 -------------------------------------------------
            base *= (1 - self.eta)
            base[requested] += self.eta * (1.0 if rec_files[requested] else 0.5)
            veh.interest_vector = base / base.sum()

        # ---- 奖励 / 命中率 -------------------------------------------------
        hit_mask = rec * cache[None, :]
        miss_mask = rec * (1 - cache)[None, :]

        hit = hit_mask.sum()
        total_util = hit + 0.001 * miss_mask.sum()

        hit = (req_vec * cache).sum()
        total_req = req_vec.sum()
        hit_ratio = hit / (total_req + 1e-9)

        reward = hit_ratio + total_util * 0.5  # 或 / self.max_V

        # ---------- 4) 场景推进 ---------- #
        self.cache_status = np.zeros(self.num_files, np.int32)
        self.cache_status[list(cache_set)] = 1

        self.cross.simulate_step()
        self.state = self._pack_state()
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        return self.state, reward, done, {
            "total_utility": total_util,
            "hit_ratio": hit_ratio,
        }

    # -------- reset -------- #
    def reset(self):
        self.cross = Crossroad(width=200, height=200,
                               spawn_rate=self.cfg.spawn_rate,
                               user_interest_dict=self.cross.user_interest_dict)
        self.cache_status.fill(0)
        self.current_step = 0
        self.state = self._pack_state()
        return self.state

    def render(self, mode="human"):
        print(f"[Dynamic] step {self.current_step}, veh={self.state['mask'].sum()}")


# ===================== 2. GNNRecCacheEnv ============================= #
class GNNRecCacheEnv(gym.Env):
    """
    obs = {"graph":Data, "cache_status":[F]}
    act = {"cache":[F], "recommend":[max_V,F]}
    """
    def __init__(self, cfg, *, user_interest_dict=None):
        self.cfg, self.num_files = cfg, cfg.num_files
        self.max_V = cfg.gnn_max_vehicles
        self.cross = Crossroad(width=200, height=200,
                               spawn_rate=cfg.spawn_rate,
                               user_interest_dict=user_interest_dict)

        self.cache_cap = cfg.cache_capacity_rsu
        self.lambda_r  = cfg.lambda_rate
        self.eta       = cfg.preference_update_rate
        self.steps_per_episode = cfg.steps_per_episode

        self.current_step = 0
        self.cache_status = np.zeros(self.num_files, np.int32)

        # 占位 spaces
        self.observation_space = spaces.Dict({
            "cache_status": spaces.Box(0, 1, (self.num_files,), np.int32)
        })
        self.action_space = spaces.Dict({
            "cache": spaces.Box(0, 1, (self.num_files,), np.int32),
            "recommend": spaces.Box(0, 1,
                                    (self.max_V, self.num_files), np.int32)
        })
        self.state = self._pack_state()

    # -------- 打包状态 -------- #
    def _pack_state(self):
        g, _ = _build_rsu_vehicle_graph(self.cross.vehicles,
                                        self.num_files, self.max_V+1)
        return {"graph": g, "cache_status": self.cache_status.copy()}

    def _violated(self, msg):
        self.current_step += 1
        self.cross.simulate_step()
        self.state = self._pack_state()
        return self.state, -self.max_V, False, {"error": msg}

    # -------- step -------- #
    def step(self, action):
        cache, rec = action["cache"], action["recommend"]
        mask = self.state["graph"].mask.numpy()[1:]      # 车辆部分

        if cache.sum() > self.cache_cap:                                   return self._violated("cache cap")
        if (rec * mask[:, None]).sum(1).max() > self.cfg.max_recommend_per_vehicle:  return self._violated("rec cap")
        if np.any((rec==1) & (cache[np.newaxis,:]==0) & (mask[:,None]==1)):          return self._violated("rec non-cached")

        total_util, hit = 0.0, 0
        for idx, v in enumerate(self.cross.vehicles):
            base, rec_files = v.interest_vector, rec[idx]
            k = rec_files.sum();  bonus = (1/k) if k else 0
            eff = (1-self.lambda_r)*base + self.lambda_r*rec_files*bonus
            eff /= eff.sum()
            f = np.random.choice(self.num_files, p=eff)
            if cache[f]:  total_util += 1.0 if rec_files[f] else 0.5; hit += 1
            upd = (1-self.eta)*base; upd[f] += self.eta*(1.0 if rec_files[f] else 0.5)
            v.interest_vector = upd / upd.sum()

        reward   = total_util
        hit_ratio= hit / max(len(self.cross.vehicles), 1)

        self.cache_status = cache.copy()
        self.cross.simulate_step()
        self.state = self._pack_state()
        self.current_step += 1
        done = self.current_step >= self.steps_per_episode
        return self.state, reward, done, {"total_utility": total_util,
                                          "hit_ratio": hit_ratio}

    def reset(self):
        self.cross = Crossroad(width=200, height=200,
                               spawn_rate=self.cfg.spawn_rate,
                               user_interest_dict=self.cross.user_interest_dict)
        self.cache_status.fill(0)
        self.current_step = 0
        self.state = self._pack_state()
        return self.state

    def render(self, mode="human"):
        print(f"[GNN] step {self.current_step}, veh={len(self.cross.vehicles)}")
