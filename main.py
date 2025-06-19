# main.py  ──────────────────────────────────────────────────────────────
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
# from sqlalchemy.testing.suite.test_reflection import users
from tqdm import tqdm

from config import get_config
from preprocess import preprocess_movielens

# 环境
from env import DynamicRecCacheEnv, GNNRecCacheEnv
from replaybuffer import ReplayBuffer
from agent import DSACAgent


# ---------- 统一扁平化 / 还原工具 ----------
def flatten_state_generic(state):
    """
    Dynamic  : -> 1-D numpy
    GNN      : -> 原 dict (由 agent 内部处理)
    """
    if "preferences" in state:                       # Dynamic
        prefs = state["preferences"].reshape(-1)
        cache = state["cache_status"]
        return np.concatenate([prefs, cache])
    elif "graph" in state:                           # GNN
        return state                                 # 保留原结构
    else:
        raise ValueError("Unknown state format")


# ----------------------------------------------------------------------
def main():
    cfg = get_config()

    # -------- MovieLens 预处理，只在 Dynamic 模式下可选 --------
    preferences = None
    if getattr(cfg, "preferences_path", None) and os.path.exists(cfg.preferences_path):
        preferences = np.load(cfg.preferences_path)

        print("Loaded preferences from", cfg.preferences_path)
    else:
        print("Preprocessing MovieLens ...")
        preferences, user_to_index, _ = preprocess_movielens(
            ratings_path=cfg.ratings_path,
            user_limit=cfg.user_limit,
            movie_limit=cfg.movie_limit,
            min_user_ratings=cfg.min_user_ratings,
            min_movie_ratings=cfg.min_movie_ratings,

        )

        if getattr(cfg, "preferences_output_path", None):
            np.save(cfg.preferences_output_path, preferences)

    # -------- 创建环境 --------
    user_interest_dict = {
        u_id: preferences[idx]  # idx = 行号
        for u_id, idx in user_to_index.items()
    }
    env = GNNRecCacheEnv(cfg, user_interest_dict=user_interest_dict) if cfg.use_gnn else DynamicRecCacheEnv(cfg,  user_interest_dict=user_interest_dict)

    # -------- 决定 state_dim（仅 MLP 用） --------
    first_state = env.reset()
    state_dim = (
        flatten_state_generic(first_state).shape[0]  # Dynamic 向量长度
        if not cfg.use_gnn else 0                    # GNN 不用
    )

    # -------- 构造 Agent --------
    agent = DSACAgent(
        state_dim   = state_dim,
        hidden_dim  = getattr(cfg, "hidden_dim", 128),
        num_files   = cfg.num_files,
        num_vehicles= cfg.gnn_max_vehicles,
        actor_lr    = cfg.lr_policy,
        critic_lr   = cfg.lr_q,
        gamma       = cfg.discount_factor,
        tau         = cfg.soft_update_step,
        alpha       = cfg.temperature,
        device      = cfg.device,
        use_gnn     = cfg.use_gnn,
        conv_type   = cfg.conv_type,
        node_feat_dim = cfg.num_files + 1 if cfg.use_gnn else None
    )

    # -------- ReplayBuffer --------
    buffer = ReplayBuffer(cfg.replay_buffer_size)

    # -------- 训练循环 --------
    episode_rewards = []
    avg_hits = []
    for ep in tqdm(range(cfg.episode), desc="Training"):
        state = env.reset()
        done, ep_reward = False, 0.0
        hit_ratios = 0.0
        utilities = 0.0
        while not done:
            obs_for_agent = state if cfg.use_gnn else flatten_state_generic(state)
            action_dict   = agent.select_action(obs_for_agent)

            next_state, reward, done, info = env.step(action_dict)
            hit_ratios += info["hit_ratio"]
            utilities  += info["total_utility"]
            state   = next_state
            ep_reward += reward
            buffer.push(state, action_dict, reward, next_state, done)
            if len(buffer) >= cfg.batch_size:
                batch = buffer.sample(cfg.batch_size)
                agent.update(batch, flatten_state_generic)

        episode_rewards.append(ep_reward)
        avg_hit = hit_ratios / cfg.steps_per_episode
        avg_utilities = utilities / cfg.steps_per_episode
        avg_hits.append(avg_hit)
        print(f"Episode {ep+1}/{cfg.episode}  reward={ep_reward:.1f} avg_hits={avg_hit:.3f} avg_utilities={avg_utilities:.1f}")

    # -------- 绘图 --------
    plt.plot(avg_hits)
    plt.xlabel("Episode"); plt.ylabel("hit_ratio")
    plt.title("DSAC Training hits"); plt.grid(True); plt.show()
    plt.savefig("results/training_hits.png")


if __name__ == "__main__":
    import torch
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()
