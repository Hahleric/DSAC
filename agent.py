# agent.py  ──────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.running_stats import RunningStats
import config

cfg = config.get_config()
# optional: only used when --use_gnn
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv, global_mean_pool
)
import torch_geometric

# ────────────────────────────────────────────────────────────────────────
# 0. 通用小工具
# ────────────────────────────────────────────────────────────────────────
def _scatter_onehot(indices, length, device):
    """indices: (B,C) or (B,V,K)  →  one-hot (B,length) / (B,V,length)"""
    if indices.ndim == 2:      # cache
        B, C = indices.shape
        out = torch.zeros(B, length, device=device)
        out.scatter_(1, indices, 1.0)
        return out
    else:                      # recommend
        B, V, K = indices.shape
        out = torch.zeros(B, V, length, device=device)
        for v in range(V):
            out[torch.arange(B, device=device), v]\
                .scatter_(1, indices[:, v, :], 1.0)
        return out

def flatten_action(action_dict):
    # cache_idx : (B, C)   int32/64
    # recommend_idx : (B, V, K)
    cache = action_dict["cache_idx"].float()        # (B,C)
    rec   = action_dict["recommend_idx"].float().view(cache.size(0), -1)  # (B,V·K)
    return torch.cat([cache, rec], dim=1)           # (B, C+V·K)



# ────────────────────────────────────────────────────────────────────────
# 1. Actor — MLP 版
# ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPActor(nn.Module):
    """
    输出：
      cache_idx     :  [B, C]          （每行 C 个文件 id）
      recommend_idx :  [B, V, K]       （每辆车 K 个 id）  -> 扁平后给 env
      logp          :  [B]             （对应动作的对数概率）
    """
    def __init__(self,
                 state_dim        : int,
                 hidden_dim       : int,
                 num_files        : int,    # F
                 num_vehicles     : int,    # V
                 cache_topk       : int,    # C
                 rec_topk         : int):   # K
        super().__init__()
        self.F = num_files
        self.V = num_vehicles
        self.C = cache_topk
        self.K = rec_topk

        # ── MLP trunk ──────────────────────────
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # ── 输出两个 logits 向量 ────────────────
        # ① 缓存选择：尺寸 F
        self.cache_head = nn.Linear(hidden_dim, num_files)

        # ② 推荐选择：对每辆车都给一个 F-dim logits
        self.rec_head = nn.Linear(hidden_dim, num_vehicles * num_files)

    # ------------------------------------------------------------
    # util：Gumbel-Top-K 采样  (hard = True => 纯离散  +  logπ)
    # ------------------------------------------------------------
    def _gumbel_topk(self, logits: torch.Tensor, k: int):
        """
        logits : [B, F]
        return : indices[B, k],   logp[B]
        """
        g = -torch.empty_like(logits).exponential_().log()  # ~ Gumbel(0,1)
        y = (logits + g).topk(k, dim=-1)
        idx = y.indices                                    # [B,k]
        # 取每个选中元素的 softmax prob 再 log；这里用 log-softmax 快速取
        log_probs = F.log_softmax(logits, dim=-1).gather(1, idx)  # [B,k]
        return idx, log_probs.sum(-1)                              # 和起来得 logπ

    # ------------------------------------------------------------
    def forward(self, s):
        """
        s : [B, state_dim]  （这里只算一次 trunk + 两个 logits）
        """
        h = F.relu(self.fc1(s))
        h = F.relu(self.fc2(h))
        cache_logits = self.cache_head(h)                        # [B,F]
        rec_logits   = self.rec_head(h).view(-1, self.V, self.F) # [B,V,F]
        return cache_logits, rec_logits

    # ------------------------------------------------------------
    def sample(self, s):
        cache_logits, rec_logits = self.forward(s)  # logits
        B = s.size(0)

        # 1. 缓存 Top-C
        cache_idx, logp_cache = self._gumbel_topk(cache_logits, self.C)  # (B,C)

        # 2. 推荐 Top-K / vehicle
        rec_idx = torch.empty(B, self.V, self.K, dtype=torch.long, device=s.device)
        logp_rec = torch.zeros(B, device=s.device)
        for v in range(self.V):
            idx_v, lp_v = self._gumbel_topk(rec_logits[:, v, :], self.K)  # (B,K)
            rec_idx[:, v, :] = idx_v
            logp_rec += lp_v

        act = {"cache_idx": cache_idx,  # (B,C)  long
               "recommend_idx": rec_idx}  # (B,V,K) long
        return act, (logp_cache + logp_rec)  # (B,)


# ────────────────────────────────────────────────────────────────────────
# 2. Actor — GNN 版
# ────────────────────────────────────────────────────────────────────────
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hid, conv_type="gcn", heads=4):
        super().__init__()
        conv = conv_type.lower()
        if conv == "gcn":
            self.g1, self.g2 = GCNConv(in_dim, hid), GCNConv(hid, hid)
        elif conv == "gat":
            self.g1 = GATConv(in_dim, hid // heads, heads=heads, concat=True)
            self.g2 = GATConv(hid, hid, heads=1, concat=False)
        elif conv == "transformer":
            self.g1 = TransformerConv(in_dim, hid, heads=heads)
            self.g2 = TransformerConv(hid*heads, hid)
        else:
            raise ValueError("conv_type must be gcn | gat | transformer")

    def forward(self, x, edge_index, batch):
        h = F.relu(self.g1(x, edge_index))
        h = self.g2(h, edge_index)
        rsu = h[batch == 0]                            # [B,hid]
        veh = h[batch > 0]
        veh_pool = global_mean_pool(
            veh, batch[batch > 0]) if veh.shape[0] else torch.zeros_like(rsu)
        return torch.cat([rsu, veh_pool], dim=-1)      # [B,2*hid]


class GNNActor(nn.Module):
    def __init__(self, num_files, max_veh,
                 node_feat_dim, hid, conv_type):
        super().__init__()
        self.num_files, self.max_veh = num_files, max_veh
        self.enc = GNNEncoder(node_feat_dim, hid, conv_type)
        self.fc1 = nn.Linear(2*hid + num_files, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.cache_head = nn.Linear(hid, num_files*2)
        self.rec_head   = nn.Linear(hid, max_veh*num_files*2)

    def forward(self, g, cache_status):
        z = self.enc(g.x, g.edge_index, g.batch)       # [B,2hid]
        x = torch.cat([z, cache_status.float()], dim=-1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        B = x.size(0)
        cache_logits = self.cache_head(x).view(B, self.num_files, 2)
        rec_logits   = self.rec_head(x).view(
            B, self.max_veh, self.num_files, 2)
        return F.softmax(cache_logits, -1), F.softmax(rec_logits, -1)

    def sample(self, g, cache_status):
        cache_p, rec_p = self.forward(g, cache_status)
        cd, rd = torch.distributions.Categorical(cache_p), \
                 torch.distributions.Categorical(rec_p)
        cache_a = cd.sample(); rec_a = rd.sample()
        rec_a   = rec_a * cache_a.unsqueeze(1)
        logp = cd.log_prob(cache_a).sum(1) + \
               rd.log_prob(rec_a).sum([1,2])
        return {"cache_idx": cache_a, "recommend_idx": rec_a}, logp


# ────────────────────────────────────────────────────────────────────────
# 3. Critic（共享）
# ────────────────────────────────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.f1 = nn.Linear(state_dim + action_dim, hidden)
        self.f2 = nn.Linear(hidden, hidden)
        self.q  = nn.Linear(hidden, 1)

    def forward(self, s, a_flat):
        x = torch.cat([s, a_flat], dim=1)
        x = F.relu(self.f1(x)); x = F.relu(self.f2(x))
        return self.q(x)


class DSACAgent:
    def __init__(self,
                 state_dim, hidden_dim,
                 num_files, num_vehicles,
                 actor_lr=1e-4, critic_lr=2e-4,
                 gamma=0.9, tau=0.01, alpha=0.05,
                 device="cpu",
                 use_gnn=False, conv_type="gcn",
                 node_feat_dim=None        # GNN 时必填 = num_files+1
                 ):
        self.use_gnn = use_gnn
        self.stats = RunningStats(maxlen=100)
        self.device  = torch.device(device)
        self.num_files, self.num_veh = num_files, num_vehicles
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.C = cfg.cache_capacity_rsu  # 缓存 top-C
        self.K = cfg.max_recommend_per_vehicle  # 每车 top-K

        # 把旧的 act_dim 替换掉
        self.act_dim = self.C + num_vehicles * self.K  # <<< 只有 index 数量

        # debug data
        self.update_steps = 0
        self.print_every = 64

        # ---------- Actor ----------
        if use_gnn:
            assert node_feat_dim is not None
            self.actor = GNNActor(num_files, num_vehicles,
                                  node_feat_dim, hidden_dim, conv_type).to(self.device)
            self.state_dim = 2*hidden_dim + num_files
        else:
            self.actor = MLPActor(state_dim, hidden_dim,
                                      num_files, num_vehicles,
                                       cache_topk = cfg.cache_capacity_rsu, rec_topk = cfg.max_recommend_per_vehicle).to(self.device)
            self.state_dim = state_dim

        # ---------- Critic & target ----------
        self.critic1 = Critic(self.state_dim, self.act_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.act_dim, hidden_dim).to(self.device)
        self.targ1   = Critic(self.state_dim, self.act_dim, hidden_dim).to(self.device)
        self.targ2   = Critic(self.state_dim, self.act_dim, hidden_dim).to(self.device)
        self.targ1.load_state_dict(self.critic1.state_dict())
        self.targ2.load_state_dict(self.critic2.state_dict())

        self.act_opt  = optim.Adam(self.actor.parameters(),   lr=actor_lr)
        self.cri1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.cri2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

    # ── 行为策略 ───────────────────────────────────────────────
    def select_action(self, state):
        """
        state:
          - GNN  : {"graph": Data, "cache_status": np.array}
          - MLP  : flat numpy array
        """
        self.actor.eval()
        with torch.no_grad():
            if self.use_gnn:
                g = state["graph"].to(self.device)
                cache = torch.FloatTensor(state["cache_status"]).unsqueeze(0).to(self.device)
                act_idx, _ = self.actor.sample(g, cache)
            else:
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                act_idx, _ = self.actor.sample(s)
            self.actor.train()

        return {  # 保持 long/int 索引
            "cache_idx": act_idx["cache_idx"].squeeze(0).cpu().numpy().astype(np.int32),
            "recommend_idx": act_idx["recommend_idx"].squeeze(0).cpu().numpy().astype(np.int32)
        }

    # ── 内部：batch-state → tensor ─────────────────────────────
    def _state_batch_to_tensor(self, states):
        if self.use_gnn:                                     # list[dict]
            g_list, cache = zip(*[(s["graph"], s["cache_status"]) for s in states])
            batched = g_list[0].__class__.from_data_list(g_list).to(self.device)
            cache = torch.FloatTensor(np.stack(cache)).to(self.device)
            g_emb = self.actor.enc(batched.x, batched.edge_index, batched.batch)
            return torch.cat([g_emb, cache], dim=-1)         # [B,state_dim]
        else:                                                # list[np.ndarray]
            arr = np.stack(states).astype(np.float32)
            return torch.FloatTensor(arr).to(self.device)

    # ── 更新参数 ──────────────────────────────────────────────
    def update(self, batch, flatten_state_fn):
        s_b, a_b, r_b, ns_b, d_b = batch
        S   = self._state_batch_to_tensor([flatten_state_fn(s) for s in s_b])
        S2  = self._state_batch_to_tensor([flatten_state_fn(s) for s in ns_b])
        A = torch.from_numpy(np.stack([self._flatten_action_dict(a) for a in a_b])
                             ).float().to(self.device)
        R   = torch.FloatTensor(r_b).unsqueeze(1).to(self.device)
        D   = torch.FloatTensor(d_b).unsqueeze(1).to(self.device)

        # ------- target Q -------
        with torch.no_grad():
            if self.use_gnn:
                next_act, lp2 = self.actor.sample(
                    torch_geometric.loader.DataLoader([s["graph"] for s in ns_b]).collate(),
                    torch.FloatTensor([s["cache_status"] for s in ns_b]).to(self.device))
            else:
                next_act, lp2 = self.actor.sample(S2)
            A2 = flatten_action(next_act)
            tq1 = self.targ1(S2, A2); tq2 = self.targ2(S2, A2)
            target = R + (1-D) * self.gamma * (torch.min(tq1, tq2) - self.alpha*lp2.unsqueeze(1))

        # ------- critic -------
        for critic,opt in [(self.critic1,self.cri1_opt),(self.critic2,self.cri2_opt)]:
            loss = F.smooth_l1_loss(critic(S, A), target)  # Huber loss
            opt.zero_grad(); loss.backward(); opt.step()

        c1_loss = F.smooth_l1_loss(self.critic1(S, A), target)
        c2_loss = F.smooth_l1_loss(self.critic2(S, A), target)

        # ------- actor -------
        if self.use_gnn:
            new_act, lp = self.actor.sample(
                torch_geometric.loader.DataLoader([s["graph"] for s in s_b]).collate(),
                torch.FloatTensor([s["cache_status"] for s in s_b]).to(self.device))
        else:
            new_act, lp = self.actor.sample(S)
        A_new = flatten_action(new_act)
        q_new = torch.min(self.critic1(S, A_new), self.critic2(S, A_new))
        act_loss = (self.alpha * lp.unsqueeze(1) - q_new).mean()
        self.act_opt.zero_grad(); act_loss.backward(); self.act_opt.step()

        # ------- soft-update -------
        self._soft_update(self.critic1, self.targ1)
        self._soft_update(self.critic2, self.targ2)

        self.stats.push(c1_loss.item(),
                        c2_loss.item(),
                        act_loss.item(),
                        lp2.mean().item(),  # 上面算 target-Q 时得到的 logπ
                        tq1.abs().max().item())  # 同样在 target-Q 处已算出

        self.update_steps += 1
        # ---------- extra debug ----------
        if self.update_steps % self.print_every == 0:
            with torch.no_grad():
                # policy 熵（越大越随机）
                ent_cache = -(F.softmax(self.actor.cache_head(self.actor.fc2(F.relu(self.actor.fc1(S)))),
                                        dim=-1) * F.log_softmax(
                    self.actor.cache_head(self.actor.fc2(F.relu(self.actor.fc1(S)))), dim=-1)
                              ).sum(-1).mean()
                ent_rec = -(F.softmax(
                    self.actor.rec_head(self.actor.fc2(F.relu(self.actor.fc1(S)))).view(-1, self.num_veh,
                                                                                        self.num_files),
                    dim=-1)
                            * F.log_softmax(
                            self.actor.rec_head(self.actor.fc2(F.relu(self.actor.fc1(S)))).view(-1, self.num_veh,
                                                                                                self.num_files),
                            dim=-1)
                            ).sum([-1]).mean()

                # Q 分布统计
                q1_vals = self.critic1(S, A).detach().cpu().numpy()
                q_std = q1_vals.std()

                # 梯度范数
                total_norm = 0.0
                for p in self.actor.parameters():
                    if p.grad is None: continue
                    total_norm += (p.grad.data.norm(2) ** 2).item()
                total_norm = total_norm ** 0.5

            print(f"[upd {self.update_steps:>6}] "
                  f"Q1_L {c1_loss.item():.3e} | "
                  f"Act_L {act_loss.item():.3e} | "
                  f"‖∇π‖ {total_norm:.2e} | "
                  f"H(cache) {ent_cache:.2f} H(rec) {ent_rec:.2f} | "
                  f"logπ {lp2.mean().item():.1f} | "
                  f"Q_std {q_std:.1f} | "
                  f"TQ_max {tq1.abs().max().item():.1e}")

    # ── 辅助 ─────────────────────────────────────────────────
    def _soft_update(self, net, targ):
        for p,tp in zip(net.parameters(), targ.parameters()):
            tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)

    def _flatten_action_dict(self, a):
        idx = np.concatenate([a["cache_idx"].reshape(-1),
                              a["recommend_idx"].reshape(-1)]).astype(np.float32)
        return idx / self.num_files   # 或者 (idx - μ) / σ
