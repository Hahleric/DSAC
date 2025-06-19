
from collections import deque
import numpy as np
class RunningStats:
    """把最近 N 次 update 的标量保存起来，方便做均值 / 最大值等统计"""
    def __init__(self, maxlen=100):
        self.q1 = deque(maxlen=maxlen)
        self.q2 = deque(maxlen=maxlen)
        self.al = deque(maxlen=maxlen)
        self.lp = deque(maxlen=maxlen)        # 平均 log π
        self.tq = deque(maxlen=maxlen)        # target-Q 最大值

    def push(self, q1_loss, q2_loss, actor_loss, logp_mean, tq_max):
        self.q1.append(q1_loss)
        self.q2.append(q2_loss)
        self.al.append(actor_loss)
        self.lp.append(logp_mean)
        self.tq.append(tq_max)

    def summary(self):
        fmt = lambda x: f"{np.mean(x):7.3e}"
        return (f"Q1_L {fmt(self.q1)} | "
                f"Q2_L {fmt(self.q2)} | "
                f"Act_L {fmt(self.al)} | "
                f"logπ {fmt(self.lp)} | "
                f"TQ_max {fmt(self.tq)}")
