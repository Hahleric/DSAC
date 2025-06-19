import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化 replay buffer

        参数：
          capacity: buffer 的最大容量
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        存储一个 transition 到 buffer 中

        参数：
          state: 当前状态（一般为字典，如 {"preferences": ..., "cache_status": ...}）
          action: 当前动作（字典形式，如 {"cache": ..., "recommend": ...}）
          reward: 本步奖励
          next_state: 下一状态（同 state 格式）
          done: 是否 episode 结束（True/False）
        """
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        随机采样一批 transition

        返回：
          state_batch, action_batch, reward_batch, next_state_batch, done_batch
        """
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return (state_batch,
                action_batch,
                np.array(reward_batch, dtype=np.float32),
                next_state_batch,
                np.array(done_batch, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DataLoader:
    def __init__(self, replay_buffer, batch_size):
        """
        DataLoader 封装 replay buffer 的采样功能，用于训练时迭代获取 minibatch

        参数：
          replay_buffer: 一个 ReplayBuffer 实例
          batch_size: 每个 minibatch 的样本数
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.replay_buffer) < self.batch_size:
            raise StopIteration
        return self.replay_buffer.sample(self.batch_size)
