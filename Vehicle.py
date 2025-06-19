# vehicle_env.py
import numpy as np
import uuid
import config

args = config.get_config()

# --------------------  Vehicle  --------------------
class Vehicle:
    """
    车辆（或用户）实体

    Parameters
    ----------
    user_id : int | str
        唯一标识符
    position : array-like, shape (2,)
        初始位置 (x, y)
    speed : array-like, shape (2,)
        速度向量 (vx, vy)  — 每个 step 默认位移 = speed
    interest_vector : array-like, shape (F,)
        内容偏好分布
    request_frequency : float | None, optional
        **当前版本未使用**。预留给“车辆个体请求速率”特性，后续可根据
        `np.random.poisson(lam=request_frequency)` 生成请求事件。
    """

    def __init__(
            self,
            user_id,
            position,
            speed,
            interest_vector,
            request_frequency=None  # <-- 现在默认为 None
    ):
        self.user_id = user_id
        self.position = np.asarray(position, dtype=float)
        self.speed = np.asarray(speed, dtype=float)
        self.interest_vector = np.asarray(interest_vector, dtype=float)
        self.request_frequency = request_frequency  # 未使用，占位

    # 其余方法保持不变
    def update_position(self, dt=1.0):
        self.position += self.speed * dt

    def get_bandwidth(self, rsu_position, B0=10.0, alpha=0.1):
        distance = np.linalg.norm(self.position - rsu_position)
        bandwidth = B0 * np.exp(-alpha * distance) + np.random.normal(0, 0.5)
        return max(bandwidth, 0.0)

    def get_position(self):
        return self.position


# --------------------  RSU  --------------------
class RSU:
    def __init__(self, position, B0=10.0, alpha=0.1):
        self.position = np.asarray(position, dtype=float)
        self.B0 = B0
        self.alpha = alpha

    def compute_bandwidth(self, vehicle_position):
        distance = np.linalg.norm(np.asarray(vehicle_position) - self.position)
        bandwidth = self.B0 * np.exp(-self.alpha * distance) + np.random.normal(0, 0.5)
        return max(bandwidth, 0.0)


# --------------------  Crossroad  --------------------
class Crossroad:
    """
    简单“十字路口”场景：车辆随时间进入 / 离开，中心 RSU 提供带宽。

    Parameters
    ----------
    width, height : float
        场景边界大小
    rsu_position : (2,), optional
        RSU 位置；默认放在中心
    spawn_rate : float
        每个 step 期望生成的新车数量 (Poisson λ)
    user_interest_dict : dict[int, np.ndarray] | None
        可选：映射 user_id -> 预定义兴趣向量
    """

    def __init__(
            self,
            width,
            height,
            rsu_position=None,
            B0=10.0,
            alpha=0.1,
            spawn_rate=1.2,
            user_interest_dict=None,
    ):
        self.width = width
        self.height = height
        rsu_position = rsu_position or [width / 2, height / 2]
        self.rsu = RSU(rsu_position, B0, alpha)

        self.spawn_rate = spawn_rate
        self.user_interest_dict = user_interest_dict

        self.vehicles = []
        self.vehicle_counter = 0
        self.max_distance = np.linalg.norm([width, height])

    # ---------- 车辆生成 ----------
    def generate_vehicle(
            self,
            user_id=None,
            position=None,
            speed=None,
            interest_vector=None,
    ):
        if user_id is None:
            user_id = np.random.choice(list(self.user_interest_dict)) \
                if self.user_interest_dict else uuid.uuid4().hex

        position = position or np.random.uniform([0, 0], [self.width, self.height])
        speed = speed or np.random.uniform(-1, 1, size=2)

        if interest_vector is None:
            if self.user_interest_dict and user_id in self.user_interest_dict:
                interest_vector = self.user_interest_dict[user_id].copy()
            else:
                interest_vector = np.random.rand(args.num_files)
                interest_vector /= interest_vector.sum()

        # request_frequency 不再传入，保持 None
        v = Vehicle(
            user_id=user_id,
            position=position,
            speed=speed,
            interest_vector=interest_vector,
            request_frequency=None
        )
        self.vehicles.append(v)
        return v

    # ---------- 从场景边缘生成新车 ----------
    def spawn_vehicle_from_edge(self):
        edge = np.random.choice(['left', 'right', 'top', 'bottom'])
        if edge == 'left':
            x, y = 0, np.random.uniform(0, self.height)
            speed = [np.random.uniform(0.5, 1.5), np.random.uniform(-1, 1)]
        elif edge == 'right':
            x, y = self.width, np.random.uniform(0, self.height)
            speed = [np.random.uniform(-1.5, -0.5), np.random.uniform(-1, 1)]
        elif edge == 'top':
            x, y = np.random.uniform(0, self.width), self.height
            speed = [np.random.uniform(-1, 1), np.random.uniform(-1.5, -0.5)]
        else:  # bottom
            x, y = np.random.uniform(0, self.width), 0
            speed = [np.random.uniform(-1, 1), np.random.uniform(0.5, 1.5)]
        self.generate_vehicle(position=[x, y], speed=speed)

    # ---------- 每一步仿真 ----------
    def update_vehicles(self, dt=1.0):
        for v in self.vehicles:
            v.update_position(dt)

    def remove_exited_vehicles(self):
        self.vehicles = [
            v for v in self.vehicles
            if 0 <= v.position[0] <= self.width and 0 <= v.position[1] <= self.height
        ]

    def spawn_new_vehicles(self):
        lam = self.spawn_rate
        n_new = np.random.poisson(lam)
        for _ in range(n_new):
            self.spawn_vehicle_from_edge()

    def simulate_step(self, dt=1.0):
        self.update_vehicles(dt)
        self.remove_exited_vehicles()
        self.spawn_new_vehicles()
        # 返回每辆车的带宽字典 {user_id: bandwidth}
        return {
            v.user_id: v.get_bandwidth(self.rsu.position)
            for v in self.vehicles
        }
