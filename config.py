import argparse


def get_config():
    parser = argparse.ArgumentParser(description="DSAC-based JRCTO Configuration")

    # 仿真实验参数
    parser.add_argument('--num_vehicles', type=int, default=30,
                        help='车辆数量')
    parser.add_argument('--num_files', type=int, default=3260,
                        help='视频文件数量')
    parser.add_argument('--cache_capacity_rsu', type=int, default=20,
                        help='RSU缓存容量（单位：个），可视为可存放文件数量')
    parser.add_argument('--max_delay', type=int, default=5,
                        help='全局默认最大延迟容忍（秒）')
    parser.add_argument('--top_k_pref', type=int, default=1,
                        help='每个车辆的偏好列表长度（K）')
    parser.add_argument('--bloom_B', type=int, default=256,
                        help='Bloom filter 位数（B）')
    parser.add_argument('--bloom_H', type=int, default=4,
                        help='Bloom filter 哈希函数数量（H）')
    parser.add_argument('--ratings_path', type=str,default='ml-1m/ratings.dat',
                        help="MovieLens 数据 CSV 文件路径")
    parser.add_argument('--user_limit', type=int, default=None,
                        help="可选：限制抽取的用户数量（车辆数量），若不指定则使用所有满足条件的用户")
    parser.add_argument('--movie_limit', type=int, default=None,
                        help="可选：限制抽取的电影数量（文件数量），若不指定则使用所有满足条件的电影")
    parser.add_argument('--min_user_ratings', type=int, default=10,
                        help="用户至少需要的评分数")
    parser.add_argument('--hist_window', type=int, default=20,
                        help="历史窗口大小（H）")
    parser.add_argument('--min_movie_ratings', type=int, default=10,
                        help="电影至少需要的评分数")
    parser.add_argument('--output_path', type=str, default="preferences.npy",
                        help="保存预处理结果的文件路径")
    parser.add_argument('--max_recommend_per_vehicle', type=int, default=1,
                        help="每个车辆最多推荐的文件数量")
    parser.add_argument('--lambda_rate', type=float, default=0.9,
                        help="推荐采纳比例")
    parser.add_argument('--cross_dt', type=float, default=5, help='路口模拟的时间步长')
    parser.add_argument('--spawn_rate', type=float, default=2, help='每个时间步生成新车辆的期望数量')
    parser.add_argument('--max_speed', type=float, default=1.0, help='车辆的最大速度')
    parser.add_argument('--max_position', type=float, default=2, help='车辆的最大位置')
    #parser.add_argument('--lambda_rsu', type=float, default=0.5,
     #                   help="RSU缓存更新比例")
    parser.add_argument('--preference_update_rate', type=float, default=0.8,
                        help="偏好更新率")
    # 训练参数
    parser.add_argument('--device', type=str, default='mps',
                        help='设备：cuda 或 mps')

    parser.add_argument('--episode', type=int, default=1000,
                        help='训练总轮数')
    parser.add_argument('--steps_per_episode', type=int, default=256,
                        help='每个轮次的步数')
    parser.add_argument('--discount_factor', type=float, default=0.9,
                        help='折扣因子 γ')
    parser.add_argument('--replay_buffer_size', type=int, default=50000,
                        help='经验回放缓冲区大小')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='小批量训练的batch size')
    parser.add_argument('--soft_update_step', type=float, default=0.01,
                        help='目标网络软更新步长')

    # 学习率设置
    parser.add_argument('--lr_policy', type=float, default=5e-5,
                        help='策略网络学习率')
    parser.add_argument('--lr_q', type=float, default=1e-4,
                        help='Q网络学习率')

    # 通信相关参数（可根据实际需要扩展）
    parser.add_argument('--bandwidth_v2i', type=int, default=100,
                        help='RSU与车辆之间V2I通信带宽（MHz）')

    # DSAC 相关参数
    parser.add_argument('--temperature', type=float, default=1.2,
                        help='DSAC 算法中的温度参数，用于调节策略熵')


    # GNN 相关参数
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ("true", "1", "yes")
    parser.add_argument('--use_gnn', type=str2bool, default=False, help='是否使用 GNN 作为特征提取器')
    parser.add_argument('--gnn_max_vehicles', type=int, default=10, help='GNN 中的最大车辆数')
    parser.add_argument('--conv_type', type=str, default='gcn', help='GNN 中的卷积类型（gcn 或 gnn）')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_config()
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
