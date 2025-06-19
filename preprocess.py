import argparse
import pandas as pd
import numpy as np
from config import get_config

def preprocess_movielens(ratings_path, user_limit=None, movie_limit=None, min_user_ratings=20, min_movie_ratings=20):
    """
    加载 MovieLens 数据，并构造归一化的用户-电影偏好矩阵。

    参数：
      ratings_path: MovieLens 数据文件路径（CSV 格式，需包含 userId, movieId, rating, timestamp）
      user_limit: 可选参数，限制抽取的用户数量；若为 None，则使用所有满足 min_user_ratings 条件的用户
      movie_limit: 可选参数，限制抽取的电影数量；若为 None，则使用所有满足 min_movie_ratings 条件的电影
      min_user_ratings: 用户至少需要的评分数，低于此值的用户将被过滤掉
      min_movie_ratings: 电影至少需要的评分数，低于此值的电影将被过滤掉

    返回：
      rating_matrix: NumPy 数组，形状为 (num_users, num_movies)，每行归一化为概率分布
      user_to_index: 用户ID到矩阵行索引的映射字典
      movie_to_index: 电影ID到矩阵列索引的映射字典
    """
    # 读取数据
    df = pd.read_csv(ratings_path, sep="::", engine="python",
                     names=["userId", "movieId", "rating", "timestamp"])
    # 确保 userId 和 movieId 为整数（若原始数据已调整为从0开始，可忽略此步）
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)

    # 根据最小评分数过滤用户和电影
    user_counts = df['userId'].value_counts()
    movie_counts = df['movieId'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df = df[df['userId'].isin(valid_users) & df['movieId'].isin(valid_movies)]

    # 如果指定了 movie_limit，则选取评分数量最多的前 movie_limit 部电影
    if movie_limit is not None:
        top_movies = df['movieId'].value_counts().nlargest(movie_limit).index
        df = df[df['movieId'].isin(top_movies)]

    # 如果指定了 user_limit，则从满足条件的用户中随机抽取 user_limit 个用户
    if user_limit is not None:
        available_users = df['userId'].unique()
        selected_users = np.random.choice(available_users, size=user_limit, replace=False)
        df = df[df['userId'].isin(selected_users)]

    # 最终获取用户和电影的有序列表
    users = sorted(df['userId'].unique())
    movies = sorted(df['movieId'].unique())

    # 构造映射字典：将实际的 userId 和 movieId 映射到矩阵的索引
    user_to_index = {user: idx for idx, user in enumerate(users)}
    movie_to_index = {movie: idx for idx, movie in enumerate(movies)}

    num_users = len(users)
    num_movies = len(movies)

    # 构造评分矩阵
    rating_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
    for row in df.itertuples():
        u = user_to_index[row.userId]
        m = movie_to_index[row.movieId]
        rating_matrix[u, m] = row.rating

    # 对每个用户的评分向量归一化为概率分布（若一行全为 0，则设为均匀分布）
    for i in range(num_users):
        row_sum = rating_matrix[i].sum()
        if row_sum > 0:
            rating_matrix[i] /= row_sum
        else:
            rating_matrix[i] = np.ones(num_movies, dtype=np.float32) / num_movies

    return rating_matrix, user_to_index, movie_to_index


def save_preferences(preferences, output_path):
    """
    将预处理得到的偏好矩阵保存为 .npy 文件，便于后续加载到环境中使用。
    """
    np.save(output_path, preferences)
    print(f"Preferences matrix saved to {output_path}")


if __name__ == "__main__":

    args = get_config()

    preferences, user_to_index, movie_to_index = preprocess_movielens(
        args.ratings_path,
        user_limit=args.user_limit,
        movie_limit=args.movie_limit,
        min_user_ratings=args.min_user_ratings,
        min_movie_ratings=args.min_movie_ratings
    )

    print("Processed preference matrix shape:", preferences.shape)
    save_preferences(preferences, args.output_path)
