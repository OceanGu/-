import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """读取数据"""
    df = pd.read_excel(file_path)
    print("数据读取成功！")
    print(f"数据形状: {df.shape}")
    return df

def preprocess_data(df):
    """数据预处理 - 修复版本"""
    print("\n数据预处理...")

    # 确保数值列正确
    numeric_cols = ['DOUBAN_SCORE', 'TOTAL_RATINGS', 'RATING_1_COUNT', 'RATING_2_COUNT',
                    'RATING_3_COUNT', 'RATING_4_COUNT', 'RATING_5_COUNT']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 去除缺失值
    df_clean = df.dropna(subset=numeric_cols)

    # 过滤有效数据
    df_filtered = df_clean[
        (df_clean['TOTAL_RATINGS'] >= 50) &
        (df_clean['DOUBAN_SCORE'] >= 1) &
        (df_clean['DOUBAN_SCORE'] <= 10)
        ]

    print(f"有效数据量: {len(df_filtered)}")
    return df_filtered


def define_rating_levels():
    """定义评分等级区间 - 基于10分制"""
    levels = {
        'A_很差': (0, 4.0),
        'B_较差': (4.0, 5.5),
        'C_一般': (5.5, 7.0),
        'D_良好': (7.0, 8.5),
        'E_优秀': (8.5, 10.1)
    }
    return levels


def generate_samples_from_ratings(row, scale=2):
    """
    根据评分分布生成样本点
    scale=2: 将1-5星映射到2-10分（豆瓣评分制）
    """
    ratings = []
    # 1星 -> 2分, 2星 -> 4分, 3星 -> 6分, 4星 -> 8分, 5星 -> 10分
    star_mapping = {1: 2.0, 2: 4.0, 3: 6.0, 4: 8.0, 5: 10.0}

    for star in range(1, 6):
        count = int(row[f'RATING_{star}_COUNT'])
        rating_value = star_mapping[star]
        ratings.extend([rating_value] * count)

    return ratings


def backward_cloud_generator(samples):
    """逆向云发生器 - 修复版本"""
    if len(samples) == 0:
        return 0, 0, 0

    samples = np.array(samples)
    Ex = np.mean(samples)
    En = np.std(samples)

    # 修复He的计算：使用样本熵的标准差
    if len(samples) > 1:
        # 将样本分成若干组，计算每组的熵，然后求这些熵的标准差
        n_groups = min(10, len(samples) // 10)  # 至少10个样本一组
        if n_groups > 1:
            grouped_samples = np.array_split(samples, n_groups)
            group_entropies = [np.std(group) for group in grouped_samples]
            He = np.std(group_entropies)
        else:
            He = 0.1  # 默认小值
    else:
        He = 0.1

    # 确保数值合理
    En = max(En, 0.1)
    He = max(He, 0.05)

    return Ex, En, He


def forward_cloud_generator(Ex, En, He, n=1000):
    """正向云发生器 - 修复版本"""
    cloud_drops = []

    for _ in range(n):
        # 生成正态分布的En_i
        En_i = np.random.normal(En, He)
        En_i = max(En_i, 0.01)  # 避免负值或过小

        # 生成评分
        x_i = np.random.normal(Ex, En_i)

        # 计算确定度
        if En_i > 0.001:
            y_i = np.exp(-(x_i - Ex) ** 2 / (2 * En_i ** 2))
        else:
            y_i = 1.0 if abs(x_i - Ex) < 0.1 else 0.0

        cloud_drops.append((x_i, y_i))

    return np.array(cloud_drops)


def calculate_membership(x, Ex, En, He=0.1):
    """计算确定度 - 修复版本"""
    if En < 0.001:
        return 1.0 if abs(x - Ex) < 0.1 else 0.0

    # 考虑超熵的影响，使用蒙特卡洛方法
    n_samples = 100
    memberships = []

    for _ in range(n_samples):
        En_i = np.random.normal(En, He)
        En_i = max(En_i, 0.01)
        membership = np.exp(-(x - Ex) ** 2 / (2 * En_i ** 2))
        memberships.append(membership)

    return np.mean(memberships)


def main_analysis_fixed(file_path):
    """修复后的主分析函数"""

    # 1. 读取和预处理数据
    df = load_data(file_path)
    df_clean = preprocess_data(df)

    # 2. 定义等级
    rating_levels = define_rating_levels()

    # 3. 为每个等级收集样本
    print("\n为每个等级收集样本...")
    level_samples = {}
    level_movies = {}

    for level_name, (low, high) in rating_levels.items():
        mask = (df_clean['DOUBAN_SCORE'] >= low) & (df_clean['DOUBAN_SCORE'] < high)
        level_movies_df = df_clean[mask]
        level_movies[level_name] = level_movies_df

        print(f"{level_name} ({low}-{high}分): {len(level_movies_df)} 部电影")

        # 生成样本
        all_samples = []
        for _, row in level_movies_df.iterrows():
            movie_samples = generate_samples_from_ratings(row, scale=2)
            all_samples.extend(movie_samples)

        level_samples[level_name] = all_samples

    # 4. 计算云模型参数
    print("\n计算云模型参数...")
    cloud_models = {}

    for level_name, samples in level_samples.items():
        if len(samples) > 10:  # 确保有足够样本
            Ex, En, He = backward_cloud_generator(samples)
            cloud_models[level_name] = {
                'Ex': Ex, 'En': En, 'He': He, 'sample_size': len(samples)
            }
            print(f"{level_name}: Ex={Ex:.3f}, En={En:.3f}, He={He:.3f}, 样本数={len(samples)}")
        else:
            # 使用默认值
            default_ex = (rating_levels[level_name][0] + rating_levels[level_name][1]) / 2
            cloud_models[level_name] = {
                'Ex': default_ex, 'En': 1.0, 'He': 0.3, 'sample_size': len(samples)
            }
            print(f"{level_name}: 样本不足，使用默认值")

    # 5. 可视化云图
    print("\n生成云图...")
    plt.figure(figsize=(14, 8))
    colors = ['red', 'orange', 'gold', 'lightgreen', 'darkgreen']

    for i, (level_name, params) in enumerate(cloud_models.items()):
        if params['sample_size'] > 10:
            cloud_drops = forward_cloud_generator(params['Ex'], params['En'], params['He'], 1000)
            plt.scatter(cloud_drops[:, 0], cloud_drops[:, 1],
                        alpha=0.6, s=8, label=level_name, color=colors[i])

    plt.xlabel('豆瓣评分 (0-10分)')
    plt.ylabel('确定度')
    plt.title('电影评分等级的云模型可视化 (修复版)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 10)  # 限制x轴范围
    plt.ylim(0, 1.1)  # 限制y轴范围
    plt.show()

    # 6. 电影分析
    print("\n具体电影分析...")
    # 选择有代表性的电影
    sample_indices = []
    for level_name in rating_levels:
        movies_in_level = level_movies[level_name]
        if len(movies_in_level) > 0:
            sample_indices.append(movies_in_level.index[0])

    for idx in sample_indices[:5]:  # 分析前5部
        movie = df_clean.loc[idx]
        print(f"\n电影: {movie['NAME']}, 豆瓣评分: {movie['DOUBAN_SCORE']:.2f}")
        print("属于各等级的确定度:")

        max_membership = 0
        best_level = ""

        for level_name, params in cloud_models.items():
            membership = calculate_membership(movie['DOUBAN_SCORE'], params['Ex'], params['En'], params['He'])
            print(f"  {level_name}: {membership:.3f}")

            if membership > max_membership:
                max_membership = membership
                best_level = level_name

        print(f"主要归属: {best_level} (确定度: {max_membership:.3f})")

    # 7. 争议度分析
    print("\n争议度分析...")

    def calculate_controversy(row):
        """计算争议度"""
        star_counts = [row['RATING_1_COUNT'], row['RATING_2_COUNT'], row['RATING_3_COUNT'],
                       row['RATING_4_COUNT'], row['RATING_5_COUNT']]
        total = sum(star_counts)
        if total == 0:
            return 0

        # 计算加权平均星数
        stars = [1, 2, 3, 4, 5]
        avg_star = sum(s * c for s, c in zip(stars, star_counts)) / total

        # 计算标准差作为争议度
        variance = sum(c * (s - avg_star) ** 2 for s, c in zip(stars, star_counts)) / total
        return np.sqrt(variance)

    df_clean['争议度'] = df_clean.apply(calculate_controversy, axis=1)

    # 显示争议度最高和最低的电影
    print("\n争议度最高的3部电影:")
    controversial = df_clean.nlargest(3, '争议度')
    for _, movie in controversial.iterrows():
        print(f"  {movie['NAME']}: 评分{movie['DOUBAN_SCORE']:.2f}, 争议度{movie['争议度']:.3f}")

    print("\n争议度最低的3部电影:")
    consistent = df_clean.nsmallest(3, '争议度')
    for _, movie in consistent.iterrows():
        print(f"  {movie['NAME']}: 评分{movie['DOUBAN_SCORE']:.2f}, 争议度{movie['争议度']:.3f}")

    # 8. 保存结果
    cloud_params_df = pd.DataFrame(cloud_models).T
    cloud_params_df.to_excel('cloud_model_parameters_fixed.xlsx')
    df_clean.to_excel('movie_analysis_fixed.xlsx', index=False)

    print("\n分析完成！结果已保存。")
    return df_clean, cloud_models


# 运行修复版本
if __name__ == "__main__":
    file_path = "a.xls"
    df_result, cloud_models = main_analysis_fixed(file_path)