import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

def weighted_centroid(points, traffics):
    """计算业务量加权的质心"""
    total_traffic = sum(traffics)
    if total_traffic == 0:
        return (0, 0)
    x = sum(p[0] * d for p, d in zip(points, traffics)) / total_traffic
    y = sum(p[1] * d for p, d in zip(points, traffics)) / total_traffic
    return (round(x), round(y))

def main():
    # 读取数据
    weak_df = pd.read_csv('weak.csv')
    current_df = pd.read_csv('current.csv')

    # 提取弱信号点
    weak_points = [(row['x'], row['y'], row['traffic']) for _, row in weak_df.iterrows()]
    existing_stations = [(row['x'], row['y']) for _, row in current_df.iterrows()]

    # 筛选未被现有基站覆盖的弱信号点
    existing_tree = KDTree(existing_stations) if existing_stations else None
    uncovered = []
    for x, y, traffic in weak_points:
        if existing_tree is None:
            uncovered.append((x, y, traffic))
            continue
        dist, _ = existing_tree.query([x, y], k=1)
        if dist > 30:  # 现有基站覆盖半径为30
            uncovered.append((x, y, traffic))

    if not uncovered:
        print("所有弱信号点已被现有基站覆盖")
        return

    # 初始化
    total_traffic = sum(d for _, _, d in uncovered)
    target = 0.9 * total_traffic
    covered = np.zeros(len(uncovered), dtype=bool)
    selected_stations = []

    iteration = 0
    while sum(d for i, d in enumerate([d for _, _, d in uncovered]) if not covered[i]) > 0:
        iteration += 1

        # 获取当前未被覆盖的点
        current_points = [uncovered[i] for i in range(len(uncovered)) if not covered[i]]
        coords = np.array([(x, y) for x, y, _ in current_points])
        traffics = np.array([d for _, _, d in current_points])

        # 聚类
        clustering = DBSCAN(eps=30, min_samples=1).fit(coords)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # 构建候选基站列表
        candidates = []
        for label in set(labels):
            if label == -1:
                # 单独处理噪声点（未被聚类的点）
                for idx in np.where(labels == label)[0]:
                    x, y, d = current_points[idx]
                    candidates.append({'center': (x, y), 'points': [(x, y)], 'traffics': [d]})
                continue
            indices = np.where(labels == label)[0]
            cluster_points = [current_points[i] for i in indices]
            points = [(x, y) for x, y, _ in cluster_points]
            traffics_cluster = [d for _, _, d in cluster_points]
            center = weighted_centroid(points, traffics_cluster)
            candidates.append({'center': center, 'points': points, 'traffics': traffics_cluster})

        # 构建已选基站的KDTree
        selected_coords = [(x, y) for x, y, _ in selected_stations]
        selected_tree = KDTree(selected_coords) if selected_coords else None
        existing_tree = KDTree(existing_stations) if existing_stations else None

        best_gain = 0
        best_candidate = None
        best_type = None

        for candidate in candidates:
            cx, cy = candidate['center']
            points = candidate['points']
            traffics_cluster = candidate['traffics']

            # 检查与现有基站的距离约束
            if existing_tree is not None:
                neighbors = existing_tree.query_ball_point([cx, cy], r=10 + 1e-9)
                if neighbors:
                    continue

            # 检查与已选基站的距离约束
            if selected_tree is not None:
                neighbors = selected_tree.query_ball_point([cx, cy], r=10 + 1e-9)
                if neighbors:
                    continue

            # 评估宏基站和微基站的覆盖效果
            for station_type in ['macro', 'micro']:
                radius = 30 if station_type == 'macro' else 10
                tree = KDTree(coords)
                inside_indices = tree.query_ball_point([cx, cy], r=radius + 1e-9)
                gain = sum(traffics[i] for i in inside_indices)

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = (cx, cy)
                    best_type = station_type

        # 更新覆盖状态
        if best_gain == 0:
            print("无法继续覆盖，无法达到目标业务量")
            break

        selected_stations.append((*best_candidate, best_type))
        cx, cy = best_candidate
        tree = KDTree(coords)
        inside_indices = tree.query_ball_point([cx, cy], r=30 + 1e-9)
        for i in inside_indices:
            global_idx = [j for j, p in enumerate(uncovered) if p[:2] == current_points[i][:2]][0]
            covered[global_idx] = True

        # 输出进度
        current_coverage = sum(uncovered[i][2] for i in range(len(uncovered)) if covered[i])
        print(f"迭代 {iteration}: 当前覆盖业务量 {current_coverage:.2f}/{target:.2f}, 新增 {best_gain:.2f}")

    # 输出结果
    result = pd.DataFrame(selected_stations, columns=['x', 'y', 'type'])
    result.to_csv('selected_stations.csv', index=False)
    print(f"共选择 {len(selected_stations)} 个基站，覆盖业务量 {current_coverage / total_traffic * 100:.2f}%")

if __name__ == '__main__':
    main()