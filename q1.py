import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from math import sqrt

# 读取数据
weak = pd.read_csv('weak.csv')  # 列：x, y, traffic
current = pd.read_csv('current.csv')  # 列：x, y

# 总业务量
total_traffic = weak['traffic'].sum()
target_traffic = total_traffic * 0.9

# 过滤弱覆盖点，保留 traffic >= 50 的点
weak = weak[weak['traffic'] >= 50].reset_index(drop=True)

# 初始化覆盖状态
covered = np.zeros(len(weak), dtype=bool)

# 构建 KDTree 用于邻域查询
points = weak[['x', 'y']].values
tree = KDTree(points)

# 现有基站列表
existing_sites = current[['x', 'y']].values.tolist()

# 已选基站列表（格式：(x, y, type)）
selected_sites = []

# 距离约束函数
def is_valid_site(x: int, y: int, existing_sites: list, min_distance: float = 10) -> bool:
    for sx, sy in existing_sites:
        if sqrt((x - sx)**2 + (y - sy)**2) <= min_distance:
            return False
    return True

# 主循环
current_covered_traffic = 0.0
while current_covered_traffic < target_traffic:
    best_score = 0.0
    best_site = None
    best_type = None

    # 构建当前已选基站的坐标列表
    selected_coords = [(x, y) for x, y, _ in selected_sites]
    all_sites = existing_sites + selected_coords

    # 遍历所有未被覆盖的弱覆盖点
    for i in range(len(weak)):
        # print(f"检查点 {i + 1}/{len(weak)}: ({weak.iloc[i]['x']}, {weak.iloc[i]['y']})")
        if covered[i]:
            continue
        x, y = points[i]

        # 检查宏基站覆盖
        macro_radius = 30
        macro_indices = tree.query_ball_point((x, y), macro_radius)
        macro_covered = [j for j in macro_indices if not covered[j]]
        macro_traffic = sum(weak.iloc[j]['traffic'] for j in macro_covered)

        # 检查微基站覆盖
        micro_radius = 10
        micro_indices = tree.query_ball_point((x, y), micro_radius)
        micro_covered = [j for j in micro_indices if not covered[j]]
        micro_traffic = sum(weak.iloc[j]['traffic'] for j in micro_covered)

        # 选择覆盖更高的基站类型
        if macro_traffic > best_score and is_valid_site(x, y, all_sites):
            best_score = macro_traffic
            best_site = (x, y)
            best_type = 'macro'
        if micro_traffic > best_score and is_valid_site(x, y, all_sites):
            best_score = micro_traffic
            best_site = (x, y)
            best_type = 'micro'

    # 如果没有找到可用基站，终止
    if best_site is None:
        print("无法找到满足条件的基站")
        break

    # 更新覆盖状态
    x0, y0 = best_site
    radius = 30 if best_type == 'macro' else 10
    indices = tree.query_ball_point((x0, y0), radius)
    for i in indices:
        if not covered[i]:
            current_covered_traffic += weak.iloc[i]['traffic']
        covered[i] = True

    # 添加到已选基站列表
    selected_sites.append((x0, y0, best_type))

    print(f"已覆盖业务量: {current_covered_traffic:.2f} / {target_traffic:.2f}")

# 输出结果
print("\n选择的基站：")
for x, y, t in selected_sites:
    print(f"站点: ({x}, {y}), 类型: {t}")