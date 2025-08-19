from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse # 保留，因为您原始代码中有
import numpy as np
import pandas as pd
import pickle

# --- 文件路径和常量定义 ---
# CSV 文件路径 (包含 'from', 'to', 'cost' 列)
# 请确保 distance_df 是从这个文件加载的，并且这个文件定义了节点间的距离
distance_file_path = 'data/PEMS07L/distance.csv' # 假设这是您包含连接信息的 distance.csv
output_pkl_filename = 'data/sensor_graph/adj_mx_07L.pkl'

# 根据您的说明，我们知道总共有1026个节点，ID从0到1025
TOTAL_NODES = 1026
EXPECTED_NODE_IDS = list(range(TOTAL_NODES))


# --- 函数定义 ---
def create_adjacency_matrix(distance_df, authoritative_node_ids, normalized_k=0.1):
    """
    根据距离数据创建邻接矩阵。
    使用 authoritative_node_ids (例如 list(range(1026))) 来确定节点总数和映射。
    distance_df 用于填充节点间的成本。
    """
    
    # 1. 基于权威节点列表确定节点和映射
    # 确保权威列表中的ID是唯一的，并排序以保证映射的一致性
    # 由于我们知道ID是0-1025，可以直接使用
    if not authoritative_node_ids:
        print("错误：权威节点列表为空！")
        return None, None
        
    nodes = sorted(list(set(authoritative_node_ids))) # 对于 list(range(N))，这一步后 nodes 仍是 [0, 1, ..., N-1]
    num_nodes = len(nodes)

    if num_nodes != TOTAL_NODES: # 验证一下传入的权威列表是否符合预期
        print(f"警告：权威节点列表包含 {num_nodes} 个唯一节点，与预期的 {TOTAL_NODES} 不符。将按实际列表处理。")
    
    print(f"将基于 {num_nodes} 个权威节点创建邻接矩阵。")
    # 节点ID直接映射到索引 (因为我们期望ID是0到num_nodes-1)
    node_to_index = {node_id: int(node_id) for node_id in nodes} 
    # 如果权威ID不是0-N-1的整数，上面的映射需要改为:
    # node_to_index = {node_id: idx for idx, node_id in enumerate(nodes)}
    # 但根据您的描述“序号是0-1025”，直接映射是合适的。

    # 2. 初始化邻接矩阵
    adj_matrix = np.inf * np.ones((num_nodes, num_nodes), dtype=float)
    np.fill_diagonal(adj_matrix, 0)  # 自环的成本设置为 0

    # 3. 根据 distance_df 填充邻接矩阵
    edges_processed_count = 0
    edges_skipped_due_to_invalid_node_id = 0
    
    for _, row in distance_df.iterrows():
        from_node_original = row['from']
        to_node_original = row['to']
        cost = row['cost']

        # 尝试将CSV中的ID转换为整数，以匹配权威列表的整数ID
        try:
            from_node = int(from_node_original)
            to_node = int(to_node_original)
        except ValueError:
            # print(f"警告：distance.csv 中的边 ({from_node_original}, {to_node_original}) 包含非整数节点ID，已跳过。")
            edges_skipped_due_to_invalid_node_id += 1
            continue
            
        # 检查节点ID是否在我们的权威节点索引中 (即0到TOTAL_NODES-1)
        if from_node in node_to_index and to_node in node_to_index:
            from_idx = node_to_index[from_node]
            to_idx = node_to_index[to_node]
            adj_matrix[from_idx, to_idx] = cost
            edges_processed_count += 1
        else:
            # print(f"警告：distance.csv 中的边 ({from_node}, {to_node}) 的节点ID超出了预期的0-{TOTAL_NODES-1}范围，已跳过。")
            edges_skipped_due_to_invalid_node_id += 1
            
    if edges_skipped_due_to_invalid_node_id > 0:
        print(f"警告：因节点ID无效或超出预期范围，从 distance.csv 中跳过了 {edges_skipped_due_to_invalid_node_id} 条边。")
    print(f"已处理 {edges_processed_count} 条有效边到初始成本矩阵。")

    # 4. 归一化处理 (与您之前的逻辑保持一致)
    # 提取实际存在的距离值（非无穷大），不包括对角线的0
    off_diagonal_finite_distances = adj_matrix[
        (adj_matrix != np.inf) & (np.arange(num_nodes)[:, None] != np.arange(num_nodes))
    ]

    if off_diagonal_finite_distances.size == 0:
        std = 1.0 
        print("警告: 未找到有效的非对角线距离来计算标准差。归一化时将使用 std=1.0。")
    else:
        std = off_diagonal_finite_distances.std()
        if std == 0:
            std = 1.0
            print("警告: 距离的标准差为零。归一化时将使用 std=1.0。")

    adj_mx = np.exp(-np.square(adj_matrix / std))
    adj_mx[adj_matrix == np.inf] = 0  # 无连接或原始距离为inf的设为0
    adj_mx[adj_mx < normalized_k] = 0 # 应用稀疏阈值

    # node_to_index 的键现在就是 0 到 TOTAL_NODES-1 的ID
    # pickle 保存时，传感器ID列表可以直接用 list(node_to_index.keys()) 或 EXPECTED_NODE_IDS
    return adj_mx, node_to_index


# --- 主脚本执行部分 ---
try:
    distance_df_main = pd.read_csv(distance_file_path)
    print(f"\n从 '{distance_file_path}' 成功加载 distance_df。")
    print("distance_df 前5行:")
    print(distance_df_main.head())
except FileNotFoundError:
    print(f"错误：未能找到距离文件 '{distance_file_path}'。请检查路径。")
    exit()
except Exception as e:
    print(f"读取距离文件时发生错误: {e}")
    exit()

# 使用定义的权威节点列表 (0 到 1025)
authoritative_node_list_for_adj = EXPECTED_NODE_IDS

print(f"\n准备创建邻接矩阵，期望节点数: {TOTAL_NODES} (ID范围 0-{TOTAL_NODES-1})")
adj_matrix_final, node_to_index_final = create_adjacency_matrix(
    distance_df_main,
    authoritative_node_ids=authoritative_node_list_for_adj, # 传递权威节点列表
    normalized_k=0.1 # 您可以按需调整 normalized_k
)

if adj_matrix_final is not None and node_to_index_final is not None:
    print(f"\n最终生成的邻接矩阵形状: {adj_matrix_final.shape}")
    # print(f"最终的 node_to_index 映射 (示例前5项): {list(node_to_index_final.items())[:5]}")
    # print(f"最终的 node_to_index 映射中的节点数量: {len(node_to_index_final)}")

    # 保存结果到 PKL 文件
    # 确保PKL文件名和路径正确
    # 权威的传感器ID列表现在就是 authoritative_node_list_for_adj (即0-1025)
    # node_to_index_final 将是 {0:0, 1:1, ..., 1025:1025}
    
    # 为了与您原始pickle文件的结构 ['id_list_or_label', node_to_index_map, adj_matrix] 保持一致:
    # 1. 'id_list_or_label': 可以是实际的ID列表 authoritative_node_list_for_adj
    # 2. 'node_to_index_map': node_to_index_final
    # 3. 'adj_matrix': adj_matrix_final
    
    # 如果 'id' 只是一个标签字符串，而 node_to_index 才是关键映射：
    # data_to_pickle = ['PEMS07L_node_ids', node_to_index_final, adj_matrix_final]
    # 或者，如果您希望第一个元素是实际的节点ID列表（这通常更有用）：
    data_to_pickle = [authoritative_node_list_for_adj, node_to_index_final, adj_matrix_final]

    try:
        with open(output_pkl_filename, 'wb') as f:
            pickle.dump(data_to_pickle, f, protocol=2) # protocol=2 是为了Python 2的兼容性，如果不需要可以去掉
            print(f"邻接矩阵和节点映射已成功保存到 {output_pkl_filename}")
    except Exception as e:
        print(f"保存PKL文件时出错: {e}")
else:
    print("未能创建邻接矩阵。")