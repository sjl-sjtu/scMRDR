import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.stats import spearmanr,pearsonr
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['figure.dpi'] = 300

adata_methy = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/methy_mCG_imputed.h5ad")
adata_rna = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/rna_imputed.h5ad")
rna = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/rna.h5ad")

# protein_genes = rna[:,adata_rna.var_names].var[rna[:,adata_rna.var_names].var['gene_type'] == "protein_coding"].index
# adata_rna = adata_rna[:,protein_genes].copy()

# exp_cells = ['Layer2/3','Layer4/5','Layer6']
# adata_rna_subset = adata_rna[adata_rna.obs['harmonic_celltype'].isin(exp_cells),:].copy()
# sc.pp.highly_variable_genes(adata_rna_subset,n_top_genes=3000)
# featlist = adata_rna_subset.var.index[adata_rna_subset.var['highly_variable']==True]
# adata_rna = adata_rna[:,featlist].copy()
# adata_methy = adata_methy[:, featlist].copy()

sc.pp.highly_variable_genes(adata_rna,n_top_genes=3000)
featlist = adata_rna.var.index[adata_rna.var['highly_variable']==True]
adata_rna = adata_rna[:,featlist].copy()
adata_methy = adata_methy[:, featlist].copy()


n_before = adata_rna.n_obs
mask = ~np.isnan(adata_rna.obsm["X_spatial_imputed"]).any(axis=1)
n_after = mask.sum()
print(f"去掉 {n_before - n_after} 个 NA spot, 保留 {n_after} 个")
adata_rna = adata_rna[mask].copy()

n_before = adata_methy.n_obs
mask = ~np.isnan(adata_methy.obsm["X_spatial_imputed"]).any(axis=1)
n_after = mask.sum()
print(f"去掉 {n_before - n_after} 个 NA spot, 保留 {n_after} 个")
adata_methy = adata_methy[mask].copy()


print(adata_rna.obs['harmonic_celltype'].unique())
print(adata_methy.obs['harmonic_celltype'].unique())

# expr_matrix = adata_rna.layers['normalized'].toarray()  # genes × RNA spots
# # expr_matrix = adata_rna.layers['counts'].toarray()
# meth_matrix = adata_methy.layers['normalized'].toarray()  # genes × meth spots

expr_matrix = adata_rna.layers['norm'].toarray()  # genes × RNA spots
# expr_matrix = adata_rna.layers['counts'].toarray()
meth_matrix = adata_methy.layers['norm'].toarray()  # genes × meth spots
rna_positions = adata_rna.obsm['X_spatial_imputed']  # RNA spots × (x,y)
meth_positions = adata_methy.obsm['X_spatial_imputed']  # Meth spots × (x,y)
rna_ids = adata_rna.obs_names
meth_ids = adata_methy.obs_names

rna_celltypes = adata_rna.obs.harmonic_celltype  # RNA spots × celltypes
meth_celltypes = adata_methy.obs.harmonic_celltype  # Meth spots × celltypes
gene_names = adata_rna.var_names

def aggregate_spatial_spots(data_matrix, positions, spot_ids, annotations, precision=2):
    """
    根据空间坐标和注释类型聚合数据矩阵。

    Args:
        data_matrix (np.ndarray): Spotxgene 的数据矩阵。
        positions (np.ndarray): Spotx(x, y) 的坐标矩阵。
        spot_ids (pd.Index): Spot 的原始 ID。
        annotations (pd.Series or list): Spot 的注释类型，长度与 spot_ids 相同。
        precision (int): 坐标四舍五入的精度。

    Returns:
        tuple: (聚合后的数据矩阵 (Spot × Gene), 聚合后的坐标矩阵, 聚合后的 ID, 聚合后的注释类型)
    """
    import pandas as pd
    import numpy as np

    # 转换为 DataFrame 方便操作
    df = pd.DataFrame(data_matrix, index=spot_ids)  # Spot × Gene
    pos_df = pd.DataFrame(positions, index=spot_ids, columns=['x', 'y'])
    anno_df = pd.Series(annotations, index=spot_ids, name="annotation")

    # 创建分组 ID（位置 + 类型）
    pos_df['x_rounded'] = pos_df['x'].round(precision)
    pos_df['y_rounded'] = pos_df['y'].round(precision)
    group_id = (
        pos_df['x_rounded'].astype(str) + "_" +
        pos_df['y_rounded'].astype(str) + "_" +
        anno_df.astype(str)
    )

    # 分组聚合 (均值)
    data_pooled = df.groupby(group_id).mean()
    pos_pooled = pos_df.groupby(group_id)[['x', 'y']].mean()
    anno_pooled = anno_df.groupby(group_id).first()  # 类型一样的，不需要再合并

    # 转回 numpy
    data_pooled_matrix = data_pooled.values  # Spot × Gene
    pos_pooled_matrix = pos_pooled.values
    new_spot_ids = data_pooled.index
    new_annotations = anno_pooled.values

    return data_pooled_matrix, pos_pooled_matrix, new_spot_ids, new_annotations

# 设定聚合精度
SPATIAL_PRECISION = 2 

# -----------------------------------------------------------------
# 步骤 0.1: 对 RNA 数据进行聚合
# -----------------------------------------------------------------
print("--- 步骤 0.1: RNA 数据聚合 ---")
expr_matrix_pooled, rna_positions_pooled, rna_ids_pooled, rna_annotations = aggregate_spatial_spots(
    expr_matrix, rna_positions, rna_ids, rna_celltypes, SPATIAL_PRECISION
)

print(f"RNA 原始 spots: {expr_matrix.shape[0]} -> 聚合后 spots: {rna_positions_pooled.shape[0]}")

# -----------------------------------------------------------------
# 步骤 0.2: 对 Methylation 数据进行聚合
# -----------------------------------------------------------------
print("--- 步骤 0.2: Methylation 数据聚合 ---")
meth_matrix_pooled, meth_positions_pooled, meth_ids_pooled, meth_annotations = aggregate_spatial_spots(
    meth_matrix, meth_positions, meth_ids, meth_celltypes, SPATIAL_PRECISION
)
print(f"Meth 原始 spots: {meth_matrix.shape[0]} -> 聚合后 spots: {meth_positions_pooled.shape[0]}")



from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

print("\n--- 步骤 1: Spot 对齐 (Gated Linear Assignment) ---")

# ------------------------------------------------------------------
# 关键超参数：设置一个你认为合理的“最大匹配距离”
# 任何超过这个距离的匹配都会被禁止
# 这个值需要你根据数据的物理尺度来设定（例如，20个像素？50微米？）
MAX_DISTANCE_THRESHOLD = 1000.0  
# ------------------------------------------------------------------

# 设置一个非常大的惩罚值，代表“禁止匹配”
PENALTY_VALUE = 1e9 

matched_spots_list = []

# 遍历所有类型
unique_types = np.unique(rna_annotations)
for cell_type in unique_types:
    rna_mask = rna_annotations == cell_type
    meth_mask = meth_annotations == cell_type

    num_rna = np.sum(rna_mask)
    num_meth = np.sum(meth_mask)

    if num_rna == 0 or num_meth == 0:
        continue

    rna_pos = rna_positions_pooled[rna_mask]
    meth_pos = meth_positions_pooled[meth_mask]

    # 1. 计算 M x N 的完整距离矩阵
    cost_matrix = cdist(meth_pos, rna_pos)

    # 2. 【新步骤】应用“门控”：将所有 > 阈值的距离设置为高额惩罚
    cost_matrix[cost_matrix > MAX_DISTANCE_THRESHOLD] = PENALTY_VALUE

    # 3. 求解指派问题
    # 算法现在会自动避免选择那些被惩罚的“极端”匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. 提取匹配的距离
    distances = cost_matrix[row_ind, col_ind]

    # 5. 【新步骤】过滤掉那些“别无选择”的匹配
    # 这一步是必要的，因为如果某个 spot A 的所有邻居都在阈值之外，
    # LAP 仍然会“强制”把它匹配给一个 PENALTY_VALUE。我们必须把这些过滤掉。
    valid_matches_mask = distances < PENALTY_VALUE
    
    row_ind = row_ind[valid_matches_mask]
    col_ind = col_ind[valid_matches_mask]
    distances = distances[valid_matches_mask]
    
    if len(row_ind) == 0:
        continue # 这个类型里没有任何在阈值内的匹配

    # 6. 保存匹配结果
    original_meth_indices = np.where(meth_mask)[0][row_ind]
    original_rna_indices = np.where(rna_mask)[0][col_ind]
    
    matched_df = pd.DataFrame({
        "meth_spot": np.array(meth_ids_pooled)[original_meth_indices],
        "rna_spot": np.array(rna_ids_pooled)[original_rna_indices],
        "meth_index": original_meth_indices,
        "rna_index": original_rna_indices,
        "distance": distances,
        "cell_type": cell_type
    })
    matched_spots_list.append(matched_df)

# 合并所有类型的匹配结果
if len(matched_spots_list) > 0:
    matched_spots = pd.concat(matched_spots_list, ignore_index=True)
    print("匹配完成，过滤后的有效对数: ", matched_spots.shape)
else:
    print("匹配完成，没有找到任何在阈值内的有效匹配。")
    # 你可能需要在这里处理空 DataFrame 的情况
    matched_spots = pd.DataFrame(columns=["meth_spot", "rna_spot", "meth_index", "rna_index", "distance", "cell_type"])

# --- 步骤 2 不变 ---
# (注意：如果 matched_spots 为空，后续步骤会出错，你可能需要加个检查)
if matched_spots.empty:
    print("没有可对齐的矩阵。")
    aligned_meth_matrix = np.empty((0, meth_matrix_pooled.shape[1]))
    aligned_expr_matrix = np.empty((0, expr_matrix_pooled.shape[1]))
else:
    print("\n--- 步骤 2: 提取对齐后的最终矩阵 ---")
    aligned_meth_matrix = meth_matrix_pooled[matched_spots["meth_index"].values, :]
    aligned_expr_matrix = expr_matrix_pooled[matched_spots["rna_index"].values, :]

    print("最终表达矩阵 (aligned_expr_matrix):", aligned_expr_matrix.shape)
    print("最终甲基化矩阵 (aligned_meth_matrix):", aligned_meth_matrix.shape)


# -----------------------------------------------------------------
# 步骤 3: 构建最终的输出文件
# -----------------------------------------------------------------
# expr_df/meth_df: 匹配后的 spot × gene
expr_df = pd.DataFrame(aligned_expr_matrix, columns=gene_names, index=matched_spots['rna_spot'])
meth_df = pd.DataFrame(aligned_meth_matrix, columns=gene_names, index=matched_spots['rna_spot'])

# meta: 匹配后的 spot × [x, y, layer]
# 获取聚合后 RNA spot 的坐标
rna_pos_df = pd.DataFrame(rna_positions_pooled, columns=['x', 'y'], index=rna_ids_pooled)
rna_pos_df['layer'] = rna_annotations.copy()
meta = rna_pos_df.loc[expr_df.index].copy()

# 清理 meta
meta = meta[['x', 'y', 'layer']].copy() 

expr_df.to_csv("./expr_mCG_df_pooled_matched.csv")
meth_df.to_csv("./meth_mCG_df_pooled_matched.csv")
meta.to_csv("./meta_mCG_df_pooled_matched.csv")
print("\n新的 '先聚合，后匹配' 的数据已保存。")


#######################################
adata_methy = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/methy_mCH_imputed.h5ad")
adata_rna = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/rna_imputed.h5ad")
rna = sc.read_h5ad("/home/bingxing2/ailab/group/ai4bio/sunjianle/mouse_brain_simo/rna.h5ad")

# protein_genes = rna[:,adata_rna.var_names].var[rna[:,adata_rna.var_names].var['gene_type'] == "protein_coding"].index
# adata_rna = adata_rna[:,protein_genes].copy()

sc.pp.highly_variable_genes(adata_rna,n_top_genes=3000)
featlist = adata_rna.var.index[adata_rna.var['highly_variable']==True]
adata_rna = adata_rna[:,featlist].copy()
adata_methy = adata_methy[:, featlist].copy()

n_before = adata_rna.n_obs
mask = ~np.isnan(adata_rna.obsm["X_spatial_imputed"]).any(axis=1)
n_after = mask.sum()
print(f"去掉 {n_before - n_after} 个 NA spot, 保留 {n_after} 个")
adata_rna = adata_rna[mask].copy()

n_before = adata_methy.n_obs
mask = ~np.isnan(adata_methy.obsm["X_spatial_imputed"]).any(axis=1)
n_after = mask.sum()
print(f"去掉 {n_before - n_after} 个 NA spot, 保留 {n_after} 个")
adata_methy = adata_methy[mask].copy()


print(adata_rna.obs['harmonic_celltype'].unique())
print(adata_methy.obs['harmonic_celltype'].unique())

# expr_matrix = adata_rna.layers['normalized'].toarray()  # genes × RNA spots
# # expr_matrix = adata_rna.layers['counts'].toarray()
# meth_matrix = adata_methy.layers['normalized'].toarray()  # genes × meth spots

expr_matrix = adata_rna.layers['norm'].toarray()  # genes × RNA spots
meth_matrix = adata_methy.layers['norm'].toarray()  # genes × meth spots
rna_positions = adata_rna.obsm['X_spatial_imputed']  # RNA spots × (x,y)
meth_positions = adata_methy.obsm['X_spatial_imputed']  # Meth spots × (x,y)
rna_ids = adata_rna.obs_names
meth_ids = adata_methy.obs_names

rna_celltypes = adata_rna.obs.harmonic_celltype  # RNA spots × celltypes
meth_celltypes = adata_methy.obs.harmonic_celltype  # Meth spots × celltypes
gene_names = adata_rna.var_names

def aggregate_spatial_spots(data_matrix, positions, spot_ids, annotations, precision=2):
    """
    根据空间坐标和注释类型聚合数据矩阵。

    Args:
        data_matrix (np.ndarray): Spotxgene 的数据矩阵。
        positions (np.ndarray): Spotx(x, y) 的坐标矩阵。
        spot_ids (pd.Index): Spot 的原始 ID。
        annotations (pd.Series or list): Spot 的注释类型，长度与 spot_ids 相同。
        precision (int): 坐标四舍五入的精度。

    Returns:
        tuple: (聚合后的数据矩阵 (Spot × Gene), 聚合后的坐标矩阵, 聚合后的 ID, 聚合后的注释类型)
    """
    import pandas as pd
    import numpy as np

    # 转换为 DataFrame 方便操作
    df = pd.DataFrame(data_matrix, index=spot_ids)  # Spot × Gene
    pos_df = pd.DataFrame(positions, index=spot_ids, columns=['x', 'y'])
    anno_df = pd.Series(annotations, index=spot_ids, name="annotation")

    # 创建分组 ID（位置 + 类型）
    pos_df['x_rounded'] = pos_df['x'].round(precision)
    pos_df['y_rounded'] = pos_df['y'].round(precision)
    group_id = (
        pos_df['x_rounded'].astype(str) + "_" +
        pos_df['y_rounded'].astype(str) + "_" +
        anno_df.astype(str)
    )

    # 分组聚合 (均值)
    data_pooled = df.groupby(group_id).mean()
    pos_pooled = pos_df.groupby(group_id)[['x', 'y']].mean()
    anno_pooled = anno_df.groupby(group_id).first()  # 类型一样的，不需要再合并

    # 转回 numpy
    data_pooled_matrix = data_pooled.values  # Spot × Gene
    pos_pooled_matrix = pos_pooled.values
    new_spot_ids = data_pooled.index
    new_annotations = anno_pooled.values

    return data_pooled_matrix, pos_pooled_matrix, new_spot_ids, new_annotations

# 设定聚合精度
SPATIAL_PRECISION = 2 

# -----------------------------------------------------------------
# 步骤 0.1: 对 RNA 数据进行聚合
# -----------------------------------------------------------------
print("--- 步骤 0.1: RNA 数据聚合 ---")
expr_matrix_pooled, rna_positions_pooled, rna_ids_pooled, rna_annotations = aggregate_spatial_spots(
    expr_matrix, rna_positions, rna_ids, rna_celltypes, SPATIAL_PRECISION
)

print(f"RNA 原始 spots: {expr_matrix.shape[0]} -> 聚合后 spots: {rna_positions_pooled.shape[0]}")

# -----------------------------------------------------------------
# 步骤 0.2: 对 Methylation 数据进行聚合
# -----------------------------------------------------------------
print("--- 步骤 0.2: Methylation 数据聚合 ---")
meth_matrix_pooled, meth_positions_pooled, meth_ids_pooled, meth_annotations = aggregate_spatial_spots(
    meth_matrix, meth_positions, meth_ids, meth_celltypes, SPATIAL_PRECISION
)
print(f"Meth 原始 spots: {meth_matrix.shape[0]} -> 聚合后 spots: {meth_positions_pooled.shape[0]}")


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

print("\n--- 步骤 1: Spot 对齐 (Gated Linear Assignment) ---")


MAX_DISTANCE_THRESHOLD = 1000.0  
PENALTY_VALUE = 1e9 

matched_spots_list = []

# 遍历所有类型
unique_types = np.unique(rna_annotations)
for cell_type in unique_types:
    rna_mask = rna_annotations == cell_type
    meth_mask = meth_annotations == cell_type

    num_rna = np.sum(rna_mask)
    num_meth = np.sum(meth_mask)

    if num_rna == 0 or num_meth == 0:
        continue

    rna_pos = rna_positions_pooled[rna_mask]
    meth_pos = meth_positions_pooled[meth_mask]

    # 1. 计算 M x N 的完整距离矩阵
    cost_matrix = cdist(meth_pos, rna_pos)

    # 2. 【新步骤】应用“门控”：将所有 > 阈值的距离设置为高额惩罚
    cost_matrix[cost_matrix > MAX_DISTANCE_THRESHOLD] = PENALTY_VALUE

    # 3. 求解指派问题
    # 算法现在会自动避免选择那些被惩罚的“极端”匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. 提取匹配的距离
    distances = cost_matrix[row_ind, col_ind]

    # 5. 【新步骤】过滤掉那些“别无选择”的匹配
    # 这一步是必要的，因为如果某个 spot A 的所有邻居都在阈值之外，
    # LAP 仍然会“强制”把它匹配给一个 PENALTY_VALUE。我们必须把这些过滤掉。
    valid_matches_mask = distances < PENALTY_VALUE
    
    row_ind = row_ind[valid_matches_mask]
    col_ind = col_ind[valid_matches_mask]
    distances = distances[valid_matches_mask]
    
    if len(row_ind) == 0:
        continue # 这个类型里没有任何在阈值内的匹配

    # 6. 保存匹配结果
    original_meth_indices = np.where(meth_mask)[0][row_ind]
    original_rna_indices = np.where(rna_mask)[0][col_ind]
    
    matched_df = pd.DataFrame({
        "meth_spot": np.array(meth_ids_pooled)[original_meth_indices],
        "rna_spot": np.array(rna_ids_pooled)[original_rna_indices],
        "meth_index": original_meth_indices,
        "rna_index": original_rna_indices,
        "distance": distances,
        "cell_type": cell_type
    })
    matched_spots_list.append(matched_df)

# 合并所有类型的匹配结果
if len(matched_spots_list) > 0:
    matched_spots = pd.concat(matched_spots_list, ignore_index=True)
    print("匹配完成，过滤后的有效对数: ", matched_spots.shape)
else:
    print("匹配完成，没有找到任何在阈值内的有效匹配。")
    # 你可能需要在这里处理空 DataFrame 的情况
    matched_spots = pd.DataFrame(columns=["meth_spot", "rna_spot", "meth_index", "rna_index", "distance", "cell_type"])

# --- 步骤 2 不变 ---
# (注意：如果 matched_spots 为空，后续步骤会出错，你可能需要加个检查)
if matched_spots.empty:
    print("没有可对齐的矩阵。")
    aligned_meth_matrix = np.empty((0, meth_matrix_pooled.shape[1]))
    aligned_expr_matrix = np.empty((0, expr_matrix_pooled.shape[1]))
else:
    print("\n--- 步骤 2: 提取对齐后的最终矩阵 ---")
    aligned_meth_matrix = meth_matrix_pooled[matched_spots["meth_index"].values, :]
    aligned_expr_matrix = expr_matrix_pooled[matched_spots["rna_index"].values, :]

    print("最终表达矩阵 (aligned_expr_matrix):", aligned_expr_matrix.shape)
    print("最终甲基化矩阵 (aligned_meth_matrix):", aligned_meth_matrix.shape)


# -----------------------------------------------------------------
# 步骤 3: 构建最终的输出文件
# -----------------------------------------------------------------
# expr_df/meth_df: 匹配后的 spot × gene
expr_df = pd.DataFrame(aligned_expr_matrix, columns=gene_names, index=matched_spots['rna_spot'])
meth_df = pd.DataFrame(aligned_meth_matrix, columns=gene_names, index=matched_spots['rna_spot'])

# meta: 匹配后的 spot × [x, y, layer]
# 获取聚合后 RNA spot 的坐标
rna_pos_df = pd.DataFrame(rna_positions_pooled, columns=['x', 'y'], index=rna_ids_pooled)
rna_pos_df['layer'] = rna_annotations.copy()
meta = rna_pos_df.loc[expr_df.index].copy()

# 清理 meta
meta = meta[['x', 'y', 'layer']].copy() 

expr_df.to_csv("./expr_mCH_df_pooled_matched.csv")
meth_df.to_csv("./meth_mCH_df_pooled_matched.csv")
meta.to_csv("./meta_mCH_df_pooled_matched.csv")
print("\n新的 '先聚合，后匹配' 的数据已保存。")
