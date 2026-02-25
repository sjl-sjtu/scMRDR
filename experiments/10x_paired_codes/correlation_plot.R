library(tidyverse)
library(data.table)

setwd("D:/multi-omics")

counts <- read_csv("celltype_counts.csv")
counts

top_celltypes <- counts %>% arrange(desc(count)) %>% head(8) %>% pull(cell_type)
top_celltypes <- gsub(" ", "_", top_celltypes)

df <- fread("mop_10x_mean_expression_mse.csv")
df <- as_tibble(df)
df

true <- df %>% filter(celltype=="CD4_TCM", method=="true", modality=="RNA") %>%
  dplyr::select(-celltype, -method, -modality) %>%
  as.numeric()
pred <- df %>% filter(celltype=="CD4_TCM", method=="ot", modality=="RNA") %>%
  dplyr::select(-celltype, -method, -modality) %>%
  as.numeric()
cor(true, pred)

df %>% pivot_longer(-c(celltype, method, modality), names_to = "gene", values_to = "value") -> 
  df_long
df_long

df_long %>% pivot_wider(names_from = method, values_from = value) -> 
  df_wide
df_wide

df_wide %>% pivot_longer(-c(celltype, modality, gene, true), names_to = "method", values_to = "predict") -> 
  df_long2
df_long2

df_long2 <- df_long2 %>% filter(celltype %in% top_celltypes)

library(ggplot2)
library(dplyr)
library(tidyr)
library(ggthemes)
library(ggsci)
library(broom) # 用于提取模型统计量
library(Cairo)

# ----------------------------------------------------------------------
# 步骤 1 & 2: 分组计算 r 和 p 值
# ----------------------------------------------------------------------
stats_df <- df_long2 %>%
  
  # 按 pair (分面) 和 celltype (分组) 分组
  group_by(modality, method, celltype) %>%
  
  # 使用 cor.test() 一次性计算相关系数和 P 值，简化代码
  summarise(
    # 核心修正：使用 cor.test() 结果
    cor_test_result = list(cor.test(predict, true)),
    
    # 直接提取 r (PCC)
    r_value = cor_test_result[[1]]$estimate,
    
    # 直接提取 P-value
    p_value = cor_test_result[[1]]$p.value,
    .groups = 'drop'
  ) %>%
  # ----------------------------------------------------------------------
# 步骤 3: 格式化标签字符串 (保持不变)
# ----------------------------------------------------------------------
  mutate(
    p_value_fmt = case_when(
      p_value < 0.001 ~ "p < 0.001",
      TRUE ~ paste0("p = ", format.pval(p_value, digits = 2, eps = 0.001))
    ),
    r_value_fmt = sprintf("r = %.2f", r_value),
    
    # 核心：创建最终的标签格式: 细胞类型名称, r=, p=
    label = paste0(celltype, ", ", r_value_fmt, ", ", p_value_fmt)
  )

# ----------------------------------------------------------------------
# 步骤 4: 绘制图形并添加标签
# ----------------------------------------------------------------------

# 核心修正 1：获取全局 X, Y 轴范围用于定位 (计算 y_min)
range_df <- df_long2 %>%
  group_by(modality, method) %>%
  summarise(
    # 标签的 X 位置 (接近最小值)
    x_min = min(true), # 修正: 变量名改为 x_min
    x_max = max(true), # 修正: 添加 x_max
    # 标签的 Y 位置的底部 (最小值)
    y_min = min(predict), # 修正: 添加 y_min
    # 标签的 Y 位置的顶部 (最大值)
    y_max = max(predict)
  ) %>%
  ungroup()

# 合并位置信息
stats_df_labeled <- stats_df %>%
  group_by(modality, method) %>%
  mutate(
    # 计算组内行号，用于堆叠标签
    label_y_offset = row_number() 
  ) %>%
  ungroup() %>%
  left_join(range_df, by = c("modality","method")) %>%
  
  mutate(
    y_range = y_max - y_min,
    
    # 核心修正：Y 轴位置从 Y_min 开始向上偏移
    # label_y = Y_min + (Y轴范围 * 堆叠因子 * 偏移量)
    label_y = y_max - y_range * 0.04 * label_y_offset
  )



df_long2 %>% 
  ggplot(aes(x=true, y=predict, color=celltype)) +
  # facet_grid(modality ~ method, 
  #            labeller = labeller(
  #   modality = c("ATAC" = "scATAC", "RNA" = "scRNA"),
  #   method = c("ot" = "OT", "knn" = "kNN"))
  #   ) +
  # 改为 facet_grid 横向排列
  facet_grid(~modality + method, 
             scales="free",
             labeller = labeller(
               modality = c("ATAC" = "scATAC", "RNA" = "scRNA"),
               method = c("ot" = "OT", "knn" = "kNN"))
  ) +
  geom_point(alpha=0.6,size=0.3) +
  
  # 拟合回归线 (按 celltype 分组)
  geom_smooth(method="lm", se=FALSE) + 
  
  # 添加统计量标签 (统一在左上角)
  geom_text(
    data = stats_df_labeled,
    # 核心修正 1: X 坐标使用 x_max
    aes(x = x_min, y = label_y, label = label, color = celltype), 
    inherit.aes = FALSE,
    size = 3, 
    hjust = 0, 
    vjust = 1, 
    lineheight = 1,
    # 核心修正 4: X 轴左推偏移 (使标签不超出右边界)
    nudge_x = -stats_df_labeled$y_range * 0.01, 
    
    show.legend = FALSE # 确保不产生额外的 'a' 图例
  ) +
  labs(x="True", y="Predict") +
  theme_minimal() +
  scale_color_bmj(alpha = 1) +
  theme(legend.position="bottom")

ggsave("corr.pdf", width=15, height=5, device = cairo_pdf)
ggsave("corr.png", width=15, height=5,dpi=300)

##########################################################
########################
# ——————————————————————————————————————————————————
# 方案 2：分离标签图层，使用 patchwork 组合
# ————————————————————————————————————————————————
####################3
# ----------------------------------------------------------------------
# 步骤 1 & 2: 分组计算 r 和 p 值 (保持不变)
# ----------------------------------------------------------------------
stats_df <- df_long2 %>%
  group_by(modality, method, celltype) %>%
  summarise(
    cor_test_result = list(cor.test(predict, true)),
    r_value = cor_test_result[[1]]$estimate,
    p_value = cor_test_result[[1]]$p.value,
    .groups = 'drop'
  ) %>%
  mutate(
    p_value_fmt = case_when(
      p_value < 0.001 ~ "p < 0.001",
      TRUE ~ paste0("p = ", format.pval(p_value, digits = 2, eps = 0.001))
    ),
    r_value_fmt = sprintf("r = %.2f", r_value),
    # 核心：创建最终的标签格式: r=, p=
    label = paste0(sprintf("%-20s", celltype), "  |  ", 
                   r_value_fmt, ", ", p_value_fmt)
  )

# ----------------------------------------------------------------------
# 步骤 3: 颜色和布局设置
# ----------------------------------------------------------------------
library(paletteer)
# 1. 颜色方案：获取 19 种离散颜色
N_CELLTYPES <- length(unique(df_long2$celltype))
# 使用 paletteer 从一个包含足够颜色的调色板中获取 19 种颜色
# 'ggthemes::Classic_20' 提供 20 种颜色
color_palette <- paletteer_d("ggthemes::Classic_20")[1:N_CELLTYPES]


# ----------------------------------------------------------------------
# 步骤 4: 绘制图形 (修改：移除 geom_text，使用 scale_color_manual)
# ----------------------------------------------------------------------

# 绘制主图 P_main
p_main <- df_long2 %>% 
  ggplot(aes(x=true, y=predict, color=celltype)) +
  facet_grid(~modality + method, 
             scales="free",
             labeller = labeller(
               modality = c("ATAC" = "scATAC", "RNA" = "scRNA"),
               method = c("ot" = "OT", "knn" = "kNN"))
  ) +
  geom_point(alpha=0.6) +
  geom_smooth(method="lm", se=FALSE) + 
  labs(x="True", y="Predict", color = "Cell Type") +
  theme_minimal() +
  
  # 核心修正 1：使用手动颜色标度 (Manual Scale)
  scale_color_manual(values = color_palette) + 
  
  # 将图例放在底部，待 patchwork 统一调整
  theme(legend.position="bottom",
        legend.box = "horizontal",
        legend.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))

# ----------------------------------------------------------------------
# 步骤 5: 创建标签图 (作为单独的图层)
# ----------------------------------------------------------------------

# 聚合标签，按分面分组 (modality + method)
label_df_plot <- stats_df %>%
  group_by(modality, method) %>%
  # 将同一分面的所有标签连接成一个字符串，并换行
  summarise(
    label_text = paste(label, collapse = "\n"),
    .groups = 'drop'
  ) %>%
  
  # 添加 labeller 信息用于分面
  mutate(
    modality = factor(modality, levels = c("ATAC", "RNA"), labels = c("scATAC", "scRNA")),
    method = factor(method, levels = c("ot", "knn"), labels = c("OT", "kNN")),
    facet_label = paste(modality, method, sep = " | ")
  )

# 绘制文本图 (p_notes)
p_notes <- ggplot(label_df_plot) +
  # 使用 geom_text 在图层上居中显示标签文本
  geom_text(aes(x=0.5, y=0.5, label=label_text),
            hjust = 0, vjust = 1, size = 2.5, lineheight = 1.2, family = "mono") +
  
  # 分面，以便标签与主图对齐
  facet_grid(~facet_label) +
  
  # 移除所有美学元素，只保留文本和分面条
  theme_void() + 
  theme(
    strip.text = element_blank(), # 移除顶部分面标签条（因为它在主图上已经有了）
    # 如果想保留标签条，可以设置：strip.text = element_text(face = "bold", size = 8)
    plot.margin = margin(t = 0, r = 5.5, b = 5.5, l = 5.5, unit = "pt")
  )

# ----------------------------------------------------------------------
# 步骤 6: 组合图形 (patchwork)
# ----------------------------------------------------------------------
library(patchwork)
# 1. 垂直堆叠：将主图放在上方，将标签图放在下方
combined_stacked <- p_main / p_notes + 
  # 设置标签图的相对高度很小 (例如 0.15)
  plot_layout(heights = c(1, 0.15)) 

# 2. 最终布局：使用 plot_layout(guides = 'collect') 将图例统一收集到右侧
# 移除主图的图例（已移至右侧），然后用 patchwork 收集
p_main_no_legend <- p_main + theme(legend.position = "none")

# 重新组合，并收集图例
final_plot <- p_main_no_legend / p_notes + 
  plot_layout(heights = c(1, 0.15), guides = "collect") & 
  theme(legend.position = "right") # 将收集的图例放在最右侧

final_plot
# GGSAVE
ggsave("corr_final.pdf", final_plot, width=15, height=8, device = cairo_pdf)
