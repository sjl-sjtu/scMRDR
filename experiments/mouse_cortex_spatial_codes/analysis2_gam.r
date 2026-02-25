
library(gamm4)
library(tidyverse)
library(ape)
library(mgcv)


analyze_gene_gamm <- function(gene, expr_df, meth_df, meta_df) { # Added k as an argument
  # 1. Construct analysis data
  df <- data.frame(
    expr = expr_df[[gene]],
    meth = meth_df[[gene]],
    x = meta_df$x,
    y = meta_df$y,
    layer = factor(meta_df$layer)
  ) %>% na.omit()

  # # Optional: A check for sufficient data points
  # if (nrow(df) < (k + 5)) { # Ensure enough data for the model
  #    message(paste("Gene", gene, "skipped: insufficient data points."))
  #    return(NULL)
  # }

  tryCatch({
    # 2. Fit the GAMM model
    # I've made the 'k' parameter an argument to the function for flexibility
    model <- gamm4(expr ~ meth + s(x, y, bs = "gp"),
                   random = ~(1 + meth | layer),
                   data = df)
     
    conv_code <- model$mer@optinfo$conv$opt

    # 3. Extract key metrics from the model summaries
    gam_summary <- summary(model$gam)
    mer_summary <- summary(model$mer)

    # --- Fixed effect for 'meth' ---
    p_table <- gam_summary$p.table
    coef_meth <- p_table["meth", "Estimate"]
    se_meth <- p_table["meth", "Std. Error"] # ADDED: Standard Error
    pval_meth <- p_table["meth", "Pr(>|t|)"]

    # --- Overall model fit ---
    adj_r_sq <- gam_summary$r.sq # ADDED: Adjusted R-squared

    # --- Spatial smooth term diagnostics ---
    s_table <- gam_summary$s.table
    edf_spatial <- s_table["s(x,y)", "edf"] # ADDED: Effective Degrees of Freedom
    pval_spatial <- s_table["s(x,y)", "p-value"] # ADDED: P-value for the smooth
    # # ADDED: A crucial diagnostic ratio. If this is close to 1, you need to increase k.
    # k_edf_ratio <- edf_spatial / (k - 1) 
    
    # --- Random effect variances ---
    ranef_vc <- as.data.frame(VarCorr(model$mer))
    # ADDED: Variance of the random intercept and slope
    # var_intercept_layer <- ranef_vc$vcov[ranef_vc$grp == "layer" & ranef_vc$var1 == "(Intercept)"]
    var_intercept_layer <- ranef_vc$vcov[
        ranef_vc$grp == "layer" & 
        ranef_vc$var1 == "(Intercept)" & 
        is.na(ranef_vc$var2)
    ]
    var_meth_layer <- ranef_vc$vcov[ranef_vc$grp == "layer" & ranef_vc$var1 == "meth"]

    # --- Layer-specific random effects (as you had before) ---
    ranef_layer <- ranef(model$mer)$layer
    meth_layer <- ranef_layer[, "meth"]
    meth_layer_named <- as.list(meth_layer)
    names(meth_layer_named) <- paste0("meth_layer_", rownames(ranef_layer))

    #
    res <- residuals(model$gam)
    coords <- df[, c("x", "y")]
    dist_inv <- 1 / as.matrix(dist(coords))
    diag(dist_inv) <- 0
    dist_inv[is.infinite(dist_inv)] <- 0
    moran_test_result <- ape::Moran.I(res, dist_inv)
    residual_p_value <- moran_test_result$p.value

    # 4. Compile all results into a single data frame row
    results <- list(
      gene = gene,
      beta_meth = coef_meth,
      se_meth = se_meth, # ADDED
      pval_meth = pval_meth,
      adj_r_sq = adj_r_sq, # ADDED
      edf_spatial = edf_spatial, # ADDED
      pval_spatial = pval_spatial, # ADDED
      # k_edf_ratio = k_edf_ratio, # ADDED
      var_intercept_layer = var_intercept_layer, # ADDED
      var_meth_layer = var_meth_layer, # ADDED
      residual_p_value = residual_p_value,
      conv_code = conv_code
    )
    
    results <- c(results, meth_layer_named) # Append layer-specific effects
    results <- as.data.frame(results)

    # Optional: A print statement to track progress
    print(paste0("Finish ", gene))
    return(results)
  }, error = function(e) {
    # Improved error message
    message(paste("Gene", gene, "failed with error:", e$message))
    return(NULL)
  })
}

# 我们将函数重命名为 analyze_gene_gam，因为它不再是 GAMM
analyze_gene_gam <- function(gene, expr_df, meth_df, meta_df) {
  # 1. 构建分析数据 (与之前相同)
  df <- data.frame(
    expr = expr_df[[gene]],
    meth = meth_df[[gene]],
    x = meta_df$x,
    y = meta_df$y,
    layer = factor(meta_df$layer)
  ) %>% na.omit()

  # 可选：数据检查 (与之前相同)
  # if (nrow(df) < (k + 5)) { ... }

  tryCatch({
    # 2. 拟合 GAM 模型
    # --- 更改 ---
    # 我们不再使用 gamm4，而是使用 mgcv::gam
    # 固定效应公式变为 expr ~ meth * layer
    # 'meth * layer' 展开为 'meth + layer + meth:layer'
    # 这为每个 'layer' 提供了独立的截距和 'meth' 斜率
    model <- gam(expr ~ meth + layer + meth:layer + s(x, y, bs = "gp"),
                 data = df,
                 method = "REML") # 使用 REML/ML 估计更稳定

    # --- 更改 ---
    # gam 对象的收敛检查更简单
    conv_status <- model$converged # 这将是 TRUE 或 FALSE

    # 3. 从模型摘要中提取关键指标
    # --- 更改 ---
    # gam 对象只有一个摘要，而不是 $gam 和 $mer
    model_summary <- summary(model)

    # --- 'meth' 的固定效应 ---
    # p.table 包含所有参数项
    p_table <- model_summary$p.table
    
    # 注意：这里的 "meth" 系数现在是 *参考层* (reference layer) 的斜率
    coef_meth <- p_table["meth", "Estimate"]
    se_meth <- p_table["meth", "Std. Error"]
    pval_meth <- p_table["meth", "Pr(>|t|)"]

    # --- 整体模型拟合 ---
    adj_r_sq <- model_summary$r.sq # 调整后的 R-squared

    # --- 空间平滑项诊断 ---
    s_table <- model_summary$s.table
    edf_spatial <- s_table["s(x,y)", "edf"]
    pval_spatial <- s_table["s(x,y)", "p-value"]

    # --- 更改：移除所有随机效应提取 ---
    # ranef_vc, var_intercept_layer, var_meth_layer 都被移除
    # 因为它们不再是模型的一部分

    # --- 更改：提取固定效应交互项 ---
    # 我们不再提取 'ranef'，而是提取 'meth:layer' 交互项的固定系数
    # 这些系数表示其他 'layer' 的斜率与 *参考层* 斜率之间的 *差异*
    all_coefs <- coefficients(model)
    interaction_coefs <- all_coefs[grepl("meth:layer", names(all_coefs))]
    
    # 转换为与您之前格式类似的列表
    interaction_list <- as.list(interaction_coefs)
    names(interaction_list) <- gsub(":", "_", names(interaction_list)) # 重命名以匹配 data.frame
    
    # --- 更改：残差提取 ---
    res <- residuals(model) # 直接从 model 对象提取
    
    # --- 残差空间自相关 (与之前相同) ---
    coords <- df[, c("x", "y")]
    dist_inv <- 1 / as.matrix(dist(coords))
    diag(dist_inv) <- 0
    dist_inv[is.infinite(dist_inv)] <- 0
    moran_test_result <- ape::Moran.I(res, dist_inv)
    residual_p_value <- moran_test_result$p.value

    # 4. 将所有结果编译为单个 data frame 行
    # --- 更改：更新了结果列表 ---
    results <- list(
      gene = gene,
      beta_meth = coef_meth, # 明确这是参考层的 beta
      se_meth = se_meth,
      pval_meth = pval_meth,
      adj_r_sq = adj_r_sq,
      edf_spatial = edf_spatial,
      pval_spatial = pval_spatial,
      residual_p_value = residual_p_value,
      converged = conv_status # 更改了收敛标志
    )
    
    # 将交互项（斜率差异）附加到列表中
    if (length(interaction_list) > 0) {
      results <- c(results, interaction_list)
    }
    
    results <- as.data.frame(results)

    print(paste0("Finish ", gene))
    return(results)
    
  }, error = function(e) {
    message(paste("Gene", gene, "failed with error:", e$message))
    return(NULL)
  })
}

# 并行运行
# library(parallel)
# num_workers <- detectCores() %/% 2

# genes <- colnames(expr_df)[-1]
# results <- mclapply(genes, function(g) {
#   analyze_gene_gamm(g, expr_df, meth_df, meta_df)
# }, mc.cores = num_workers)

# results <- lapply(genes, function(g) {
#   analyze_gene_gam(g, expr_df, meth_df, meta_df)
# })

# results_df <- do.call(rbind, results) %>% as.data.frame()
# results_df <- results_df[order(as.numeric(results_df$pval_meth)), ]

# head(results_df)
# class(results_df)
# write.csv(results_df, "results/rna_methy_cis_gam4_gp.csv", row.names = FALSE)


# 假设数据框：expr_df, meth_df, meta_df
# genes 是基因名列表
expr_df <- read_csv("expr_mCG_df_pooled_matched.csv")
meth_df <- read_csv("meth_mCG_df_pooled_matched.csv")
meta_df <- read_csv("meta_mCG_df_pooled_matched.csv")

print(table(meta_df$layer))

genes <- colnames(expr_df)[-1]

results <- lapply(genes, function(g) {
  analyze_gene_gam(g, expr_df, meth_df, meta_df)
})

results_df <- do.call(rbind, results) %>% as.data.frame()
results_df <- results_df[order(as.numeric(results_df$pval_meth)), ]

head(results_df)
class(results_df)
write.csv(results_df, "results/rna_methy_mCG_cis_gam4_gp.csv", row.names = FALSE)

expr_df <- read_csv("expr_mCH_df_pooled_matched.csv")
meth_df <- read_csv("meth_mCH_df_pooled_matched.csv")
meta_df <- read_csv("meta_mCH_df_pooled_matched.csv")

print(table(meta_df$layer))

genes <- colnames(expr_df)[-1]

results <- lapply(genes, function(g) {
  analyze_gene_gam(g, expr_df, meth_df, meta_df)
})

results_df <- do.call(rbind, results) %>% as.data.frame()
results_df <- results_df[order(as.numeric(results_df$pval_meth)), ]

head(results_df)
class(results_df)
write.csv(results_df, "results/rna_methy_mCH_cis_gam4_gp.csv", row.names = FALSE)


########
# 假设数据框：expr_df, meth_df, meta_df
# genes 是基因名列表
expr_df <- read_csv("expr_mCG_df_pooled_matched.csv")
meth_df <- read_csv("meth_mCG_df_pooled_matched.csv")
meta_df <- read_csv("meta_mCG_df_pooled_matched.csv")

print(table(meta_df$layer))

genes <- colnames(expr_df)[-1]

# subset layers with > 100 spots
layers <- c('Layer2/3','Layer4/5','Layer6')
meta_df <- meta_df %>% filter(layer %in% layers)
expr_df <- expr_df %>% filter(rna_spot %in% meta_df$rna_spot)
meth_df <- meth_df %>% filter(rna_spot %in% meta_df$rna_spot)

results <- lapply(genes, function(g) {
  analyze_gene_gam(g, expr_df, meth_df, meta_df)
})

results_df <- do.call(rbind, results) %>% as.data.frame()
results_df <- results_df[order(as.numeric(results_df$pval_meth)), ]

head(results_df)
class(results_df)
write.csv(results_df, "results/rna_methy_mCG_cis_gam4_subset_gp.csv", row.names = FALSE)
# write.csv(results_df, "results/rna_methy_mCG_cis_gam4_gp.csv", row.names = FALSE)


# expr_df <- read_csv("expr_mCH_df_pooled_matched.csv")
# meth_df <- read_csv("meth_mCH_df_pooled_matched.csv")
# meta_df <- read_csv("meta_mCH_df_pooled_matched.csv")

# print(table(meta_df$layer))

# genes <- colnames(expr_df)[-1]

# # subset layers with > 100 spots
# layers <- c('Layer2/3','Layer4/5','Layer6')
# meta_df <- meta_df %>% filter(layer %in% layers)
# expr_df <- expr_df %>% filter(rna_spot %in% meta_df$rna_spot)
# meth_df <- meth_df %>% filter(rna_spot %in% meta_df$rna_spot)

# results <- lapply(genes, function(g) {
#   analyze_gene_gam(g, expr_df, meth_df, meta_df)
# })

# results_df <- do.call(rbind, results) %>% as.data.frame()
# results_df <- results_df[order(as.numeric(results_df$pval_meth)), ]

# head(results_df)
# class(results_df)
# write.csv(results_df, "results/rna_methy_mCH_cis_gam4_subset_gp.csv", row.names = FALSE)
# # write.csv(results_df, "results/rna_methy_mCH_cis_gam4_gp.csv", row.names = FALSE)