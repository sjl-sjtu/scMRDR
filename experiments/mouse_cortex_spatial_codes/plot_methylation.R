# ============================================================
# 1. 依赖包
# ============================================================
library(readr)
library(dplyr)
library(ggplot2)
library(biomaRt)
library(ggrepel)
library(patchwork)
library(Cairo)

setwd("D:/multi-omics")

# ============================================================
# 2. 读入分析结果
# ============================================================
model_list <- c("gamm", "gam")
sites_list <- c("mCG", "mCH")
for(i in 1:length(model_list)){
  for(j in 1:length(sites_list)){
    model <- model_list[i]
    sites <- sites_list[j]
    
    if(model=="gamm"){
      if(sites=="mCG"){
        results_df <- read_csv("results/rna_methy_mCG_cis_gamm4_gp.csv")
      } else if(sites=="mCH"){
        results_df <- read_csv("results/rna_methy_mCH_cis_gamm4_gp.csv")
      }
      # results_df <- results_df %>% filter(conv_code==0 & residual_p_value>0.05)
    }else if(model=="gam"){
      if(sites=="mCG"){
        results_df <- read_csv("results/rna_methy_mCG_cis_gam4_gp.csv")
      } else if(sites=="mCH"){
        results_df <- read_csv("results/rna_methy_mCH_cis_gam4_gp.csv")
      }
      # results_df <- results_df %>% filter(converged==TRUE & residual_p_value>0.05)
    }
    
    # 确保gene列存在
    stopifnot("gene" %in% colnames(results_df))
    
    # ============================================================
    # 3. 基因注释 (biomaRt)
    # ============================================================
    # mart <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
    mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
    
    # annot <- getBM(
    #   attributes=c("hgnc_symbol", "chromosome_name", "start_position"),
    #   filters="hgnc_symbol",
    #   values=results_df$gene, #|> toupper(),
    #   mart=mart
    # )
    annot <- getBM(
      attributes=c("mgi_symbol", "chromosome_name", "start_position"),
      filters="mgi_symbol",
      values=results_df$gene, #|> toupper(),
      mart=mart
    )
    
    
    # 合并
    results_anno <- results_df %>%
      # mutate(gene_ori=gene) %>%
      # mutate(gene=toupper(gene)) %>%
      # left_join(annot, by=c("gene"="hgnc_symbol"))
      left_join(annot, by=c("gene"="mgi_symbol"))
    results_anno$FDR <- p.adjust(results_anno$pval_meth, method="BH")
    
    # ============================================================
    # 4. 火山图
    # ============================================================
    if(sites=="mCG"){
      beta_thres <- 0.05
    }else if(sites=="mCH"){
      beta_thres <- 0.05
    }
    p_thres <- 0.05
    # results_anno <- results_anno %>%
    #   mutate(
    #     neglog10p = -log10(pval_meth),
    #     sig = ifelse(pval_meth < p_thres & abs(beta_meth) > beta_thres, "Significant", "NS")
    #   )
    # 
    # volcano <- ggplot(results_anno, aes(x=beta_meth, y=neglog10p, color=sig)) +
    #   geom_point(alpha=0.7) +
    #   geom_hline(yintercept=-log10(p_thres), linetype="dashed", color="red") +
    #   geom_vline(xintercept=c(-beta_thres,beta_thres), linetype="dashed", color="blue") +
    #   geom_text_repel(data=results_anno %>% filter(sig=="Significant") %>% head(10),
    #                   aes(label=gene), size=3, max.overlaps=10) +
    #   scale_color_manual(values=c("grey","red")) +
    #   theme_minimal() +
    #   labs(x="Effect size (β for meth)", y="-log10(FDR)") #, title="Volcano plot")
    # volcano
    
    results_anno <- results_anno %>%
      mutate(
        neglog10p = -log10(FDR),
        # sig = ifelse(FDR < p_thres & abs(beta_meth) > beta_thres, "Significant", "NS")
        sig = ifelse(FDR < p_thres & beta_meth > beta_thres, "Significant Pos",
                     ifelse(FDR < p_thres & beta_meth < -beta_thres, "Significant Neg", "NS"))
      )
    results_anno$sig <- factor(results_anno$sig, levels=c("NS", "Significant Neg", "Significant Pos"))
    
    volcano <- ggplot(results_anno %>% filter(abs(beta_meth)<10), 
                      aes(x=beta_meth, y=neglog10p, color=sig)) +
      geom_point(alpha=0.7,size=1) +
      geom_hline(yintercept=-log10(p_thres), linetype="dashed", color="red") +
      geom_vline(xintercept=c(-beta_thres,beta_thres), linetype="dashed", color="black") +
      geom_text_repel(data=results_anno %>% 
                        # filter(abs(beta_meth)<10) %>% 
                        filter(beta_meth>0) %>%
                        filter(sig=="Significant Pos") %>% head(8),
                      aes(label=gene), size=3, max.overlaps=10,show.legend = FALSE) +
      geom_text_repel(data=results_anno %>% 
                        # filter(abs(beta_meth)<10) %>% 
                        filter(beta_meth<0) %>%
                        filter(sig=="Significant Neg") %>% head(8),
                      aes(label=gene), size=3, max.overlaps=10,show.legend = FALSE) +
      scale_color_manual(values=c("grey", "blue", "red"),
                         labels=c("NS", "Significant Neg", "Significant Pos")) +
      theme_minimal() +
      labs(x="Effect size (fixed-effect beta)", y="-log10(FDR)") #, title="Volcano plot")
    print(volcano)
    ggsave(paste0("results/volcano_plot_",sites, "_",model,".pdf"), volcano, width=8, height=5.5, device = cairo_pdf)
    
    # ============================================================
    # 5. 曼哈顿图
    # ============================================================
    
    library(dplyr)
    library(ggplot2)
    
    library(dplyr)
    library(ggplot2)
    
    # 准备数据（假设 results_anno 已经在环境里）
    manhattan_df <- results_anno %>%
      filter(chromosome_name %in% c(1:22, "X", "Y")) %>%
      mutate(
        chr = factor(chromosome_name, levels = c(as.character(1:22), "X", "Y")),
        start_position = as.numeric(start_position),   # 转为 double
        pval_meth = as.numeric(pval_meth)              # 确保是数值
      )
    
    # 每条染色体长度
    chr_info <- manhattan_df %>%
      group_by(chr) %>%
      summarise(chr_len = max(start_position, na.rm = TRUE)) %>%
      mutate(
        chr_start = lag(cumsum(chr_len), default = 0),
        chr_center = chr_start + chr_len / 2
      )
    
    # 把所有染色体串到一个连续坐标
    manhattan_df2 <- manhattan_df %>%
      inner_join(chr_info %>% dplyr::select(chr, chr_start), by = "chr") %>%
      mutate(pos = start_position + chr_start)
    
    sig_threshold <- 1e-5 #(0.01/1000)
    cutoff_line <- -log10(sig_threshold)
    
    # 选出显著基因（这里选择 p < 5e-8，你也可以改成 top N）
    top_hits <- manhattan_df2 %>%
      filter(pval_meth < 5e-8) %>%
      arrange(pval_meth) %>%
      distinct(gene, .keep_all = TRUE)   # 避免重复基因名
    
    # 画图
    manhattan <- ggplot(manhattan_df2,
                        aes(x = pos, y = -log10(pval_meth), color = as.factor(chr))) +
      geom_point(alpha = 0.6, size = 0.8) +
      geom_hline(yintercept = cutoff_line, color = "red", 
                 linetype = "dashed", size = 0.6) +
      geom_text_repel(
        data = top_hits,
        aes(label = gene),
        size = 2.5, color = "black",
        box.padding = 0.3,
        point.padding = 0.2,
        max.overlaps = Inf
      ) +
      scale_x_continuous(
        label = chr_info$chr,
        breaks = chr_info$chr_center
      ) +
      scale_color_manual(values = rep(c("steelblue", "darkorange"),
                                      length.out = nrow(chr_info))) +
      theme_bw() +
      theme(
        legend.position = "none",
        panel.border = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
      ) +
      labs(x = "Chromosome", y = "-log10(p)") #, title = "Manhattan plot")
    
    print(manhattan)
    
    ggsave(paste0("results/manhattan_plot_",sites,"_",model,".pdf"), manhattan, width=10, height=4, device = cairo_pdf)
    
  }
}

############################################################
#===========================================
###########################################################3
setwd("D:/multi-omics")

# ============================================================
# 2. 读入分析结果
# ============================================================
model_list <- c("gamm", "gam")
sites_list <- c("mCG", "mCH")
for(i in 1:length(model_list)){
  for(j in 1:length(sites_list)){
    model <- model_list[i]
    sites <- sites_list[j]
    
    if(model=="gamm"){
      if(sites=="mCG"){
        results_df <- read_csv("results/rna_methy_mCG_cis_gamm4_subset_gp.csv")
      } else if(sites=="mCH"){
        results_df <- read_csv("results/rna_methy_mCH_cis_gamm4_subset_gp.csv")
      }
      # 过滤收敛且残差正态的基因
      results_df <- results_df %>%
        filter(conv_code==0 & residual_p_value>0.05)
    }else if(model=="gam"){
      if(sites=="mCG"){
        results_df <- read_csv("results/rna_methy_mCG_cis_gam4_subset_gp.csv")
      } else if(sites=="mCH"){
        results_df <- read_csv("results/rna_methy_mCH_cis_gam4_subset_gp.csv")
      }
      results_df <- results_df %>% filter(converged==TRUE & residual_p_value>0.05)
    }
    
    # 确保gene列存在
    stopifnot("gene" %in% colnames(results_df))
    
    # ============================================================
    # 3. 基因注释 (biomaRt)
    # ============================================================
    # mart <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
    mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
    
    annot <- getBM(
      attributes=c("mgi_symbol", "chromosome_name", "start_position"),
      filters="mgi_symbol",
      values=results_df$gene, #|> toupper(),
      mart=mart
    )
    
    
    # 合并
    results_anno <- results_df %>%
      left_join(annot, by=c("gene"="mgi_symbol"))
    results_anno$FDR <- p.adjust(results_anno$pval_meth, method="BH")
    
    # ============================================================
    # 4. 火山图
    # ============================================================
    if(sites=="mCG"){
      beta_thres <- 0.02
    }else if(sites=="mCH"){
      beta_thres <- 0.2
    }
    p_thres <- 0.05
    
    results_anno <- results_anno %>%
      mutate(
        neglog10p = -log10(FDR),
        # sig = ifelse(FDR < p_thres & abs(beta_meth) > beta_thres, "Significant", "NS")
        sig = ifelse(FDR < p_thres & beta_meth > beta_thres, "Significant Pos",
                     ifelse(FDR < p_thres & beta_meth < -beta_thres, "Significant Neg", "NS"))
      )
    results_anno$sig <- factor(results_anno$sig, levels=c("NS", "Significant Neg", "Significant Pos"))
    
    volcano <- ggplot(results_anno %>% filter(abs(beta_meth)<10), 
                      aes(x=beta_meth, y=neglog10p, color=sig)) +
      geom_point(alpha=0.7,size=1) +
      geom_hline(yintercept=-log10(p_thres), linetype="dashed", color="red") +
      geom_vline(xintercept=c(-beta_thres,beta_thres), linetype="dashed", color="black") +
      geom_text_repel(data=results_anno %>% 
                        # filter(abs(beta_meth)<10) %>% 
                        filter(beta_meth>0) %>%
                        filter(sig=="Significant Pos") %>% head(8),
                      aes(label=gene), size=3, max.overlaps=10,show.legend = FALSE) +
      geom_text_repel(data=results_anno %>% 
                        # filter(abs(beta_meth)<10) %>% 
                        filter(beta_meth<0) %>%
                        filter(sig=="Significant Neg") %>% head(8),
                      aes(label=gene), size=3, max.overlaps=10,show.legend = FALSE) +
      scale_color_manual(values=c("grey", "blue", "red"),
                         labels=c("NS", "Significant Neg", "Significant Pos")) +
      theme_minimal() +
      labs(x="Effect size (fixed-effect beta)", y="-log10(FDR)") #, title="Volcano plot")
    print(volcano)
    ggsave(paste0("results/volcano_plot_",sites, "_",model,"_subset.pdf"), volcano, width=8, height=5.5, device = cairo_pdf)
    
    # ============================================================
    # 5. 曼哈顿图
    # ============================================================
    
    library(dplyr)
    library(ggplot2)
    
    library(dplyr)
    library(ggplot2)
    
    # 准备数据（假设 results_anno 已经在环境里）
    manhattan_df <- results_anno %>%
      filter(chromosome_name %in% c(1:22, "X", "Y")) %>%
      mutate(
        chr = factor(chromosome_name, levels = c(as.character(1:22), "X", "Y")),
        start_position = as.numeric(start_position),   # 转为 double
        pval_meth = as.numeric(pval_meth)              # 确保是数值
      )
    
    # 每条染色体长度
    chr_info <- manhattan_df %>%
      group_by(chr) %>%
      summarise(chr_len = max(start_position, na.rm = TRUE)) %>%
      mutate(
        chr_start = lag(cumsum(chr_len), default = 0),
        chr_center = chr_start + chr_len / 2
      )
    
    # 把所有染色体串到一个连续坐标
    manhattan_df2 <- manhattan_df %>%
      inner_join(chr_info %>% dplyr::select(chr, chr_start), by = "chr") %>%
      mutate(pos = start_position + chr_start)
    
    sig_threshold <- 5e-8
    cutoff_line <- -log10(sig_threshold)
    
    # 选出显著基因（这里选择 p < 5e-8，你也可以改成 top N）
    top_hits <- manhattan_df2 %>%
      filter(pval_meth < 5e-8) %>%
      arrange(pval_meth) %>%
      distinct(gene, .keep_all = TRUE)   # 避免重复基因名
    
    # 画图
    manhattan <- ggplot(manhattan_df2,
                        aes(x = pos, y = -log10(pval_meth), color = as.factor(chr))) +
      geom_point(alpha = 0.6, size = 0.8) +
      geom_hline(yintercept = cutoff_line, color = "red", 
                 linetype = "dashed", size = 0.6) +
      geom_text_repel(
        data = top_hits,
        aes(label = gene),
        size = 2.5, color = "black",
        box.padding = 0.3,
        point.padding = 0.2,
        max.overlaps = Inf
      ) +
      scale_x_continuous(
        label = chr_info$chr,
        breaks = chr_info$chr_center
      ) +
      scale_color_manual(values = rep(c("steelblue", "darkorange"),
                                      length.out = nrow(chr_info))) +
      theme_bw() +
      theme(
        legend.position = "none",
        panel.border = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
      ) +
      labs(x = "Chromosome", y = "-log10(p)") #, title = "Manhattan plot")
    
    print(manhattan)
    
    ggsave(paste0("results/manhattan_plot_",sites,"_",model,"_subset.pdf"), manhattan, width=10, height=5, device = cairo_pdf)
    
  }
}













#######################################################################################

# ============================================================
results_anno|>write_csv("results/rna_methy_cis_gamm4_anno.csv")

# ============================================================
# 6. 单基因空间可视化函数
# ============================================================
plot_gene_spatial <- function(gene, expr_df, meth_df, meta_df){
  df <- data.frame(
    expr = expr_df[[gene]],
    meth = meth_df[[gene]],
    x = meta_df$x,
    y = meta_df$y,
    layer = factor(meta_df$layer)
  ) %>% na.omit()
  
  p1 <- ggplot(df, aes(x=x, y=y, color=expr)) +
    geom_point(size=1) +
    scale_color_viridis_c() +
    theme_minimal() +
    labs(title=paste(gene, "Expression"))
  
  p2 <- ggplot(df, aes(x=x, y=y, color=meth)) +
    geom_point(size=1) +
    scale_color_viridis_c() +
    theme_minimal() +
    labs(title=paste(gene, "Methylation"))
  
  p3 <- ggplot(df, aes(x=meth, y=expr, color=layer)) +
    geom_point(size=1, alpha=0.7) +
    theme_minimal() +
    labs(title=paste(gene, "Expr vs Meth"), x="Methylation", y="Expression")
  
  return(p1 + p2 + p3)
}

# 举例：画前3个显著基因的空间图
expr_df <- read_csv("results/expr_df_pooled_matched.csv")
meth_df <- read_csv("results/meth_df_pooled_matched.csv")
meta_df <- read_csv("results/meta_df_pooled_matched.csv")

top_genes <- results_anno %>% arrange(pval_meth) %>% head(3) %>% pull(gene_ori)

for(g in top_genes){
  g_plot <- plot_gene_spatial(g, expr_df, meth_df, meta_df)
  ggsave(paste0("results/", g, "_spatial.png"), g_plot, width=10, height=4)
}

# subset layers with > 100 spots
layers <- names(table(meta_df$layer))[table(meta_df$layer)>=100]
meta_df <- meta_df %>% filter(layer %in% layers)
expr_df <- expr_df %>% filter(rna_spot %in% meta_df$rna_spot)
meth_df <- meth_df %>% filter(rna_spot %in% meta_df$rna_spot)
