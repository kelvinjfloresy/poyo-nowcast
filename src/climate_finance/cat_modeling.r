# ==============================================================================
# POYO-NOWCAST: Módulo 4 - Finanzas Climáticas, Cat Modeling y Seguros Paramétricos
# Estándar Regulatorio: Solvencia II (2009/138/CE), EIOPA ORSA & Reaseguro XoL
# Autor: Kelvin Jesus Flores Yarihuaman
# Licencia: Open Science (CC BY 4.0)
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(evd)
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(patchwork)
  library(tibble)
})

# Asegurar directorios de trabajo y salida
if (basename(getwd()) == "climate_finance") setwd("../..")
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------------------
# 1. INGESTA Y VALIDACIÓN DEL CONTRATO COLUMNAR PARQUET
# ------------------------------------------------------------------------------
parquet_path <- "data/processed/flood_damage_matrix.parquet"

if (!file.exists(parquet_path)) {
  stop(paste("Contrato Parquet ausente:", parquet_path,
             "\n- Ejecute previamente el integrador 'UnifiedDamageMatrixIntegrator'."))
}

cat("[CAT MODELING] Leyendo matriz de daños unificada...\n")
damage_data <- arrow::read_parquet(parquet_path)

if ("municipality" %in% colnames(damage_data)) {
  damage_data$municipality <- as.character(damage_data$municipality)
} else {
  damage_data$municipality <- "Horta_Sud"
}

total_exposure_eur <- sum(damage_data$asset_value_eur, na.rm = TRUE)
ground_up_loss_eur <- sum(damage_data$economic_loss_eur, na.rm = TRUE)

cat(sprintf("Registros procesados: %d activos catastrales.\n", nrow(damage_data)))
cat(sprintf("Exposición total de cartera: %.2f M€ | Daño bruto (Ground-Up): %.2f M€\n", 
            total_exposure_eur / 1e6, ground_up_loss_eur / 1e6))

# ------------------------------------------------------------------------------
# 2. ANÁLISIS DE CONCENTRACIÓN ESPACIAL POR TÉRMINO MUNICIPAL
# ------------------------------------------------------------------------------
municipal_risk <- damage_data %>%
  group_by(municipality) %>%
  summarise(
    n_parcels       = n(),
    exposure_m_eur  = sum(asset_value_eur, na.rm = TRUE) / 1e6,
    loss_m_eur      = sum(economic_loss_eur, na.rm = TRUE) / 1e6,
    destruction_pct = (sum(economic_loss_eur, na.rm = TRUE) / sum(asset_value_eur, na.rm = TRUE)) * 100,
    collapse_count  = sum(structural_collapse, na.rm = TRUE),
    mean_depth_m    = mean(max_depth_m, na.rm = TRUE),
    .groups         = "drop"
  ) %>%
  arrange(desc(loss_m_eur))

cat("\n[DISTRIBUCIÓN GEOGRÁFICA DEL IMPACTO]\n")
print(as.data.frame(municipal_risk), row.names = FALSE)

# ------------------------------------------------------------------------------
# 3. VALORES EXTREMOS (GEV): LÍNEA BASE VS. ESTRÉS CLIMÁTICO (EIOPA ORSA)
# ------------------------------------------------------------------------------
set.seed(46)
historical_q_peaks <- c(
  180, 250, 310, 420, 195, 280, 520, 340, 290, 610,
  220, 380, 490, 270, 330, 850, 410, 360, 290, 1950
)

gev_fit <- fgev(historical_q_peaks)
mu_base    <- as.numeric(gev_fit$estimate["loc"])
sigma_base <- as.numeric(gev_fit$estimate["scale"])
xi_base    <- as.numeric(gev_fit$estimate["shape"])
vcov_mat   <- gev_fit$var.cov

climate_stress_factor <- 1.18
mu_stressed    <- mu_base * climate_stress_factor
sigma_stressed <- sigma_base * climate_stress_factor
xi_stressed    <- xi_base

compute_return_level <- function(T_val, mu, sigma, xi, vcov_m = NULL) {
  p <- 1 - (1 / T_val)
  y_p <- -log(p)
  
  if (abs(xi) > 1e-5) {
    z_T <- mu - (sigma / xi) * (1 - y_p^(-xi))
  } else {
    z_T <- mu - sigma * log(y_p)
  }
  
  se_zT <- 0.0
  if (!is.null(vcov_m)) {
    grad <- if (abs(xi) > 1e-5) {
      c(1.0, -(1 / xi) * (1 - y_p^(-xi)),
        (sigma / (xi^2)) * (1 - y_p^(-xi)) - (sigma / xi) * (y_p^(-xi)) * log(y_p))
    } else {
      c(1.0, -log(y_p), 0.5 * sigma * (log(y_p))^2)
    }
    se_zT <- sqrt(max(0, as.numeric(t(grad) %*% vcov_m %*% grad)))
  }
  
  z_num <- as.numeric(z_T)
  se_num <- as.numeric(se_zT)
  c(flow = z_num, se = se_num, low = max(0, z_num - 1.96 * se_num), high = z_num + 1.96 * se_num)
}

return_periods <- c(seq(2, 50, by = 2), seq(55, 200, by = 5), seq(210, 500, by = 10))

ep_curve_list <- lapply(return_periods, function(T_val) {
  rl_base     <- compute_return_level(T_val, mu_base, sigma_base, xi_base, vcov_mat)
  rl_stressed <- compute_return_level(T_val, mu_stressed, sigma_stressed, xi_stressed)
  
  flow_b <- max(0, rl_base[["flow"]])
  flow_s <- max(0, rl_stressed[["flow"]])
  low_b  <- max(0, rl_base[["low"]])
  high_b <- max(0, rl_base[["high"]])
  
  loss_base     <- ground_up_loss_eur * min(1.6, (flow_b / 1950.0)^1.35)
  loss_base_low <- ground_up_loss_eur * min(1.6, (low_b  / 1950.0)^1.35)
  loss_base_hi  <- ground_up_loss_eur * min(1.6, (high_b / 1950.0)^1.35)
  loss_stressed <- ground_up_loss_eur * min(1.6, (flow_s / 1950.0)^1.35)
  
  data.frame(
    Return_Period    = T_val,
    Exceedance_Prob  = 1 / T_val,
    Loss_Base_M      = loss_base / 1e6,
    Loss_Base_Low_M  = loss_base_low / 1e6,
    Loss_Base_High_M = loss_base_hi / 1e6,
    Loss_Stressed_M  = loss_stressed / 1e6
  )
})

ep_df <- do.call(rbind, ep_curve_list)

calc_aal <- function(p_vec, l_vec, xi) {
  valid <- !is.na(p_vec) & !is.na(l_vec)
  p_vec <- p_vec[valid]
  l_vec <- l_vec[valid]
  dp <- abs(diff(p_vec))
  mid_l <- (head(l_vec, -1) + tail(l_vec, -1)) / 2
  tail_denom <- if (as.numeric(xi) < 1) max(0.1, 1 - as.numeric(xi)) else 0.5
  tail_extrap <- (min(p_vec) * max(l_vec)) / tail_denom
  as.numeric(sum(dp * mid_l) + tail_extrap)
}

aal_base_m     <- calc_aal(ep_df$Exceedance_Prob, ep_df$Loss_Base_M, xi_base)
aal_stressed_m <- calc_aal(ep_df$Exceedance_Prob, ep_df$Loss_Stressed_M, xi_stressed)

var_995_base     <- ep_df$Loss_Base_M[which.min(abs(ep_df$Return_Period - 200))]
var_995_stressed <- ep_df$Loss_Stressed_M[which.min(abs(ep_df$Return_Period - 200))]

scr_base     <- max(0, var_995_base - aal_base_m)
scr_stressed <- max(0, var_995_stressed - aal_stressed_m)

cost_of_capital_rate <- 0.06
risk_margin_base     <- cost_of_capital_rate * scr_base
risk_margin_stressed <- cost_of_capital_rate * scr_stressed

# ------------------------------------------------------------------------------
# 4. ESTRUCTURACIÓN DE TRATADO DE REASEGURO (EXCESS OF LOSS - XoL)
# ------------------------------------------------------------------------------
xol_attachment_m <- 60.0
xol_limit_m      <- 120.0

apply_xol <- function(gross_loss, attach, limit) {
  pmin(limit, pmax(0, gross_loss - attach))
}

ep_df <- ep_df %>%
  mutate(
    Ceded_Loss_M   = apply_xol(Loss_Base_M, xol_attachment_m, xol_limit_m),
    Net_Retained_M = Loss_Base_M - Ceded_Loss_M
  )

aal_ceded_m <- calc_aal(ep_df$Exceedance_Prob, ep_df$Ceded_Loss_M, xi_base)
aal_net_m   <- calc_aal(ep_df$Exceedance_Prob, ep_df$Net_Retained_M, xi_base)

var_995_net <- ep_df$Net_Retained_M[which.min(abs(ep_df$Return_Period - 200))]
scr_net_m   <- max(0, var_995_net - aal_net_m)

# ------------------------------------------------------------------------------
# 5. PRICING ACTUARIAL DE BONO CATASTRÓFICO / COBERTURA PARAMÉTRICA
# ------------------------------------------------------------------------------
cat_bond_capacity_m <- 80.0
q_attach <- 1000.0; q_exhaust <- 1800.0
insar_attach <- 0.03; insar_exhaust <- 0.12

n_sims <- 10000
sim_flows <- rgev(n_sims, loc = mu_base, scale = sigma_base, shape = xi_base)
sim_dpm   <- pmin(0.25, pmax(0.0, 0.02 + 0.00006 * sim_flows + rnorm(n_sims, 0, 0.02)))

payout_sim <- with(data.frame(Q = sim_flows, I = sim_dpm), {
  fq <- pmin(1.0, pmax(0.0, (Q - q_attach) / (q_exhaust - q_attach)))
  fi <- pmin(1.0, pmax(0.0, (I - insar_attach) / (insar_exhaust - insar_attach)))
  cat_bond_capacity_m * sqrt(fq * fi)
})

expected_payout_m <- mean(payout_sim)
var_payout_995_m  <- as.numeric(quantile(payout_sim, 0.995))
cat_bond_scr_m    <- max(0, var_payout_995_m - expected_payout_m)

capital_cost_load    <- cost_of_capital_rate * cat_bond_scr_m
expense_load         <- 0.10 * (expected_payout_m + capital_cost_load)
commercial_premium_m <- expected_payout_m + capital_cost_load + expense_load
rate_on_line         <- (commercial_premium_m / cat_bond_capacity_m) * 100

cat("\n================ BALANCE REGULATORIO Y REASEGURO XoL ================\n")
cat(sprintf("  • AAL Bruta:                    %6.2f M€  |  AAL Neta (Post-XoL): %6.2f M€\n", aal_base_m, aal_net_m))
cat(sprintf("  • SCR Solvencia II Bruto:       %6.2f M€  |  SCR Neto (Post-XoL): %6.2f M€\n", scr_base, scr_net_m))
cat(sprintf("  • Alivio de Capital vía XoL:    %6.2f M€  (Reducción: %.1f%%)\n", 
            scr_base - scr_net_m, ((scr_base - scr_net_m) / max(0.1, scr_base)) * 100))
cat(sprintf("  • Margen de Riesgo (Risk Margin): %4.2f M€  |  Bajo Clima Estresado: %4.2f M€\n", 
            risk_margin_base, risk_margin_stressed))

cat("\n================ PRICING ACTUARIAL CAT BOND / PARAMÉTRICO ================\n")
cat(sprintf("  • Capacidad del Bono:           %6.2f M€\n", cat_bond_capacity_m))
cat(sprintf("  • Pérdida Esperada (Expected):  %6.2f M€ (%.2f%%)\n", expected_payout_m, (expected_payout_m/cat_bond_capacity_m)*100))
cat(sprintf("  • Cargo Coste de Capital (CoC): %6.2f M€\n", capital_cost_load))
cat(sprintf("  • Prima Comercial Total:        %6.2f M€\n", commercial_premium_m))
cat(sprintf("  • Rate-on-Line (ROL):           %6.2f%%\n", rate_on_line))

# ------------------------------------------------------------------------------
# A. DEMAND SURGE (INFLACIÓN POR SATURACIÓN DE RECONSTRUCCIÓN)
# ------------------------------------------------------------------------------
kappa_surge <- 0.25
gamma_surge <- 0.60

destruction_ratio <- ground_up_loss_eur / total_exposure_eur
demand_surge_factor <- 1.0 + kappa_surge * (destruction_ratio ^ gamma_surge)
loss_with_surge_eur <- ground_up_loss_eur * demand_surge_factor

cat(sprintf("\n[DEMAND SURGE] Factor de sobrecoste por demanda: x%.3f (+%.1f%%)\n",
            demand_surge_factor, (demand_surge_factor - 1.0) * 100))
cat(sprintf("  • Pérdida con Demand Surge: %.2f M€ (Sobrecoste: +%.2f M€)\n",
            loss_with_surge_eur / 1e6, (loss_with_surge_eur - ground_up_loss_eur) / 1e6))

# ------------------------------------------------------------------------------
# B. PÉRDIDAS INDIRECTAS (BUSINESS INTERRUPTION INDEXADO AL TTI DE REDES)
# ------------------------------------------------------------------------------
vab_diario_horta_sud_eur <- 12.5e6

dias_aislamiento_medio <- if ("time_to_isolation_min" %in% colnames(damage_data)) {
  mean_tti <- mean(damage_data$time_to_isolation_min, na.rm = TRUE)
  if (is.finite(mean_tti)) pmax(3.0, (mean_tti / 60) * 2.5) else 7.0
} else {
  7.0
}

loss_bi_eur <- vab_diario_horta_sud_eur * dias_aislamiento_medio * 0.65
total_economic_impact_eur <- loss_with_surge_eur + loss_bi_eur

cat(sprintf("[LUCRO CESANTE / BI] Días promedio de parálisis vial (TTI): %.1f días\n", dias_aislamiento_medio))
cat(sprintf("  • Pérdida Indirecta (Business Interruption):  %6.2f M€\n", loss_bi_eur / 1e6))
cat(sprintf("  • Impacto Económico Consolidado (PD + BI):   %6.2f M€\n", total_economic_impact_eur / 1e6))

# ------------------------------------------------------------------------------
# C. ANÁLISIS DE RIESGO DE BASE (MONTE CARLO BASIS RISK)
# ------------------------------------------------------------------------------
correlation_payout_loss <- cor(payout_sim, pmin(cat_bond_capacity_m, sim_flows * 0.05))
prob_type_ii_risk <- mean(sim_flows > 1300 & payout_sim == 0)
prob_type_i_risk  <- mean(sim_flows < 900 & payout_sim > 30)

cat("\n[AUDITORÍA DE RIESGO DE BASE (BASIS RISK)]\n")
cat(sprintf("  • Coeficiente de Correlación (Payout vs Ground-Up): %.3f\n", correlation_payout_loss))
cat(sprintf("  • Probabilidad Error Tipo I  (Riesgo Inversor):   %.2f%%\n", prob_type_i_risk * 100))
cat(sprintf("  • Probabilidad Error Tipo II (Riesgo Asegurado):  %.2f%%\n", prob_type_ii_risk * 100))

# ------------------------------------------------------------------------------
# D. ESTRUCTURACIÓN COMBINADA: CONSORCIO (CCS) VS FACILIDAD PARAMÉTRICA
# ------------------------------------------------------------------------------
total_payout_m_eur <- expected_payout_m

cobertura_ccs_eur       <- ground_up_loss_eur * 0.72
franquicia_no_cubierta  <- ground_up_loss_eur * 0.13
infraestructura_publica <- ground_up_loss_eur * 0.15

liquidity_gap_eur <- franquicia_no_cubierta + infraestructura_publica
gap_cubierto_pct  <- min(100.0, (total_payout_m_eur / (liquidity_gap_eur / 1e6)) * 100)

cat("\n[ARQUITECTURA DE FINANCIACIÓN DEL DESASTRE]\n")
cat(sprintf("  • Absorción Estimada CCS (Indemnización ordinaria): %6.2f M€ (72.0%%)\n", cobertura_ccs_eur / 1e6))
cat(sprintf("  • Vacío de Cobertura (Gap Municipal / Franquicias): %6.2f M€ (28.0%%)\n", liquidity_gap_eur / 1e6))
cat(sprintf("  • Cobertura Inmediata vía Paramétrico:              %6.2f M€ (%.1f%% del Gap)\n", 
            total_payout_m_eur, gap_cubierto_pct))

# ------------------------------------------------------------------------------
# E. TARIFICACIÓN DE PRIMA PURA POR MUNICIPIO (TASA POR MIL ‰)
# ------------------------------------------------------------------------------
municipal_tariff <- municipal_risk %>%
  mutate(
    aal_local_m_eur = loss_m_eur * (aal_base_m / (ground_up_loss_eur / 1e6)),
    pure_rate_per_thousand = (aal_local_m_eur / exposure_m_eur) * 1000.0
  ) %>%
  select(municipality, exposure_m_eur, loss_m_eur, aal_local_m_eur, pure_rate_per_thousand)

cat("\n[TARIFARIO MUNICIPAL DE RIESGO (TASA PURA ‰)]\n")
print(as.data.frame(municipal_tariff), row.names = FALSE)

# ------------------------------------------------------------------------------
# 6. EXPORTACIÓN DEL REPORTE REGULATORIO QRT (EIOPA NATCAT TEMPLATE)
# ------------------------------------------------------------------------------
pml_100 <- ep_df$Loss_Base_M[which.min(abs(ep_df$Return_Period - 100))]

solvency_summary <- tibble(
  Métrica_Regulatoria = c(
    "Exposición Bruta del Portfolio (Total Insured Value)",
    "Pérdida Anual Esperada Bruta (Gross AAL)",
    "Pérdida Anual Esperada Neta (Net AAL)",
    "PML 100 Años (VaR 99.0%)",
    "PML 200 Años (VaR 99.5% - Línea Base)",
    "PML 200 Años (VaR 99.5% - Clima Estresado ORSA)",
    "Capital de Solvencia Obligatorio Bruto (Gross SCR)",
    "Capital de Solvencia Obligatorio Neto (Net SCR)",
    "Margen de Riesgo Regulatorio (Risk Margin - Art. 77)",
    "Provisión Técnica Total (Gross Technical Provisions = AAL + RM)",
    "Prima Comercial Cat Bond / Cobertura Paramétrica"
  ),
  Importe_M_EUR = c(
    total_exposure_eur / 1e6, aal_base_m, aal_net_m,
    pml_100, var_995_base, var_995_stressed,
    scr_base, scr_net_m, risk_margin_base,
    aal_base_m + risk_margin_base, commercial_premium_m
  )
)

write.csv(solvency_summary, "data/processed/solvency_ii_qrt_summary.csv", row.names = FALSE)
cat("[REPORTE] Plantilla QRT exportada en: data/processed/solvency_ii_qrt_summary.csv\n")

# ------------------------------------------------------------------------------
# 7. DASHBOARD GRÁFICO INTEGRADO A 300 DPI
# ------------------------------------------------------------------------------
theme_actuarial <- theme_minimal(base_size = 9) +
  theme(
    plot.title = element_text(face = "bold", size = 10, hjust = 0),
    plot.subtitle = element_text(color = "#444444", size = 8),
    axis.title = element_text(face = "bold", size = 8.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#d9d9d9", fill = NA, linewidth = 0.5)
  )

p1 <- ggplot(ep_df, aes(x = Return_Period)) +
  geom_ribbon(aes(ymin = Loss_Base_Low_M, ymax = Loss_Base_High_M), fill = "#2166ac", alpha = 0.12) +
  geom_line(aes(y = Loss_Base_M, color = "Base Bruta (Ground-Up)"), linewidth = 1.0) +
  geom_line(aes(y = Loss_Stressed_M, color = "Estrés Climático (ORSA +18%)"), linewidth = 0.9, linetype = "dashed") +
  geom_line(aes(y = Net_Retained_M, color = "Retención Neta (Post-XoL)"), linewidth = 1.0) +
  geom_vline(xintercept = 200, linetype = "dotted", color = "#252525") +
  scale_color_manual(name = "Estructura", values = c("Base Bruta (Ground-Up)" = "#2166ac", 
                                                     "Estrés Climático (ORSA +18%)" = "#b2182b", 
                                                     "Retención Neta (Post-XoL)" = "#238b45")) +
  scale_x_continuous(breaks = c(2, 50, 100, 200, 300, 400, 500)) +
  scale_y_continuous(labels = dollar_format(suffix = " M€", prefix = "")) +
  labs(title = "Curva de Excedencia de Probabilidad y Tratado XoL",
       subtitle = "Modelado estocástico con incertidumbre paramétrica y estrés climático",
       x = "Periodo de Retorno T (Años)", y = "Pérdida (M€)") +
  theme_actuarial +
  theme(legend.position = "bottom")

p2 <- ggplot(head(municipal_risk, 6), aes(x = reorder(municipality, loss_m_eur), y = loss_m_eur)) +
  geom_col(fill = "#e76f51", width = 0.65) +
  geom_text(aes(label = paste0(round(loss_m_eur, 1), " M€")), hjust = -0.1, size = 2.7, fontface = "bold") +
  coord_flip(ylim = c(0, max(municipal_risk$loss_m_eur) * 1.25)) +
  scale_y_continuous(labels = dollar_format(suffix = " M€", prefix = "")) +
  labs(title = "Concentración Municipal de Pérdidas",
       subtitle = "Términos con mayor destrucción directa en la Horta Sud",
       x = NULL, y = "Pérdida Estimada (M€)") +
  theme_actuarial

dashboard_cat <- (p1 + p2) + plot_annotation(
  title = "POYO-NOWCAST: MÓDULO INTEGRADO DE FINANZAS DEL CLIMA & SOLVENCIA II",
  subtitle = "Evaluación Pericial Actuarial, Análisis de Cartera y Transferencia de Riesgos (Directiva 2009/138/CE)",
  theme = theme(plot.title = element_text(face = "bold", size = 12),
                plot.subtitle = element_text(size = 9, color = "#333333"))
)

output_fig_path <- "data/processed/cat_modeling_solvency_ii_institutional.png"
ggsave(output_fig_path, dashboard_cat, width = 11.5, height = 5.5, dpi = 300)
cat(sprintf("[GRÁFICOS] Panel institucional exportado a 300 DPI en: %s\n", output_fig_path))