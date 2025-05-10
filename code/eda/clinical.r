### -- RSCRIPT TO HELP WITH EDA FOR CLINICAL DATA

### -- LIBRARIES
library(tidyverse)
library(magrittr)
library(ggplot2)
library(glue)
library(skimr)
library(fastDummies)

### -- LOAD DATA

df <- read_csv("1_Clinical/ADNIMERGE_01Oct2024.csv", col_names = TRUE)
common_ids <- read_csv("common_ids.csv") %>% select(ID) %>% pull()## -- complete T1_MRI data (including tabular)

df %<>%
  filter(RID %in% common_ids)

print(head(df))

df %>%
  select(starts_with('DX')) %>%
  head()

df %>% 
  skim()

## -- subset to the demographic covariates and the response (DX_bl) variable. Remove rows with missing response
## -- we also convert variables to a relevant format

df_sub <-
  df %>%
    filter(VISCODE == "bl" & !is.na(DX_bl)) %>%
    select(RID, AGE, DX_bl, PTGENDER, PTETHCAT, PTMARRY, PTRACCAT, APOE4, PTEDUCAT) %>%
    mutate(across(c(DX_bl, PTGENDER, PTETHCAT, PTRACCAT, PTMARRY), as.factor)) %>%
    mutate(across(c(RID), as.character)) %>%
    mutate(across(c(AGE, APOE4, PTEDUCAT), as.numeric)) %>%
    mutate(APOE4_fac = factor(APOE4, levels = c("0", "1", "2"))) %>%
    mutate(ID = RID)

## -- DISEASE STATUS
df_sub %>%
  group_by(DX_bl) %>%
  summarise(count = n())

  ### -- combine EMCI and LMCI together, and combine SMC and CN to the CN category
df_sub %<>%
  mutate(DX_bl_v2 = 
          factor(
            case_when(
            DX_bl %in% c('EMCI', 'LMCI') ~ 'MCI',
            DX_bl == 'SMC' ~ 'CN',
            TRUE ~ DX_bl
            )
          )
        ) %>%
    select(-DX_bl)

## -- APOE4 GENE (MARGINAL AND ALSO BY DISEASE STATUS AT BL)

df_sub %>%
  group_by(APOE4, DX_bl_v2) %>%
  summarise(count = n())

  ### -- visualize this
plot_apoe4 <-
  df_sub %>%
    filter(!is.na(APOE4_fac) & !is.na(DX_bl_v2)) %>%
    count(APOE4_fac, DX_bl_v2) %>%
    ggplot(aes(x = droplevels(APOE4_fac), y = n, fill = DX_bl_v2)) +
      geom_bar(stat = "identity", position = position_dodge()) +
      labs(
        title = "APOE4 Gene and Diagnosis",
        x = "APOE4 Gene",
        y = "Count",
        fill = "Diagnosis"
      ) +
      theme_classic()
    
plot_apoe4 %>%
  ggsave(filename = "figures/apoe4_diag.png")


### -- APPLY ONE-HOT ENCODING TO CATEGORICAL

# Keep only factor columns for encoding
df_factor <- 
  df_sub[sapply(df_sub, is.factor)] %>%
    select(-DX_bl_v2)

# One-hot encode only the factor columns
encoded_factors <- dummy_cols(df_factor, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
encoded_factors <- encoded_factors[, !grepl("_NA$", names(encoded_factors))]

# Combine with non-factor columns
non_factors <- df_sub[, c(names(df_sub)[!sapply(df_sub, is.factor)], "DX_bl_v2")]
df_sub <- cbind(non_factors, encoded_factors)

### -- SAVE FINAL DATA
df_sub %>%
  write.csv(file = "1_Clinical/clinical_sub.csv", row.names = FALSE)

df_sub %>%
  saveRDS("1_Clinical/clinical_sub.RDS")