### -- SCRIPT TO SIMPLY PUT ALL TABULAR INFORMATION TOGETHER IN ONE SCRIPT

### -- LIBRARIES
library(tidyverse)
library(skimr)
library(magrittr)

### -- FIRST LOAD THEM ALL

demo <- readRDS("1_Clinical/clinical_sub.RDS")
global <- read_csv("5_T1_MRI/TabularFeature/Global_v2.csv")
roi_thick <- read_csv("5_T1_MRI/TabularFeature/ROI_Thick_v2.csv")
roi_vol <- read_csv("5_T1_MRI/TabularFeature/ROI_VolumeInVoxels_v2.csv")

### -- MAKE SURE ID DTYPES MATCH
global$ID <- as.character(global$ID)
roi_thick$ID <- as.character(roi_thick$ID)
roi_vol$ID <- as.character(roi_vol$ID)

### -- PERFORM THE MERGE

df_list <- list(demo, global, roi_thick, roi_vol)

# Merge by ID using inner_join (only common IDs)
df_tabular <- reduce(df_list, ~ inner_join(.x, .y, by = "ID", suffix = c("_thick", "_vol")))
df_tabular %>% head()
df_tabular %>% dim()

### -- STANDARDIZE ALL CONTINUOUS VARIABLES

exclude_cols <- c("APOE4", "PearsonCorr")

df_tabular %<>%
  mutate(across(
    where(is.numeric) & 
      !all_of(exclude_cols) & 
      where(~ n_distinct(., na.rm = TRUE) > 2),
    ~ scale(.)[,1]
  ))

# PRIOR TO EXPORT, BEST TO TAKE A LOOK

missing_thres <- 3.5
missing_pct <- colMeans(is.na(df_tabular)) * 100

# Keep only columns with <= missing_thres
df_tabular <- df_tabular[, missing_pct <= missing_thres]

# Keep only rows with <= missing_thres
df_tabular <- df_tabular[rowMeans(is.na(df_tabular)) == 0, ]


### -- MAKE ADJUSTMENTS TO COLUMN NAMES
colnames(df_tabular) <- gsub(" ", "_", colnames(df_tabular))
colnames(df_tabular) <- gsub("/", "_", colnames(df_tabular))


### -- EXPORT TO A CSV
df_tabular %>%
  skim()

df_tabular %>%
  write_csv(file = "tabular.csv")
