### -- PRIOR TO DL, JUST SOME PRELIM TRADITIONAL METHODS TO GET STARTED (ON DEMOGRAPHICS)

### -- LIBRARIES

library(tidyverse)
library(glue)
library(ggplot2)
library(nnet)
library(caret)
library(magrittr)

### -- LOAD DATA

df <- read_csv("tabular.csv")
str(df)

df %<>%
  select(-c("RID", "ID", "APOE4_fac_1", "APOE4_fac_2"))

colnames(df) <- gsub("^([0-9])", "prefix_\\1", colnames(df))

### -- TRAIN TEST SPLIT
set.seed(123)  # for reproducibility

# Create a 70% training index
train_index <- createDataPartition(df$DX_bl_v2, p = 0.7, list = FALSE)

# Split the data
train_data <- df[train_index, ]
test_data  <- df[-train_index, ]

### -- RUN A MULTINOMIAL CLASSIFICATION MODEL
covariates <- setdiff(names(df), "DX_bl_v2")
formula <- as.formula(paste("DX_bl_v2 ~", paste(covariates, collapse = " + ")))

model <- multinom(formula, data = train_data)
summary(model)

### -- MAKE PREDICTIONS

predicted_classes <- predict(model, newdata = test_data, type = "class")

### -- MODEL EVALUATIONS

# Compare predicted classes with actual classes in the test set
truth <- factor(test_data$DX_bl_v2)
conf_matrix <- confusionMatrix(predicted_classes, truth)
print(conf_matrix)

### -- NOTES/COMMENTS

#### -- the recall is pretty bad. We can only identify 24% of our actual AD cases correctly. 
