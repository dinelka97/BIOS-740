library(tidyverse)

folder <- "2_SNP"
file_name <- "adni12GO_imputed.bed"
path <- file.path(folder, file_name)

print(path)

df <- read.table(path, nrows = 5, header = FALSE, sep="\t",stringsAsFactors=FALSE, quote="")
print(df)