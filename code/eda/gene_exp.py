### -- DOCUMENTATION

## -- This script serves the purpose of importing and analysing the gene expression data prior to using it downstream
## -- The data used in this script can be found in the '3_GeneExpression' folder

### -- IMPORT LIBRARIES

import pandas as pd
import polars as pl


### -- LOAD GENE EXPRESSION DATA

df_ge = pd.read_csv("3_GeneExpression/ADNI_Gene_Expression_Profile.csv")
print(df_ge.head(n=25))


## -- gene expression
folder = "3_GeneExpression"
file_name = "ADNI_Gene_Expression_Profile.csv"
path = os.path.join(folder, file_name)

df_ge = pd.read_csv(path)
print(df_ge.head(n=25))


