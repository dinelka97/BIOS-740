## -- import modules

import pandas as pd
import polars as pl
import os
import random
import matplotlib.pyplot as plt

folder = "4_Protein"
file_ref = "ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv"
file_protein = "CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv"

pl.Config.set_tbl_cols(-1)

path = os.path.join(folder, file_ref)
df_ref = pl.read_csv(path)

print(df_ref.head(n=20))
print(df_ref.shape)
print(df_ref.columns)

pl.Config.set_tbl_cols(20)

path = os.path.join(folder, file_protein)
df_protein = pl.read_csv(path)
print(df_protein.head(n=20))
print(df_protein.shape)

## -- is all our protein data from BL?

print(df_protein.select(pl.col('VISCODE2').unique()))
value_counts = df_protein.select(
    pl.col('VISCODE2').value_counts()
)
print(value_counts)

## -- how many unique RIDs do we have?

print(df_protein.select(pl.col('ExtIdentifier').n_unique()))





## -- visualize a few proteins

n_prot = 20
cols_prot = df_protein.columns[7:]
selected_cols = random.sample(cols_prot, n_prot)

df_pd = df_protein.select(selected_cols).to_pandas()

# Plot in a 5x2 grid
fig, axes = plt.subplots(int(n_prot/2), 2, figsize=(12, 16))
axes = axes.flatten()

for i, col in enumerate(selected_cols):
    axes[i].hist(df_pd[col].dropna(), bins=30, color="skyblue", edgecolor="black")
    axes[i].set_title(col)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()

plt.savefig("figures/protein_hist.png", dpi=300)
plt.close()


