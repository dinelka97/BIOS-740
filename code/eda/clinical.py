### -- EDA ON CLINICAL DATA

### -- It's important to get a better understanding of our clinical data prior to using it in our analysis

### -- LOAD LIBRARIES

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

## -- folder path
folder = "1_Clinical"
file_name = "ADNIMERGE_01Oct2024.csv"
path = os.path.join(folder, file_name)

## -- read clinical data

df_clinical = pd.read_csv(path)

print("Printing the first 25 rows and 15 columns")
print(df_clinical.iloc[:25, :15])

print("Variables")
vars = list(df_clinical.columns)
print(vars)

print("Shape of the Clinical Dataset")
print(df_clinical.shape)
print(df_clinical['PTID'].nunique())

print(df_clinical['APOE4'].nunique())

print(df_clinical.loc[:25, df_clinical.columns.str.startswith(('DX', 'VISCODE', 'PTID'))])

## -- how many unique IDs (subjects) do we have in the clinical data?
unique_rids = df_clinical['RID'].unique()

# Count unique RIDs
count = len(unique_rids)

with open("eda/unique_id.txt", "w") as f:
    f.write(f"Unique RID count: {count}\n")

## -- how many unique DX?

print(df_clinical['DX'].nunique())

## -- AGE DISTRIBUTION BY SEX
  ### -- histogram plotting

# Set Seaborn style
sns.set(style="whitegrid")

# Create histogram
plt.figure(figsize=(8, 5))
sns.histplot(data=df_clinical, x='AGE', hue='PTGENDER', multiple='stack', palette='Set2', bins=10)

# Add kernel density
sns.kdeplot(data=df_clinical, x='AGE', hue='PTGENDER', common_norm=False, palette='Set2', linewidth=2)

# Titles and labels
plt.title("Age Distribution by Sex", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# Save figure
plt.savefig("figures/age_sex_hist.png", dpi=300, bbox_inches='tight')  # high-quality image
plt.close() ## -- close plot

