### -- SCRIPT USED TO IMPORT SNP DATA

### --- import libraries

import pandas as pd
import os
from pandas_plink import read_plink1_bin
from pandas_plink import read_plink

### --- set file paths
folder = "2_SNP"
basename = "adni12GO_imputed"
bed = os.path.join(folder, basename + ".bed")
bim = os.path.join(folder, basename + ".bim")
fam = os.path.join(folder, basename + ".fam")

### --- read PLINK files
#G = read_plink1_bin(bed, bim, fam, verbose=True)
(bim, fam, bed) = read_plink(os.path.join(folder, basename))

print(bed.shape())
print(bed.compute())