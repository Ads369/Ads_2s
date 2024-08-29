# %% case: Imports
import pandas as pd
from pathlib import Path

# %% case: Load Data
# load A_Z_Handwritten_Data
csv_path = Path('assets/A_Z_Handwritten_Data.csv')
data_set = pd.read_csv(csv_path)
print(data_set.shape)
