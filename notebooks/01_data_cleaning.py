
# %%
import pandas as pd
import numpy as np
from pathlib import Path

# %%
project_root = Path("/Users/jaeseokhwang/value-ofpe-pytorch/value-opfe-pytorch")
data_path = project_root / "Data" / "Processed" / "Analysis_ready"



# Load the CSV files
dat_binded = pd.read_csv(data_path / "dat_binded.csv")
info_binded = pd.read_csv(data_path / "info_binded.csv")
n_table_anony = pd.read_csv(data_path / "n_table_anony.csv")

# %%
# Check dimension of the data
print(" dat_binded shape:", dat_binded.shape)
print(" info_binded shape:", info_binded.shape)
print(" n_table_anony shape:", n_table_anony.shape)

# %%
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# import statsmodels.api as sm
from econml.dml import CausalForestDML

# %%
