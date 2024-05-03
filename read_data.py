import numpy as np
import pandas as pd

file_name = 'data/Ekman_0.001_Prandtl_1_beta_0.35/wnl_coefficients.csv'
df = pd.read_csv(file_name,index_col=0)
# Convert data to complex
df = df.applymap(lambda s: np.complex128(s))

print(df)

