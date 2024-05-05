import numpy as np
import pandas as pd
import os

data_dirs = os.listdir('data')
data_frames = []
for folder in data_dirs:
    if os.path.isdir("data/{0:s}".format(folder)):
        file_name = 'data/{0:s}/wnl_coefficients.csv'.format(folder)
        df = pd.read_csv(file_name,index_col=0)
        # Convert data to complex
        df = df.applymap(lambda s: np.complex128(s))
        data_frames.append(df)

df = pd.concat(data_frames)
print(df.T)
df.to_csv('data/wnl_merged_data.csv')
