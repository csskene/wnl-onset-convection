"""
Read data from all csv files and merge them

Usage:
    read_data.py [options]

Options:
    --data_dir=<data_dir>   Location of saved data [default: data]
"""

import numpy as np
import pandas as pd
import os
from docopt import docopt

# Read arguments
args = docopt(__doc__)
data_dir = str(args['--data_dir'])
# Get directories in data folder
data_dirs = os.listdir(data_dir)

#Get individual dataframes
data_frames = []
for folder in data_dirs:
    try:
        if os.path.isdir("{0:s}/{1:s}".format(data_dir, folder)):
            file_name = '{0:s}/{1:s}/wnl_coefficients.csv'.format(data_dir, folder)
            df = pd.read_csv(file_name,index_col=0)
            # Convert data saved as a string to complex
            df = df.map(lambda s: np.complex128(s) if isinstance(s, str) else s)
            data_frames.append(df)
    except:
        print('No file {}'.format(folder))

# Merge data
df = pd.concat(data_frames)
print(df.T)
df.to_csv('data/wnl_merged_data.csv')
