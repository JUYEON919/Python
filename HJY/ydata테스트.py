import os
import pandas as pd
from ydata_profiling import ProfileReport

file_path = 'dataset/diabetes.csv'
df = pd.read_csv(file_path)

print(df.info())