import pandas as pd
import shutil

df = pd.read_csv('raw_data.csv')
for index, row in df.iterrows():
    if row['identity'] == 7 or row['identity'] == 8:
        shutil.copy2(row['path'], f'data/test/{row["expression"]}/{row["path"].split("/")[-1]}.JPG')
    else:
        shutil.copy2(row['path'], f'data/train/{row["expression"]}/{row["path"].split("/")[-1]}.JPG')