import pandas as pd
import shutil

df = pd.read_csv('raw_data.csv')
# print(df["expression"].unique())
# print(df)

for index, row in df.iterrows():
    # print(row['path'], row['expression'], f'data/all/{row["expression"]}/{row["path"].split("/")[-1]}')
    shutil.copy2(row['path'], f'data/all/{row["expression"]}/{row["path"].split("/")[-1]}.JPG')